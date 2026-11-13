"""Writes short info cards about what was just said, using a local Ollama model."""
import contextlib
import itertools
import re
from collections import deque
from dataclasses import dataclass

import ollama

KEEP_LOADED_FOR = "1h"
OPTIONS = {"temperature": 0, "presence_penalty": 0, "num_ctx": 2048, "num_predict": 200}

SYSTEM_PROMPT = """You write info cards for someone listening to a live conversation.

You get the latest words of the conversation in <latest>. Write a card for each notable thing or idea mentioned there: people, companies, products, places, events, works, technologies, scientific terms, concepts, jargon and acronyms.

One card per line, in exactly this format:
Title | Kind | Body

Title: the proper name, spelled correctly, 1 to 4 words.
Kind: one word, such as Person, Company, Product, Place, Event, Technology, Science, Concept, History or Art.
Body: at most 20 words of useful facts from your own knowledge: what it is and why it matters. Do not repeat what was said.

Rules:
- Only things mentioned in <latest>. <earlier> is only there to help you understand it.
- Never write a card for anything in <covered>: those cards already exist.
- Ordinary words used in a technical or specialist sense count too, such as agents, tools or sub agents in AI.
- Skip everyday words, small talk, opinions, and private people such as friends, family or colleagues mentioned by first name.
- The words come from speech recognition, so names can come out misspelled or misheard. Use the context to work out what was meant, and write the card under its correct name: in a talk about web APIs, "jay son" is JSON. Skip only words you cannot match to a real thing with confidence; never invent one.
- If nothing qualifies, write only NONE.

Example:
<covered>Kubernetes</covered>
<earlier>We moved our app to Kubernetes last year.</earlier>
<latest>Kubernetes was fine, but then Jensen Hwang showed the new Blackwell chips at GTC and</latest>
Jensen Huang | Person | Co-founder and CEO of Nvidia since 1993, who led its shift from gaming graphics to AI computing.
Blackwell | Technology | Nvidia's GPU architecture announced in 2024, built to train and run large AI models.
GTC | Event | Nvidia's yearly GPU Technology Conference, where it unveils new chips and AI software.

Example:
<covered>Kubernetes, Jensen Huang, Blackwell, GTC</covered>
<earlier>Kubernetes was fine, but then Jensen Huang showed the new Blackwell chips at GTC.</earlier>
<latest>Dana from my team wants us on Postgres, but Omer says the mobble deck rollout comes first.</latest>
PostgreSQL | Technology | Open-source relational database known for reliability, standard SQL and extensions such as PostGIS.

Example:
<covered>Kubernetes, Jensen Huang, Blackwell, GTC, PostgreSQL</covered>
<earlier>Dana from my team wants us on Postgres, but Omer says the mobble deck rollout comes first.</earlier>
<latest>Haha, yeah, I told Noa about it over lunch and she loved it.</latest>
NONE"""

CLAIM_CHECK = """Now fact-check only the sentence in <check>, instead of writing info cards.
The sentence comes from speech recognition, so names can be misspelled and words misheard. That is not a factual error: never correct spelling or wording, only facts.
First write one line with your verdict:
"Verdict: wrong" if it states something factually wrong, such as a wrong number, date or record, or a common myth.
"Verdict: misheard" if the only problem is a misspelled or misheard name or word.
"Verdict: right" if it is true.
"Verdict: none" for opinions, questions, or anything you are not sure about.
Only after "Verdict: wrong", write one card: what the claim is about | Fact check | Actually, followed by the right facts in at most 30 words. When numbers are disputed, say who reports them.

For "The Eiffel Tower is like 800 meters tall":
Verdict: wrong
Eiffel Tower height | Fact check | Actually, the Eiffel Tower is 330 meters tall including its antennas.

For "Then Jensen Hwang showed the new Blackwell chips":
Verdict: misheard

For "Water freezes at zero degrees Celsius":
Verdict: right

<earlier>{earlier}</earlier>
<check>{sentence}</check>"""

LIST_MARKER = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
SENTENCE_END = re.compile(r"[.?!](?=\s|$)")
CLAUSE_ENDINGS = (",", ".", "?", "!", ";", ":")
WORDS_PER_LOOK = 8
card_ids = itertools.count(1)


def worth_a_look(text, still_talking):
    """Words go to the model in whole clauses rather than two-word scraps:
    once a comma or a sentence end is heard, every 8 words, or when the speaker pauses."""
    words = text.split()
    return not still_talking or any(word.endswith(CLAUSE_ENDINGS) for word in words) or len(words) >= WORDS_PER_LOOK


def finished_sentences(text, speaker_paused):
    """Splits heard text into the sentences that are finished, which are ready for a fact check, and the rest."""
    if speaker_paused:
        return text, ""
    ends = [match.end() for match in SENTENCE_END.finditer(text)]
    if not ends:
        return "", text
    return text[:ends[-1]].strip(), text[ends[-1]:].strip()


def topic_key(title):
    words = re.findall(r"\w+", title.lower())
    return " ".join(words[1:] if words[:1] == ["the"] else words)


@dataclass
class Card:
    title: str
    kind: str
    body: str
    id: int = 0
    done: bool = False


def parse_card_line(line):
    """Reads a 'Title | Kind | Body' line, or gives None for anything else."""
    line = LIST_MARKER.sub("", line.strip().strip("|").strip())
    parts = [part.strip().strip("*").strip() for part in line.split("|", 2)]
    if len(parts) < 3 or not all(parts):
        return None
    title, kind, body = parts
    if title.lower() == "title" or not re.search(r"\w", title):
        return None
    return Card(title, kind, body)


class CardParser:
    """Follows the model's streamed reply and reports each card while it is being written."""

    def __init__(self, is_new_topic):
        self.is_new_topic = is_new_topic
        self.line = ""
        self.card = None
        self.skipping_line = False

    def feed(self, text):
        """Takes the next piece of the reply and returns the cards it started, grew or finished."""
        *finished_lines, self.line = (self.line + text).split("\n")
        updates = []
        for line in finished_lines:
            updates += self.read(line, finished=True)
        return updates + self.read(self.line, finished=False)

    def finish(self):
        """Ends the reply, which finishes the card on the last line."""
        line, self.line = self.line, ""
        return self.read(line, finished=True)

    def read(self, line, finished):
        card = self.card_on(line)
        if finished:
            self.card, self.skipping_line = None, False
        if card is None:
            return []
        card.done = finished
        return [card]

    def card_on(self, line):
        """Gives the card being written on this line, as long as it is about a new topic."""
        if self.skipping_line:
            return None
        written = parse_card_line(line)
        if written is None:
            return None
        if self.card is None:
            if not self.is_new_topic(written.title):
                self.skipping_line = True
                return None
            written.id = next(card_ids)
            self.card = written
        self.card.body = written.body
        return self.card


class CardWriter:
    """Asks the model for cards about the latest words, and remembers which topics already have a card."""

    def __init__(self, model, fact_check_model=None):
        self.model = model
        self.fact_check_model = fact_check_model or model
        self.client = ollama.AsyncClient()
        self.topics = {}
        self.checked_claims = set()
        self.earlier = deque(maxlen=3)
        self.unanswered = ""

    def is_new_topic(self, title):
        key = topic_key(title)
        if key in self.topics:
            return False
        self.topics[key] = title
        return True

    def is_new_claim(self, title):
        key = topic_key(title)
        if key in self.checked_claims:
            return False
        self.checked_claims.add(key)
        return True

    def clear(self):
        self.topics.clear()
        self.checked_claims.clear()
        self.earlier.clear()
        self.unanswered = ""

    async def warm_up(self):
        """Loads the models and has them read the instructions once, so the first real cards come fast."""
        for model in dict.fromkeys([self.model, self.fact_check_model]):
            await self.client.chat(
                model=model, messages=self.messages_for("Hello."), think=False, options={**OPTIONS, "num_predict": 1}, keep_alive=KEEP_LOADED_FOR
            )

    async def unload(self):
        """Frees the memory the models use, so switching to another model never leaves two big ones loaded."""
        for model in dict.fromkeys([self.model, self.fact_check_model]):
            with contextlib.suppress(ConnectionError):
                await self.client.chat(model=model, messages=[], keep_alive=0)

    async def write_cards(self, words):
        """Streams the cards about newly heard words. A card comes back each time its body grows, and once more when done.
        Words that get no card get a second look together with the words that follow them."""
        latest = f"{self.unanswered} {words}".strip()
        wrote_a_card = False
        async for card in self.stream_cards(self.messages_for(latest), self.is_new_topic):
            wrote_a_card = True
            yield card

        if wrote_a_card:
            self.earlier.append(latest)
            self.unanswered = ""
        else:
            if self.unanswered:
                self.earlier.append(self.unanswered)
            self.unanswered = words

    async def check_claims(self, sentence, said_before):
        """Gives a fact check card when a finished sentence says something factually wrong.
        The model gives its verdict first, so a card that merely confirms a right claim is never shown."""
        check = CLAIM_CHECK.format(earlier=said_before, sentence=sentence)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": check}]
        options = {**OPTIONS, "num_predict": 80}  # a verdict and one card; keeps the shared model free for cards
        reply = await self.client.chat(model=self.fact_check_model, messages=messages, think=False, options=options, keep_alive=KEEP_LOADED_FOR)
        verdict, _, card_lines = reply.message.content.strip().partition("\n")
        if "wrong" not in verdict.lower():
            return []
        return CardParser(self.is_new_claim).feed(card_lines + "\n")[:1]

    async def stream_cards(self, messages, is_new):
        parser = CardParser(is_new)
        reply = await self.client.chat(model=self.model, messages=messages, stream=True, think=False, options=OPTIONS, keep_alive=KEEP_LOADED_FOR)
        async for piece in reply:
            for card in parser.feed(piece.message.content or ""):
                yield card
        for card in parser.finish():
            yield card

    def messages_for(self, phrase):
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": self.prompt_for(phrase)}]

    def prompt_for(self, phrase):
        """Shows the model the new words, what was said before them, and the topics that already have a card."""
        covered = ", ".join(list(self.topics.values())[-40:])
        earlier = " ".join(self.earlier)
        return f"<covered>{covered}</covered>\n<earlier>{earlier}</earlier>\n<latest>{phrase}</latest>"
