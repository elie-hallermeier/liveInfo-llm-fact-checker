import asyncio
import json
import urllib.request
from types import SimpleNamespace

import pytest

from app import MODEL
from cards import CardParser, CardWriter, finished_sentences, parse_card_line, worth_a_look


def test_reads_a_card_line():
    card = parse_card_line("Caravaggio | Person | Italian Baroque painter famous for dramatic light.")

    assert (card.title, card.kind, card.body) == ("Caravaggio", "Person", "Italian Baroque painter famous for dramatic light.")


def test_ignores_bullets_numbers_bold_and_table_pipes():
    for line in [
        "1. **TSMC** | Company | Taiwan's chip foundry.",
        "- TSMC | Company | Taiwan's chip foundry.",
        "| TSMC | Company | Taiwan's chip foundry. |",
    ]:
        card = parse_card_line(line)
        assert (card.title, card.kind, card.body) == ("TSMC", "Company", "Taiwan's chip foundry.")


def test_lines_that_are_not_cards_give_nothing():
    for line in ["NONE", "", "Here are the cards:", "Title | Kind | Body", "| --- | --- | --- |", "Nvidia | Company |"]:
        assert parse_card_line(line) is None


def test_a_card_shows_up_while_its_body_is_still_being_written():
    parser = CardParser(is_new_topic=lambda title: True)

    assert parser.feed("Nvidia | Comp") == []
    [card] = parser.feed("any | Designs")
    assert (card.title, card.kind, card.body, card.done) == ("Nvidia", "Company", "Designs", False)

    [same_card] = parser.feed(" GPUs.\n")
    assert same_card.id == card.id
    assert (same_card.body, same_card.done) == ("Designs GPUs.", True)


def test_finish_completes_the_last_card():
    parser = CardParser(is_new_topic=lambda title: True)
    parser.feed("TSMC | Company | Taiwan's chip foundry.")

    [card] = parser.finish()

    assert (card.body, card.done) == ("Taiwan's chip foundry.", True)


def test_each_card_gets_its_own_id():
    parser = CardParser(is_new_topic=lambda title: True)

    first, second = parser.feed("Nvidia | Company | Designs GPUs.\nTSMC | Company | Makes chips.\n")

    assert first.id != second.id
    assert first.done and second.done


def test_topics_that_already_have_a_card_are_skipped():
    parser = CardParser(is_new_topic=lambda title: title != "Nvidia")

    assert parser.feed("Nvidia | Company | Designs GPUs.\n") == []


def test_a_topic_gets_only_one_card():
    writer = CardWriter(MODEL)

    assert writer.is_new_topic("Nvidia")
    assert not writer.is_new_topic("NVIDIA")
    assert not writer.is_new_topic("The Nvidia.")
    assert writer.is_new_topic("Nvidia Shield")


def test_clear_forgets_the_topics():
    writer = CardWriter(MODEL)
    writer.is_new_topic("Nvidia")

    writer.clear()

    assert writer.is_new_topic("Nvidia")


def test_the_model_reads_the_new_words_apart_from_what_came_before():
    writer = CardWriter(MODEL)
    writer.earlier.extend(["I love Paris.", "We should use Kubernetes, or maybe just"])

    prompt = writer.prompt_for("Docker Compose since we are small.")

    assert "<earlier>I love Paris. We should use Kubernetes, or maybe just</earlier>" in prompt
    assert "<latest>Docker Compose since we are small.</latest>" in prompt


class FakeModel:
    """Stands in for Ollama: gives canned replies and keeps the prompts it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.models = []
        self.unloaded = []

    async def chat(self, model, messages=(), stream=False, **options):
        if not messages and options.get("keep_alive") == 0:
            self.unloaded.append(model)
            return None
        self.prompts.append(messages[-1]["content"])
        self.models.append(model)
        reply = SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))

        async def pieces():
            yield reply

        return pieces() if stream else reply


async def read_all(writer, *texts):
    for text in texts:
        async for _ in writer.write_cards(text):
            pass


def test_words_that_got_no_card_get_a_second_look_with_the_words_that_follow():
    writer = CardWriter(MODEL)
    writer.client = FakeModel("NONE", "Berlin | Place | Capital of Germany.")

    asyncio.run(read_all(writer, "So last week I was in Berlin for a", "conference about AI."))

    assert "<latest>So last week I was in Berlin for a conference about AI.</latest>" in writer.client.prompts[1]


def test_words_that_got_a_card_are_only_context_afterwards():
    writer = CardWriter(MODEL)
    writer.client = FakeModel("Berlin | Place | Capital of Germany.", "NONE")

    asyncio.run(read_all(writer, "So last week I was in Berlin.", "It was sunny."))

    assert "<earlier>So last week I was in Berlin.</earlier>" in writer.client.prompts[1]
    assert "<latest>It was sunny.</latest>" in writer.client.prompts[1]


def test_words_get_only_one_second_look():
    writer = CardWriter(MODEL)
    writer.client = FakeModel("NONE", "NONE", "NONE")

    asyncio.run(read_all(writer, "I had lunch", "and then a walk", "and a nap."))

    assert "<earlier>I had lunch</earlier>" in writer.client.prompts[2]
    assert "<latest>and then a walk and a nap.</latest>" in writer.client.prompts[2]


def test_claims_are_checked_one_whole_sentence_at_a_time():
    assert finished_sentences("Israel killed three million people in Gaza, which is insane. Also my", speaker_paused=False) == (
        "Israel killed three million people in Gaza, which is insane.",
        "Also my",
    )
    assert finished_sentences("Also my uncle says", speaker_paused=False) == ("", "Also my uncle says")


def test_a_pause_finishes_the_sentence():
    assert finished_sentences("and the sequel is", speaker_paused=True) == ("and the sequel is", "")


def test_a_fact_check_sees_the_whole_sentence_and_what_was_said_before_it():
    writer = CardWriter(MODEL)
    writer.client = FakeModel("Verdict: wrong\nEiffel Tower height | Fact check | Actually, it is 330 meters tall.")

    checks = asyncio.run(writer.check_claims("It's like 800 meters tall.", said_before="We went up the Eiffel Tower."))

    assert "<earlier>We went up the Eiffel Tower.</earlier>" in writer.client.prompts[0]
    assert "<check>It's like 800 meters tall.</check>" in writer.client.prompts[0]
    assert [(card.title, card.kind, card.done) for card in checks] == [("Eiffel Tower height", "Fact check", True)]


def test_fact_checks_can_use_a_bigger_model_than_the_cards():
    writer = CardWriter("small-model", fact_check_model="big-model")
    writer.client = FakeModel("NONE", "Verdict: right")

    asyncio.run(read_all(writer, "Water boils at 100 degrees."))
    asyncio.run(writer.check_claims("Water boils at 100 degrees.", said_before=""))

    assert writer.client.models == ["small-model", "big-model"]


def test_a_sentence_gets_one_fact_check_at_most():
    writer = CardWriter(MODEL)
    writer.client = FakeModel("Verdict: wrong\nEverest height | Fact check | Actually, 8,849 m.\nBitcoin | Fact check | Actually, 2009.")

    checks = asyncio.run(writer.check_claims("Everest is 2,000 m high and Bitcoin is from 2015.", said_before=""))

    assert [card.title for card in checks] == ["Everest height"]


def test_unloading_frees_every_model_the_writer_used():
    writer = CardWriter("small-model", fact_check_model="big-model")
    writer.client = FakeModel()

    asyncio.run(writer.unload())

    assert writer.client.unloaded == ["small-model", "big-model"]


def test_a_claim_judged_right_gets_no_card_even_if_the_model_writes_one():
    writer = CardWriter(MODEL)
    writer.client = FakeModel("Verdict: right\nWater boiling | Fact check | Actually, water boils at 100 degrees at sea level.")

    assert asyncio.run(writer.check_claims("Water boils at 100 degrees at sea level.", said_before="")) == []


def test_words_wait_for_the_end_of_their_clause():
    assert not worth_a_look("So last week I was in", still_talking=True)
    assert worth_a_look("who talked about AlphaFold,", still_talking=True)
    assert worth_a_look("I finished The Three-Body Problem.", still_talking=True)
    assert worth_a_look("catching up with GPT 5, which", still_talking=True)


def test_a_long_run_of_words_does_not_wait_for_punctuation():
    assert worth_a_look("So last week I was in Berlin for", still_talking=True)


def test_a_pause_sends_whatever_was_said():
    assert worth_a_look("the sequel is", still_talking=False)


def model_is_available(model):
    request = urllib.request.Request("http://127.0.0.1:11434/api/show", data=json.dumps({"model": model}).encode())
    try:
        urllib.request.urlopen(request, timeout=2)
        return True
    except OSError:
        return False


needs_model = pytest.mark.skipif(not model_is_available(MODEL), reason=f"Ollama is not running or {MODEL} is not pulled")


async def finished_cards(text):
    return [card async for card in CardWriter(MODEL).write_cards(text) if card.done]


@needs_model
def test_writes_a_card_about_a_famous_painter():
    cards = asyncio.run(finished_cards("My favorite painter is Caravaggio, his use of light is just incredible."))

    assert "Caravaggio" in [card.title for card in cards]


def fact_checks(cards):
    return [card for card in cards if card.kind.lower() == "fact check"]


@needs_model
def test_a_wrong_claim_gets_a_fact_check():
    writer = CardWriter(MODEL)

    checks = asyncio.run(writer.check_claims("World War Two ended in 1960, so my grandpa was born after it.", said_before=""))

    assert any("1945" in card.body for card in fact_checks(checks))


@needs_model
def test_misheard_names_are_not_fact_checked():
    writer = CardWriter(MODEL)

    checks = asyncio.run(writer.check_claims("The keynote was by Demis Hasabas from Deek Mind, who talked about Alpha Fold and protein folding.", said_before=""))

    assert checks == []


@needs_model
def test_a_right_claim_gets_no_fact_check():
    writer = CardWriter(MODEL)

    checks = asyncio.run(writer.check_claims("Water boils at 100 degrees Celsius at sea level, that's why we use it.", said_before=""))

    assert checks == []


@needs_model
def test_small_talk_gets_no_cards():
    cards = asyncio.run(finished_cards("Anyway, I had lunch and then we went for a walk, it was really nice outside."))

    assert cards == []


@needs_model
def test_an_idea_without_a_name_gets_a_card():
    cards = asyncio.run(finished_cards("The most practical way to turn your agent into a learning machine is to give it the right tools and the right sub agents."))

    assert any("agent" in card.title.lower() for card in cards)


@needs_model
def test_everyday_errands_get_no_cards():
    cards = asyncio.run(finished_cards("I'll grab a coffee and then call my mom, the meeting is at three anyway."))

    assert cards == []


async def finished_cards_after(earlier, text):
    writer = CardWriter(MODEL)
    writer.earlier.append(earlier)
    return [card async for card in writer.write_cards(text) if card.done]


@needs_model
def test_a_misheard_name_gets_a_card_under_its_real_name():
    cards = asyncio.run(finished_cards_after("We were debugging the deployment all morning.", "Then we talked about cooper netties and the cube control command."))

    assert any("kubernetes" in card.title.lower() for card in cards)
    assert not any("cooper" in card.title.lower() for card in cards)


@needs_model
def test_a_made_up_word_still_gets_no_card():
    cards = asyncio.run(finished_cards("Omer says the frobnik deck rollout slipped again."))

    assert not any("frobnik" in card.title.lower() for card in cards)
