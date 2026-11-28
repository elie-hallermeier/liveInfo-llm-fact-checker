# liveInfo

Listens to a conversation and shows short info cards about whatever comes up (people, companies, places, tech, concepts), live and without being asked. It also fact-checks claims. It understands English (and 24 other European languages) and Hebrew. Everything runs on your Mac, for free.

Tested on an Apple Silicon MacBook Pro (M4 Pro, 48 GB) with Python 3.14 and [Ollama](https://ollama.com). The default model needs about 24 GB of free memory.

- **Hearing:** Silero VAD finds speech, and a speech model on MLX re-reads the phrase being spoken: NVIDIA Parakeet for English, twice a second, or [ivrit.ai](https://huggingface.co/ivrit-ai)'s Hebrew Whisper (large-v3 turbo) for Hebrew, once a second, since it is heavier and shares the GPU with the card model. A word counts as settled once it ended 0.8s before the newest audio, so text keeps flowing during long sentences.
- **Language:** picked with the `LANGUAGE` knob at the top of `app.py`: `"en"` for English and 24 other European languages, `"he"` for Hebrew, including Hebrew tech meetings full of English terms. The cards are in English either way.
- **When a card is made:** settled words go to the model one clause at a time: at a comma or sentence end, every 8 words, or when the speaker pauses. Words that got no card get a second look together with the words that follow them. Each topic gets one card.
- **Fact checks:** each finished sentence is also checked. The model first gives a verdict: wrong, misheard (only a misspelled name), right, or not sure. Only "wrong" shows a red "Fact check" card saying what is actually true. For disputed numbers it says who reports them.
- **Model:** picked with the `MODEL` knob at the top of `app.py`, from a short list ordered lightest to most knowledgeable. The default, `gemma4:26b-nvfp4` (Google Gemma 4), does both jobs. It is a mixture of experts: 26B parameters of knowledge, but only about 4B are used per word, so it writes as fast as a 4B model. It uses 16 GB when loaded and grows to about 24 GB in use.
- **Speed** on an M4 Pro in High Power mode: a card shows up about 1.6s after its name is said in conversation, and about 2.4s in nonstop talk. Of that, the model takes only ~0.4s; the rest is the speech side settling words and waiting for the clause to end. A fact check shows up 1–3s after the sentence ends.
- **Hebrew speed:** in a test meeting of Hebrew with English tech terms (two synthetic voices, a term every 3 seconds), Gemma showed a card for 18 of 19 terms, a median 3s after the term was said (worst 6s). Qwen 35B was slower there and wrote many junk cards ("After" for אחרי), so Gemma is the one to pick for Hebrew. In a fast design review (182 Hebrew words a minute, 25 terms such as sharding, CQRS and exponential backoff), Gemma showed cards for 14–18 of the 25 terms, a median of about 4s after (up to 11s when speech recognition falls behind). Whole-sentence transcription gets 20 of the 25; the rest are lost to the synthetic English voice, such as "be Redis" heard as אלכוהול.

The knobs are `SETTLED_AFTER_SECONDS` and `READ_EVERY_FRAMES` in `listener.py`, and `WORDS_PER_LOOK`, the prompt and `CLAIM_CHECK` in `cards.py`.

## Setup

```bash
ollama pull gemma4:26b-nvfp4
```

To try the other models in the knob, pull them too:

```bash
ollama pull qwen3.5:4b-mlx
```

```bash
ollama pull qwen3.5:35b-mlx
```

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python app.py
```

A slim window opens: a Chrome app window, or your default browser. On first run, macOS asks for microphone access for the app you launched it from (Terminal, PyCharm…). Loading the model takes about 7 seconds.

Space pauses and resumes listening. The trash icon clears the cards.

Options:

- `--model`: another Ollama model for one run, without touching the knob in `app.py`
- `--fact-check-model`: another Ollama model for fact checks. When it differs from `--model`, fact checks run alongside card writing.
- `--language`: `en` or `he`, for one run, without touching the knob in `app.py`
- `--device`: microphone name or number (`.venv/bin/python -m sounddevice` lists them)
- `--port`: default 8765
- `--save-audio FILE`: also saves everything the microphone hears to a .wav file, so a session can be replayed to debug it

On a Mac with less memory, pick the 4B in the knob (`MODEL = MODELS[0]`). It writes more junk cards, and its fact checks are less reliable.

## Tests

```bash
.venv/bin/python -m pytest
```

The model tests need Ollama running with the model pulled. They are skipped otherwise.

## Notes

- The model only knows what it learned in training, so fact checks about recent events can be out of date. For example, it gives Gaza death tolls from early 2024.
- The app unloads its model when you quit, so switching models never leaves two big ones in memory. Don't run other large models in Ollama next to it: two don't fit in 48 GB together, and the Mac starts swapping.
- Speech recognition covers English and 24 other European languages (Parakeet v3), and Hebrew (ivrit.ai Whisper).
- Whisper normally reads 30 seconds of audio at a time. For Hebrew, the app gives it only the audio heard plus 2 seconds of silence, which halves the time a read takes: 0.3–0.6s for a few seconds of speech on an M4 Pro.
- Hebrew Whisper mangles words with ג׳, צ׳ or ז׳: it learned from subtitles that wrote them in Latin letters with "XD7XB3" for the geresh, so ג'נסן comes out as GXD7XB3NSN. The app turns these back into Hebrew letters, but some still come out garbled, so the card model has to work out the name from context. It usually does, and sometimes writes a junk card instead. English tech words said in English (Jira, merge, JSON) mostly come out fine; the mangling shows up most with names said the Hebrew way.
- The Mac's Energy Mode matters: Low Power roughly halves card speed.
