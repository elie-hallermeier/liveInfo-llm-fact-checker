import subprocess
import wave

import numpy as np

from listener import FRAME_SIZE, SAMPLE_RATE, Listener, PhraseCutter, Word, settled_words

VOICE = 0.9
QUIET = 0.05


def cut(cutter, voice_probabilities, first_number=0):
    """Feeds frames whose samples hold their frame number, and returns the frame numbers of each phrase."""
    phrases = []
    for number, voice_probability in enumerate(voice_probabilities, first_number):
        phrase = cutter.add(np.full(FRAME_SIZE, number, dtype=np.float32), voice_probability)
        if phrase is not None:
            phrases.append(phrase.reshape(-1, FRAME_SIZE)[:, 0].astype(int).tolist())
    return phrases


def test_a_phrase_ends_at_a_pause_and_keeps_a_little_lead_in():
    cutter = PhraseCutter(pause_frames=10, max_frames=500, min_voice_frames=5, lead_in_frames=2)

    phrases = cut(cutter, [QUIET] * 5 + [VOICE] * 20 + [QUIET] * 10)

    assert phrases == [list(range(3, 35))]


def test_a_click_is_not_a_phrase():
    cutter = PhraseCutter(pause_frames=10, max_frames=500, min_voice_frames=5, lead_in_frames=2)

    assert cut(cutter, [QUIET] * 5 + [VOICE] * 2 + [QUIET] * 10) == []


def test_nonstop_speech_is_cut_at_its_quietest_moments():
    cutter = PhraseCutter(pause_frames=10, max_frames=50, min_voice_frames=5, lead_in_frames=0)
    voice_probabilities = [VOICE] * 100 + [QUIET] * 10
    voice_probabilities[40] = 0.4
    voice_probabilities[80] = 0.4

    phrases = cut(cutter, voice_probabilities)

    assert phrases == [list(range(0, 40)), list(range(40, 80)), list(range(80, 110))]


def test_forgetting_the_start_of_a_long_phrase_keeps_the_rest():
    cutter = PhraseCutter(pause_frames=10, max_frames=500, min_voice_frames=5, lead_in_frames=0)
    cut(cutter, [VOICE] * 20)

    cutter.forget_first(15)

    assert cut(cutter, [QUIET] * 10, first_number=20) == [list(range(15, 30))]


SENTENCE = [Word("So", 0.0, 0.3), Word("last", 0.3, 0.5), Word("week", 0.5, 0.7), Word("in", 0.9, 1.1), Word("Berlin", 1.1, 1.5), Word("for", 1.5, 1.7)]


def texts(words):
    return [word.text for word in words]


def test_while_talking_only_words_that_ended_a_while_ago_are_settled():
    assert texts(settled_words(SENTENCE, last_word=None, heard_seconds=2.0, ending=False)) == ["So", "last", "week", "in"]


def test_when_the_phrase_ends_every_remaining_word_is_settled():
    assert texts(settled_words(SENTENCE, last_word=Word("in", 0.9, 1.1), heard_seconds=2.0, ending=True)) == ["Berlin", "for"]


def test_a_word_already_handed_out_is_not_repeated_when_its_timing_shifts():
    shifted = [Word("So", 0.0, 0.3), Word("last", 0.3, 0.5), Word("week", 0.66, 0.9), Word("in", 0.9, 1.1), Word("Berlin", 1.1, 1.5)]

    assert texts(settled_words(shifted, last_word=Word("week", 0.5, 0.7), heard_seconds=3.0, ending=False)) == ["in", "Berlin"]


def test_when_the_last_word_was_heard_differently_the_next_words_are_found_by_time():
    changed = [Word("So", 0.0, 0.3), Word("last", 0.3, 0.5), Word("weak", 0.5, 0.7), Word("in", 0.9, 1.1)]

    assert texts(settled_words(changed, last_word=Word("week", 0.5, 0.7), heard_seconds=3.0, ending=False)) == ["in"]


def spoken(text, folder):
    aiff_path, wav_path = folder / "speech.aiff", folder / "speech.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", aiff_path, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1", aiff_path, wav_path], check=True)
    with wave.open(str(wav_path)) as wav_file:
        samples = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
    return np.concatenate([samples.astype(np.float32) / 32768, np.zeros(SAMPLE_RATE, dtype=np.float32)])


def hear_all(listener, audio):
    """Returns each piece of text the listener gives, with the second of audio at which it gave it."""
    heard = []
    for start in range(0, len(audio) - FRAME_SIZE + 1, FRAME_SIZE):
        text = listener.hear(audio[start:start + FRAME_SIZE])
        if text:
            heard.append((text, (start + FRAME_SIZE) / SAMPLE_RATE))
    return heard


def test_hears_each_phrase_of_a_conversation(tmp_path):
    audio = spoken("Did you see the James Webb telescope images? [[slnc 900]] My favorite painter is Caravaggio.", tmp_path)

    text = " ".join(piece for piece, _ in hear_all(Listener(), audio))

    assert text.count("James Webb") == 1
    assert text.count("Caravaggio") == 1
    assert text.index("James Webb") < text.index("Caravaggio")


def test_a_long_sentence_is_heard_while_it_is_still_being_said(tmp_path):
    audio = spoken(
        "So last week I was in Berlin for a conference about large language models, and the opening keynote was by "
        "Demis Hassabis from DeepMind, who talked about how it cracked protein folding.",
        tmp_path,
    )

    heard = hear_all(Listener(), audio)
    text = " ".join(piece for piece, _ in heard)

    berlin_heard_at = next(second for piece, second in heard if "Berlin" in piece)
    assert berlin_heard_at < 4
    assert text.count("Berlin") == 1
    assert "protein folding" in text.lower()


def test_can_save_what_it_hears_for_a_replay(tmp_path):
    audio = spoken("Hello there, this is a recording.", tmp_path)
    listener = Listener(save_audio_to=tmp_path / "heard.wav")

    hear_all(listener, audio)

    with wave.open(str(tmp_path / "heard.wav")) as saved:
        assert saved.getframerate() == SAMPLE_RATE
        saved_audio = np.frombuffer(saved.readframes(saved.getnframes()), dtype=np.int16) / 32768
    assert len(saved_audio) == len(audio) // FRAME_SIZE * FRAME_SIZE
    assert np.abs(saved_audio - audio[:len(saved_audio)]).max() < 0.001
