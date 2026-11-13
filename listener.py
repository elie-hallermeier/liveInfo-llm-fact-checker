"""Hears the microphone and turns speech into text while it is being said, fully on-device."""
import queue
import re
import time
import warnings
import wave
from collections import deque
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import sounddevice
import torch
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel
from silero_vad import load_silero_vad

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit")
torch.set_num_threads(1)

SPEECH_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
SAMPLE_RATE = 16000
FRAME_SIZE = 512  # 32 ms, the frame size the Silero voice detector works on
FRAME_SECONDS = FRAME_SIZE / SAMPLE_RATE
VOICE_STARTS = 0.5
VOICE_CONTINUES = 0.35  # once a phrase has started, softer sounds still count as voice
SETTLED_AFTER_SECONDS = 0.8  # a word this far behind the newest audio is no longer heard differently
LONGEST_READ_SECONDS = 10  # in a long monologue, audio whose words are settled is dropped beyond this


def frames_in(seconds):
    return round(seconds / FRAME_SECONDS)


READ_EVERY_FRAMES = frames_in(0.5)  # how often a phrase is read while it is still being said


@dataclass
class Word:
    text: str
    start: float
    end: float


def same_word(first, second):
    return re.sub(r"\W", "", first.lower()) == re.sub(r"\W", "", second.lower())


def settled_words(words, last_word, heard_seconds, ending):
    """Picks the words that come after last_word and will not change any more:
    all of them once the phrase has ended, otherwise those that ended a while before the newest audio."""
    first = 0
    if last_word:
        same_as_last = [index for index, word in enumerate(words) if same_word(word.text, last_word.text) and abs(word.end - last_word.end) < 0.3]
        if same_as_last:
            first = same_as_last[0] + 1
        else:
            first = next((index for index, word in enumerate(words) if (word.start + word.end) / 2 > last_word.end), len(words))
    settled = []
    for word in words[first:]:
        if not ending and word.end > heard_seconds - SETTLED_AFTER_SECONDS:
            break
        settled.append(word)
    return settled


class PhraseCutter:
    """Groups 32 ms frames into phrases. A phrase ends at a short pause, or at its quietest moment when it runs too long."""

    def __init__(self, pause_frames=frames_in(0.4), max_frames=frames_in(15), min_voice_frames=frames_in(0.25), lead_in_frames=frames_in(0.2)):
        self.pause_frames = pause_frames
        self.max_frames = max_frames
        self.min_voice_frames = min_voice_frames
        self.lead_in = deque(maxlen=lead_in_frames)
        self.frames = []
        self.voice_probabilities = []

    @property
    def hearing_voice(self):
        return bool(self.frames)

    def add(self, frame, voice_probability):
        """Takes the next frame and returns the audio of a phrase once the phrase is complete."""
        if not self.frames and voice_probability < VOICE_STARTS:
            self.lead_in.append((frame, voice_probability))
            return None
        if not self.frames:
            for lead_in_frame, lead_in_probability in self.lead_in:
                self.frames.append(lead_in_frame)
                self.voice_probabilities.append(lead_in_probability)
            self.lead_in.clear()
        self.frames.append(frame)
        self.voice_probabilities.append(voice_probability)

        if self.quiet_frames_at_end() >= self.pause_frames:
            return self.cut_at(len(self.frames))
        if len(self.frames) >= self.max_frames:
            half = self.max_frames // 2
            return self.cut_at(half + int(np.argmin(self.voice_probabilities[half:])))
        return None

    def quiet_frames_at_end(self):
        quiet_frames = 0
        for voice_probability in reversed(self.voice_probabilities):
            if voice_probability >= VOICE_CONTINUES:
                break
            quiet_frames += 1
        return quiet_frames

    def cut_at(self, index):
        phrase_frames, self.frames = self.frames[:index], self.frames[index:]
        phrase_probabilities, self.voice_probabilities = self.voice_probabilities[:index], self.voice_probabilities[index:]
        voice_frames = sum(probability >= VOICE_CONTINUES for probability in phrase_probabilities)
        if voice_frames < self.min_voice_frames:
            return None
        return np.concatenate(phrase_frames)

    def forget_first(self, frame_count):
        self.frames = self.frames[frame_count:]
        self.voice_probabilities = self.voice_probabilities[frame_count:]


class Listener:
    """Listens to the microphone and gives the words being said, about a second after they are said."""

    def __init__(self, device=None, save_audio_to=None):
        self.device = device
        self.recording = None
        if save_audio_to:
            self.recording = wave.open(str(save_audio_to), "wb")
            self.recording.setnchannels(1)
            self.recording.setsampwidth(2)
            self.recording.setframerate(SAMPLE_RATE)
        self.voice_detector = load_silero_vad()
        self.speech_model = from_pretrained(SPEECH_MODEL)
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.cutter = PhraseCutter()
        self.last_word = None
        self.frames_since_reading = 0
        self.frames = queue.Queue()
        self.stream = None

    def hear(self, frame):
        """Takes the next 32 ms of audio and returns the words that just became certain, if any.
        While someone talks, the phrase is read twice a second; the rest of it is read when they pause."""
        if self.recording:
            self.recording.writeframes((frame * 32767).astype(np.int16).tobytes())
        voice_probability = self.voice_detector(torch.from_numpy(frame), SAMPLE_RATE).item()
        phrase = self.cutter.add(frame, voice_probability)
        if phrase is not None:
            return self.read(phrase, ending=True)
        if not self.cutter.hearing_voice:
            self.last_word, self.frames_since_reading = None, 0
            return None
        self.frames_since_reading += 1
        if self.frames_since_reading < READ_EVERY_FRAMES:
            return None
        return self.read(np.concatenate(self.cutter.frames), ending=False)

    def read(self, audio, ending):
        self.frames_since_reading = 0
        heard_seconds = len(audio) / SAMPLE_RATE
        words = settled_words(self.transcribe(audio), self.last_word, heard_seconds, ending)
        if ending:
            self.last_word = None
        elif words:
            self.last_word = words[-1]
            if heard_seconds > LONGEST_READ_SECONDS:
                self.forget_settled_audio()
        return " ".join(word.text for word in words)

    def forget_settled_audio(self):
        """Keeps reading fast in a long monologue, by dropping the audio of words that are already settled."""
        frame_count = int(self.last_word.end / FRAME_SECONDS)
        self.cutter.forget_first(frame_count)
        shift = frame_count * FRAME_SECONDS
        self.last_word = Word(self.last_word.text, self.last_word.start - shift, self.last_word.end - shift)

    def transcribe(self, audio):
        """Gives the words in the audio, each with the second it starts and ends."""
        mel = get_logmel(mx.array(audio), self.speech_model.preprocessor_config)
        words = []
        for token in self.speech_model.generate(mel)[0].tokens:
            if token.text.startswith(" ") or not words:
                words.append(Word(token.text.strip(), token.start, token.end))
            else:
                words[-1].text += token.text
                words[-1].end = token.end
        return words

    def open_microphone(self):
        self.stream = sounddevice.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_SIZE, device=self.device, callback=self.keep_frame
        )
        self.stream.start()

    def listen(self, on_voice, on_phrase, on_mic_silent):
        """Runs forever. Calls back when someone starts or stops talking, with the text of each phrase,
        and when the microphone gives nothing but digital silence (usually a missing permission)."""
        silent_frames = 0
        warned_about_lag = False
        while True:
            frame = self.frames.get()
            behind_seconds = self.frames.qsize() * FRAME_SECONDS
            if behind_seconds > 2 and not warned_about_lag:
                print(f"{time.strftime('%H:%M:%S')}  Speech recognition is {behind_seconds:.0f}s behind the microphone: the Mac is too busy.", flush=True)
                warned_about_lag = True
            elif behind_seconds < 1:
                warned_about_lag = False
            if frame is None:
                self.cutter = PhraseCutter()
                self.voice_detector.reset_states()
                continue

            if frame.any():
                if silent_frames >= frames_in(3):
                    on_mic_silent(False)
                silent_frames = 0
            else:
                silent_frames += 1
                if silent_frames == frames_in(3):
                    on_mic_silent(True)

            was_hearing_voice = self.cutter.hearing_voice
            text = self.hear(frame)
            if text:
                on_phrase(text)
            if self.cutter.hearing_voice != was_hearing_voice:
                on_voice(self.cutter.hearing_voice)

    def keep_frame(self, samples, frame_count, time_info, status):
        self.frames.put(samples[:, 0].copy())

    def pause(self):
        self.stream.stop()
        self.frames.put(None)

    def resume(self):
        self.stream.start()
