"""liveInfo: listens to a conversation and shows short info cards about whatever is mentioned, live and on-device."""
import argparse
import asyncio
import contextlib
import json
import subprocess
import threading
import time
import traceback
import webbrowser
from collections import deque
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from cards import CardWriter, finished_sentences, worth_a_look
from listener import Listener

# The model that writes the cards and the fact checks. To try another one, change the number in MODEL and restart.
# Lightest and quickest to load first, most knowledge last. All three write cards at about the same speed on an M4 Pro.
MODELS = [
    "qwen3.5:4b-mlx",  # 4 GB, loads in 3s. Writes some junk cards, and its fact checks can get numbers wrong.
    "gemma4:26b-nvfp4",  # 16 GB (about 24 GB in use), loads in 7s. Best results in the tests.
    "qwen3.5:35b-mlx",  # 22 GB (about 30 GB in use), loads in 16s. Most knowledge; first card a little slower.
]
MODEL = MODELS[0]

PAGE = Path(__file__).parent / "static" / "index.html"
MIC_SILENT_NOTE = "The microphone gives no sound. Allow microphone access for this terminal in System Settings → Privacy & Security → Microphone."


class LiveInfo:
    """Connects the microphone listener, the card writer and the open windows."""

    def __init__(self, model, fact_check_model, device, save_audio_to=None):
        self.writer = CardWriter(model, fact_check_model)
        self.device = device
        self.save_audio_to = save_audio_to
        self.listener = None
        self.state = {"mode": "starting", "note": None}
        self.cards = {}
        self.windows = set()
        self.words = []  # words heard but not read by the card writer yet, with when they arrived
        self.sentence = ""  # the sentence being said, fact-checked once it is finished
        self.said = deque(maxlen=2)  # the last finished sentences, as context for the next fact check
        self.fact_checks = set()  # fact checks running alongside the card writing
        self.talking = False
        self.words_ready = asyncio.Event()
        self.loop = None

    async def run(self):
        self.loop = asyncio.get_running_loop()
        listener_ready = self.loop.create_future()
        threading.Thread(target=self.listen, args=(listener_ready,), daemon=True).start()
        try:
            await asyncio.gather(self.writer.warm_up(), listener_ready)
        except Exception as error:
            print(f"Could not start: {error}", flush=True)
            await self.set_mode("problem", str(error))
            return
        print("Listening. Talk, and cards show up in the window. Ctrl+C to quit.", flush=True)
        await self.set_mode("listening")
        await self.write_cards_forever()

    def listen(self, listener_ready):
        try:
            self.listener = Listener(self.device, self.save_audio_to)
            self.listener.open_microphone()
        except Exception as error:
            self.loop.call_soon_threadsafe(listener_ready.set_exception, error)
            return
        self.loop.call_soon_threadsafe(listener_ready.set_result, None)
        self.listener.listen(
            on_voice=lambda talking: self.from_thread(self.voice(talking)),
            on_phrase=lambda text: self.from_thread(self.hear(text)),
            on_mic_silent=lambda silent: self.from_thread(self.set_mode(self.state["mode"], MIC_SILENT_NOTE if silent else None)),
        )

    def from_thread(self, coroutine):
        try:
            asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        except RuntimeError:  # the app is quitting and its event loop is closed: drop what the listener heard last
            coroutine.close()

    async def voice(self, talking):
        self.talking = talking
        self.check_words()
        await self.send({"type": "voice", "talking": talking})

    async def hear(self, text):
        self.words.append((text, time.monotonic()))
        self.check_words()
        await self.send({"type": "phrase", "text": text})

    def check_words(self):
        if self.words and worth_a_look(" ".join(text for text, _ in self.words), self.talking):
            self.words_ready.set()

    async def write_cards_forever(self):
        while True:
            await self.words_ready.wait()
            self.words_ready.clear()
            text, heard_at = " ".join(text for text, _ in self.words), self.words[0][1]
            self.words = []
            print(f"\n{time.strftime('%H:%M:%S')}  “{text}”", flush=True)
            try:
                async for card in self.writer.write_cards(text):
                    await self.show(card, heard_at)
                finished, self.sentence = finished_sentences(f"{self.sentence} {text}".strip(), speaker_paused=not self.talking)
                if finished:
                    fact_check = asyncio.create_task(self.fact_check(finished, " ".join(self.said), heard_at))
                    self.fact_checks.add(fact_check)
                    fact_check.add_done_callback(self.fact_checks.discard)
                    self.said.append(finished)
            except Exception:
                print("Card writing failed, carrying on with the next phrase:", flush=True)
                traceback.print_exc()

    async def fact_check(self, sentence, said_before, heard_at):
        try:
            for card in await self.writer.check_claims(sentence, said_before):
                await self.show(card, heard_at)
        except Exception:
            print("Fact check failed, carrying on:", flush=True)
            traceback.print_exc()

    async def show(self, card, heard_at):
        self.cards[card.id] = asdict(card)
        await self.send({"type": "card", **asdict(card)})
        if card.done:
            print(f"   → {card.title} · {card.kind}  ({time.monotonic() - heard_at:.2f}s)", flush=True)

    async def set_mode(self, mode, note=None):
        self.state = {"mode": mode, "note": note}
        await self.send({"type": "state", **self.state})

    async def send(self, event):
        message = json.dumps(event)
        for window in list(self.windows):
            try:
                await window.send_text(message)
            except (WebSocketDisconnect, RuntimeError):
                self.windows.discard(window)

    async def obey(self, command):
        if command == "pause" and self.state["mode"] == "listening":
            self.listener.pause()
            await self.voice(False)
            await self.set_mode("paused")
        elif command == "resume" and self.state["mode"] == "paused":
            self.listener.resume()
            await self.set_mode("listening")
        elif command == "clear":
            self.cards.clear()
            self.writer.clear()
            await self.send({"type": "cleared"})


def create_app(live_info):
    app = FastAPI()

    @app.get("/")
    async def page():
        return FileResponse(PAGE)

    @app.websocket("/live")
    async def live(websocket: WebSocket):
        await websocket.accept()
        live_info.windows.add(websocket)
        await websocket.send_text(json.dumps({"type": "hello", **live_info.state, "cards": list(live_info.cards.values())}))
        try:
            while True:
                await live_info.obey(await websocket.receive_text())
        except WebSocketDisconnect:
            live_info.windows.discard(websocket)

    return app


def open_window(url):
    """Opens the cards in a slim app window when Chrome is installed, otherwise in the default browser."""
    if Path("/Applications/Google Chrome.app").exists():
        subprocess.Popen(["open", "-na", "Google Chrome", "--args", f"--app={url}", "--window-size=460,820"])
    else:
        webbrowser.open(url)


async def serve(live_info, port):
    server = uvicorn.Server(uvicorn.Config(create_app(live_info), host="127.0.0.1", port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    url = f"http://127.0.0.1:{port}"
    print(f"liveInfo is open at {url}\nLoading the speech model and {live_info.writer.model}…", flush=True)
    open_window(url)
    running = asyncio.create_task(live_info.run())
    try:
        await serving
    finally:  # uvicorn re-raises Ctrl+C after shutting down, so this runs while the app is being cancelled
        running.cancel()
        await live_info.writer.unload()


def main():
    parser = argparse.ArgumentParser(description="Live info cards about whatever is said near your microphone.")
    parser.add_argument("--model", default=MODEL, help=f"Ollama model for the cards and fact checks (default: {MODEL}, picked at the top of app.py)")
    parser.add_argument("--fact-check-model", help="a different Ollama model for the fact checks (default: the same as --model)")
    parser.add_argument("--device", help="microphone name or number from `python -m sounddevice` (default: the system microphone)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--save-audio", metavar="FILE", help="also save everything the microphone hears to this .wav file, to replay it later")
    args = parser.parse_args()
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    with contextlib.suppress(KeyboardInterrupt):  # Ctrl+C is how you quit, so no traceback for it
        asyncio.run(serve(LiveInfo(args.model, args.fact_check_model, device, args.save_audio), args.port))


if __name__ == "__main__":
    main()
