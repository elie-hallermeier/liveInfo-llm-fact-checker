import asyncio

from app import MODEL, LiveInfo


def test_words_heard_after_the_app_stopped_are_dropped_quietly():
    live_info = LiveInfo(MODEL, None, None)
    live_info.loop = asyncio.new_event_loop()
    live_info.loop.close()

    live_info.from_thread(live_info.hear("one last word"))
