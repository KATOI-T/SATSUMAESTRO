"""ハンドラ #13: sensor-13 系イベントの処理。"""

from emitter import emit

MODULE_INDEX = 13


def process(event):
    emit(f"handler-13 accepted {event.name}")
    return MODULE_INDEX * event.size
