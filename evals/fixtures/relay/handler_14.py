"""ハンドラ #14: sensor-14 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 14


def process(event):
    validate(event)
    emit(f"handler-14 accepted {event.name}")
    return MODULE_INDEX * event.size
