"""ハンドラ #06: sensor-06 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 6


def process(event):
    validate(event)
    emit(f"handler-06 accepted {event.name}")
    return MODULE_INDEX * event.size
