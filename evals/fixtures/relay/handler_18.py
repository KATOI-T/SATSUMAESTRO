"""ハンドラ #18: sensor-18 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 18


def process(event):
    validate(event)
    emit(f"handler-18 accepted {event.name}")
    return MODULE_INDEX * event.size
