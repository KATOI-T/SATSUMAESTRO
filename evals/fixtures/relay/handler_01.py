"""ハンドラ #01: sensor-01 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 1


def process(event):
    validate(event)
    emit(f"handler-01 accepted {event.name}")
    return MODULE_INDEX * event.size
