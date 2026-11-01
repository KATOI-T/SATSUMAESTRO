"""ハンドラ #11: sensor-11 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 11


def process(event):
    validate(event)
    emit(f"handler-11 accepted {event.name}")
    return MODULE_INDEX * event.size
