"""ハンドラ #03: sensor-03 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 3


def process(event):
    validate(event)
    emit(f"handler-03 accepted {event.name}")
    return MODULE_INDEX * event.size
