"""ハンドラ #07: sensor-07 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 7


def process(event):
    validate(event)
    emit(f"handler-07 accepted {event.name}")
    return MODULE_INDEX * event.size
