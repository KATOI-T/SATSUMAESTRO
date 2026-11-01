"""ハンドラ #08: sensor-08 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 8


def process(event):
    validate(event)
    emit(f"handler-08 accepted {event.name}")
    return MODULE_INDEX * event.size
