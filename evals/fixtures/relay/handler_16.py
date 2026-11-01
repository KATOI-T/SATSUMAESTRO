"""ハンドラ #16: sensor-16 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 16


def process(event):
    validate(event)
    emit(f"handler-16 accepted {event.name}")
    return MODULE_INDEX * event.size
