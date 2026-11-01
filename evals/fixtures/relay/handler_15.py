"""ハンドラ #15: sensor-15 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 15


def process(event):
    validate(event)
    emit(f"handler-15 accepted {event.name}")
    return MODULE_INDEX * event.size
