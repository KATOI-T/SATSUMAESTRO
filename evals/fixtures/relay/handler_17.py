"""ハンドラ #17: sensor-17 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 17


def process(event):
    validate(event)
    emit(f"handler-17 accepted {event.name}")
    return MODULE_INDEX * event.size
