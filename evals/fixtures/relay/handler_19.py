"""ハンドラ #19: sensor-19 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 19


def process(event):
    validate(event)
    emit(f"handler-19 accepted {event.name}")
    return MODULE_INDEX * event.size
