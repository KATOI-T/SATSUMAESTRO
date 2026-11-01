"""ハンドラ #09: sensor-09 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 9


def process(event):
    validate(event)
    emit(f"handler-09 accepted {event.name}")
    return MODULE_INDEX * event.size
