"""ハンドラ #04: sensor-04 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 4


def process(event):
    validate(event)
    emit(f"handler-04 accepted {event.name}")
    return MODULE_INDEX * event.size
