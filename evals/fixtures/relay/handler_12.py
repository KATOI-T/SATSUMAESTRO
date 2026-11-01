"""ハンドラ #12: sensor-12 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 12


def process(event):
    validate(event)
    emit(f"handler-12 accepted {event.name}")
    return MODULE_INDEX * event.size
