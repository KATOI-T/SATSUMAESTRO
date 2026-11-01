"""ハンドラ #10: sensor-10 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 10


def process(event):
    validate(event)
    emit(f"handler-10 accepted {event.name}")
    return MODULE_INDEX * event.size
