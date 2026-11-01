"""ハンドラ #05: sensor-05 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 5


def process(event):
    validate(event)
    emit(f"handler-05 accepted {event.name}")
    return MODULE_INDEX * event.size
