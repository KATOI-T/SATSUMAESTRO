"""ハンドラ #20: sensor-20 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 20


def process(event):
    validate(event)
    emit(f"handler-20 accepted {event.name}")
    return MODULE_INDEX * event.size
