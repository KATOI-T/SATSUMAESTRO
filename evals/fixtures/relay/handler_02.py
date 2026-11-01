"""ハンドラ #02: sensor-02 系イベントの処理。"""

from emitter import emit
from validate import validate

MODULE_INDEX = 2


def process(event):
    validate(event)
    emit(f"handler-02 accepted {event.name}")
    return MODULE_INDEX * event.size
