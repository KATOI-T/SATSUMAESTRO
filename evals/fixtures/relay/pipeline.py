"""イベント処理パイプライン。"""

from importlib import import_module

from emitter import emit
from normalize import normalize
from transport import send_with_retry


def dispatch_all(events):
    """イベントを正規化し、route 対応のハンドラで処理して結果値リストを返す。"""
    results = []
    for event in map(normalize, events):
        if event.status == "ERROR":
            emit(f"drop {event.name}")
            continue
        handler = import_module(f"handler_{event.route:02d}")
        results.append(handler.process(event))
    send_with_retry(results)
    return results
