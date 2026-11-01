"""サンプル実行のエントリポイント。"""

from aggregate import summarize
from events import Event
from pipeline import dispatch_all

SAMPLE_EVENTS = [
    Event(name="ingest", size=4, route=4),
    Event(name="transform", size=7, route=13),
    Event(name="export", size=9, route=7),
]


def sample_total():
    """サンプルイベント一式を処理した集計値を返す。"""
    return summarize(dispatch_all(list(SAMPLE_EVENTS)))
