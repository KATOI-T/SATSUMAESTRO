"""応答の最初の中身が届いた時刻を、会話ループへ届ける経路。"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class StreamTiming:
    """1 回のストリームを始めた時刻と、最初の中身（思考・本文・ツール呼び出し）が届いた時刻。"""

    started_at: float = field(default_factory=time.monotonic)
    first_chunk_at: float | None = None


_current: ContextVar[StreamTiming | None] = ContextVar("stream_timing", default=None)


@contextmanager
def bind(timing: StreamTiming) -> Iterator[StreamTiming]:
    """この区間のストリームの時刻を ``timing`` に記録する。"""
    token = _current.set(timing)
    try:
        yield timing
    finally:
        _current.reset(token)


def note_first_chunk() -> None:
    """最初の中身が届いた（2 回目以降は何もしない）。"""
    timing = _current.get()
    if timing is not None and timing.first_chunk_at is None:
        timing.first_chunk_at = time.monotonic()


__all__ = ["StreamTiming", "bind", "note_first_chunk"]
