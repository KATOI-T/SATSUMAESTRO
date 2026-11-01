"""読み込み（prefill）の時間の見積もりと、最初の 1 字までの待ちの延長。"""

from __future__ import annotations

import re
import statistics
from collections import deque

_SHAPE_TOKENS = 80_000
DEFAULT_RATE = 300.0
_MIN_SAMPLE_TOKENS = 2_000
_MIN_SAMPLE_S = 1.0
_SAMPLES = 5
SAFETY = 1.5

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)?")
_UNIT_S = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001, None: 1.0}


def _work(tokens: int) -> float:
    """``tokens`` を先頭から読む手間（速さ ``r0`` で割ると秒）。"""
    return tokens + tokens * tokens / _SHAPE_TOKENS


class PrefillClock:
    """会話の中で読み込みの速さを測り、読み直しに掛かる秒を見積もる。"""

    def __init__(self) -> None:
        self._rates: deque[float] = deque(maxlen=_SAMPLES)

    @property
    def rate(self) -> float:
        """量 0 の極限での読み込みの速さ（tok/s）。"""
        return statistics.median(self._rates) if self._rates else DEFAULT_RATE

    @property
    def measured(self) -> bool:
        return bool(self._rates)

    def note(self, cached: int, total: int, seconds: float) -> None:
        """1 回の要求の標本を足す（先頭 ``cached`` はキャッシュから読み、残りを読んだ）。"""
        if total - cached < _MIN_SAMPLE_TOKENS or seconds < _MIN_SAMPLE_S:
            return
        self._rates.append((_work(total) - _work(cached)) / seconds)

    def seconds(self, cached: int, total: int) -> float:
        """先頭 ``cached`` がキャッシュにあるとき、``total`` を読み終えるまでの秒。"""
        return max(_work(total) - _work(max(cached, 0)), 0.0) / self.rate

    def first_byte_timeout(
        self, cached: int, total: int, silence_s: float, cap_s: float
    ) -> float | None:
        """最初の 1 字までに待つ秒（無音の上限＋読み込みの見積もり）。"""
        need = self.seconds(cached, total) * SAFETY
        if need <= silence_s / 2:
            return None
        return min(silence_s + need, max(cap_s, silence_s))


def keep_alive_seconds(value: str | None) -> float | None:
    """Ollama の ``keep_alive``（``30m``・``1h30m``・``300``）を秒にする。"""
    if not value:
        return None
    text = value.strip()
    if text.startswith("-"):
        return None
    total = 0.0
    position = 0
    for match in _DURATION_PART.finditer(text):
        if match.start() != position:
            return None
        total += float(match.group(1)) * _UNIT_S[match.group(2)]
        position = match.end()
    return total if position == len(text) else None


__all__ = ["DEFAULT_RATE", "SAFETY", "PrefillClock", "keep_alive_seconds"]
