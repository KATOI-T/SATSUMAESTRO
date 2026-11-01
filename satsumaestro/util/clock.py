"""壁時計と単調時計のミリ秒。"""

from __future__ import annotations

import time


def now_ms() -> int:
    """現在時刻を UNIX エポックからのミリ秒（整数）で返す。"""
    return time.time_ns() // 1_000_000


def monotonic_ms() -> int:
    """経過時間を測るための単調時計のミリ秒（値そのものに意味は無く、差だけを使う）。"""
    return time.monotonic_ns() // 1_000_000
