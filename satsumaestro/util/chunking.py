"""1 回の書き込みの分量の目安と、段の予算の換算定数。"""

from __future__ import annotations

CHUNK_LINES = 150
CHUNK_KB = 6

LINES = f"~{CHUNK_LINES} lines"
LINES_AND_SIZE = f"{LINES} (~{CHUNK_KB} KB)"
SIZE = f"~{CHUNK_KB} KB"

CHARS_PER_LINE = 30
CHARS_PER_TOKEN = 2.5
STEP_BUDGET_SAFETY = 0.5
MIN_RATE_SAMPLE_TOKENS = 300
STEP_BUDGET_MIN_LINES = 60
STEP_BUDGET_MAX_LINES = 400


def lines_estimate(tokens: float) -> int:
    """トークン数を行数へ換算する（丸めも上下限も無し）。"""
    return int(tokens * CHARS_PER_TOKEN / CHARS_PER_LINE)
