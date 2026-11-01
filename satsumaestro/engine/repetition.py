"""応答の反復をストリーム中に**数値で**検知する。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

PARAGRAPH_MIN_CHARS = 40
PARAGRAPH_WINDOW = 12
PARAGRAPH_MAX_SEEN = 10
LINE_MIN_CHARS = 20
LINE_WINDOW = 30
LINE_MAX_SEEN = 27
SNIPPET_CHARS = 60

Unit = Literal["paragraph", "line"]


@dataclass(frozen=True)
class RepetitionVerdict:
    """反復と判定した根拠（差し戻し文・画面・記録に載せる数字）。"""

    unit: Unit
    seen: int
    window: int
    repeats: int
    snippet: str
    chars: int


class RepetitionAbort(Exception):
    """ストリームを打ち切る内部信号（``_stream_render`` → ``run_turn``）。"""

    def __init__(self, verdict: RepetitionVerdict) -> None:
        self.verdict = verdict
        super().__init__(
            f"repetition detected: {verdict.seen}/{verdict.window} recent {verdict.unit}s "
            f"already seen (top {verdict.unit} x{verdict.repeats})"
        )


class _Head:
    """1 種類の単位（段落／行）についての既出率の追跡。"""

    def __init__(self, min_chars: int, window: int, max_seen: int) -> None:
        self.min_chars = min_chars
        self.window = window
        self.max_seen = max_seen
        self._counts: dict[str, int] = {}
        self._recent: deque[bool] = deque(maxlen=window)

    def push(self, unit: str) -> bool:
        """単位を 1 つ観測し、反復の閾値に達したら ``True``。"""
        text = unit.strip()
        if len(text) < self.min_chars:
            return False
        seen = text in self._counts
        self._counts[text] = self._counts.get(text, 0) + 1
        self._recent.append(seen)
        return len(self._recent) == self.window and sum(self._recent) >= self.max_seen

    @property
    def seen(self) -> int:
        return sum(self._recent)

    def top(self) -> tuple[str, int]:
        """最多の単位とその出現回数。"""
        text, count = max(self._counts.items(), key=lambda item: item[1])
        return text, count


class RepetitionDetector:
    """本文のデルタを食わせ、反復の閾値に達した時点で :class:`RepetitionVerdict` を返す。"""

    def __init__(self) -> None:
        self._chars = 0
        self._pending = ""
        self._paragraph = ""
        self._paragraphs = _Head(PARAGRAPH_MIN_CHARS, PARAGRAPH_WINDOW, PARAGRAPH_MAX_SEEN)
        self._lines = _Head(LINE_MIN_CHARS, LINE_WINDOW, LINE_MAX_SEEN)

    @property
    def chars(self) -> int:
        return self._chars

    def feed(self, text: str) -> RepetitionVerdict | None:
        self._chars += len(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            verdict = self._line(line)
            if verdict is not None:
                return verdict
        return None

    def _line(self, line: str) -> RepetitionVerdict | None:
        if self._lines.push(line):
            return self._verdict("line", self._lines)
        if line.strip():
            self._paragraph = f"{self._paragraph}\n{line}" if self._paragraph else line
            return None
        paragraph, self._paragraph = self._paragraph, ""
        if paragraph and self._paragraphs.push(paragraph):
            return self._verdict("paragraph", self._paragraphs)
        return None

    def _verdict(self, unit: Unit, head: _Head) -> RepetitionVerdict:
        text, count = head.top()
        return RepetitionVerdict(
            unit=unit,
            seen=head.seen,
            window=head.window,
            repeats=count,
            snippet=" ".join(text.split())[:SNIPPET_CHARS],
            chars=self._chars,
        )
