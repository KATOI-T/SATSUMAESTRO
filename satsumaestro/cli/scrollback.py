"""端末の幅が変わったら、それまでの出力を新しい幅で描き直すための Console。"""

from __future__ import annotations

from collections import deque
from typing import Any, TextIO

from rich.console import Console
from rich.control import Control
from rich.segment import ControlType, Segment
from rich.text import Text, TextType

# 覚えておく出力の数（print の回数）。古いものから捨て、描き直しの時間もこれで頭打ちになる
MAX_RECORDS = 2000
# 画面を消す → スクロールバックを消す → 左上へ。control を空でない組にして幅 0 と数えさせる
_CLEAR_ALL = Segment("\x1b[2J\x1b[3J\x1b[H", None, ((ControlType.CURSOR_FORWARD, 0),))


class ScrollbackConsole(Console):
    """``keep_scrollback`` の後の端末への出力を覚え、``redraw`` で今の幅に描き直す Console。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._records: deque[tuple[tuple[Any, ...], dict[str, Any]]] = deque(maxlen=MAX_RECORDS)
        self._keeping = False
        self._widths: set[int] = set()
        self._partial = False

    def keep_scrollback(self) -> None:
        """ここから後の出力を覚え始める（端末でなければ何もしない）。"""
        self._keeping = self.is_terminal

    def print(self, *objects: Any, **kwargs: Any) -> None:
        # 最下部の描き直し領域（Control だけの出力）は覚えない
        if self._keeping and not all(isinstance(obj, Control) for obj in objects or (None,)):
            self._remember(objects, kwargs)
        super().print(*objects, **kwargs)

    def input(
        self,
        prompt: TextType = "",
        *,
        markup: bool = True,
        emoji: bool = True,
        password: bool = False,
        stream: TextIO | None = None,
    ) -> str:
        answer = super().input(prompt, markup=markup, emoji=emoji, password=password, stream=stream)
        # 答えは端末が映すだけで print を通らないので、問いの後ろに続けて覚える
        if self._keeping:
            self._remember((Text("" if password else answer),), {})
        return answer

    def needs_redraw(self) -> bool:
        """覚えている出力に今と違う幅で描いたものがあるか（末尾だけ出し直した後も真）。"""
        return bool(self._records) and (self._partial or self._widths != {self.width})

    def redraw(self, rows: int | None = None) -> bool:
        """画面とスクロールバックを消し、末尾の ``rows`` 行分か全部を出し直す（全部かを返す）。"""
        records = list(self._records)
        start = 0 if rows is None else self._tail_start(records, rows)
        clear = Control()
        clear.segment = _CLEAR_ALL
        with self:  # まとめて 1 回で書く（消えた画面を見せない）
            self.control(clear)
            for objects, kwargs in records[start:]:
                super().print(*objects, **kwargs)
        self._widths = {self.width}
        self._partial = start > 0
        return not self._partial

    def _tail_start(self, records: list[tuple[tuple[Any, ...], dict[str, Any]]], rows: int) -> int:
        """末尾から ``rows`` 行を埋めるのに要る、最初の出力の位置。"""
        lines = 0
        for index in range(len(records) - 1, 0, -1):
            objects, kwargs = records[index]
            with self.capture() as capture:
                super().print(*objects, **kwargs)
            lines += capture.get().count("\n")
            if lines >= rows:
                return index
        return 0

    def _remember(self, objects: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._records.append((objects, kwargs))
        self._widths.add(self.width)
