"""画面最下部の描き直し領域（Rich の ``Live`` の置き換え）。"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

from rich.console import (
    Console,
    ConsoleOptions,
    ConsoleRenderable,
    RenderableType,
    RenderHook,
    RenderResult,
)
from rich.control import Control
from rich.segment import ControlType, Segment

_NOOP: tuple[tuple[ControlType, int], ...] = ((ControlType.CURSOR_FORWARD, 0),)
_SYNC_BEGIN = Segment("\x1b[?2026h", None, _NOOP)
_SYNC_END = Segment("\x1b[?2026l", None, _NOOP)
_ERASE_LINE = Control((ControlType.ERASE_IN_LINE, 2))
_CR = Control(ControlType.CARRIAGE_RETURN)


def _same_line(a: list[Segment], b: list[Segment]) -> bool:
    """行が同じか（文字と色）。"""
    return a == b


class _Frame:
    """1 フレーム分の出力。"""

    def __init__(self, region: LiveRegion, *, full: bool) -> None:
        self._region = region
        self._full = full

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        region = self._region
        old = region._lines
        renderable = region._renderable
        new = console.render_lines(renderable, options, pad=False) if renderable is not None else []
        if region._width != options.max_width:
            self._full = True
            region._width = options.max_width
        if not old and not new:
            return
        start = 0
        if not self._full:
            limit = min(len(old), len(new))
            while start < limit and _same_line(old[start], new[start]):
                start += 1
            if start == len(old) == len(new):
                return
        yield _SYNC_BEGIN
        if start < len(old):
            up = len(old) - 1 - start
            yield _CR
            if up > 0:
                yield Control((ControlType.CURSOR_UP, up))
        for index in range(start, len(new)):
            if index > start or (old and start == len(old)):
                yield Segment.line()
            yield _ERASE_LINE
            yield from new[index]
        if len(new) < len(old):
            extra = len(old) - len(new)
            if start < len(new):
                for _ in range(extra):
                    yield Segment.line()
                    yield _ERASE_LINE
                yield Control((ControlType.CURSOR_UP, extra))
            else:
                yield _ERASE_LINE
                for _ in range(extra - 1):
                    yield Segment.line()
                    yield _ERASE_LINE
                up = extra if new else extra - 1
                if up > 0:
                    yield Control((ControlType.CURSOR_UP, up))
            yield _CR
        yield _SYNC_END
        region._lines = new


class _RefreshThread(threading.Thread):
    """一定間隔で :meth:`LiveRegion.refresh` を呼ぶ（Rich の Live と同じ）。"""

    def __init__(self, region: LiveRegion, fps: float) -> None:
        super().__init__(daemon=True)
        self._region = region
        self._interval = 1 / fps
        self.done = threading.Event()

    def run(self) -> None:
        while not self.done.wait(self._interval):
            if not self.done.is_set():
                self._region.refresh()


class LiveRegion(RenderHook):
    """画面最下部の描き直し領域。"""

    def __init__(
        self, console: Console, *, fps: float = 12.5, on_print: Callable[[], None] | None = None
    ) -> None:
        self.console = console
        self._fps = fps
        self._on_print = on_print
        self._lock = threading.RLock()
        self._renderable: RenderableType | None = None
        self._lines: list[list[Segment]] = []
        self._width = console.options.max_width
        self._full_next = True
        self._started = False
        self._refreshing = False
        self._thread: _RefreshThread | None = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._lines = []
            self._full_next = True
            self.console.show_cursor(False)
            self.console.push_render_hook(self)
            if self._fps > 0:
                self._thread = _RefreshThread(self, self._fps)
                self._thread.start()

    def update(self, renderable: RenderableType) -> None:
        """次のフレームで描く内容を差し替える（描くのはタイマー。``Live.update`` と同じ）。"""
        with self._lock:
            self._renderable = renderable

    def refresh(self) -> None:
        """いまの内容を描く（差分だけ）。"""
        with self._lock:
            if not self._started:
                return
            self._refreshing = True
            try:
                with self.console:
                    self.console.print(Control())
            finally:
                self._refreshing = False

    def stop(self) -> None:
        """領域を消して畳む。"""
        with self._lock:
            if not self._started:
                return
            if self._thread is not None:
                self._thread.done.set()
                self._thread = None
            self._renderable = None
            try:
                self.refresh()
            finally:
                self._started = False
                self.console.pop_render_hook()
                self.console.show_cursor(True)

    def process_renderables(self, renderables: list[ConsoleRenderable]) -> list[ConsoleRenderable]:
        """``console.print`` を受ける。"""
        with self._lock:
            if self._refreshing:
                frame = _Frame(self, full=self._full_next)
                self._full_next = False
                return [frame]
            if self._on_print is not None:
                self._on_print()
            cleared = _Clear(self._lines)
            self._lines = []
            self._full_next = False
            frame = _Frame(self, full=True)
            return [_Raw([_SYNC_BEGIN]), cleared, *renderables, frame, _Raw([_SYNC_END])]


class _Clear:
    """前フレームの全行を消し、カーソルを領域の先頭に置く（Rich の ``position_cursor`` と同じ）。"""

    def __init__(self, lines: list[list[Segment]]) -> None:
        self._height = len(lines)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self._height == 0:
            return
        yield _CR
        yield _ERASE_LINE
        for _ in range(self._height - 1):
            yield Control((ControlType.CURSOR_UP, 1))
            yield _ERASE_LINE


class _Raw:
    """生の制御セグメントをそのまま出す renderable。"""

    def __init__(self, segments: Iterable[Segment]) -> None:
        self._segments = list(segments)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield from self._segments
