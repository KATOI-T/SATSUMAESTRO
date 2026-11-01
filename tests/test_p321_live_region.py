"""画面最下部の描き直し領域（``cli/live_region.py``）。"""

from __future__ import annotations

import io
import re

from rich.console import Console, Group
from rich.text import Text

from satsumaestro.cli.live_region import LiveRegion
from satsumaestro.cli.render import RichRenderer
from satsumaestro.engine.events import TextDelta

BSU, ESU = "\x1b[?2026h", "\x1b[?2026l"
ERASE = "\x1b[2K"
UP = re.compile(r"\x1b\[(\d+)A")
HIDE, SHOW = "\x1b[?25l", "\x1b[?25h"


def _console(width: int = 40) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, force_terminal=True, width=width, height=20, color_system=None), buf


def _take(buf: io.StringIO) -> str:
    out = buf.getvalue()
    buf.seek(0)
    buf.truncate()
    return out


def _lines(*texts: str) -> Group:
    return Group(*(Text(t) for t in texts))


def test_first_frame_writes_every_line_inside_one_synchronized_update() -> None:
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    assert _take(buf) == HIDE
    region.update(_lines("body 1", "body 2", "status"))
    region.refresh()
    out = _take(buf)
    assert out.startswith(BSU) and out.endswith(ESU)
    assert out.count(ERASE) == 3 and UP.search(out) is None
    assert out.index("body 1") < out.index("body 2") < out.index("status")


def test_unchanged_frame_writes_nothing() -> None:
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("body", "status"))
    region.refresh()
    _take(buf)
    region.refresh()
    assert _take(buf) == ""


def test_only_the_changed_last_line_is_rewritten() -> None:
    """スピナーだけが動く通常のフレーム: 状態行の 1 行だけ消して書く。"""
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("body 1", "body 2", "status A"))
    region.refresh()
    _take(buf)
    region.update(_lines("body 1", "body 2", "status B"))
    region.refresh()
    out = _take(buf)
    assert out == f"{BSU}\r{ERASE}status B{ESU}"


def test_change_in_the_middle_rewrites_from_that_line_down() -> None:
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("a", "b", "c", "status"))
    region.refresh()
    _take(buf)
    region.update(_lines("a", "B", "c", "status"))
    region.refresh()
    out = _take(buf)
    assert UP.findall(out) == ["2"]
    assert out == f"{BSU}\r\x1b[2A{ERASE}B\n{ERASE}c\n{ERASE}status{ESU}"


def test_appended_lines_go_below_without_touching_the_old_ones() -> None:
    """本文が 1 行伸びたフレーム（末尾の段落が折り返した）: 前の行は消さず、下に書く。"""
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("body", "status"))
    region.refresh()
    _take(buf)
    region.update(_lines("body", "more", "status"))
    region.refresh()
    out = _take(buf)
    assert UP.findall(out) == []
    assert out == f"{BSU}\r{ERASE}more\n{ERASE}status{ESU}"


def test_pure_append_keeps_the_last_old_line() -> None:
    """前の行が全部そのままで下に増えるだけ: 最終行を消さずに改行して書く。"""
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("a", "b"))
    region.refresh()
    _take(buf)
    region.update(_lines("a", "b", "c"))
    region.refresh()
    assert _take(buf) == f"{BSU}\n{ERASE}c{ESU}"


def test_shrinking_frame_erases_only_the_leftover_lines() -> None:
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("a", "b", "c", "status"))
    region.refresh()
    _take(buf)
    region.update(_lines("a", "status"))
    region.refresh()
    out = _take(buf)
    assert UP.findall(out) == ["2", "2"]
    assert out.count(ERASE) == 3
    assert out.endswith(f"\r{ESU}")


def test_shrinking_without_a_changed_line_erases_from_the_cut() -> None:
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("a", "b", "c"))
    region.refresh()
    _take(buf)
    region.update(_lines("a"))
    region.refresh()
    out = _take(buf)
    assert out == f"{BSU}\r\x1b[1A{ERASE}\n{ERASE}\x1b[2A\r{ESU}"


def test_printing_while_live_clears_the_region_and_redraws_it_below() -> None:
    """確定内容（``console.print``）は領域の上に出て、領域は全行描き直される。"""
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("tail", "status"))
    region.refresh()
    _take(buf)
    console.print(Text("committed"))
    out = _take(buf)
    assert out.startswith(BSU) and out.endswith(ESU)
    head, _, rest = out.partition("committed")
    assert head.count(ERASE) == 2 and UP.findall(head) == ["1"]
    assert rest.count(ERASE) == 2 and "tail" in rest and "status" in rest
    assert out.index("committed") < out.index("tail") < out.index("status")


def test_stop_erases_the_region_and_shows_the_cursor() -> None:
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("tail", "status"))
    region.refresh()
    _take(buf)
    region.stop()
    out = _take(buf)
    assert out == f"{BSU}\r\x1b[1A{ERASE}\n{ERASE}\x1b[1A\r{ESU}{SHOW}"
    console.print(Text("after"))
    assert _take(buf) == "after\n"


def test_width_change_forces_a_full_redraw() -> None:
    console, buf = _console()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("a", "b"))
    region.refresh()
    _take(buf)
    console.width = 30
    region.update(_lines("a", "b"))
    region.refresh()
    out = _take(buf)
    assert out.count(ERASE) == 2 and UP.findall(out) == ["1"]


def test_renderer_uses_the_region_and_leaves_scrollback_unchanged() -> None:
    """RichRenderer 経由: 確定したスクロールバックは Live のときと同じ 1 本の Markdown。"""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=60, height=20, color_system=None)
    r = RichRenderer(console=console)
    r.start_turn()
    assert isinstance(r._live, LiveRegion)
    r.on_assistant_event(TextDelta(text="first paragraph\n\nsecond "))
    r.on_assistant_event(TextDelta(text="paragraph"))
    r.end_turn()
    out = buf.getvalue()
    assert r._live is None
    assert out.count(BSU) == out.count(ESU) >= 1
    plain = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", out)
    assert "first paragraph" in plain and "second paragraph" in plain
    tail = plain.rstrip().split("\r")[-1]
    assert tail.startswith("second paragraph") and tail.rstrip().endswith("— 出力 8")


def test_full_width_committed_line_is_not_cropped_by_the_sync_markers() -> None:
    """同期出力の列は幅 0 として扱われ、同じ行に出る確定本文を切り詰めない。"""
    console, buf = _console(width=20)
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(_lines("status"))
    region.refresh()
    _take(buf)
    console.print(Text("x" * 20))
    out = _take(buf)
    assert "x" * 20 in out
