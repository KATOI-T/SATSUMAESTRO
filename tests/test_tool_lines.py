"""ツールの行（``⏺ 動作(対象)`` と ``⎿`` の結果の行）と本文の ``⏺``。"""

from __future__ import annotations

import io
import re
from pathlib import Path

from rich.console import Console
from rich.text import Text

from satsumaestro.cli.markdown import ReplyMarkdown
from satsumaestro.cli.render import RichRenderer, _Gutter
from satsumaestro.engine.events import TextDelta, ToolUse
from satsumaestro.session.models import ToolResultBlock

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _renderer(*, color: bool = False, cwd: Path | None = None) -> tuple[RichRenderer, io.StringIO]:
    """TTY のレンダラ。最下部の領域の描き直しは混ぜず、確定した中身だけを見る。"""
    buf = io.StringIO()
    console = Console(
        file=buf, force_terminal=True, width=100, color_system="standard" if color else None
    )
    r = RichRenderer(console=console, cwd=cwd)
    r._refresh_live = lambda: None  # type: ignore[method-assign]
    return r, buf


def _live_text(r: RichRenderer) -> str:
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=100).print(r._live_renderable())
    return out.getvalue()


def _call(name: str, cid: str = "1", **inp: object) -> ToolUse:
    return ToolUse(id=cid, name=name, input=inp)


def _result(call: ToolUse, output: str = "", *, error: bool = False) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=call.id, tool_name=call.name, output=output, is_error=error)


def test_tool_line_waits_in_the_live_area_until_the_result() -> None:
    r, buf = _renderer()
    r.start_turn()
    call = _call("bash", command="make test")
    r.on_tool_start(call)
    assert "コマンド実行" not in buf.getvalue()
    assert "⏺ コマンド実行(make test)" in _live_text(r)
    r.on_tool_result(_result(call, "ok\n"))
    assert "⏺ コマンド実行(make test)\n  ⎿  ok\n     ✓ 完了" in buf.getvalue()
    assert "コマンド実行(" not in _live_text(r)


def test_mark_is_green_on_success_and_red_on_failure() -> None:
    r, buf = _renderer(color=True)
    r.start_turn()
    ok, bad = _call("read_file", "1", path="a.py"), _call("read_file", "2", path="b.py")
    r.on_tool_start(ok)
    r.on_tool_result(_result(ok, "x"))
    r.on_tool_start(bad)
    r.on_tool_result(_result(bad, "file not found: b.py", error=True))
    out = buf.getvalue()
    assert "\x1b[32m⏺" in out and "\x1b[31m⏺" in out


def test_pause_shows_the_running_line_before_the_card(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a\nb\nc\n", encoding="utf-8")
    r, buf = _renderer(cwd=tmp_path)
    r.start_turn()
    call = _call("edit_file", path="a.py", old="b", new="B")
    r.on_tool_start(call)
    r.pause()  # 権限カードの直前
    before_card = buf.getvalue()
    # 差分は実行の前に作ったもの（カードの文脈として先に出る）。
    assert "⏺ 編集(a.py)\n  ⎿  " in before_card and " 2 +B" in before_card
    (tmp_path / "a.py").write_text("a\nB\nc\n", encoding="utf-8")  # 実行
    r.resume()
    r.on_tool_result(_result(call, "replaced 1 occurrence(s) in a.py"))
    out = buf.getvalue()
    assert out.count("⏺ 編集(a.py)") == 1
    assert out[len(before_card) :] == "  ⎿  ✓ 完了\n"


def test_end_of_turn_keeps_a_tool_that_never_returned() -> None:
    r, buf = _renderer()
    r.start_turn()
    r.on_tool_start(_call("bash", command="sleep 30"))
    r.end_turn()  # 中断などで結果が来なかった
    assert "⏺ コマンド実行(sleep 30)" in buf.getvalue()


def test_live_area_lists_running_tools_above_the_status() -> None:
    r, _ = _renderer()
    r.start_turn()
    r.on_tool_start(_call("bash", "1", command="make a"))
    r.on_tool_start(_call("bash", "2", command="make b"))
    rows = [ln for ln in _live_text(r).splitlines() if ln.strip()]
    assert rows[0].startswith("⏺ コマンド実行(make a)")
    assert rows[1].startswith("⏺ コマンド実行(make b)")
    assert "コマンド実行中…" in rows[-1]


def test_body_starts_each_segment_with_the_mark() -> None:
    r, buf = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="first paragraph\n\nsecond paragraph\n\n"))
    call = _call("read_file", path="a.py")
    r.on_tool_start(call)
    r.on_tool_result(_result(call, "x"))
    r.on_assistant_event(TextDelta(text="after the tool"))
    r.end_turn()
    lines = _ANSI.sub("", buf.getvalue()).splitlines()
    assert any(ln.startswith("⏺ first paragraph") for ln in lines)
    assert any(ln.startswith("  second paragraph") for ln in lines)
    assert any(ln.startswith("⏺ after the tool") for ln in lines)


def _gutter_lines(renderable: object, mark: str) -> list[str]:
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=40).print(_Gutter(renderable, Text(mark)))  # type: ignore[arg-type]
    return out.getvalue().split("\n")[:-1]


def test_gutter_marks_the_first_visible_line_and_keeps_empty_lines_empty() -> None:
    # 先頭のリストの前には Rich が空行を置く。印は項目の行に付き、空行は字下げしない。
    lines = _gutter_lines(ReplyMarkdown("- a\n- b"), "⏺ ")
    assert lines[0] == ""
    assert lines[1].startswith("⏺ - a") and lines[2].startswith("  - b")
    lines = _gutter_lines(ReplyMarkdown("one\n\ntwo"), "⏺ ")
    assert lines[0].startswith("⏺ one") and lines[1] == "" and lines[2].startswith("  two")
