"""ツール実行の表示のテスト。"""

from __future__ import annotations

import io
import time
from pathlib import Path

from rich.console import Console

from satsumaestro.cli.render import RichRenderer, _done_line, _state_label, change_preview
from satsumaestro.cli.tool_display import describe, known_tool_names, result_note
from satsumaestro.engine.events import TextDelta, ToolUse
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.tools import default_registry
from satsumaestro.tools.builtins.rag_search import RagSearch
from satsumaestro.tools.builtins.skill import Skill
from satsumaestro.tools.mcp_bridge import mcp_tool_name


def _renderer(*, tty: bool = False, cwd: Path | None = None) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=tty, width=100, color_system=None)
    return RichRenderer(console=console, cwd=cwd), buf


def _call(name: str, cid: str = "1", **inp: object) -> ToolUse:
    return ToolUse(id=cid, name=name, input=inp)


def _result(call: ToolUse, output: str = "", *, error: bool = False) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=call.id, tool_name=call.name, output=output, is_error=error)


def _plain(renderable: object) -> str:
    c = Console(file=io.StringIO(), force_terminal=False, width=100)
    c.print(renderable)
    assert isinstance(c.file, io.StringIO)
    return c.file.getvalue()


def test_every_registered_tool_has_a_label() -> None:
    registry = default_registry(strict_schema=False, orchestration=True, workers=True)
    registry.register(RagSearch)
    registry.register(Skill)
    names = {s["name"] for s in registry.core_schemas() + registry.schemas_for(set())}
    names |= set(registry.deferred_names())
    missing = names - known_tool_names()
    assert not missing, f"表示ラベル未定義: {sorted(missing)}"
    for name in names:
        assert describe(_call(name)).label != name


def test_mcp_and_unknown_tools() -> None:
    mcp = describe(_call(mcp_tool_name("demo", "echo")))
    assert mcp.label == "MCP（demo）" and mcp.items == ("echo",)
    unknown = describe(_call("mystery_tool"))
    assert unknown.label == "mystery_tool" and unknown.items == ()


def test_details_are_single_line_and_clipped() -> None:
    d = describe(_call("bash", command="echo a\necho b" + "x" * 200))
    assert "\n" not in d.items[0] and d.items[0].endswith("…")


def test_parallel_reads_group_under_one_heading() -> None:
    r, buf = _renderer()
    r.start_turn()
    a = _call("read_file", "a", path="docs/guide.md")
    b = _call("read_file", "b", path="docs/spec/generation-limits.md")
    r.on_tool_start(a)
    r.on_tool_start(b)
    r.on_tool_result(_result(a, "x\n" * 60))
    r.on_tool_result(_result(b, "y\n" * 70))
    r.end_turn()
    out = buf.getvalue()
    assert out.count("・読み込み") == 1
    assert "    - docs/guide.md" in out
    assert "    - docs/spec/generation-limits.md" in out
    assert "read_file" not in out and "行" not in out


def test_other_output_breaks_the_group() -> None:
    r, buf = _renderer()
    r.start_turn()
    a = _call("read_file", "a", path="a.py")
    r.on_tool_start(a)
    r.on_tool_result(_result(a, "x"))
    r.on_assistant_event(TextDelta(text="確認しました。"))
    r.on_tool_start(b := _call("read_file", "b", path="b.py"))
    r.on_tool_result(_result(b, "y"))
    r.end_turn()
    assert buf.getvalue().count("・読み込み") == 2


def test_different_labels_get_their_own_headings() -> None:
    r, buf = _renderer()
    r.start_turn()
    g = _call("grep_search", "g", pattern="def total")
    r.on_tool_start(g)
    r.on_tool_result(_result(g, "cart.py:12:def total():"))
    b = _call("bash", "b", command="make test")
    r.on_tool_start(b)
    r.on_tool_result(_result(b, "ok"))
    r.end_turn()
    out = buf.getvalue()
    assert '・検索\n    - "def total"（1 件）' in out
    assert "・コマンド実行\n    - make test" in out and "ok" in out


def test_error_is_shown_on_the_item() -> None:
    r, buf = _renderer()
    r.start_turn()
    a = _call("read_file", "a", path="missing.py")
    r.on_tool_start(a)
    r.on_tool_result(_result(a, "file not found: missing.py\ntrace", error=True))
    e = _call("edit_file", "e", path="a.py", old="x", new="y")
    r.on_tool_start(e)
    r.on_tool_result(_result(e, "old string not found", error=True))
    r.end_turn()
    out = buf.getvalue()
    assert "- missing.py  ✗ file not found: missing.py\n      trace" in out
    assert "✗ a.py — old string not found" in out


def test_status_line_uses_label_and_tracks_remaining_parallel_tools() -> None:
    r, _ = _renderer()
    r.start_turn()
    a = _call("read_file", "a", path="a.py")
    b = _call("bash", "b", command="sleep 1")
    r.on_tool_start(a)
    r.on_tool_start(b)
    assert r._phase_label == "コマンド実行中…"
    r.on_tool_result(_result(a, "x"))
    assert r._phase_label == "コマンド実行中…"
    r.on_tool_result(_result(b, "ok"))
    assert r._phase_label is None and r._state == "waiting_next"


def test_search_notes() -> None:
    glob = _call("glob_search", pattern="**/*.py")
    assert result_note(glob, _result(glob, "a.py\nb.py")) == "2 件"
    assert result_note(glob, _result(glob, "")) == "該当なし"
    assert result_note(glob, _result(glob, "a.py\n... (truncated at 1 entries)")) == "1 件以上"
    grep = _call("grep_search", pattern="x")
    content = "a.py:3:x = 1\na.py-4-y\n--\nb.py:9:x()"
    assert result_note(grep, _result(grep, content)) == "2 件"
    files = _call("grep_search", pattern="x", output_mode="files_with_matches")
    assert result_note(files, _result(files, "a.py\nb.py")) == "2 ファイル"
    count = _call("grep_search", pattern="x", output_mode="count")
    assert result_note(count, _result(count, "a.py:3\nb.py:4")) == "7 件"
    read = _call("read_file", path="a.py")
    assert result_note(read, _result(read, "a\nb")) is None


def test_edit_diff_uses_file_line_numbers(tmp_path: Path) -> None:
    src = "".join(f"line{i}\n" for i in range(1, 31))
    (tmp_path / "m.py").write_text(src, encoding="utf-8")
    call = _call("edit_file", path="m.py", old="line20\n", new="LINE20\nadded\n")
    out = _plain(change_preview(call, tmp_path))
    assert "20 - line20" in out and "20 + LINE20" in out and "21 + added" in out
    assert "18   line18" in out
    assert "line1\n" not in out and "@@" not in out


def test_edit_replace_all_shows_every_site(tmp_path: Path) -> None:
    src = "".join("foo\n" if i in (3, 25) else f"l{i}\n" for i in range(1, 31))
    (tmp_path / "m.py").write_text(src, encoding="utf-8")
    call = _call("edit_file", path="m.py", old="foo", new="bar", replace_all=True)
    out = _plain(change_preview(call, tmp_path))
    assert "3 + bar" in out and "25 + bar" in out and "⋮" in out


def test_write_existing_file_shows_only_changed_lines(tmp_path: Path) -> None:
    before = "".join(f"row{i}\n" for i in range(1, 51))
    (tmp_path / "w.txt").write_text(before, encoding="utf-8")
    after = before.replace("row40\n", "ROW40\n")
    out = _plain(change_preview(_call("write_file", path="w.txt", content=after), tmp_path))
    assert "40 - row40" in out and "40 + ROW40" in out
    assert "row10" not in out


def test_write_new_file_is_marked(tmp_path: Path) -> None:
    call = _call("write_file", path="new.py", content="a\nb\n")
    out = _plain(change_preview(call, tmp_path))
    assert "（新規ファイル）" in out and "1 + a" in out and "2 + b" in out


def test_falls_back_when_file_unreadable_or_outside(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("x = 1\n", encoding="utf-8")
    call = _call("edit_file", path=str(outside), old="x = 1", new="x = 2")
    out = _plain(change_preview(call, ws))
    assert "-x = 1" in out and "1 - x = 1" not in out
    (ws / "a.py").write_text("y = 0\n", encoding="utf-8")
    out = _plain(change_preview(_call("edit_file", path="a.py", old="zzz", new="w"), ws))
    assert "-zzz" in out


def test_renderer_shows_numbered_diff_under_item(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a\nb\nc\n", encoding="utf-8")
    r, buf = _renderer(tty=True, cwd=tmp_path)
    r.start_turn()
    e = _call("edit_file", "e", path="a.py", old="b", new="B")
    r.on_tool_start(e)
    r.on_tool_result(_result(e, "replaced 1 occurrence(s) in a.py"))
    r.end_turn()
    out = buf.getvalue()
    assert "・編集" in out and "- a.py" in out
    assert "2 - b" in out and "2 + B" in out
    assert "replaced 1 occurrence" not in out


def test_silent_command_shows_that_it_finished() -> None:
    r, buf = _renderer()
    r.start_turn()
    call = _call("bash", command="mkdir -p dashboard")
    r.on_tool_start(call)
    r.on_tool_result(_result(call, ""))
    out = buf.getvalue()
    assert "・コマンド実行" in out and "- mkdir -p dashboard" in out
    assert "✓ 完了" in out
    assert r._state == "waiting_next"
    assert _state_label(r._state) == "次の操作を考え中…"


def test_turn_start_keeps_the_plain_waiting_label() -> None:
    r, _ = _renderer()
    r.start_turn()
    assert r._state == "waiting" and _state_label(r._state) == "考え中…"


def test_failed_tool_shows_the_error_not_a_done_line() -> None:
    r, buf = _renderer()
    r.start_turn()
    call = _call("bash", command="false")
    r.on_tool_start(call)
    r.on_tool_result(_result(call, "exit 1", error=True))
    out = buf.getvalue()
    assert "✗" in out and "✓ 完了" not in out


def test_done_line_shows_elapsed() -> None:
    assert _done_line(time.monotonic() - 0.15).strip().startswith("✓ 完了 0.")
    assert _done_line(time.monotonic() - 42.0).strip() == "✓ 完了 42s"
    assert _done_line(time.monotonic()).strip() == "✓ 完了"
    assert _done_line(None).strip() == "✓ 完了"
