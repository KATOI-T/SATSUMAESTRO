"""REPL の描画とツールの表示の英語の画面（日本語の画面は既存のテストが見ている）。"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli.render import (
    RichRenderer,
    _done_line,
    _state_label,
    action_preview,
    change_preview,
)
from satsumaestro.cli.tool_display import describe, error_reason, known_tool_names, result_note
from satsumaestro.engine.events import ToolUse
from satsumaestro.engine.repetition import RepetitionVerdict
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.security.jail import OUTSIDE_WORKSPACE, PATH_ESCAPE_PREFIX
from satsumaestro.session.models import PlanStepRecord, TokenUsage, ToolResultBlock

_JAPANESE = re.compile(r"[\u3000-\u30ff\u4e00-\u9fff\uff00-\uffef]")


@pytest.fixture(autouse=True)
def _english_screen() -> Iterator[None]:
    with pinned_ui_language("en"):
        yield


def _call(name: str, cid: str = "1", **inp: object) -> ToolUse:
    return ToolUse(id=cid, name=name, input=inp)


def _result(call: ToolUse, output: str = "", *, error: bool = False) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=call.id, tool_name=call.name, output=output, is_error=error)


def _renderer(width: int = 160) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=width, color_system=None)
    return RichRenderer(console=console), buf


def _plain(renderable: object) -> str:
    c = Console(file=io.StringIO(), force_terminal=False, width=120)
    c.print(renderable)
    assert isinstance(c.file, io.StringIO)
    return c.file.getvalue()


@pytest.mark.parametrize(
    ("name", "label", "status"),
    [
        ("read_file", "Read", "Reading…"),
        ("glob_search", "Find files", "Finding files…"),
        ("write_file", "Write", "Writing…"),
        ("bash", "Run command", "Running command…"),
        ("agent", "Subagent", "Running subagent…"),
        ("grep_search", "Search", "Searching…"),
        ("lsp", "Code analysis", "Analyzing code…"),
        ("todo_write", "Update ToDo", "Updating ToDo…"),
        ("worker_await_ready", "Wait for worker", "Waiting for worker…"),
    ],
)
def test_tool_labels_and_status_lines(name: str, label: str, status: str) -> None:
    shown = describe(_call(name))
    assert (shown.label, shown.status()) == (label, status)


def test_every_known_tool_has_english_text() -> None:
    for name in known_tool_names():
        shown = describe(_call(name))
        assert not _JAPANESE.search(shown.label + shown.status()), name
    mcp = describe(_call("mcp__demo__echo"))
    assert (mcp.label, mcp.status()) == ("MCP (demo)", "MCP (demo)…")
    assert describe(_call("mystery_tool")).status() == "mystery_tool…"


@pytest.mark.parametrize(
    ("call", "output", "note"),
    [
        (_call("glob_search"), "a.py\nb.py\nc.py", "3 matches"),
        (_call("glob_search"), "a.py", "1 match"),
        (_call("glob_search"), "a.py\nb.py\n... (truncated at 2 results)", "2+ matches"),
        (_call("glob_search"), "", "no matches"),
        (_call("list_directory"), "", "empty directory"),
        (_call("list_directory"), "a.py\nsub/\n(2 entries)", "2 entries"),
        (_call("grep_search", output_mode="files_with_matches"), "a.py\nb.py", "2 files"),
        (_call("grep_search"), "a.py:3:x\na.py:9:y", "2 matches"),
    ],
)
def test_result_notes(call: ToolUse, output: str, note: str) -> None:
    assert result_note(call, _result(call, output)) == note


def test_short_failure_reason() -> None:
    call = _call("read_file", path="/etc/passwd")
    block = _result(call, f"{PATH_ESCAPE_PREFIX}{OUTSIDE_WORKSPACE}: /etc/passwd", error=True)
    assert error_reason(block) == "Outside the working directory"


def test_tool_heading_and_done_line() -> None:
    r, buf = _renderer()
    r.start_turn()
    call = _call("read_file", path="README.md")
    r.on_tool_start(call)
    r.on_tool_result(_result(call, "# hello"))
    listing = _call("list_directory", "3", path=".")
    r.on_tool_start(listing)
    r.on_tool_result(_result(listing, "notes.txt  6 B\n(1 entries)"))
    write = _call("write_file", "2", path="a.txt", content="x\n")
    r.on_tool_start(write)
    r.on_tool_result(_result(write, "wrote 2 bytes"))
    r.end_turn()
    out = buf.getvalue()
    assert "• Read" in out and "- README.md" in out
    assert "• List" in out and "- . (1 entry)" in out
    assert not _JAPANESE.search(out)
    assert "• Write" in out and "✓ Done" in out
    assert _done_line(None).strip() == "✓ Done"


def test_state_labels() -> None:
    assert _state_label("waiting") == "Thinking…"
    assert _state_label("waiting_next") == "Thinking about the next step…"
    assert _state_label("recovering") == "Waiting for the retry…"
    assert _state_label("unknown") is None


def test_step_lines() -> None:
    r, buf = _renderer()
    r.start_turn()
    r.on_plan_step(2, 7, "Write the sidebar")
    r.on_plan_sendback("step 2 is over budget")
    r.on_plan_step_done(
        PlanStepRecord(
            index=2,
            total=7,
            estimated_lines=120,
            budget_lines=280,
            actual_lines=143,
            max_write_lines=143,
            writes=1,
            max_response_s=38.2,
        )
    )
    r.on_plan_step_done(
        PlanStepRecord(
            index=3,
            total=7,
            budget_lines=280,
            actual_lines=400,
            max_write_lines=400,
            edits=1,
            edited_lines=12,
        )
    )
    r.end_turn()
    out = buf.getvalue()
    assert "— Step 2/7: Write the sidebar" in out
    assert "— Plan sent back: step 2 is over budget" in out
    assert (
        "— Step 2/7 done: estimate 120 lines / actual 143 lines / largest write 143 lines / "
        "longest response 38 s" in out
    )
    assert (
        "— Step 3/7 done: estimate none / actual 400 lines / largest write 400 lines (over budget)"
        in out
    )
    assert "/ edited 12 lines" in out


def test_timeout_and_repetition_lines() -> None:
    r, buf = _renderer(width=200)
    r.start_turn()
    r.on_stream_aborted(None)
    r.on_stream_aborted(120.0)
    r.on_step_timeout_resume(120.0, 2, 5, 2, 2)
    verdict = RepetitionVerdict(
        unit="paragraph", seen=10, window=12, repeats=7, snippet="x", chars=100
    )
    r.on_repetition_aborted(verdict, False)
    r.on_repetition_aborted(verdict, True)
    r.end_turn()
    out = buf.getvalue()
    assert "— The response was cut off" in out
    assert "— No response for 120 s, so it was stopped and asked to resend in smaller parts" in out
    assert (
        "asked to resume step 2/5 in smaller parts (attempt 2 in this step; it stops if this "
        "happens once more)" in out
    )
    assert "— Detected repeated content (the same paragraph 7 times), so it was stopped" in out
    assert "asked to write the conclusion only once" in out
    assert not _JAPANESE.search(out)


def test_turn_summary_line() -> None:
    r, _buf = _renderer()
    r._out_done = 3400
    r._requests, r._in_total, r._ctx_last, r._context_limit = 5, 60_200, 14_100, 52_000
    line = r._turn_summary_line(TokenUsage(input_tokens=40_100, output_tokens=5_600))
    assert line == (
        "— output 3.4K · input 60.2K (5 requests) · delegated input 40.1K / output 5.6K · "
        "context 14.1K/52K"
    )


def test_permission_card_and_previews(tmp_path: Path) -> None:
    card = _plain(
        action_preview(_call("write_file", path="a.txt", content="x\ny\n"), "workspace-write")
    )
    assert "Permission: write_file (workspace-write)" in card
    assert "(2 lines, 4 bytes)" in card and "(content shown above; d for full text)" in card
    append = _plain(action_preview(_call("append_file", path="a.txt", content="z\n"), "w"))
    assert "(append, 1 line, 2 bytes)" in append
    delete = _plain(action_preview(_call("delete_path", path="d", recursive=True), "w"))
    assert "(deletes the directory and everything in it)" in delete
    assert "Deletion cannot be undone. This is confirmed every time" in delete
    new_file = _plain(change_preview(_call("write_file", path="b.txt", content="hi\n"), tmp_path))
    assert "(new file)" in new_file
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.txt").write_text("x", encoding="utf-8")
    contents = _plain(change_preview(_call("delete_path", path="d", recursive=True), tmp_path))
    assert "Deletes everything inside (1 entry)" in contents
    for text in (card, append, delete, new_file, contents):
        assert not _JAPANESE.search(text), text
