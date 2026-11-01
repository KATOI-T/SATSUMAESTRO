"""ツール失敗の詳細表示のテスト。"""

from __future__ import annotations

import io

from rich.console import Console

from satsumaestro.cli.render import RichRenderer
from satsumaestro.cli.tool_display import ERROR_TRUNCATED, error_detail
from satsumaestro.engine.events import ToolUse
from satsumaestro.session.models import ToolResultBlock

_PYDANTIC = (
    "invalid input: invalid input for tool todo_write: 1 validation error for Input\n"
    "todos.0.activeForm\n"
    "  Field required [type=missing, input_value={'content': 'x'}, input_type=dict]\n"
    "    For further information visit https://errors.pydantic.dev/2.11/v/missing"
)


def _block(output: str, name: str = "bash") -> ToolResultBlock:
    return ToolResultBlock(tool_use_id="1", tool_name=name, output=output, is_error=True)


def _render_failure(name: str, output: str, **inp: object) -> str:
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=False, width=200, color_system=None))
    r.start_turn()
    call = ToolUse(id="1", name=name, input=inp)
    r.on_tool_start(call)
    r.on_tool_result(_block(output, name))
    r.end_turn()
    return buf.getvalue()


def test_single_line_error_has_no_detail() -> None:
    assert error_detail(_block("old string not found")) == []
    assert error_detail(_block("")) == []


def test_detail_drops_blank_lines_and_trailing_space_but_keeps_indent() -> None:
    assert error_detail(_block("head\n\n  field   \n\n    reason")) == ["  field", "    reason"]


def test_detail_drops_pydantic_reference_url() -> None:
    detail = error_detail(_block(_PYDANTIC))
    assert detail[0] == "todos.0.activeForm"
    assert detail[1].startswith("  Field required [type=missing")
    assert not any("errors.pydantic.dev" in line for line in detail)


def test_detail_is_capped_by_lines() -> None:
    output = "\n".join(f"line {i}" for i in range(30))
    detail = error_detail(_block(output))
    assert detail[:-1] == [f"line {i}" for i in range(1, 11)]
    assert detail[-1] == ERROR_TRUNCATED


def test_detail_is_capped_by_chars() -> None:
    output = "head\n" + "\n".join("x" * 300 for _ in range(3))
    detail = error_detail(_block(output))
    assert [len(line) for line in detail[:-1]] == [300, 100]
    assert detail[-1] == ERROR_TRUNCATED


def test_detail_within_limits_is_not_marked_truncated() -> None:
    output = "\n".join(f"line {i}" for i in range(11))
    assert ERROR_TRUNCATED not in error_detail(_block(output))


def test_validation_error_shows_the_field_for_action_tools() -> None:
    out = _render_failure("edit_file", _PYDANTIC, path="a.py", old="x", new="y")
    assert "✗ a.py — invalid input" in out
    assert "\n      todos.0.activeForm\n" in out
    assert "\n        Field required [type=missing" in out
    assert "errors.pydantic.dev" not in out


def test_validation_error_shows_the_field_for_reference_tools() -> None:
    out = _render_failure("read_file", _PYDANTIC, path="missing.py")
    assert "- missing.py  ✗ invalid input" in out
    assert "\n      todos.0.activeForm\n" in out


def test_single_line_error_output_is_unchanged() -> None:
    out = _render_failure("bash", "exit 1", command="false")
    assert "✗ false — exit 1" in out
    assert "\n      " not in out
    assert ERROR_TRUNCATED not in out


def test_long_error_does_not_flood_the_screen() -> None:
    output = "\n".join(f"line {i}" for i in range(200))
    out = _render_failure("bash", output, command="make test")
    assert "      line 10\n" in out and "line 11" not in out
    assert ERROR_TRUNCATED in out


def test_tool_result_for_the_model_is_untouched() -> None:
    block = _block(_PYDANTIC)
    before = block.model_dump()
    error_detail(block)
    assert block.model_dump() == before


_LONG_FIRST = (
    "FileNotFoundError: no such file or directory: "
    + "../" * 20
    + "etc/passwd-and-a-long-tail-name.txt"
)


def test_long_single_line_error_puts_the_full_line_in_detail() -> None:
    assert len(_LONG_FIRST) > 80
    assert error_detail(_block(_LONG_FIRST)) == [_LONG_FIRST]


def test_short_single_line_error_still_has_no_detail() -> None:
    assert error_detail(_block("x" * 80)) == []


def test_long_first_line_is_followed_by_the_rest() -> None:
    detail = error_detail(_block(_LONG_FIRST + "\nsecond\n\nthird"))
    assert detail == [_LONG_FIRST, "second", "third"]


def test_clipped_tail_is_visible_on_screen_for_both_tool_kinds() -> None:
    tail = "passwd-and-a-long-tail-name.txt"
    for name, inp in (("read_file", {"path": "x.md"}), ("bash", {"command": "cat x"})):
        out = _render_failure(name, _LONG_FIRST, **inp)
        assert "…" in out, name
        assert tail in out, name
