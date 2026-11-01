"""tool_call 引数のストリーミング描画のテスト。"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace as NS

from rich.console import Console

from satsumaestro.cli.render import RichRenderer
from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    ToolUse,
    ToolUseDelta,
    fold_events,
)
from satsumaestro.providers.openai_provider import _assemble_openai
from satsumaestro.session.models import ToolUseBlock
from satsumaestro.util.partial_json import partial_json_string


def test_extracts_complete_value() -> None:
    assert partial_json_string('{"path": "a.py", "new": "x = 2"}', "new") == "x = 2"


def test_extracts_partial_value_while_streaming() -> None:
    assert partial_json_string('{"new": "def total(', "new") == "def total("


def test_returns_none_before_key_arrives() -> None:
    assert partial_json_string('{"pa', "new") is None
    assert partial_json_string('{"path": "a.py"', "new") is None
    assert partial_json_string('{"new"', "new") is None
    assert partial_json_string('{"new":', "new") is None


def test_decodes_escapes() -> None:
    assert partial_json_string(r'{"new": "a\nb\t\"c\""}', "new") == 'a\nb\t"c"'
    assert partial_json_string(r'{"new": "C:\\tmp"}', "new") == "C:\\tmp"
    assert partial_json_string(r'{"new": "\u3042\u3044"}', "new") == "あい"


def test_drops_incomplete_escape_at_tail() -> None:
    assert partial_json_string('{"new": "ab\\', "new") == "ab"
    assert partial_json_string('{"new": "ab\\u30', "new") == "ab"
    assert partial_json_string('{"new": "ab\\u3042', "new") == "ab\u3042"


def test_escaped_backslash_before_u_is_not_treated_as_unicode() -> None:
    assert partial_json_string('{"new": "a\\\\u30"', "new") == "a\\u30"


def test_value_that_is_not_a_string_is_ignored() -> None:
    assert partial_json_string('{"new": 12', "new") is None
    assert partial_json_string('{"new": {"a": 1}', "new") is None


def test_key_is_matched_exactly() -> None:
    assert partial_json_string('{"new_path": "z"}', "new") is None


def test_key_inside_a_value_does_not_shadow_the_real_key() -> None:
    buf = '{"old": "say \\"new\\": here", "new": "done"}'
    assert partial_json_string(buf, "new") == "done"


def _tc(index: int, id: str | None = None, name: str | None = None, args: str | None = None) -> NS:
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def _chunk(tool_calls: list[NS] | None = None, finish: str | None = None) -> NS:
    return NS(
        choices=[NS(delta=NS(content=None, tool_calls=tool_calls), finish_reason=finish)],
        usage=None,
    )


async def _chunks(items: list[NS]) -> AsyncIterator[NS]:
    for it in items:
        yield it


async def test_openai_emits_tool_use_delta_per_fragment() -> None:
    stream = _chunks(
        [
            _chunk([_tc(0, id="c1", name="edit_file", args='{"new": "a')]),
            _chunk([_tc(0, args='b"}')]),
            _chunk(finish="tool_calls"),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream)]
    deltas = [ev for ev in out if isinstance(ev, ToolUseDelta)]
    assert [d.args_delta for d in deltas] == ['{"new": "a', 'b"}']
    assert all(d.name == "edit_file" for d in deltas)
    tools = [ev for ev in out if isinstance(ev, ToolUse)]
    assert len(tools) == 1 and tools[0].input == {"new": "ab"}


async def test_tool_use_delta_is_not_folded_into_the_message() -> None:
    async def _events() -> AsyncIterator[AssistantEvent]:
        yield ToolUseDelta(index=0, name="edit_file", args_delta='{"new": "x"}')
        yield ToolUse(id="c1", name="edit_file", input={"new": "x"})
        yield MessageStop()

    msg = await fold_events(_events())
    assert len(msg.blocks) == 1
    assert isinstance(msg.blocks[0], ToolUseBlock)


def _tty_renderer(*, stream_tool_args: bool = True) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=70)
    return RichRenderer(console=console, stream_tool_args=stream_tool_args), buf


def _live_text(r: RichRenderer) -> str:
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=70).print(r._live_renderable())
    return out.getvalue()


def _feed(r: RichRenderer, args: str, *, step: int = 12) -> None:
    for i in range(0, len(args), step):
        r.on_assistant_event(ToolUseDelta(index=0, name="edit_file", args_delta=args[i : i + step]))


def test_live_shows_decoded_content_while_streaming() -> None:
    r, _ = _tty_renderer()
    r.start_turn()
    args = json.dumps({"path": "cart.py", "new": "def total():\n    return 1\n"})
    _feed(r, args)
    text = _live_text(r)
    assert "def total():" in text and "return 1" in text
    assert "\\n" not in text


def test_live_panel_clears_once_the_tool_call_is_final() -> None:
    r, _ = _tty_renderer()
    r.start_turn()
    _feed(r, json.dumps({"new": "abc"}))
    assert "abc" in _live_text(r)
    r.on_assistant_event(ToolUse(id="c1", name="edit_file", input={"new": "abc"}))
    assert "abc" not in _live_text(r)


def test_streaming_can_be_disabled() -> None:
    r, _ = _tty_renderer(stream_tool_args=False)
    r.start_turn()
    _feed(r, json.dumps({"new": "abc"}))
    assert "abc" not in _live_text(r)


def test_write_file_content_key_is_also_streamed() -> None:
    r, _ = _tty_renderer()
    r.start_turn()
    _feed(r, json.dumps({"path": "b.py", "content": "print('hi')\n"}))
    assert "print('hi')" in _live_text(r)
