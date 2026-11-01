"""OpenAI ストリーム組み立てテスト。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, ToolUseDelta, Usage
from satsumaestro.providers.openai_provider import _assemble_openai


def _chunk(
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish: str | None = None,
    usage: Any | None = None,
) -> NS:
    if usage is not None:
        return NS(choices=[], usage=usage)
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=None)


def _tc(index: int, id: str | None = None, name: str | None = None, args: str | None = None) -> NS:
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


async def _chunks(items: list[NS]) -> AsyncIterator[NS]:
    for it in items:
        yield it


async def test_openai_assemble() -> None:
    stream = _chunks(
        [
            _chunk(content="Hi"),
            _chunk(tool_calls=[_tc(0, id="call_1", name="search", args='{"q":')]),
            _chunk(tool_calls=[_tc(0, args='"cats"}')]),
            _chunk(finish="tool_calls"),
            _chunk(usage=NS(prompt_tokens=10, completion_tokens=20)),
        ]
    )

    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]

    assert out == [
        TextDelta(text="Hi"),
        ToolUse(id="call_1", name="search", input={"q": "cats"}),
        Usage(input_tokens=10, output_tokens=20),
        MessageStop(stop_reason="tool_use"),
    ]


async def test_openai_assemble_orders_tool_calls_by_index() -> None:
    stream = _chunks(
        [
            _chunk(tool_calls=[_tc(1, id="call_b", name="second", args="{}")]),
            _chunk(tool_calls=[_tc(0, id="call_a", name="first", args="{}")]),
            _chunk(finish="tool_calls"),
        ]
    )
    tools = [ev async for ev in _assemble_openai(stream) if isinstance(ev, ToolUse)]
    assert [t.id for t in tools] == ["call_a", "call_b"]


async def test_openai_assemble_synth_id_when_missing() -> None:
    stream = _chunks(
        [
            _chunk(tool_calls=[_tc(0, name="do", args="{}")]),
            _chunk(finish="tool_calls"),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    tool = next(e for e in out if isinstance(e, ToolUse))
    assert tool.id == "call_synth_0"
    assert tool.input == {}


async def test_openai_assemble_flushes_pending_tool_calls_at_stream_end() -> None:
    stream = _chunks(
        [
            _chunk(tool_calls=[_tc(0, id="call_1", name="search", args='{"q":')]),
            _chunk(tool_calls=[_tc(0, args='"cats"}')]),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [
        ToolUse(id="call_1", name="search", input={"q": "cats"}),
        MessageStop(stop_reason="end_turn"),
    ]


async def test_openai_assemble_drops_nameless_pending_at_stream_end() -> None:
    stream = _chunks([_chunk(tool_calls=[_tc(0, id="call_1", args="{}")])])
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [MessageStop(stop_reason="end_turn")]


async def test_openai_assemble_flushes_fragments_arriving_after_finish() -> None:
    stream = _chunks(
        [
            _chunk(content="done", finish="stop"),
            _chunk(tool_calls=[_tc(0, id="call_l", name="late", args="{}")]),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [
        TextDelta(text="done"),
        ToolUse(id="call_l", name="late", input={}),
        MessageStop(stop_reason="end_turn"),
    ]


async def test_openai_assemble_empty_stream_raises_retryable() -> None:
    import pytest

    from satsumaestro.providers.retry import is_retryable, is_tool_call_parse_error
    from satsumaestro.util.errors import ProviderError

    with pytest.raises(ProviderError) as excinfo:
        _ = [ev async for ev in _assemble_openai(_chunks([]))]
    assert is_retryable(excinfo.value) is True
    assert is_tool_call_parse_error(excinfo.value) is False


def test_to_openai_messages_empty_assistant_content() -> None:
    from satsumaestro.providers.base import ApiRequest
    from satsumaestro.providers.openai_provider import _to_openai_messages
    from satsumaestro.session.models import ConversationMessage, TextBlock, ToolUseBlock

    messages = [
        ConversationMessage(role="user", blocks=[TextBlock(text="q")]),
        ConversationMessage(role="assistant", blocks=[]),
        ConversationMessage(role="assistant", blocks=[ToolUseBlock(id="c1", name="t", input={})]),
    ]
    req = ApiRequest(model="m", system="", messages=messages)
    out = _to_openai_messages(req)
    assert out[1] == {"role": "assistant", "content": ""}
    assert out[2]["content"] is None and out[2]["tool_calls"]
