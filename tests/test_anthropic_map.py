"""Anthropic SDK イベント → AssistantEvent 変換テスト。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace as NS

from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    ToolUseDelta,
    Usage,
    fold_events,
)
from satsumaestro.providers.anthropic_provider import _map_anthropic_event
from satsumaestro.session.models import TextBlock, ThinkingBlock, ToolUseBlock

ANTHROPIC_STREAM = [
    NS(type="message_start"),
    NS(type="content_block_start"),
    NS(type="thinking", thinking="Hmm", snapshot="Hmm"),
    NS(type="signature", signature="sig-abc"),
    NS(
        type="content_block_stop",
        index=0,
        content_block=NS(type="thinking", thinking="Hmm, let me think.", signature="sig-abc"),
    ),
    NS(type="text", text="Hello ", snapshot="Hello "),
    NS(type="text", text="world", snapshot="Hello world"),
    NS(type="input_json", partial_json='{"q":', snapshot='{"q":'),
    NS(
        type="content_block_stop",
        index=1,
        content_block=NS(type="tool_use", id="toolu_1", name="search", input={"q": "cats"}),
    ),
    NS(
        type="message_delta",
        usage=NS(
            input_tokens=12,
            output_tokens=34,
            cache_read_input_tokens=5,
            cache_creation_input_tokens=0,
        ),
    ),
    NS(type="message_stop", message=NS(stop_reason="tool_use")),
]


def test_anthropic_map() -> None:
    got = [m for ev in ANTHROPIC_STREAM if (m := _map_anthropic_event(ev)) is not None]
    assert got == [
        Thinking(text="Hmm, let me think.", signature="sig-abc"),
        TextDelta(text="Hello "),
        TextDelta(text="world"),
        ToolUseDelta(args_delta='{"q":'),
        ToolUse(id="toolu_1", name="search", input={"q": "cats"}),
        Usage(input_tokens=12, output_tokens=34, cache_read=5, cache_write=0),
        MessageStop(stop_reason="tool_use"),
    ]


async def test_anthropic_map_folds_to_message() -> None:
    events = [m for ev in ANTHROPIC_STREAM if (m := _map_anthropic_event(ev)) is not None]

    async def _aiter() -> AsyncIterator[AssistantEvent]:
        for e in events:
            yield e

    msg = await fold_events(_aiter())

    assert msg.role == "assistant"
    thinking, text, tool = msg.blocks
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.thinking == "Hmm, let me think."
    assert thinking.signature == "sig-abc"
    assert isinstance(text, TextBlock)
    assert text.text == "Hello world"
    assert isinstance(tool, ToolUseBlock)
    assert tool.id == "toolu_1"
    assert tool.input == {"q": "cats"}
    assert msg.usage is not None
    assert msg.usage.input_tokens == 12
    assert msg.usage.cache_read == 5
