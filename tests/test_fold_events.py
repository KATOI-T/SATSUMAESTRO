"""fold_events テスト。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    Usage,
    fold_events,
)
from satsumaestro.session.models import TextBlock, ThinkingBlock, ToolUseBlock


async def _aiter(events: list[AssistantEvent]) -> AsyncIterator[AssistantEvent]:
    for ev in events:
        yield ev


async def test_fold_events_basic() -> None:
    events: list[AssistantEvent] = [
        Thinking(text="考える", signature="sig-1"),
        TextDelta(text="Hello"),
        TextDelta(text=", "),
        TextDelta(text="world"),
        ToolUse(id="t1", name="bash", input={"cmd": "ls"}),
        Usage(input_tokens=12, output_tokens=8, cache_read=1, cache_write=2),
        MessageStop(stop_reason="tool_use"),
    ]

    msg = await fold_events(_aiter(events))

    assert msg.role == "assistant"
    assert len(msg.blocks) == 3

    thinking, text, tool = msg.blocks
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.thinking == "考える"
    assert thinking.signature == "sig-1"

    assert isinstance(text, TextBlock)
    assert text.text == "Hello, world"

    assert isinstance(tool, ToolUseBlock)
    assert tool.id == "t1"
    assert tool.name == "bash"
    assert tool.input == {"cmd": "ls"}

    assert msg.usage is not None
    assert msg.usage.input_tokens == 12
    assert msg.usage.output_tokens == 8
    assert msg.usage.cache_read == 1
    assert msg.usage.cache_write == 2


async def test_fold_events_stops_at_message_stop() -> None:
    events: list[AssistantEvent] = [
        TextDelta(text="keep"),
        MessageStop(),
        TextDelta(text="dropped"),
    ]

    msg = await fold_events(_aiter(events))

    assert len(msg.blocks) == 1
    assert isinstance(msg.blocks[0], TextBlock)
    assert msg.blocks[0].text == "keep"
    assert msg.usage is None
