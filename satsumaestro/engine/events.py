"""ストリームイベントと畳み込み。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from itertools import groupby
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from satsumaestro.session.models import (
    ContentBlock,
    ConversationMessage,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolUseBlock,
)

_log = logging.getLogger(__name__)


class Thinking(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolUse(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    input_fallback: bool = False


class ToolUseDelta(BaseModel):
    """tool_call 引数の生成途中。"""

    type: Literal["tool_use_delta"] = "tool_use_delta"
    index: int = 0
    name: str = ""
    args_delta: str = ""


class Usage(BaseModel):
    type: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0


class MessageStop(BaseModel):
    type: Literal["message_stop"] = "message_stop"
    stop_reason: str = "end_turn"


AssistantEvent = Annotated[
    Thinking | TextDelta | ToolUse | ToolUseDelta | Usage | MessageStop,
    Field(discriminator="type"),
]


async def aclose_quietly(events: AsyncIterator[AssistantEvent]) -> None:
    """イベントストリームを閉じ、プロバイダ側の HTTP 接続を解放する。"""
    aclose = getattr(events, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception as exc:
        _log.debug("could not close the event stream: %s", exc)


@asynccontextmanager
async def closing_stream(
    events: AsyncIterator[AssistantEvent],
) -> AsyncIterator[AsyncIterator[AssistantEvent]]:
    """イベントストリームを確実に閉じる ``async with``（``contextlib.aclosing`` 相当）。"""
    try:
        yield events
    finally:
        await aclose_quietly(events)


async def fold_events(events: AsyncIterator[AssistantEvent]) -> ConversationMessage:
    """ストリームを 1 つの assistant メッセージへ畳み込む。"""
    parts: list[Thinking | TextDelta | ToolUse] = []
    usage: TokenUsage | None = None
    async for ev in events:
        if isinstance(ev, MessageStop):
            break
        if isinstance(ev, Usage):
            usage = TokenUsage(
                input_tokens=ev.input_tokens,
                output_tokens=ev.output_tokens,
                cache_read=ev.cache_read,
                cache_write=ev.cache_write,
            )
        elif not isinstance(ev, ToolUseDelta):
            parts.append(ev)
    else:
        _log.warning("the event stream ended without MessageStop (parts=%d)", len(parts))

    blocks: list[ContentBlock] = []
    for is_text, run in groupby(parts, key=lambda part: isinstance(part, TextDelta)):
        if is_text:
            text = "".join(part.text for part in run if isinstance(part, TextDelta))
            blocks.append(TextBlock(text=text))
        else:
            blocks.extend(_as_block(part) for part in run)
    return ConversationMessage(role="assistant", blocks=blocks, usage=usage)


def _as_block(part: Thinking | TextDelta | ToolUse) -> ContentBlock:
    if isinstance(part, Thinking):
        return ThinkingBlock(thinking=part.text, signature=part.signature)
    if isinstance(part, ToolUse):
        return ToolUseBlock(id=part.id, name=part.name, input=part.input)
    return TextBlock(text=part.text)
