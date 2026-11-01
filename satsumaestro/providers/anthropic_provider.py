"""Anthropic プロバイダ。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from anthropic import NOT_GIVEN, APIError, AsyncAnthropic

from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    ToolUseDelta,
    Usage,
)
from satsumaestro.i18n.text import ui
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.retry import (
    describe,
    retry_after_of,
    status_code_of,
    transport_retryable,
)
from satsumaestro.session.models import (
    ConversationMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.util.errors import ProviderError


def _block_to_wire(block: Any) -> dict[str, Any] | None:
    """1 つの ``ContentBlock`` を Anthropic のワイヤ content ブロックへ変換する。"""
    match block:
        case TextBlock():
            return {"type": "text", "text": block.text}
        case ThinkingBlock():
            if block.signature is None:
                return None
            return {
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            }
        case ToolUseBlock():
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
        case ToolResultBlock():
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.output,
                "is_error": block.is_error,
            }
    return None


def _to_wire(messages: list[ConversationMessage]) -> list[dict[str, Any]]:
    """``ConversationMessage`` 列を Anthropic Messages API の messages 配列へ変換する。"""
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "assistant" if message.role == "assistant" else "user"
        content = [w for w in (_block_to_wire(b) for b in message.blocks) if w is not None]
        if not content:
            continue
        if wire and role == "user" and wire[-1]["role"] == "user":
            wire[-1]["content"].extend(content)
        else:
            wire.append({"role": role, "content": content})
    return wire


def _map_anthropic_event(ev: Any) -> AssistantEvent | None:
    """SDK ストリームイベント → ``AssistantEvent``（該当なしは ``None``）。"""
    kind = ev.type
    if kind == "text":
        return TextDelta(text=ev.text)

    if kind == "content_block_stop":
        block = ev.content_block
        block_type = block.type
        if block_type == "thinking":
            return Thinking(text=block.thinking, signature=block.signature)
        if block_type == "tool_use":
            return ToolUse(id=block.id, name=block.name, input=dict(block.input))
        return None

    if kind == "input_json":
        return ToolUseDelta(args_delta=ev.partial_json)

    if kind == "message_delta":
        usage = ev.usage
        return Usage(
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    if kind == "message_stop":
        return MessageStop(stop_reason=ev.message.stop_reason or "end_turn")

    return None


class AnthropicProvider:
    kind = ProviderKind.ANTHROPIC

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        *,
        timeout: float | httpx.Timeout | None = None,
    ) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout if timeout is not None else NOT_GIVEN,
        )

    def _build_params(self, req: ApiRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "system": [
                {"type": "text", "text": req.system_text(), "cache_control": {"type": "ephemeral"}}
            ],
            "messages": _to_wire(req.messages),
            "output_config": {"effort": req.effort},
        }
        params["thinking"] = {"type": "adaptive"} if req.thinking else NOT_GIVEN
        if req.tools:
            params["tools"] = req.tools
            params["tool_choice"] = {"type": "auto"}
        return params

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        params = self._build_params(req)
        try:
            async with self._client.messages.stream(**params) as stream:
                async for ev in stream:
                    mapped = _map_anthropic_event(ev)
                    if mapped is not None:
                        yield mapped
        except (APIError, httpx.HTTPError) as e:
            raise ProviderError(
                ui("providers.anthropic_provider.anthropic_api_call_failed", error=describe(e)),
                status_code=status_code_of(e),
                retry_after=retry_after_of(e),
                retryable=transport_retryable(e),
            ) from e

    async def count_tokens(self, req: ApiRequest) -> int:
        try:
            result = await self._client.messages.count_tokens(
                model=req.model,
                system=req.system_text(),
                messages=_to_wire(req.messages),  # type: ignore[arg-type]
                tools=req.tools or NOT_GIVEN,  # type: ignore[arg-type]
            )
        except (APIError, httpx.HTTPError) as e:
            raise ProviderError(
                ui("providers.anthropic_provider.anthropic_count_tokens_failed", error=describe(e)),
                status_code=status_code_of(e),
                retry_after=retry_after_of(e),
                retryable=transport_retryable(e),
            ) from e
        return result.input_tokens
