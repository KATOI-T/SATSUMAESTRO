"""Anthropic プロバイダ。"""

from __future__ import annotations

import logging
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
from satsumaestro.providers import notices
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.detect import AnthropicModelInfo, anthropic_model_info_for
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

_log = logging.getLogger(__name__)

_LOOKUP_TIMEOUT_S = 10.0
_EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
_EFFORT_ALIASES = {"none": "low", "minimal": "low"}
_MIN_THINKING_BUDGET = 1024


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


def _supported(node: Any) -> frozenset[str]:
    """Models API の機能の木のうち、``supported`` が真の子の名前。"""
    if not isinstance(node, dict):
        return frozenset()
    return frozenset(k for k, v in node.items() if isinstance(v, dict) and v.get("supported"))


def _learn_model(info: AnthropicModelInfo, found: Any) -> None:
    """Models API の応答から、要求の形に要る機能を記録する。"""
    info.max_tokens = getattr(found, "max_tokens", None) or None
    caps = getattr(found, "capabilities", None)
    if caps is None:
        return
    tree = caps.model_dump() if hasattr(caps, "model_dump") else dict(caps)
    thinking = tree.get("thinking") or {}
    if thinking.get("supported"):
        info.thinking_types = _supported(thinking.get("types"))
    effort = tree.get("effort") or {}
    if effort.get("supported"):
        info.effort_levels = _supported(effort) & frozenset(_EFFORT_ORDER)
    info.known = True


def _effort_for(requested: str, levels: frozenset[str]) -> str | None:
    """モデルが受ける段階のうち、``requested`` 以下でいちばん近いもの（無ければ上で近いもの）。"""
    if requested in levels:
        return requested
    rank = _EFFORT_ORDER.index(requested)
    lower = [lv for lv in _EFFORT_ORDER[:rank] if lv in levels]
    if lower:
        return lower[-1]
    return next((lv for lv in _EFFORT_ORDER[rank + 1 :] if lv in levels), None)


def _thinking_budget(max_tokens: int) -> int | None:
    """``budget_tokens`` の値。出力の上限の半分で、下限に届かなければ thinking を付けない。"""
    budget = max_tokens // 2
    return budget if budget >= _MIN_THINKING_BUDGET else None


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

    def _build_params(
        self, req: ApiRequest, info: AnthropicModelInfo | None = None
    ) -> dict[str, Any]:
        """要求の引数。モデルの機能が分かっていれば、受けない引数を外して上限に合わせる。"""
        known = info if info is not None and info.known else None
        max_tokens = req.max_tokens
        if info is not None and info.max_tokens:
            max_tokens = min(max_tokens, info.max_tokens)
        params: dict[str, Any] = {
            "model": req.model,
            "max_tokens": max_tokens,
            "system": [
                {"type": "text", "text": req.system_text(), "cache_control": {"type": "ephemeral"}}
            ],
            "messages": _to_wire(req.messages),
            "thinking": NOT_GIVEN,
        }
        effort = self._effort(req, known)
        if effort is not None:
            params["output_config"] = {"effort": effort}
        if req.thinking:
            if known is None or "adaptive" in known.thinking_types:
                params["thinking"] = {"type": "adaptive"}
            elif "enabled" in known.thinking_types:
                budget = _thinking_budget(max_tokens)
                if budget is not None:
                    params["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if req.tools:
            params["tools"] = req.tools
            params["tool_choice"] = {"type": "auto"}
        return params

    @staticmethod
    def _effort(req: ApiRequest, known: AnthropicModelInfo | None) -> str | None:
        """送る effort。利用者の指定（``/effort``）を優先し、モデルが受ける段階に合わせる。"""
        chosen = req.sampling.reasoning_effort if req.sampling is not None else None
        effort: str = chosen or req.effort
        requested = _EFFORT_ALIASES.get(effort, effort)
        return requested if known is None else _effort_for(requested, known.effort_levels)

    async def _model_info(self, model: str) -> AnthropicModelInfo:
        """モデルの機能を Models API で 1 回だけ引く。引けなければ今までどおりの形で送る。"""
        info = anthropic_model_info_for(str(self._client.base_url), model)
        if info.looked_up:
            return info
        try:
            found = await self._client.models.retrieve(model, timeout=_LOOKUP_TIMEOUT_S)
        except (APIError, httpx.HTTPError) as e:
            status = status_code_of(e)
            # 4xx は引き直しても変わらない。通信の失敗は次の要求で引き直す
            if status is not None and 400 <= status < 500:
                info.looked_up = True
            _log.debug("could not look up %s in the Models API: %s", model, e)
            return info
        _learn_model(info, found)
        info.looked_up = True
        return info

    def _notify_effort(self, req: ApiRequest, info: AnthropicModelInfo, sent: str | None) -> None:
        """利用者が選んだ effort を変えて送ったとき、モデルごとに 1 回だけ知らせる。"""
        chosen = req.sampling.reasoning_effort if req.sampling is not None else None
        if chosen is None or chosen == sent or info.notice_shown:
            return
        info.notice_shown = True
        notices.notify(
            notices.ProviderNotice(
                kind="effort_adjusted", model=req.model, requested=chosen, sent=sent or ""
            )
        )

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        try:
            info = await self._model_info(req.model)
            params = self._build_params(req, info)
            async with self._client.messages.stream(**params) as stream:
                self._notify_effort(req, info, params.get("output_config", {}).get("effort"))
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
