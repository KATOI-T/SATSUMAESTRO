"""Anthropic プロバイダ。"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
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
    status_is_transient,
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
_BINDING_BETA = "thinking-binding-controls-2026-08-01"
_UNBOUND_THINKING = "bound to a different conversation"
_THINKING_TYPES = frozenset({"thinking", "redacted_thinking"})
_TOOL_ID = re.compile(r"^[a-zA-Z0-9_-]+$")


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
    _fix_tool_ids(wire)
    return wire


def _fix_tool_ids(wire: list[dict[str, Any]]) -> None:
    """繰り返す ID と形の合わない ID（OpenAI 互換のサーバで作った履歴）を一意の ID に付け直す。"""
    blocks = [b for m in wire for b in m["content"]]
    taken = {b["id"] for b in blocks if b["type"] == "tool_use"}
    seen: set[str] = set()
    latest: dict[str, str] = {}
    counter = 0
    for block in blocks:
        if block["type"] == "tool_use":
            original = block["id"]
            if not _TOOL_ID.match(original) or original in seen:
                while f"toolu_{counter}" in taken:
                    counter += 1
                block["id"] = f"toolu_{counter}"
                taken.add(block["id"])
            seen.add(block["id"])
            latest[original] = block["id"]
        elif block["type"] == "tool_result":
            block["tool_use_id"] = latest.get(block["tool_use_id"], block["tool_use_id"])


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


def _is_unbound_thinking(e: APIError) -> bool:
    """履歴を書き換えたため、送り返した thinking の block が会話と合わないという 400 か。"""
    return status_code_of(e) == 400 and _UNBOUND_THINKING in str(e)


def _drop_unbound(params: dict[str, Any]) -> dict[str, Any] | None:
    """合わない thinking の block を API に外させる形。thinking を送らない要求では作れない。"""
    thinking = params.get("thinking")
    if not isinstance(thinking, dict):
        return None
    binding = {"prefix_mismatch_behavior": "drop_block"}
    return {
        **params,
        "thinking": {**thinking, "block_binding": binding},
        "extra_headers": {"anthropic-beta": _BINDING_BETA},
    }


def _without_thinking(params: dict[str, Any]) -> dict[str, Any]:
    """履歴から thinking の block を除いた形（text と tool_use は残す）。"""
    messages = []
    for message in params["messages"]:
        content = [b for b in message["content"] if b.get("type") not in _THINKING_TYPES]
        if content:
            messages.append({**message, "content": content})
    return {**params, "messages": messages}


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
        """送る effort。機能が分かれば利用者の指定を段階に合わせ、分からなければ今までどおり。"""
        if known is None:
            return req.effort
        chosen = req.sampling.reasoning_effort if req.sampling is not None else None
        effort: str = chosen or req.effort
        return _effort_for(_EFFORT_ALIASES.get(effort, effort), known.effort_levels)

    async def _model_info(self, model: str) -> AnthropicModelInfo:
        """モデルの機能を Models API で 1 回だけ引く。引けなければ今までどおりの形で送る。"""
        info = anthropic_model_info_for(str(self._client.base_url), model)
        if info.looked_up:
            return info
        try:
            found = await self._client.models.retrieve(model, timeout=_LOOKUP_TIMEOUT_S)
        except (APIError, httpx.HTTPError) as e:
            status = status_code_of(e)
            # 404 などは引き直しても変わらない。通信の失敗と 429 などは次の要求で引き直す
            if status is not None and status >= 400 and not status_is_transient(status):
                info.looked_up = True
            _log.debug("could not look up %s in the Models API: %s", model, e)
            return info
        _learn_model(info, found)
        info.looked_up = True
        return info

    def _notify_effort(self, req: ApiRequest, info: AnthropicModelInfo, sent: str | None) -> None:
        """利用者が選んだ effort を変えて送ったとき、モデルごとに 1 回だけ知らせる。"""
        chosen = req.sampling.reasoning_effort if req.sampling is not None else None
        if not info.known or chosen is None or chosen == sent or info.notice_shown:
            return
        info.notice_shown = True
        notices.notify(
            notices.ProviderNotice(
                kind="effort_adjusted", model=req.model, requested=chosen, sent=sent or ""
            )
        )

    async def _enter(
        self, stack: AsyncExitStack, params: dict[str, Any], info: AnthropicModelInfo
    ) -> Any:
        """ストリームを開く。thinking の block が会話と合わない 400 なら、外す形で送り直す。"""
        dropped = _drop_unbound(params)
        first = dropped if info.drop_unbound_thinking and dropped is not None else params
        try:
            return await self._open(stack, first)
        except APIError as e:
            if not _is_unbound_thinking(e):
                raise
        if first is params and dropped is not None:
            try:
                opened = await self._open(stack, dropped)
            except APIError as e:
                if status_code_of(e) != 400:
                    raise
            else:
                info.drop_unbound_thinking = True
                _log.warning(
                    "%s: thinking blocks no longer match the edited history, so they are dropped",
                    params["model"],
                )
                return opened
        # 外させる形が使えない（thinking を送らない・ベータを受けない）なら自分で除く
        return await self._open(stack, _without_thinking(params))

    async def _open(self, stack: AsyncExitStack, params: dict[str, Any]) -> Any:
        return await stack.enter_async_context(self._client.messages.stream(**params))

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        try:
            info = await self._model_info(req.model)
            params = self._build_params(req, info)
            async with AsyncExitStack() as stack:
                stream = await self._enter(stack, params, info)
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
