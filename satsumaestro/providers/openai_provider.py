"""OpenAI 互換プロバイダ。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from openai import NOT_GIVEN, APIError, AsyncOpenAI

from satsumaestro.config.models import OpenAIConfig
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
from satsumaestro.providers import notices, timing
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.detect import (
    ModelQuirks,
    capabilities_for,
    model_quirks_for,
    note_model_used,
    touch_keep_alive,
)
from satsumaestro.providers.retry import (
    describe,
    retry_after_of,
    status_code_of,
    transport_retryable,
)
from satsumaestro.session.models import TextBlock, ToolResultBlock, ToolUseBlock
from satsumaestro.tools.schema import strict_eligible
from satsumaestro.util.errors import ProviderError
from satsumaestro.util.tokens import estimate_tokens

_log = logging.getLogger(__name__)

_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}

_KEEP_ALIVE_WAIT_S = 1.0
_XAI_HOST = "api.x.ai"

# 400 を見て形を変える再送の上限。拒まれた要求は課金されないので、コストは待ち時間だけ
_MAX_NEGOTIATIONS = 6
_TOKEN_CAP_PARAMS = ("max_tokens", "max_completion_tokens")
_SAMPLING_PARAMS = ("temperature", "top_p", "reasoning_effort")
# 上限超過の本文は 2 種類ある: "Expected a value <= N" と "supports at most N completion tokens"
_CAP_RE = re.compile(r"(?:<=|at most)\s*(\d+)")
# xAI は param を返さず、本文で引数を名指しする: "does not support parameter reasoningEffort" など
_NAMED_PARAM_RES = (
    re.compile(r"does not support parameter `?(\w+)"),
    re.compile(r"`(\w+)` value"),
)


def _stop_reason_for(finish_reason: str | None) -> str:
    """``finish_reason`` を ``stop_reason`` の語に直す。"""
    if not finish_reason:
        return "end_turn"
    return _STOP_REASONS.get(finish_reason, finish_reason)


def _safe_json_args(raw: str) -> tuple[dict[str, Any], bool]:
    """壊れた ``arguments`` を例外にせず ``{}`` へ落とす。"""
    try:
        parsed = json.loads(raw or "{}")
    except ValueError:
        _log.warning("tool_call arguments are not valid JSON; using {}: %.200r", raw)
        return {}, True
    if not isinstance(parsed, dict):
        _log.warning("tool_call arguments are not a JSON object; using {}: %.200r", raw)
        return {}, True
    return parsed, False


_TRUNCATED_TOOL_CALL = "error parsing tool call"


_REASONING_FIELDS = ("reasoning", "reasoning_content")


def _has_content(chunk: Any) -> bool:
    """チャンクに中身（思考・本文・ツール呼び出し・usage）が載っているか。"""
    if getattr(chunk, "usage", None) is not None:
        return True
    for choice in getattr(chunk, "choices", None) or []:
        delta = choice.delta
        if _reasoning_text(delta) or getattr(delta, "content", None):
            return True
        if getattr(delta, "tool_calls", None) or choice.finish_reason:
            return True
    return False


async def _watch_chunks(stream: AsyncIterator[Any], silence_s: float | None) -> AsyncIterator[Any]:
    """最初の中身が届いた時刻を記録し、``silence_s`` があれば、その後の無音を見る。"""
    iterator = stream.__aiter__()
    started = False
    while True:
        try:
            if started and silence_s is not None:
                chunk = await asyncio.wait_for(iterator.__anext__(), silence_s)
            else:
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise httpx.ReadTimeout(f"no data for {silence_s:.0f} s after the reply began") from exc
        if not started and _has_content(chunk):
            started = True
            timing.note_first_chunk()
        yield chunk


def _reasoning_text(delta: Any) -> str:
    """delta に載った思考の断片。"""
    for field in _REASONING_FIELDS:
        text = getattr(delta, field, None)
        if isinstance(text, str) and text:
            return text
    return ""


def _synth_id(index: int) -> str:
    """tool_call id 欠落時の決定的な合成 id（uuid/random は使わない）。"""
    return f"call_synth_{index}"


def _join_text(blocks: list[Any]) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def _drain_tool_calls(partial: dict[int, dict[str, str]]) -> list[ToolUse]:
    """蓄積済み tool_call 断片を index 昇順の :class:`ToolUse` 列へ確定し、バッファを空にする。"""
    out: list[ToolUse] = []
    for index in sorted(partial):
        slot = partial[index]
        if not slot["name"]:
            _log.warning("dropped a tool_call fragment without a name: index=%d", index)
            continue
        args, fallback = _safe_json_args(slot["args"])
        out.append(
            ToolUse(
                id=slot["id"] or _synth_id(index),
                name=slot["name"],
                input=args,
                input_fallback=fallback,
            )
        )
    partial.clear()
    return out


def _request_kwargs(req: ApiRequest, official: bool = False) -> dict[str, Any]:
    """``chat.completions.create`` の付加引数。OpenAI 公式の推論モデルは ``max_tokens`` を拒む。"""
    kwargs: dict[str, Any] = {
        "stream_options": {"include_usage": True},
        ("max_completion_tokens" if official else "max_tokens"): req.max_tokens,
    }
    sampling = req.sampling
    if sampling is not None:
        if sampling.temperature is not None:
            kwargs["temperature"] = sampling.temperature
        if sampling.top_p is not None:
            kwargs["top_p"] = sampling.top_p
        if sampling.reasoning_effort is not None:
            kwargs["reasoning_effort"] = sampling.reasoning_effort
    return kwargs


@dataclass
class _Shape:
    """1 回の ``create`` に渡す形。400 のたびに変えて再送する。"""

    tools: list[dict[str, Any]]
    strict: bool
    kwargs: dict[str, Any]
    strict_dropped: bool = False
    forced_effort_none: bool = False
    tools_dropped: bool = False
    rejected: frozenset[tuple[str, str]] = frozenset()


def _cap_param(kwargs: dict[str, Any]) -> str | None:
    return next((k for k in _TOKEN_CAP_PARAMS if k in kwargs), None)


def _param_of(e: APIError) -> str | None:
    """400 が名指しした引数。xAI は ``param`` を返さないので本文から読む。"""
    if e.param:
        return e.param
    text = e.message or ""
    for pattern in _NAMED_PARAM_RES:
        found = pattern.search(text)
        if found:
            return re.sub(r"(?<!^)(?=[A-Z])", "_", found.group(1)).lower()
    return "reasoning_effort" if "invalid reasoning effort" in text.lower() else None


def _not_a_chat_model(e: APIError) -> bool:
    """モデルが Chat Completions で使えない旨の応答か（OpenAI の pro と xAI の multi-agent）。"""
    text = (e.message or "").lower()
    if "not allowed on chat completions" in text:
        return True
    return e.param == "model" and ("chat/completions" in text or "responses" in text)


def _at_capacity(e: BaseException) -> bool:
    """xAI がストリームの途中で返す混雑の通知か（HTTP の状態が無いので再試行の判定から漏れる）。"""
    return isinstance(e, APIError) and e.code == "resource-exhausted" and status_code_of(e) is None


def _close_objects(node: dict[str, Any]) -> dict[str, Any]:
    """全オブジェクトノードへ ``additionalProperties: false`` を付けた深いコピー。"""
    out: dict[str, Any] = {}
    for key, value in node.items():
        if isinstance(value, dict) and key in ("properties", "$defs", "definitions"):
            out[key] = {
                k: _close_objects(v) if isinstance(v, dict) else v for k, v in value.items()
            }
        elif isinstance(value, dict):
            out[key] = _close_objects(value)
        elif isinstance(value, list):
            out[key] = [_close_objects(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    if "properties" in out:
        out.setdefault("additionalProperties", False)
    return out


def _strict_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """strict 対象ツールへ ``strict: true`` / ``additionalProperties: false`` を付けたコピー。"""
    out: list[dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function") or {}
        params = fn.get("parameters") or {}
        if not isinstance(params, dict) or not strict_eligible(params):
            out.append(tool)
            continue
        out.append(
            {**tool, "function": {**fn, "strict": True, "parameters": _close_objects(params)}}
        )
    return out


def _to_openai_messages(req: ApiRequest) -> list[dict[str, Any]]:
    """``ApiRequest`` を OpenAI Chat Completions の messages 配列へ変換する。"""
    out: list[dict[str, Any]] = []
    system = req.system_text()
    if system:
        out.append({"role": "system", "content": system})

    for message in req.messages:
        if message.role == "system":
            out.append({"role": "system", "content": _join_text(message.blocks)})
        elif message.role == "tool":
            for block in message.blocks:
                if isinstance(block, ToolResultBlock):
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.output,
                        }
                    )
        elif message.role == "user":
            out.append({"role": "user", "content": _join_text(message.blocks)})
        else:
            text = _join_text(message.blocks)
            tool_calls = [
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
                for block in message.blocks
                if isinstance(block, ToolUseBlock)
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            elif not text:
                msg["content"] = ""
            out.append(msg)

    return out


async def _assemble_openai(stream: AsyncIterator[Any]) -> AsyncIterator[AssistantEvent]:
    """Chat Completions のストリームチャンクを ``AssistantEvent`` へ組み立てる。"""
    partial: dict[int, dict[str, str]] = {}
    stop_reason: str | None = None
    received = False
    produced = False
    think_buf: list[str] = []

    async for chunk in stream:
        received = True
        choice = chunk.choices[0] if chunk.choices else None
        if choice is not None:
            delta = choice.delta
            reasoning = _reasoning_text(delta)
            if reasoning:
                think_buf.append(reasoning)
            content = getattr(delta, "content", None)
            has_tool_call = bool(getattr(delta, "tool_calls", None))
            if think_buf and (content or has_tool_call or choice.finish_reason):
                yield Thinking(text="".join(think_buf), signature=None)
                think_buf.clear()
            if content:
                produced = True
                yield TextDelta(text=content)
            for tc in getattr(delta, "tool_calls", None) or []:
                produced = True
                slot = partial.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
                        yield ToolUseDelta(
                            index=tc.index, name=slot["name"], args_delta=fn.arguments
                        )
            if choice.finish_reason:
                stop_reason = choice.finish_reason
                for ev in _drain_tool_calls(partial):
                    yield ev

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            produced = True
            details = getattr(usage, "prompt_tokens_details", None)
            yield Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cache_read=getattr(details, "cached_tokens", 0) or 0,
            )

    if received and stop_reason is None and not produced:
        raise ProviderError(
            ui(
                "providers.openai_provider.the_stream_closed_with_neither",
                marker=_TRUNCATED_TOOL_CALL,
            ),
            retryable=True,
        )

    if think_buf:
        yield Thinking(text="".join(think_buf), signature=None)

    pending = _drain_tool_calls(partial)
    if pending:
        _log.warning(
            "flushed %d pending tool_call(s) at an end without finish_reason", len(pending)
        )
        for ev in pending:
            yield ev

    if not received:
        raise ProviderError(
            ui("providers.openai_provider.the_assistant_stream_ended_without"),
            retryable=True,
        )

    yield MessageStop(stop_reason=_stop_reason_for(stop_reason))


def _without_tools(shape: _Shape, req: ApiRequest) -> _Shape:
    """ツールを外し、強制した ``none`` を利用者の指定に戻す。"""
    kwargs = dict(shape.kwargs)
    if shape.forced_effort_none:
        user = req.sampling.reasoning_effort if req.sampling is not None else None
        if user is None or any(name == "reasoning_effort" for name, _ in shape.rejected):
            kwargs.pop("reasoning_effort", None)
        else:
            kwargs["reasoning_effort"] = user
    return replace(shape, tools=[], kwargs=kwargs, forced_effort_none=False, tools_dropped=True)


async def _close_sdk_stream(stream: object) -> None:
    """SDK のストリーム（``AsyncStream``）を閉じて HTTP 接続を解放する。"""
    close = getattr(stream, "close", None) or getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        _log.debug("could not close the SDK stream: %s", exc)


class OpenAICompatProvider:
    """xAI / DashScope / Ollama / mlx-lm / OpenRouter / local を base_url 差で統一。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        kind: ProviderKind = ProviderKind.OPENAI,
        *,
        timeout: float | httpx.Timeout | None = None,
        ollama_root: str | None = None,
        keep_alive: str | None = None,
        openai: OpenAIConfig | None = None,
    ) -> None:
        self.kind = kind
        self._base_url = base_url
        self._ollama_root = ollama_root
        tuning = openai if openai is not None else OpenAIConfig()
        host = urlparse(base_url if "://" in base_url else f"http://{base_url}").hostname or ""
        self._official = ollama_root is None and host in {h.lower() for h in tuning.hosts}
        # xAI も 400 が名指しした引数から学ぶ（base_url での指定も含む）。ツールの手探りは公式だけ
        self._learns = self._official or (
            ollama_root is None and (kind is ProviderKind.XAI or host == _XAI_HOST)
        )
        self._negotiate_400 = tuning.negotiate
        self._keep_alive = keep_alive if (ollama_root and keep_alive) else None
        self._bg: set[asyncio.Task[None]] = set()
        self._timeout = timeout if isinstance(timeout, httpx.Timeout) else None
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=0,
            timeout=timeout if timeout is not None else NOT_GIVEN,
        )

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in req.tools
        ]
        caps = capabilities_for(self._base_url)
        want_strict = req.strict_tools and bool(tools) and not caps.strict_tools_unsupported
        saw_stop = False
        stream: AsyncIterator[Any] | None = None
        try:
            stream = await self._open(req, tools, strict=want_strict)
            if self._ollama_root is not None:
                note_model_used(self._ollama_root, req.model)
            silence = (
                self._timeout.read
                if self._timeout is not None and self._first_byte_timeout(req) is not None
                else None
            )
            async for ev in _assemble_openai(_watch_chunks(stream, silence)):
                if isinstance(ev, MessageStop):
                    saw_stop = True
                yield ev
        except (APIError, httpx.HTTPError) as e:
            raise ProviderError(
                ui(
                    "providers.openai_provider.openai_compatible_api_call_failed", error=describe(e)
                ),
                status_code=status_code_of(e),
                retry_after=retry_after_of(e),
                retryable=True if _at_capacity(e) else transport_retryable(e),
            ) from e
        finally:
            if stream is not None:
                await _close_sdk_stream(stream)
            if saw_stop:
                await self._touch_keep_alive(req.model)

    async def _open(
        self, req: ApiRequest, tools: list[dict[str, Any]], *, strict: bool
    ) -> AsyncIterator[Any]:
        """``create`` を発行し、400 なら形を変えて再送する（OpenAI 公式と xAI は引数も調整）。"""
        quirks = model_quirks_for(self._base_url, req.model) if self._learns else None
        shape = self._initial_shape(req, tools, strict, quirks)
        error: APIError | None = None
        error_strict = False
        for _ in range(_MAX_NEGOTIATIONS + 1):
            try:
                stream = await self._create(req, shape)
            except APIError as e:
                if _not_a_chat_model(e):
                    text = ui(
                        "providers.openai_provider.model_needs_the_responses_api", model=req.model
                    )
                    raise ProviderError(text, status_code=status_code_of(e), retryable=False) from e
                if status_code_of(e) != 400:
                    raise
                if error is None or error_strict:
                    error, error_strict = e, shape.strict
                following = self._negotiate(e, shape, req, quirks)
                if following is None:
                    break
                _log.debug(
                    "400 from %s, so re-sending in another shape (param=%s, code=%s): %s",
                    req.model,
                    e.param,
                    e.code,
                    e.message,
                )
                shape = following
                continue
            self._commit(shape, req, quirks)
            return stream
        assert error is not None
        raise error

    def _initial_shape(
        self, req: ApiRequest, tools: list[dict[str, Any]], strict: bool, quirks: ModelQuirks | None
    ) -> _Shape:
        """記憶済みの形を反映した最初の要求。"""
        kwargs = _request_kwargs(req, self._official)
        if quirks is None:
            return _Shape(tools=tools, strict=strict, kwargs=kwargs)
        cap = _cap_param(kwargs)
        if cap is not None and quirks.no_token_cap:
            del kwargs[cap]
        elif cap is not None and quirks.token_cap is not None:
            kwargs[cap] = min(kwargs[cap], quirks.token_cap)
        # 拒まれた記憶は (引数名, 値) の組。同じ値のときだけ外し、別の値は改めて試す
        rejected = frozenset(
            (name, value)
            for name, value in quirks.rejected_values
            if name in kwargs and str(kwargs[name]) == value
        )
        for name, _ in rejected:
            del kwargs[name]
        shape = _Shape(tools=tools, strict=strict, kwargs=kwargs, rejected=rejected)
        if quirks.tools_unsupported and tools:
            return replace(shape, tools=[], tools_dropped=True)
        if quirks.tools_need_effort_none and tools:
            kwargs["reasoning_effort"] = "none"
            return replace(shape, forced_effort_none=True)
        return shape

    def _negotiate(
        self, e: APIError, shape: _Shape, req: ApiRequest, quirks: ModelQuirks | None
    ) -> _Shape | None:
        """400 の内容から次に試す形を決める。None なら諦める。"""
        param, message = _param_of(e), e.message or ""
        if param == "model":
            return None
        negotiable = quirks is not None and self._negotiate_400
        if negotiable:
            # param で原因が分かる 400 は strict を残したまま直す
            identified = self._fix_named_param(param, message, shape, req)
            if identified is not None:
                return identified
        if shape.strict:
            return replace(shape, strict=False, strict_dropped=True)
        if not negotiable or not self._official or not shape.tools or shape.tools_dropped:
            return None
        # ここから先は手探りの変更なので、strict 外しが効いたとは言えず、記憶しない
        kwargs = dict(shape.kwargs)
        if kwargs.get("reasoning_effort") != "none" and not shape.forced_effort_none:
            kwargs["reasoning_effort"] = "none"
            return replace(shape, kwargs=kwargs, forced_effort_none=True, strict_dropped=False)
        return replace(_without_tools(shape, req), strict_dropped=False)

    @staticmethod
    def _fix_named_param(
        param: str | None, message: str, shape: _Shape, req: ApiRequest
    ) -> _Shape | None:
        """400 が名指しした引数を直した形。名指しが無い・直せないなら None。"""
        kwargs = dict(shape.kwargs)
        cap = _cap_param(kwargs)
        if cap is not None and param in _TOKEN_CAP_PARAMS:
            found = _CAP_RE.search(message)
            if found and int(found.group(1)) > 0:
                kwargs[cap] = int(found.group(1))
            else:
                del kwargs[cap]
            return replace(shape, kwargs=kwargs, strict_dropped=False)
        if param == "reasoning_effort" and shape.forced_effort_none:
            return replace(_without_tools(shape, req), strict_dropped=False)
        if param in _SAMPLING_PARAMS and param in kwargs:
            rejected = shape.rejected | {(param, str(kwargs.pop(param)))}
            return replace(shape, kwargs=kwargs, rejected=rejected, strict_dropped=False)
        return None

    def _commit(self, shape: _Shape, req: ApiRequest, quirks: ModelQuirks | None) -> None:
        """成功した形を記憶し、ツールに関わる変更は 1 回だけ利用者に知らせる。"""
        if shape.strict_dropped:
            caps = capabilities_for(self._base_url)
            caps.mark_strict_unsupported()
            if not caps.strict_fallback_logged:
                caps.strict_fallback_logged = True
                _log.warning(
                    "strict tools were refused with a 400, so they are no longer sent (%s)",
                    self._base_url,
                )
        if quirks is None:
            return
        cap = _cap_param(shape.kwargs)
        if cap is None:
            quirks.no_token_cap = True
        elif shape.kwargs[cap] < req.max_tokens:
            quirks.token_cap = shape.kwargs[cap]
        quirks.rejected_values |= set(shape.rejected)
        kind: notices.NoticeKind | None = None
        if shape.tools_dropped:
            quirks.tools_unsupported = True
            kind = "tools_dropped"
        elif shape.forced_effort_none:
            quirks.tools_need_effort_none = True
            kind = "effort_forced"
        if kind is None or quirks.notice_shown:
            return
        quirks.notice_shown = True
        _log.warning("%s: %s", req.model, kind)
        notices.notify(notices.ProviderNotice(kind=kind, model=req.model))

    async def _create(self, req: ApiRequest, shape: _Shape) -> AsyncIterator[Any]:
        tools = _strict_tools(shape.tools) if shape.strict else shape.tools
        call: dict[str, Any] = {
            "model": req.model,
            "messages": _to_openai_messages(req),
            "tools": tools or NOT_GIVEN,
            "stream": True,
            **shape.kwargs,
        }
        first_byte = self._first_byte_timeout(req)
        if first_byte is not None and self._timeout is not None:
            t = self._timeout
            call["timeout"] = httpx.Timeout(
                connect=t.connect, read=first_byte, write=t.write, pool=t.pool
            )
        return cast("AsyncIterator[Any]", await self._client.chat.completions.create(**call))

    def _first_byte_timeout(self, req: ApiRequest) -> float | None:
        """この要求で延ばす最初の中身までの待ち。"""
        wait = req.first_byte_timeout_s
        if wait is None or self._timeout is None or self._timeout.read is None:
            return None
        return wait if wait > self._timeout.read else None

    async def _touch_keep_alive(self, model: str) -> None:
        """Ollama の常駐期限をネイティブ API で延長する。"""
        if self._keep_alive is None or self._ollama_root is None:
            return
        task = asyncio.create_task(touch_keep_alive(self._ollama_root, model, self._keep_alive))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        await asyncio.wait({task}, timeout=_KEEP_ALIVE_WAIT_S)

    async def count_tokens(self, req: ApiRequest) -> int:
        blob = json.dumps(_to_openai_messages(req), ensure_ascii=False)
        if req.tools:
            blob += json.dumps(req.tools, ensure_ascii=False)
        return estimate_tokens(blob)
