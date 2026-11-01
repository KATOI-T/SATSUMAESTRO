"""モデル連携堅牢化のテスト。"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import httpx
import openai
import pytest
from rich.console import Console

from satsumaestro.cli.render import RichRenderer
from satsumaestro.config.models import ProviderConfig, RetryConfig, RuntimeConfig
from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    ToolUseDelta,
    Usage,
)
from satsumaestro.providers.base import ApiRequest, Provider, ProviderKind
from satsumaestro.providers.factory import build_bare, build_provider
from satsumaestro.providers.fallback import FallbackProvider
from satsumaestro.providers.openai_provider import _assemble_openai, _stop_reason_for
from satsumaestro.providers.prompt_cache import PromptCacheMonitor
from satsumaestro.providers.retry import (
    RetryingProvider,
    RetryPolicy,
    backoff_delay,
    is_retryable,
    is_tool_call_parse_error,
    retry_after_of,
)
from satsumaestro.session.models import ConversationMessage, TextBlock
from satsumaestro.util.errors import ProviderError, ProviderTimeoutError, RetriesExhaustedError
from tests.fixtures.mock_provider import MockProvider, ScriptedFailureProvider

_FAST = RetryPolicy(max_retries=3, initial_backoff_s=1.0, max_backoff_s=8.0, timeout_s=600.0)


def _req(model: str = "gpt-oss:120b", *, messages: list[ConversationMessage] | None = None):
    return ApiRequest(model=model, system="sys", messages=messages or [], tools=[])


def _msg(text: str) -> ConversationMessage:
    return ConversationMessage(role="user", blocks=[TextBlock(text=text)])


def _ok(text: str = "hi") -> list[AssistantEvent | BaseException]:
    return [TextDelta(text=text), MessageStop(stop_reason="end_turn")]


def _http_error(status: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    """SDK の APIStatusError を実体で作る（duck-typing ではなく本物で分類を検証する）。"""
    response = httpx.Response(
        status, request=httpx.Request("POST", "http://localhost:1"), headers=headers or {}
    )
    return openai.APIStatusError("boom", response=response, body=None)


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:1"))


def _wrapped(exc: BaseException, **kwargs: Any) -> ProviderError:
    """プロバイダ境界が行う変換（``ProviderError`` へ包んで ``__cause__`` を張る）を再現。"""
    err = ProviderError(f"失敗: {exc}", **kwargs)
    err.__cause__ = exc
    return err


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """``asyncio.sleep`` を差し替え、要求された待機秒だけを記録する（実待機なし）。"""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


async def _drain(provider: Provider, req: ApiRequest | None = None) -> list[AssistantEvent]:
    return [ev async for ev in provider.stream(req if req is not None else _req())]


async def test_retry_recovers_after_transient(slept: list[float]) -> None:
    inner = ScriptedFailureProvider(
        [_wrapped(_connection_error()), _wrapped(_connection_error()), _ok("done")]
    )
    out = await _drain(RetryingProvider(inner, _FAST))

    assert out == [TextDelta(text="done"), MessageStop(stop_reason="end_turn")]
    assert inner.attempts == 3
    assert len(slept) == 2


async def test_retry_gives_up_and_raises(slept: list[float]) -> None:
    inner = ScriptedFailureProvider([_wrapped(_http_error(503))])
    with pytest.raises(RetriesExhaustedError) as excinfo:
        await _drain(RetryingProvider(inner, _FAST))

    assert inner.attempts == _FAST.max_retries + 1
    assert excinfo.value.attempts == _FAST.max_retries + 1
    assert excinfo.value.retryable is False


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_non_retryable_not_retried(status: int, slept: list[float]) -> None:
    inner = ScriptedFailureProvider([_wrapped(_http_error(status))])
    with pytest.raises(ProviderError):
        await _drain(RetryingProvider(inner, _FAST))

    assert inner.attempts == 1
    assert slept == []


async def test_no_retry_after_first_yield(slept: list[float]) -> None:
    inner = ScriptedFailureProvider(
        [[TextDelta(text="partial"), _wrapped(_http_error(503))], _ok("recovered")]
    )
    collected: list[AssistantEvent] = []
    with pytest.raises(ProviderError):
        async for ev in RetryingProvider(inner, _FAST).stream(_req()):
            collected.append(ev)

    assert collected == [TextDelta(text="partial")]
    assert inner.attempts == 1
    assert slept == []


async def test_retry_after_header_preferred(slept: list[float]) -> None:
    inner = ScriptedFailureProvider(
        [_wrapped(_http_error(429, {"retry-after": "5"})), _ok()],
    )
    await _drain(RetryingProvider(inner, _FAST))
    assert slept == [5.0]


async def test_retry_budget_exhausted_raises_timeout(slept: list[float]) -> None:
    policy = RetryPolicy(max_retries=8, initial_backoff_s=1.0, max_backoff_s=8.0, timeout_s=0.5)
    inner = ScriptedFailureProvider([_wrapped(_http_error(503))])
    with pytest.raises(ProviderTimeoutError) as excinfo:
        await _drain(RetryingProvider(inner, policy))

    assert isinstance(excinfo.value, RetriesExhaustedError)
    assert inner.attempts == 1
    assert slept == []


async def test_consumer_athrow_is_not_retried(slept: list[float]) -> None:
    inner = ScriptedFailureProvider([_ok("first"), _ok("second")])
    stream = RetryingProvider(inner, _FAST).stream(_req())
    assert await stream.__anext__() == TextDelta(text="first")

    with pytest.raises(RuntimeError, match="consumer side"):
        await stream.athrow(RuntimeError("consumer side"))

    assert inner.attempts == 1
    assert slept == []


async def test_inner_stream_is_aclosed_between_attempts(slept: list[float]) -> None:
    inner = ScriptedFailureProvider([_wrapped(_connection_error()), _ok()])
    await _drain(RetryingProvider(inner, _FAST))
    assert inner.aclose_calls == inner.attempts == 2


async def test_count_tokens_is_not_retried() -> None:
    class Boom:
        kind = ProviderKind.OPENAI
        calls = 0

        def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:  # pragma: no cover
            raise AssertionError("未使用")

        async def count_tokens(self, req: ApiRequest) -> int:
            Boom.calls += 1
            raise _wrapped(_http_error(503))

    with pytest.raises(ProviderError):
        await RetryingProvider(Boom(), _FAST).count_tokens(_req())  # type: ignore[arg-type]
    assert Boom.calls == 1


_TOOL_PARSE_500 = (
    'error parsing tool call: raw=\'{"limit":5","offset":0,"path":"handler_05..."}\', '
    "err=invalid character '\"' after object key:value pair"
)


def _tool_parse_failure() -> ProviderError:
    err = _http_error(500)
    err.args = (_TOOL_PARSE_500,)
    return _wrapped(err)


def test_is_tool_call_parse_error_detection() -> None:
    assert is_tool_call_parse_error(_tool_parse_failure()) is True
    assert is_tool_call_parse_error(_wrapped(_http_error(500))) is False


async def test_tool_call_parse_error_retry_is_capped(slept: list[float]) -> None:
    inner = ScriptedFailureProvider([_tool_parse_failure()])
    policy = RetryPolicy(max_retries=8, max_backoff_s=16.0, tool_parse_max_retries=1)
    with pytest.raises(RetriesExhaustedError) as excinfo:
        await _drain(RetryingProvider(inner, policy))

    assert inner.attempts == 2
    assert excinfo.value.attempts == 2
    assert len(slept) == 1


async def test_tool_call_parse_error_can_still_recover(slept: list[float]) -> None:
    inner = ScriptedFailureProvider([_tool_parse_failure(), _ok("recovered")])
    out = await _drain(RetryingProvider(inner, _FAST))
    assert out[0] == TextDelta(text="recovered")
    assert inner.attempts == 2


async def test_ordinary_500_still_uses_full_retry_budget(slept: list[float]) -> None:
    inner = ScriptedFailureProvider([_wrapped(_http_error(503))])
    with pytest.raises(RetriesExhaustedError):
        await _drain(RetryingProvider(inner, _FAST))
    assert inner.attempts == _FAST.max_retries + 1


async def test_tool_parse_exhaustion_still_triggers_fallback() -> None:
    primary = ScriptedFailureProvider([_tool_parse_failure()])
    secondary = MockProvider([TextDelta(text="from-fallback"), MessageStop()])
    policy = RetryPolicy(max_retries=8, tool_parse_max_retries=0)
    out = await _drain(_fallback(RetryingProvider(primary, policy), secondary))
    assert out[0] == TextDelta(text="from-fallback")
    assert primary.attempts == 1


def test_backoff_delay_doubles_from_the_first_wait_up_to_the_cap() -> None:
    policy = RetryPolicy(initial_backoff_s=0.5, max_backoff_s=8.0)
    floors = [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
    for attempt, floor in enumerate(floors, 1):
        for _ in range(20):
            assert floor <= backoff_delay(attempt, policy) <= 2 * floor
    assert backoff_delay(10_000, policy) <= 16.0


def test_is_retryable_classification() -> None:
    for status in (408, 409, 429, 500, 501, 502, 503, 504, 529):
        assert is_retryable(_wrapped(_http_error(status))) is True
    for status in (400, 401, 403, 404, 413, 422):
        assert is_retryable(_wrapped(_http_error(status))) is False
    assert is_retryable(_wrapped(_connection_error(), retryable=True)) is True
    assert is_retryable(ProviderError("tool_call 引数が JSON として不正")) is False
    assert is_retryable(RetriesExhaustedError(9, 1.0)) is False
    assert is_retryable(asyncio.CancelledError()) is False


def test_retry_after_parses_seconds_and_http_date() -> None:
    assert retry_after_of(_wrapped(_http_error(429, {"retry-after": "12"}))) == 12.0
    past = _wrapped(_http_error(429, {"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}))
    assert retry_after_of(past) == 0.0
    assert retry_after_of(_wrapped(_http_error(500))) is None


def _fallback(
    primary: Provider, fallback: Provider, model: str = "claude-sonnet-5"
) -> FallbackProvider:
    return FallbackProvider(primary, build_fallback=lambda: fallback, fallback_model=model)


async def test_fallback_switches_on_permanent_failure(slept: list[float]) -> None:
    primary = ScriptedFailureProvider([_wrapped(_http_error(503))])
    secondary = MockProvider([TextDelta(text="from-fallback"), MessageStop()])
    provider = _fallback(RetryingProvider(primary, _FAST), secondary)

    out = await _drain(provider)
    assert out[0] == TextDelta(text="from-fallback")
    assert secondary.requests[0].model == "claude-sonnet-5"
    assert provider.kind is secondary.kind

    attempts_after_first = primary.attempts
    await _drain(provider)
    assert primary.attempts == attempts_after_first
    assert len(secondary.requests) == 2


async def test_fallback_absent_propagates() -> None:
    inner = ScriptedFailureProvider([_wrapped(_http_error(401))])
    with pytest.raises(ProviderError):
        await _drain(RetryingProvider(inner, _FAST))


async def test_fallback_skips_hopeless_status() -> None:
    primary = ScriptedFailureProvider([_wrapped(_http_error(400))])
    secondary = MockProvider([TextDelta(text="unused"), MessageStop()])
    with pytest.raises(ProviderError):
        await _drain(_fallback(primary, secondary))
    assert secondary.requests == []


async def test_fallback_not_switched_after_first_yield() -> None:
    primary = ScriptedFailureProvider([[TextDelta(text="partial"), _wrapped(_http_error(401))]])
    secondary = MockProvider([TextDelta(text="unused"), MessageStop()])
    collected: list[AssistantEvent] = []
    with pytest.raises(ProviderError):
        async for ev in _fallback(primary, secondary).stream(_req()):
            collected.append(ev)
    assert collected == [TextDelta(text="partial")]
    assert secondary.requests == []


async def test_fallback_same_model_is_not_switched() -> None:
    primary = ScriptedFailureProvider([_wrapped(_http_error(401))])
    secondary = MockProvider([TextDelta(text="unused"), MessageStop()])
    provider = _fallback(primary, secondary, model="gpt-oss:120b")
    with pytest.raises(ProviderError):
        await _drain(provider, _req("gpt-oss:120b"))
    assert secondary.requests == []


async def test_fallback_build_failure_propagates_primary() -> None:
    primary = ScriptedFailureProvider([_wrapped(_http_error(401))])

    def broken() -> Provider:
        raise ProviderError("ANTHROPIC_API_KEY is not set")

    provider = FallbackProvider(primary, build_fallback=broken, fallback_model="claude-sonnet-5")
    with pytest.raises(ProviderError) as excinfo:
        await _drain(provider)
    assert "ANTHROPIC_API_KEY" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, openai.APIStatusError)


async def _chunks(items: list[NS]) -> AsyncIterator[NS]:
    for item in items:
        yield item


def _delta(**kwargs: Any) -> NS:
    base: dict[str, Any] = {"content": None, "tool_calls": None}
    base.update(kwargs)
    return NS(**base)


async def test_openai_reasoning_becomes_thinking() -> None:
    stream = _chunks(
        [
            NS(
                choices=[NS(delta=_delta(reasoning_content="考え中"), finish_reason=None)],
                usage=None,
            ),
            NS(choices=[NS(delta=_delta(content="答え"), finish_reason="stop")], usage=None),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [
        Thinking(text="考え中", signature=None),
        TextDelta(text="答え"),
        MessageStop(stop_reason="end_turn"),
    ]


async def test_openai_reasoning_deltas_collapse_into_one_thinking() -> None:
    stream = _chunks(
        [
            NS(choices=[NS(delta=_delta(reasoning_content="ま"), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=_delta(reasoning="ず"), finish_reason=None)], usage=None),
            NS(
                choices=[NS(delta=_delta(reasoning_content="考える"), finish_reason=None)],
                usage=None,
            ),
            NS(choices=[NS(delta=_delta(content="答え"), finish_reason="stop")], usage=None),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [
        Thinking(text="まず考える", signature=None),
        TextDelta(text="答え"),
        MessageStop(stop_reason="end_turn"),
    ]


async def test_openai_reasoning_closes_before_tool_call() -> None:
    tc = NS(index=0, id="call_1", function=NS(name="read", arguments="{}"))
    stream = _chunks(
        [
            NS(
                choices=[NS(delta=_delta(reasoning_content="読もう"), finish_reason=None)],
                usage=None,
            ),
            NS(choices=[NS(delta=_delta(tool_calls=[tc]), finish_reason="tool_calls")], usage=None),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [
        Thinking(text="読もう", signature=None),
        ToolUse(id="call_1", name="read", input={}),
        MessageStop(stop_reason="tool_use"),
    ]


async def test_openai_reasoning_only_close_is_a_tool_call_parse_failure() -> None:
    stream = _chunks(
        [NS(choices=[NS(delta=_delta(reasoning_content="途中"), finish_reason=None)], usage=None)]
    )
    out: list[object] = []
    with pytest.raises(ProviderError) as info:
        async for ev in _assemble_openai(stream):
            out.append(ev)
    assert out == []
    assert is_tool_call_parse_error(info.value)
    assert is_retryable(info.value)


async def test_openai_reasoning_with_usage_but_no_finish_still_closes() -> None:
    stream = _chunks(
        [
            NS(
                choices=[NS(delta=_delta(reasoning_content="途中"), finish_reason=None)], usage=None
            ),
            NS(choices=[], usage=NS(prompt_tokens=3, completion_tokens=2)),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert Thinking(text="途中", signature=None) in out
    assert out[-1] == MessageStop(stop_reason="end_turn")


async def test_openai_truncated_stream_flushes_thinking_then_tool_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tc = NS(index=0, id="call_1", function=NS(name="read", arguments="{}"))
    stream = _chunks(
        [
            NS(
                choices=[NS(delta=_delta(reasoning_content="読もう"), finish_reason=None)],
                usage=None,
            ),
            NS(choices=[NS(delta=_delta(tool_calls=[tc]), finish_reason=None)], usage=None),
        ]
    )
    with caplog.at_level(logging.WARNING):
        out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [
        Thinking(text="読もう", signature=None),
        ToolUse(id="call_1", name="read", input={}),
        MessageStop(stop_reason="end_turn"),
    ]
    assert "pending tool_call(s) at an end without finish_reason" in caplog.text


async def test_retry_recovers_from_empty_stream(slept: list[float]) -> None:
    inner = ScriptedFailureProvider(
        [
            ProviderError("アシスタントストリームが 1 イベントも返さず終了した", retryable=True),
            _ok("done"),
        ]
    )
    out = await _drain(RetryingProvider(inner, _FAST))
    assert out == [TextDelta(text="done"), MessageStop(stop_reason="end_turn")]
    assert inner.attempts == 2
    assert len(slept) == 1


@pytest.mark.parametrize(
    ("fields", "thinking"),
    [
        ({"reasoning": "t"}, "t"),
        ({"reasoning_content": "t"}, "t"),
        ({"reasoning": "r", "reasoning_content": "c"}, "r"),
        ({"reasoning": "", "reasoning_content": "c"}, "c"),
    ],
)
async def test_openai_reasoning_fields(fields: dict[str, str], thinking: str) -> None:
    stream = _chunks([NS(choices=[NS(delta=_delta(**fields), finish_reason="stop")], usage=None)])
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out[0] == Thinking(text=thinking, signature=None)


async def test_openai_malformed_args_fallback_empty(caplog: pytest.LogCaptureFixture) -> None:
    tc = NS(index=0, id="call_1", function=NS(name="read", arguments="思考テキスト"))
    stream = _chunks(
        [
            NS(choices=[NS(delta=_delta(tool_calls=[tc]), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=_delta(), finish_reason="tool_calls")], usage=None),
        ]
    )
    with caplog.at_level(logging.WARNING):
        out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]

    assert out[0] == ToolUse(id="call_1", name="read", input={}, input_fallback=True)
    assert "not valid JSON" in caplog.text


async def test_openai_tool_calls_null_and_id_synth() -> None:
    tc = NS(index=0, id=None, function=NS(name="do", arguments="{}"))
    stream = _chunks(
        [
            NS(choices=[NS(delta=_delta(tool_calls=None), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=_delta(tool_calls=[tc]), finish_reason="tool_calls")], usage=None),
        ]
    )
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [
        ToolUse(id="call_synth_0", name="do", input={}),
        MessageStop(stop_reason="tool_use"),
    ]


async def test_openai_nameless_tool_call_is_dropped(caplog: pytest.LogCaptureFixture) -> None:
    tc = NS(index=0, id="call_1", function=NS(name=None, arguments="{}"))
    stream = _chunks(
        [NS(choices=[NS(delta=_delta(tool_calls=[tc]), finish_reason="tool_calls")], usage=None)]
    )
    with caplog.at_level(logging.WARNING):
        out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [MessageStop(stop_reason="tool_use")]
    assert "without a name" in caplog.text


async def test_openai_usage_reports_cached_tokens() -> None:
    usage = NS(prompt_tokens=100, completion_tokens=5, prompt_tokens_details=NS(cached_tokens=80))
    stream = _chunks([NS(choices=[], usage=usage)])
    out = [ev async for ev in _assemble_openai(stream) if not isinstance(ev, ToolUseDelta)]
    assert out == [
        Usage(input_tokens=100, output_tokens=5, cache_read=80),
        MessageStop(stop_reason="end_turn"),
    ]


@pytest.mark.parametrize(
    ("finish_reason", "stop_reason"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        ("content_filter", "refusal"),
        (None, "end_turn"),
        ("", "end_turn"),
        ("eos", "eos"),
    ],
)
def test_stop_reason_for(finish_reason: str | None, stop_reason: str) -> None:
    assert _stop_reason_for(finish_reason) == stop_reason


def test_cache_loss_with_the_same_head_is_unexpected() -> None:
    monitor = PromptCacheMonitor(min_drop=2000)
    assert monitor.observe(_req(messages=[_msg("a")]), Usage(cache_read=10_000)) is None

    loss = monitor.observe(_req(messages=[_msg("a"), _msg("b")]), Usage(cache_read=0))
    assert loss is not None and loss.unexpected
    assert (loss.before, loss.after, loss.lost, loss.changed) == (10_000, 0, 10_000, None)


def test_cache_loss_names_the_first_part_that_changed() -> None:
    monitor = PromptCacheMonitor(min_drop=2000)
    monitor.observe(_req(messages=[_msg("a"), _msg("b")]), Usage(cache_read=9_000))
    loss = monitor.observe(_req(messages=[_msg("要約")]), Usage(cache_read=0))
    assert loss is not None and not loss.unexpected
    assert loss.changed == "message 1 of the conversation"


@pytest.mark.parametrize(
    ("change", "part"),
    [
        ({"model": "other"}, "the model"),
        ({"tools": [{"name": "t"}]}, "the tool list"),
        ({"system": "別の sys"}, "the fixed part of system"),
        (
            {"system_dynamic": "git: dirty"},
            "the changing part of system (git state, TODOs and so on)",
        ),
    ],
)
def test_cache_loss_names_each_part_of_the_head(change: dict[str, Any], part: str) -> None:
    monitor = PromptCacheMonitor(min_drop=1)
    monitor.observe(_req(), Usage(cache_read=5_000))
    loss = monitor.observe(_req().model_copy(update=change), Usage(cache_read=0))
    assert loss is not None and loss.changed == part


def test_cache_loss_below_the_threshold_is_not_reported() -> None:
    monitor = PromptCacheMonitor(min_drop=2000)
    monitor.observe(_req(messages=[_msg("a")]), Usage(cache_read=5_000))
    assert monitor.observe(_req(messages=[_msg("a"), _msg("b")]), Usage(cache_read=3_001)) is None


def _render_thinking(*, show_thinking: bool) -> str:
    buf = io.StringIO()
    renderer = RichRenderer(
        console=Console(file=buf, force_terminal=False, width=80), show_thinking=show_thinking
    )
    renderer.start_turn()
    renderer.on_assistant_event(Thinking(text="ユーザーの意図を推測する"))
    renderer.on_assistant_event(TextDelta(text="こんにちは"))
    renderer.end_turn()
    return buf.getvalue()


def test_thinking_is_hidden_by_default() -> None:
    out = _render_thinking(show_thinking=False)
    assert "こんにちは" in out
    assert "推測する" not in out


def test_thinking_shown_when_enabled() -> None:
    out = _render_thinking(show_thinking=True)
    assert "推測する" in out and "こんにちは" in out


def test_show_thinking_defaults_to_false() -> None:
    assert RuntimeConfig().show_thinking is False


def _cfg(**provider_kwargs: Any) -> RuntimeConfig:
    return RuntimeConfig(provider=ProviderConfig(**provider_kwargs))


def test_factory_wraps_with_retry_only_by_default() -> None:
    env = {"OLLAMA_HOST": "http://localhost:11434"}
    provider = build_provider(_cfg(), "qwen2.5-coder", env)
    assert isinstance(provider, RetryingProvider)
    assert provider.kind is ProviderKind.OPENAI


def test_factory_wraps_with_fallback_when_configured() -> None:
    env = {"OLLAMA_HOST": "http://localhost:11434", "ANTHROPIC_API_KEY": "sk-ant-x"}
    provider = build_provider(_cfg(fallback_model="claude-sonnet-5"), "qwen2.5-coder", env)
    assert isinstance(provider, FallbackProvider)
    assert isinstance(provider._primary, RetryingProvider)


def test_factory_skips_fallback_when_same_model() -> None:
    env = {"OLLAMA_HOST": "http://localhost:11434"}
    provider = build_provider(_cfg(fallback_model="qwen2.5-coder"), "qwen2.5-coder", env)
    assert isinstance(provider, RetryingProvider)


def test_factory_does_not_require_fallback_key_at_build_time() -> None:
    env = {"OLLAMA_HOST": "http://localhost:11434"}
    assert isinstance(
        build_provider(_cfg(fallback_model="claude-sonnet-5"), "qwen2.5-coder", env),
        FallbackProvider,
    )


def test_factory_passes_request_timeout_to_client() -> None:
    cfg = RuntimeConfig(
        provider=ProviderConfig(retry=RetryConfig(request_timeout_s=30.0)), api_timeout_s=600.0
    )
    bare = build_bare(cfg, "qwen2.5-coder", {"OLLAMA_HOST": "http://localhost:11434"})
    timeout = bare._client.timeout  # type: ignore[attr-defined]
    assert timeout.read == 30.0
    assert bare._client.max_retries == 0  # type: ignore[attr-defined]


def test_request_timeout_is_clamped_to_total_budget() -> None:
    cfg = RuntimeConfig(
        provider=ProviderConfig(retry=RetryConfig(request_timeout_s=120.0)), api_timeout_s=45.0
    )
    bare = build_bare(cfg, "qwen2.5-coder", {"OLLAMA_HOST": "http://localhost:11434"})
    assert bare._client.timeout.read == 45.0  # type: ignore[attr-defined]


def test_connect_timeout_is_short_and_independent_of_read() -> None:
    """接続確立は read とは別に短く保つ。"""
    cfg = RuntimeConfig(provider=ProviderConfig(retry=RetryConfig(request_timeout_s=600.0)))
    bare = build_bare(cfg, "qwen2.5-coder", {"OLLAMA_HOST": "http://localhost:11434"})
    timeout = bare._client.timeout  # type: ignore[attr-defined]
    assert timeout.connect == 10.0
    assert timeout.read == 600.0


def test_anthropic_client_gets_the_same_phase_split() -> None:
    cfg = RuntimeConfig(
        provider=ProviderConfig(retry=RetryConfig(request_timeout_s=30.0)), api_timeout_s=600.0
    )
    bare = build_bare(cfg, "claude-opus-4-8", {"ANTHROPIC_API_KEY": "k"})
    timeout = bare._client.timeout  # type: ignore[attr-defined]
    assert (timeout.connect, timeout.read) == (10.0, 30.0)


@pytest.mark.parametrize(
    "retry",
    [
        {"max_retries": -1},
        {"initial_backoff_s": 0.0},
        {"max_backoff_s": -1.0},
        {"request_timeout_s": 0.0},
        {"cache_break_min_drop": -1},
        {"unknown_key": 1},
    ],
)
def test_retry_config_rejects_invalid(retry: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RuntimeConfig.model_validate({"provider": {"retry": retry}})


def test_retry_config_defaults_match_design() -> None:
    retry = RuntimeConfig().provider.retry
    assert (retry.max_retries, retry.initial_backoff_s) == (8, 1.0)
    assert retry.max_backoff_s == 16.0
    assert retry.tool_parse_max_retries == 1
    assert retry.cache_break_min_drop == 2000


def test_worst_case_backoff_total_is_bounded() -> None:
    policy = RetryPolicy()
    worst = sum(2 * min(policy.initial_backoff_s * 2**n, policy.max_backoff_s) for n in range(8))
    assert worst < 160.0
