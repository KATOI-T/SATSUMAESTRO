"""無音タイムアウトの 120 秒契約と、その間の状態行。"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from satsumaestro.cli.render import RichRenderer
from satsumaestro.config.models import HookConfig, RetryConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ApiRequest, Provider
from satsumaestro.providers.factory import _policy
from satsumaestro.providers.fallback import FallbackProvider
from satsumaestro.providers.retry import RetryingProvider, RetryPolicy
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ProviderError, RetriesExhaustedError
from tests.fixtures.mock_provider import MockProvider, ScriptedFailureProvider

_REQUEST = httpx.Request("POST", "http://localhost:11434")


def _wrapped(cause: Exception) -> ProviderError:
    """プロバイダ境界が作るのと同じ形（``retryable=True`` + ``__cause__``）を再現する。"""
    error = ProviderError(f"OpenAI 互換 API 呼び出しに失敗: {cause}", retryable=True)
    error.__cause__ = cause
    return error


def _silence() -> ProviderError:
    """モデルが無音のまま read タイムアウトに達した（実測の形）。"""
    return _wrapped(httpx.ReadTimeout("timed out", request=_REQUEST))


def _unreachable() -> ProviderError:
    """エンドポイントに繋がらない（別問題。これまでどおり再試行してよい）。"""
    return _wrapped(httpx.ConnectTimeout("connect timed out", request=_REQUEST))


def _http_500() -> ProviderError:
    error = ProviderError("boom", retryable=True)
    error.status_code = 500  # type: ignore[attr-defined]
    return error


def _done() -> list[AssistantEvent]:
    return [TextDelta(text="ok"), MessageStop()]


def _fast(**kwargs: Any) -> RetryPolicy:
    return RetryPolicy(initial_backoff_s=0.001, max_backoff_s=0.001, **kwargs)


async def _drain(provider: Provider) -> list[AssistantEvent]:
    return [ev async for ev in provider.stream(ApiRequest(model="m", system="s", messages=[]))]


async def test_silent_timeout_is_not_retried() -> None:
    inner = ScriptedFailureProvider([_silence(), _done()])
    with pytest.raises(ProviderError) as caught:
        await _drain(RetryingProvider(inner, _fast()))

    assert inner.attempts == 1
    assert not isinstance(caught.value, RetriesExhaustedError)
    assert isinstance(caught.value.__cause__, httpx.ReadTimeout)
    assert caught.value.retryable is True
    assert "リトライ上限" not in str(caught.value)


async def test_connect_timeout_and_server_error_are_still_retried() -> None:
    for failure in (_unreachable(), _http_500()):
        inner = ScriptedFailureProvider([failure, _done()])
        assert await _drain(RetryingProvider(inner, _fast())) == _done()
        assert inner.attempts == 2, failure


async def test_the_cap_is_configurable() -> None:
    inner = ScriptedFailureProvider([_silence(), _done()])
    assert await _drain(RetryingProvider(inner, _fast(read_timeout_max_retries=1))) == _done()
    assert inner.attempts == 2

    twice = ScriptedFailureProvider([_silence(), _silence(), _done()])
    with pytest.raises(ProviderError) as caught:
        await _drain(RetryingProvider(twice, _fast(read_timeout_max_retries=1)))
    assert twice.attempts == 2
    assert isinstance(caught.value.__cause__, httpx.ReadTimeout)


def test_the_default_comes_from_config() -> None:
    assert RetryConfig().read_timeout_max_retries == 0
    assert _policy(RuntimeConfig()).read_timeout_max_retries == 0
    cfg = RuntimeConfig()
    cfg.provider.retry = RetryConfig(read_timeout_max_retries=2)
    assert _policy(cfg).read_timeout_max_retries == 2


async def test_silent_timeout_does_not_switch_models() -> None:
    """生成が大きかっただけでモデルが変わってはいけない（この契約の肝）。"""
    primary = ScriptedFailureProvider([_silence()])
    fallback = MockProvider(_done())
    switching = FallbackProvider(
        RetryingProvider(primary, _fast()),
        build_fallback=lambda: fallback,
        fallback_model="other-model",
    )
    with pytest.raises(ProviderError):
        await _drain(switching)
    assert primary.attempts == 1
    assert fallback.requests == []


def _runtime(steps: list[Any], tmp_path: Path) -> ConversationRuntime:
    """プロバイダ層の再試行を挟んだ状態で、会話層の復帰が従来どおり働くかを見る。"""
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=RetryingProvider(ScriptedFailureProvider(steps), _fast()),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.READ_ONLY),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


async def test_conversation_layer_still_recovers_once(tmp_path: Path) -> None:
    rt = _runtime([_silence(), _done()], tmp_path)
    summary = await rt.run_turn("hi")
    assert summary.timeout_recoveries == 1
    assert summary.final_text == "ok"


async def test_second_silence_in_the_turn_still_fails(tmp_path: Path) -> None:
    rt = _runtime([_silence(), _silence(), _done()], tmp_path)
    with pytest.raises(ProviderError):
        await rt.run_turn("hi")


def _aborted(timeout_recovery_s: float | None) -> tuple[RichRenderer, str]:
    """中断を通知した直後のレンダラと、スクロールバックへ出た本文を返す。"""
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=True, width=80, color_system=None))
    r.start_turn()
    r.on_stream_aborted(timeout_recovery_s)
    return r, buf.getvalue()


def _status(r: RichRenderer) -> str:
    """いま Live に出ている状態行（``_live_renderable`` は Live の更新内容そのもの）。"""
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=100).print(r._live_renderable())
    return out.getvalue()


def test_status_line_comes_back_after_a_recovery_abort() -> None:
    r, printed = _aborted(120.0)
    assert "120 秒応答がなかったので中断し、分けて出し直すよう依頼しました" in printed
    status = _status(r)
    assert "出し直しを待っています…" in status
    assert "考え中…" not in status


def test_elapsed_restarts_from_the_abort() -> None:
    r, _ = _aborted(120.0)
    r._phase_since -= 45.0
    assert "45s" in _status(r)


def test_status_line_stays_down_when_the_turn_will_fail() -> None:
    r, printed = _aborted(None)
    assert "応答が途中で切れました" in printed
    assert "出し直しを待っています…" not in _status(r)
    assert r._live is None


def test_recovering_state_does_not_leak_into_the_next_turn() -> None:
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=True, width=80, color_system=None))
    r.start_turn()
    r.on_stream_aborted(120.0)
    r.end_turn()
    before = time.monotonic()
    r.start_turn()
    assert r._state == "waiting" and r._phase_since >= before
