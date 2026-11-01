"""再試行・モデルの切り替えを、ユーザー向けの 1 行で画面へ届ける。"""

from __future__ import annotations

import io

from rich.console import Console

from satsumaestro.cli.errors_ui import broken_output_hint, turn_failure
from satsumaestro.cli.render import RichRenderer
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.providers import notices
from satsumaestro.providers.fallback import FallbackProvider
from satsumaestro.providers.notices import ProviderNotice
from satsumaestro.providers.retry import RetryingProvider, RetryPolicy
from satsumaestro.util.errors import ProviderError, RetriesExhaustedError
from tests.fixtures.mock_provider import MockProvider, ScriptedFailureProvider
from tests.test_p15_model_robustness import _drain, _http_error, _wrapped

_FAST = RetryPolicy(initial_backoff_s=0.0, max_backoff_s=0.0)


def _broken() -> ProviderError:
    return ProviderError("error parsing tool call: ストリームが…閉じた", retryable=True)


async def test_retry_is_announced_once_to_the_bound_sink() -> None:
    seen: list[ProviderNotice] = []
    provider = RetryingProvider(
        ScriptedFailureProvider([_broken(), [TextDelta(text="ok"), MessageStop()]]), _FAST
    )
    with notices.bind(seen.append):
        out = await _drain(provider)
    assert [n.kind for n in seen] == ["retry"]
    assert seen[0].reason == "tool_parse" and (seen[0].attempt, seen[0].limit) == (1, 2)
    assert out == [TextDelta(text="ok"), MessageStop()]


async def test_without_a_sink_nothing_happens() -> None:
    provider = RetryingProvider(
        ScriptedFailureProvider([_broken(), [TextDelta(text="ok"), MessageStop()]]), _FAST
    )
    assert await _drain(provider) == [TextDelta(text="ok"), MessageStop()]


async def test_fallback_is_announced_and_still_switches() -> None:
    seen: list[ProviderNotice] = []
    primary = RetryingProvider(ScriptedFailureProvider([_wrapped(_http_error(503))]), _FAST)
    secondary = MockProvider([TextDelta(text="from-fallback"), MessageStop()])
    provider = FallbackProvider(
        primary, build_fallback=lambda: secondary, fallback_model="claude-sonnet-5"
    )
    with notices.bind(seen.append):
        out = await _drain(provider)
    assert TextDelta(text="from-fallback") in out
    assert seen[-1].kind == "fallback" and seen[-1].to_model == "claude-sonnet-5"


def test_the_screen_gets_a_user_facing_line() -> None:
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=True, width=100, color_system=None))
    r.start_turn()
    r.on_provider_notice(ProviderNotice(kind="retry", attempt=1, limit=2, reason="tool_parse"))
    r.on_provider_notice(ProviderNotice(kind="retry", attempt=2, limit=9, delay_s=3.2))
    r.on_provider_notice(ProviderNotice(kind="fallback", from_model="a", to_model="b"))
    r.end_turn()
    out = buf.getvalue()
    assert "— モデルの出力が途中で壊れたので、もう一度生成します（1/2）" in out
    assert "— モデルの応答に失敗したので、3 秒後にもう一度試します（2/9）" in out
    assert "— 'a' が応答しないので 'b' に切り替えます" in out
    for developer_words in ("error parsing tool call", "finish_reason", "上限を縮小"):
        assert developer_words not in out


def test_exhausted_tool_call_failures_are_reworded() -> None:
    exc = RetriesExhaustedError(2, 25.4, last_error="error parsing tool call: ストリームが…")
    assert turn_failure(exc).reason == "モデルの出力（ツール呼び出しの引数）が繰り返し壊れました"
    assert turn_failure(exc).next_step == broken_output_hint()
    other = turn_failure(RetriesExhaustedError(9, 60.0, last_error="503"))
    assert other.reason.startswith("モデルとの通信に失敗しました（")
    assert other.next_step != broken_output_hint()
    assert turn_failure(ProviderError("接続できません")).reason == (
        "モデルとの通信に失敗しました（接続できません）"
    )
