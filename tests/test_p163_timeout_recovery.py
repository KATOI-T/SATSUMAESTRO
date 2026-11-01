"""読み取りタイムアウトからの復帰のテスト。"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from rich.console import Console

from satsumaestro.cli.render import RichRenderer
from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta
from satsumaestro.engine.reminders import timeout_note
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import NullRenderer
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.retry import describe
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import ScriptedFailureProvider


def _timeout(message: str = "timed out") -> ProviderError:
    """プロバイダ境界が作るのと同じ形（``__cause__`` に httpx 例外）を再現する。"""
    error = ProviderError(f"OpenAI 互換 API 呼び出しに失敗: {message}", retryable=True)
    error.__cause__ = httpx.ReadTimeout(
        message, request=httpx.Request("POST", "http://localhost:11434")
    )
    return error


def _runtime(
    steps: list[Any], tmp_path: Path, cfg: RuntimeConfig | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg or RuntimeConfig(),
    )


def _done() -> list[AssistantEvent]:
    return [TextDelta(text="分けて書きました"), MessageStop()]


def _injected(rt: ConversationRuntime) -> list[str]:
    """合成 user メッセージとして積まれた system-reminder の本文。"""
    return [
        block.text
        for message in rt.session.messages
        if message.role == "user"
        for block in message.blocks
        if isinstance(block, TextBlock) and "<system-reminder>" in block.text
    ]


async def test_read_timeout_recovers_once_within_the_turn(tmp_path: Path) -> None:
    rt = _runtime([_timeout(), _done()], tmp_path)
    summary = await rt.run_turn("300 行の HTML を作って")

    assert summary.final_text == "分けて書きました"
    assert summary.iterations == 2
    assert summary.timeout_recoveries == 1
    assert timeout_note(120.0) in _injected(rt)[0]
    assert "120 seconds" in _injected(rt)[0]


async def test_second_timeout_in_the_same_turn_is_raised(tmp_path: Path) -> None:
    rt = _runtime([_timeout(), _timeout("again"), _done()], tmp_path)
    with pytest.raises(ProviderError, match="again"):
        await rt.run_turn("300 行の HTML を作って")
    assert len(_injected(rt)) == 1


async def test_recovery_is_per_turn_not_per_session(tmp_path: Path) -> None:
    rt = _runtime([_timeout(), _done(), _timeout("2nd turn"), _done()], tmp_path)
    first = await rt.run_turn("1 つめ")
    second = await rt.run_turn("2 つめ")
    assert (first.timeout_recoveries, second.timeout_recoveries) == (1, 1)


async def test_non_timeout_provider_error_is_not_recovered(tmp_path: Path) -> None:
    failure = ProviderError("エンドポイントに接続できない", retryable=True)
    rt = _runtime([failure, _done()], tmp_path)
    with pytest.raises(ProviderError, match="接続できない"):
        await rt.run_turn("hi")
    assert _injected(rt) == []


async def test_recovery_can_be_disabled(tmp_path: Path) -> None:
    cfg = RuntimeConfig(guidance=GuidanceConfig(timeout_recovery=False))
    rt = _runtime([_timeout(), _done()], tmp_path, cfg)
    with pytest.raises(ProviderError):
        await rt.run_turn("hi")


async def test_recovery_does_not_fire_on_the_last_iteration(tmp_path: Path) -> None:
    cfg = RuntimeConfig(max_iterations=1)
    rt = _runtime([_timeout(), _done()], tmp_path, cfg)
    with pytest.raises(ProviderError):
        await rt.run_turn("hi")
    assert _injected(rt) == []


async def test_history_stays_well_formed_after_recovery(tmp_path: Path) -> None:
    rt = _runtime([_timeout(), _done()], tmp_path)
    await rt.run_turn("hi")
    roles = [m.role for m in rt.session.messages]
    assert roles == ["user", "user", "assistant"]
    store = JSONLSessionStore()
    restored = store.load(store.path_for(tmp_path, rt.session.session_id))
    assert [m.role for m in restored.messages] == roles


async def test_renderer_is_told_the_stream_was_aborted(tmp_path: Path) -> None:
    """``MessageStop`` が来ないので、描画側に打ち切りを伝える契機が必要。"""

    class _Recording(NullRenderer):
        seen: ClassVar[list[float | None]] = []

        def on_stream_aborted(self, timeout_recovery_s: float | None = None) -> None:
            type(self).seen.append(timeout_recovery_s)

    rt = _runtime([_timeout(), _done()], tmp_path)
    await rt.run_turn("hi", render=_Recording())
    assert _Recording.seen == [rt.config.read_timeout_s()]


async def test_renderer_is_not_told_recovery_when_turn_fails(tmp_path: Path) -> None:
    """復帰しないときに秒数を渡すと「依頼しました」が嘘になる。"""

    class _Recording(NullRenderer):
        seen: ClassVar[list[float | None]] = []

        def on_stream_aborted(self, timeout_recovery_s: float | None = None) -> None:
            type(self).seen.append(timeout_recovery_s)

    rt = _runtime([_timeout(), _timeout()], tmp_path)
    with pytest.raises(ProviderError):
        await rt.run_turn("hi", render=_Recording())
    assert _Recording.seen == [rt.config.read_timeout_s(), None]


def _rendered(timeout_recovery_s: float | None) -> str:
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=True, width=80, color_system=None))
    r.start_turn()
    r.on_stream_aborted(timeout_recovery_s)
    r.end_turn()
    return buf.getvalue()


def test_recovery_notice_is_one_user_facing_line() -> None:
    out = _rendered(120.0)
    assert "120 秒応答がなかったので中断し、分けて出し直すよう依頼しました" in out
    assert "read タイムアウト" not in out and "応答が途中で切れました" not in out


def test_abort_without_recovery_keeps_the_plain_notice() -> None:
    out = _rendered(None)
    assert "応答が途中で切れました" in out and "依頼しました" not in out


def test_provider_error_text_does_not_end_with_a_bare_colon() -> None:
    assert describe(httpx.ReadTimeout("")) == "ReadTimeout"
    assert describe(httpx.ReadTimeout("timed out")) == "timed out"
