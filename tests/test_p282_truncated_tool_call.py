"""思考だけで閉じたストリームを「本文の空」でなく tool_call パース失敗として扱う。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.openai_provider import _assemble_openai
from satsumaestro.providers.retry import RetryingProvider, RetryPolicy
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import ScriptedFailureProvider


async def _truncated_error() -> ProviderError:
    """実測と同じ形のストリーム（思考 → content "" → 何も無く EOF）を組み立てて失敗を得る。"""

    async def chunks() -> AsyncIterator[NS]:
        for item in (
            NS(choices=[NS(delta=NS(reasoning="Use write_file."), finish_reason=None)], usage=None),
            NS(choices=[NS(delta=NS(content=""), finish_reason=None)], usage=None),
        ):
            yield item

    with pytest.raises(ProviderError) as info:
        async for _ in _assemble_openai(chunks()):
            pass
    return info.value


async def test_truncated_stream_is_resent_instead_of_an_empty_reply(tmp_path: Path) -> None:
    inner = ScriptedFailureProvider(
        [
            await _truncated_error(),
            [
                ToolUse(
                    id="w1",
                    name="write_file",
                    input={"path": "dashboards/index.html", "content": "<html></html>\n"},
                ),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="作成しました。"), MessageStop()],
        ]
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=RetryingProvider(inner, RetryPolicy(initial_backoff_s=0.0, max_backoff_s=0.0)),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    summary = await rt.run_turn("ダッシュボードを作って")

    assert inner.attempts == 3
    assert (tmp_path / "dashboards" / "index.html").exists()
    assert summary.final_text == "作成しました。"
    reminders = [
        b.text
        for m in rt.session.messages
        if m.role == "user"
        for b in m.blocks
        if isinstance(b, TextBlock) and "<system-reminder>" in b.text
    ]
    assert reminders == []
    assert all(
        any(b.type in ("text", "tool_use") for b in m.blocks)
        for m in rt.session.messages
        if m.role == "assistant"
    )
