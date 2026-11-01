"""圧縮の直後の確かめは、登録簿にある読み取りのツールで行う。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.events import MessageStop, TextDelta, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.tool_check import check_tool_executor
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ConversationMessage, TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.registry import ToolRegistry
from tests.fixtures.mock_provider import MockProvider


def _runtime(
    tmp_path: Path, tools: ToolRegistry, *, compacting: bool = False
) -> ConversationRuntime:
    provider = MockProvider([TextDelta(text="ok"), Usage(input_tokens=100), MessageStop()])
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=tools,
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(compact_token_threshold=1 if compacting else 0),
        compactor=Compactor(keep_last=2),
    )
    rt.session.messages.extend(
        ConversationMessage(
            role="user" if i % 2 == 0 else "assistant",
            blocks=[TextBlock(text=f"message {i} " + "x" * 300)],
        )
        for i in range(8)
    )
    return rt


async def test_a_narrowed_registry_is_not_taken_for_a_broken_one(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, default_registry().filtered({"read_file"}))
    assert await check_tool_executor(rt) is None


async def test_list_directory_is_used_when_glob_search_is_left_out(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    rt = _runtime(tmp_path, default_registry().filtered({"read_file", "list_directory"}))
    assert await check_tool_executor(rt) is None


async def test_an_empty_registry_is_still_reported(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, ToolRegistry())
    reason = await check_tool_executor(rt)
    assert reason is not None and "no tool is registered" in reason


async def test_compaction_with_allowed_tools_does_not_abort_the_turn(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, default_registry().filtered({"read_file"}), compacting=True)
    summary = await rt.run_turn("go")
    assert summary.compacted is True
    assert summary.health_warning is None
