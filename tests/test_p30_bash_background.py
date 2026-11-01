"""bash バックグラウンド実行のテスト。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import PathEscapeError
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.bash import Bash
from satsumaestro.tools.builtins.bash_output import BashOutput
from satsumaestro.util.proc import BackgroundProcessRegistry
from tests.fixtures.mock_provider import SequencedMockProvider


@pytest.fixture
def registry(tmp_path: Path) -> BackgroundProcessRegistry:
    return BackgroundProcessRegistry(tmp_path / ".satsuma" / "background" / "sess")


def _bg_ctx(tool_ctx: ToolContext, registry: BackgroundProcessRegistry) -> ToolContext:
    return tool_ctx.model_copy(update={"background": registry})


async def _bash_call(ctx: ToolContext, command: str, **kwargs: object):  # type: ignore[no-untyped-def]
    return await Bash().run(Bash.Input(command=command, **kwargs), ctx)  # type: ignore[arg-type]


async def _run_output(ctx: ToolContext, task_id: str, wait_s: float = 0.0):  # type: ignore[no-untyped-def]
    return await BashOutput().run(BashOutput.Input(task_id=task_id, wait_s=wait_s), ctx)


async def test_background_returns_task_id_immediately(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    res = await _bash_call(ctx, "sleep 30", run_in_background=True)
    assert not res.is_error
    assert "bg-1" in res.output and "bash_output" in res.output
    task = registry.get("bg-1")
    assert task is not None and task.running
    await registry.aclose()


async def test_bash_output_reads_output_and_exit_code(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    await _bash_call(ctx, "echo hello; exit 3", run_in_background=True)
    res = await _run_output(ctx, "bg-1", wait_s=10.0)
    assert not res.is_error
    assert "exited with code 3" in res.output and "hello" in res.output
    assert "echo hello" in res.output
    res2 = await _run_output(ctx, "bg-1")
    assert "(no new output)" in res2.output
    await registry.aclose()


async def test_bash_output_incremental_chunks(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    gate = tool_ctx.workspace_root / "gate"
    await _bash_call(
        ctx,
        f"echo first; while [ ! -e {gate} ]; do sleep 0.05; done; echo second",
        run_in_background=True,
    )
    task = registry.get("bg-1")
    assert task is not None
    for _ in range(100):
        if "first" in task.log_path.read_text():
            break
        await asyncio.sleep(0.05)
    res1 = await _run_output(ctx, "bg-1")
    assert "running" in res1.output and "first" in res1.output and "second" not in res1.output
    gate.touch()
    res2 = await _run_output(ctx, "bg-1", wait_s=10.0)
    assert "exited with code 0" in res2.output
    assert "second" in res2.output and "first" not in res2.output.splitlines()[1:]
    await registry.aclose()


async def test_command_preview_truncated_to_80_chars(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    long_cmd = "echo " + "x" * 200
    await _bash_call(ctx, long_cmd, run_in_background=True)
    task = registry.get("bg-1")
    assert task is not None
    assert task.command_preview == long_cmd[:80] + "…"
    await registry.aclose()


async def test_unknown_task_id_lists_known(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    await _bash_call(ctx, "echo hi", run_in_background=True)
    res = await _run_output(ctx, "bg-9")
    assert res.is_error and "bg-9" in res.output and "bg-1" in res.output
    await registry.aclose()


async def test_degrades_without_registry(tool_ctx: ToolContext) -> None:
    res = await _bash_call(tool_ctx, "echo hi", run_in_background=True)
    assert res.is_error and "unavailable" in res.output
    res2 = await _run_output(tool_ctx, "bg-1")
    assert res2.is_error and "no background tasks" in res2.output


async def test_validation_applies_to_background(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    with pytest.raises(PathEscapeError):
        await _bash_call(ctx, "echo hi > /etc/satsuma-test", run_in_background=True)
    assert registry.known_ids() == []
    await registry.aclose()


def test_run_in_background_description_guides_usage() -> None:
    desc = Bash.spec.input_schema["properties"]["run_in_background"]["description"]
    assert "long-running" in desc


def test_spec_modes_unchanged() -> None:
    assert Bash.spec.required_mode is PermissionMode.DANGER_FULL_ACCESS
    assert BashOutput.spec.required_mode is PermissionMode.READ_ONLY
    assert BashOutput.spec.parallel_safe
    assert len(Bash.spec.input_schema["properties"]) <= 8


async def test_aclose_terminates_and_cleans_logs(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    await _bash_call(ctx, "sleep 30", run_in_background=True)
    task = registry.get("bg-1")
    assert task is not None and task.running
    await registry.aclose()
    assert not task.running
    assert not task.log_path.exists()


async def test_take_completed_unnotified_is_once(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    await _bash_call(ctx, "echo done", run_in_background=True)
    task = registry.get("bg-1")
    assert task is not None and task.waiter is not None
    await task.waiter
    done = registry.take_completed_unnotified()
    assert [t.task_id for t in done] == ["bg-1"]
    assert registry.take_completed_unnotified() == []
    await registry.aclose()


async def test_bash_output_collection_suppresses_notice(
    tool_ctx: ToolContext, registry: BackgroundProcessRegistry
) -> None:
    ctx = _bg_ctx(tool_ctx, registry)
    await _bash_call(ctx, "echo done", run_in_background=True)
    await _run_output(ctx, "bg-1", wait_s=10.0)
    assert registry.take_completed_unnotified() == []
    await registry.aclose()


def _runtime(
    provider: object,
    tmp_path: Path,
    registry: BackgroundProcessRegistry,
    cfg: RuntimeConfig | None = None,
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg or RuntimeConfig(),
        background=registry,
    )


async def _finished_registry(tmp_path: Path) -> BackgroundProcessRegistry:
    registry = BackgroundProcessRegistry(tmp_path / ".satsuma" / "background" / "sess")
    task = await registry.spawn("echo done", cwd=tmp_path, env={})
    assert task.waiter is not None
    await task.waiter
    return registry


async def test_runtime_notifies_completion_once(tmp_path: Path) -> None:
    registry = await _finished_registry(tmp_path)
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="c1", name="glob_search", input={"pattern": "*"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [
                ToolUse(id="c2", name="glob_search", input={"pattern": "*"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, registry)
    summary = await rt.run_turn("hi")
    assert summary.final_text == "done"
    tool_msgs = [m for m in rt.session.messages if m.role == "tool"]
    first = tool_msgs[0].blocks[-1]
    assert isinstance(first, ToolResultBlock)
    assert "<system-reminder>" in first.output and "bg-1" in first.output
    assert "exited with code 0" in first.output and "echo done" in first.output
    second = tool_msgs[1].blocks[-1]
    assert isinstance(second, ToolResultBlock)
    assert "bg-1" not in second.output
    assert summary.reminders == 1
    await registry.aclose()


async def test_runtime_notifies_even_without_guidance_reminders(tmp_path: Path) -> None:
    registry = await _finished_registry(tmp_path)
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="c1", name="glob_search", input={"pattern": "*"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    cfg = RuntimeConfig(guidance=GuidanceConfig(reminders=False))
    rt = _runtime(provider, tmp_path, registry, cfg)
    summary = await rt.run_turn("hi")
    tool_msgs = [m for m in rt.session.messages if m.role == "tool"]
    last = tool_msgs[0].blocks[-1]
    assert isinstance(last, ToolResultBlock) and "bg-1" in last.output
    assert summary.reminders == 0
    await registry.aclose()
