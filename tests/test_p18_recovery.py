"""失敗の手当て（recovery）のテスト。"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field

from satsumaestro.config.models import HookConfig, RecoveryConfig, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.recovery import (
    Fallback,
    Fault,
    Playbook,
    PlaybookOutcome,
    PlaybookStep,
    RecoveryLedger,
    Remedy,
    SessionAbortError,
    fault_of,
    playbook_for,
    run_playbook,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.trace import RecoveryAttempt, TurnAborted, TurnEvent, guarded
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ConversationMessage, TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.registry import ToolRegistry
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.errors import ProviderError, SessionError, ToolError
from tests.fixtures.mock_provider import MockProvider, SequencedMockProvider


@dataclass
class _ScriptedRemedies:
    """何手目から通らなくなるかを台本で決める :class:`RemedyRunner`。"""

    fail_from: int | None = None
    raise_from: int | None = None
    yield_control: bool = False
    applied: list[tuple[Remedy, Fault]] = field(default_factory=list)

    async def apply(self, remedy: Remedy, fault: Fault) -> bool:
        index = len(self.applied)
        self.applied.append((remedy, fault))
        if self.yield_control:
            await asyncio.sleep(0)
        if self.raise_from is not None and index >= self.raise_from:
            raise RuntimeError("remedy blew up")
        return self.fail_from is None or index < self.fail_from


class _SeenRecovery:
    """手当ての行とターンの失敗だけを溜める受け手。"""

    def __init__(self) -> None:
        self.recovery: list[PlaybookStep] = []
        self.failed: list[Exception] = []

    def observe(self, event: TurnEvent) -> None:
        if isinstance(event, RecoveryAttempt):
            self.recovery.append(event.step)
        elif isinstance(event, TurnAborted):
            self.failed.append(event.error)


async def test_all_remedies_pass_and_the_ledger_shows_each() -> None:
    ledger = RecoveryLedger(attempt_limit=1)
    runner = _ScriptedRemedies()
    outcome = await run_playbook(Fault.PROVIDER, ledger, runner=runner)
    assert outcome is PlaybookOutcome.SUCCEEDED
    steps = ledger.steps()
    assert [s.outcome for s in steps] == [PlaybookOutcome.SUCCEEDED] * 2
    assert [s.remedy for s in steps] == [Remedy.RETRY_PROVIDER, Remedy.SWITCH_MODEL]
    assert [s.attempt for s in steps] == [1, 1]
    assert ledger.tries(Fault.PROVIDER) == 1
    assert ledger.is_spent(Fault.PROVIDER) is True
    assert ledger.is_spent(Fault.UNHEALTHY_SESSION) is False


async def test_second_use_of_the_same_playbook_is_handed_off() -> None:
    ledger = RecoveryLedger(attempt_limit=1)
    runner = _ScriptedRemedies()
    assert await run_playbook(Fault.PROVIDER, ledger, runner=runner) is PlaybookOutcome.SUCCEEDED
    outcome = await run_playbook(Fault.PROVIDER, ledger, runner=runner)
    assert outcome is PlaybookOutcome.HANDED_OFF
    last = ledger.steps()[-1]
    assert last.remedy is None
    assert last.outcome is PlaybookOutcome.HANDED_OFF
    assert last.attempt == 1
    assert "limit" in last.note
    assert len(runner.applied) == 2


async def test_failure_at_the_first_remedy_is_handed_off() -> None:
    ledger = RecoveryLedger(attempt_limit=1)
    outcome = await run_playbook(Fault.PROVIDER, ledger, runner=_ScriptedRemedies(fail_from=0))
    assert outcome is PlaybookOutcome.HANDED_OFF
    assert ledger.steps() == [
        PlaybookStep(Fault.PROVIDER, Remedy.RETRY_PROVIDER, PlaybookOutcome.HANDED_OFF, 1)
    ]
    assert ledger.is_spent(Fault.PROVIDER) is True


async def test_failure_after_a_passing_remedy_is_incomplete() -> None:
    ledger = RecoveryLedger(attempt_limit=1)
    outcome = await run_playbook(Fault.PROVIDER, ledger, runner=_ScriptedRemedies(fail_from=1))
    assert outcome is PlaybookOutcome.INCOMPLETE
    steps = ledger.steps()
    assert steps[0].outcome is PlaybookOutcome.SUCCEEDED
    assert steps[1].outcome is PlaybookOutcome.INCOMPLETE
    assert steps[1].remedy is Remedy.SWITCH_MODEL


async def test_a_raising_runner_counts_as_a_failed_remedy() -> None:
    ledger = RecoveryLedger(attempt_limit=1)
    outcome = await run_playbook(Fault.PROVIDER, ledger, runner=_ScriptedRemedies(raise_from=1))
    assert outcome is PlaybookOutcome.INCOMPLETE
    assert ledger.steps()[1].note == "RuntimeError: remedy blew up"


async def test_attempt_limit_zero_never_applies_a_remedy() -> None:
    ledger = RecoveryLedger(attempt_limit=0)
    runner = _ScriptedRemedies()
    outcome = await run_playbook(Fault.PROVIDER, ledger, runner=runner)
    assert outcome is PlaybookOutcome.HANDED_OFF
    assert runner.applied == []
    assert ledger.tries(Fault.PROVIDER) == 0


def test_steps_is_a_copy() -> None:
    ledger = RecoveryLedger()
    ledger.steps().append(PlaybookStep(Fault.PROVIDER, None, PlaybookOutcome.HANDED_OFF, 0))
    assert ledger.steps() == []


def test_fault_of_known_exceptions() -> None:
    assert fault_of(ProviderError("boom")) is Fault.PROVIDER
    assert fault_of(SessionError("broken")) is Fault.UNHEALTHY_SESSION
    assert fault_of(SessionAbortError("aborted")) is Fault.UNHEALTHY_SESSION
    assert fault_of(ToolError("bad input")) is None
    assert fault_of(ValueError("nope")) is None


def test_every_fault_has_a_playbook_with_one_attempt() -> None:
    for fault in Fault:
        book = playbook_for(fault)
        assert book.attempt_limit == 1
        assert book.remedies
    assert playbook_for(Fault.MCP_INIT).fallback is Fallback.ABORT_TURN
    assert playbook_for(Fault.UNHEALTHY_SESSION) == Playbook(
        (Remedy.ASK_FOR_NEW_SESSION,), Fallback.ABORT_TURN
    )
    assert playbook_for(Fault.BRANCH_BEHIND).remedies == (Remedy.REBASE,)
    assert playbook_for(Fault.PROVIDER).fallback is Fallback.RETURN_ERROR


def _text_msg(role: str, text: str) -> ConversationMessage:
    return ConversationMessage(role=role, blocks=[TextBlock(text=text)])  # type: ignore[arg-type]


def _long_history(n: int) -> list[ConversationMessage]:
    return [
        _text_msg("user" if i % 2 == 0 else "assistant", f"message {i} " + "x" * 300)
        for i in range(n)
    ]


def _compacting_runtime(
    tmp_path: Path,
    *,
    tools: ToolRegistry | None = None,
    config: RuntimeConfig | None = None,
) -> ConversationRuntime:
    """1 往復目で必ず圧縮→プローブが走る runtime（圧縮テストの型を流用）。"""
    provider = MockProvider([TextDelta(text="ok"), Usage(input_tokens=100), MessageStop()])
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=tools if tools is not None else default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config or RuntimeConfig(compact_token_threshold=1),
        compactor=Compactor(keep_last=2),
    )
    rt.session.messages.extend(_long_history(8))
    return rt


async def test_tool_check_passes_after_compaction(tmp_path: Path) -> None:
    rt = _compacting_runtime(tmp_path)
    summary = await rt.run_turn("go")
    assert summary.compacted is True
    assert summary.health_warning is None


async def test_tool_check_failure_aborts_the_turn(tmp_path: Path) -> None:
    rt = _compacting_runtime(tmp_path, tools=ToolRegistry())
    seen = _SeenRecovery()
    with pytest.raises(SessionAbortError) as exc_info:
        await rt.run_turn("go", observer=seen)
    assert "新しいセッション" in str(exc_info.value)
    assert [s.fault for s in seen.recovery] == [Fault.UNHEALTHY_SESSION]
    assert seen.recovery[0].remedy is Remedy.ASK_FOR_NEW_SESSION
    assert seen.recovery[0].outcome is PlaybookOutcome.HANDED_OFF
    assert seen.failed and isinstance(seen.failed[0], SessionAbortError)


async def test_recovery_disabled_falls_back(tmp_path: Path) -> None:
    config = RuntimeConfig(compact_token_threshold=1, recovery=RecoveryConfig(enabled=False))
    rt = _compacting_runtime(tmp_path, tools=ToolRegistry(), config=config)
    summary = await rt.run_turn("go")
    assert summary.health_warning is not None
    assert "no tool is registered" in summary.health_warning


def _failure_hook_cmd(tmp_path: Path, name: str, *, exit_code: int = 0) -> tuple[str, Path]:
    """発火をマーカーファイルで記録し、payload の形を検査するフックスクリプト。"""
    marker = tmp_path / f"{name}.fired"
    body = (
        "import json, sys, pathlib\n"
        "payload = json.load(sys.stdin)\n"
        f"pathlib.Path({str(marker)!r}).write_text('fired', encoding='utf-8')\n"
        "assert payload['event'] == 'post_tool_use_failure', payload\n"
        "assert payload['is_error'] is True, payload\n"
        "print(json.dumps({'output': 'rewritten by failure hook'}))\n"
    )
    if exit_code:
        body += f"sys.exit({exit_code})\n"
    path = tmp_path / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return f"{sys.executable} {path}", marker


async def test_post_tool_use_failure_fires_only_on_error(tmp_path: Path) -> None:
    cmd, marker = _failure_hook_cmd(tmp_path, "onfail")
    runner = HookRunner(HookConfig(post_tool_use_failure=[cmd]), tmp_path)
    call = ToolUse(id="1", name="bash", input={"command": "true"})

    ok = await runner.post_tool_use_failure(call, ToolResult(output="done"))
    assert ok.output == "done" and ok.is_error is False
    assert not marker.exists()

    ng = await runner.post_tool_use_failure(call, ToolResult(output="boom", is_error=True))
    assert marker.exists()
    assert ng.output == "rewritten by failure hook"
    assert ng.is_error is True


async def test_post_tool_use_failure_fails_open(tmp_path: Path) -> None:
    cmd, marker = _failure_hook_cmd(tmp_path, "boomhook", exit_code=3)
    runner = HookRunner(HookConfig(post_tool_use_failure=[cmd]), tmp_path)
    call = ToolUse(id="1", name="bash", input={"command": "true"})
    res = await runner.post_tool_use_failure(call, ToolResult(output="boom", is_error=True))
    assert marker.exists()
    assert res.output == "boom" and res.is_error is True


def _make_flaky_tool(failures: int) -> type[Tool]:
    """最初の ``failures`` 回だけ ProviderError を投げるスタブツールを作る。"""

    class FlakyEcho:
        calls: ClassVar[int] = 0

        class Input(BaseModel):
            text: str = Field(description="Text to echo back.")

        spec: ClassVar[ToolSpec] = ToolSpec(
            name="flaky_echo",
            description="Echo the given text. Fails transiently for testing recovery.",
            input_schema=model_tool_schema(Input),
            required_mode=PermissionMode.READ_ONLY,
            parallel_safe=True,
        )

        async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
            type(self).calls += 1
            if type(self).calls <= failures:
                raise ProviderError("transient upstream failure")
            return ToolResult(output=f"echo:{inp.text}")  # type: ignore[attr-defined]

    return FlakyEcho


def _make_session_error_tool() -> type[Tool]:
    """常に SessionError を投げるスタブツール（UNHEALTHY_SESSION → ABORT_TURN 経路用）。"""

    class BrokenEcho:
        class Input(BaseModel):
            text: str = Field(description="Text to echo back.")

        spec: ClassVar[ToolSpec] = ToolSpec(
            name="flaky_echo",
            description="Echo the given text. Always fails with a session error.",
            input_schema=model_tool_schema(Input),
            required_mode=PermissionMode.READ_ONLY,
            parallel_safe=True,
        )

        async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
            raise SessionError("session state corrupted")

    return BrokenEcho


def _runtime_with_tools(
    tmp_path: Path, tool: type[Tool], *, recovery_runner: object | None = None
) -> tuple[ConversationRuntime, SequencedMockProvider]:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="t1", name="flaky_echo", input={"text": "hi"}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    registry = default_registry()
    registry.register(tool)
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=registry,
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
        recovery_runner=recovery_runner,  # type: ignore[arg-type]
    )
    return rt, provider


async def test_recovered_tool_failure_reexecutes(tmp_path: Path) -> None:
    rt, _ = _runtime_with_tools(
        tmp_path, _make_flaky_tool(failures=1), recovery_runner=_ScriptedRemedies()
    )
    seen = _SeenRecovery()
    summary = await rt.run_turn("go", observer=seen)
    assert summary.final_text == "done"
    tool_msg = rt.session.messages[2]
    assert tool_msg.blocks[0].is_error is False  # type: ignore[union-attr]
    assert tool_msg.blocks[0].output == "echo:hi"  # type: ignore[union-attr]
    assert [s.fault for s in seen.recovery] == [Fault.PROVIDER, Fault.PROVIDER]
    assert all(s.outcome is PlaybookOutcome.SUCCEEDED for s in seen.recovery)


async def test_concurrent_recovery_is_reported_once_per_ledger_row(tmp_path: Path) -> None:
    rt, _ = _runtime_with_tools(
        tmp_path,
        _make_flaky_tool(failures=0),
        recovery_runner=_ScriptedRemedies(yield_control=True),
    )
    ledger = RecoveryLedger(attempt_limit=1)
    seen = _SeenRecovery()
    notify = guarded(seen)
    first, second = await asyncio.gather(
        rt._run_recovery(Fault.PROVIDER, ledger, notify),
        rt._run_recovery(Fault.PROVIDER, ledger, notify),
    )
    assert first is PlaybookOutcome.SUCCEEDED
    assert second is PlaybookOutcome.HANDED_OFF
    assert ledger.tries(Fault.PROVIDER) == 1
    assert len(ledger.steps()) == 3
    assert len(seen.recovery) == 3


async def test_tool_abort_escalation_reports_the_aborted_turn(tmp_path: Path) -> None:
    rt, _ = _runtime_with_tools(tmp_path, _make_session_error_tool())
    seen = _SeenRecovery()
    with pytest.raises(SessionAbortError):
        await rt.run_turn("go", observer=seen)
    assert seen.recovery[0].fault is Fault.UNHEALTHY_SESSION
    assert seen.failed and isinstance(seen.failed[0], SessionAbortError)


async def test_return_error_keeps_is_error(tmp_path: Path) -> None:
    rt, _ = _runtime_with_tools(tmp_path, _make_flaky_tool(failures=99))
    seen = _SeenRecovery()
    summary = await rt.run_turn("go", observer=seen)
    assert summary.final_text == "done"
    tool_msg = rt.session.messages[2]
    assert tool_msg.blocks[0].is_error is True  # type: ignore[union-attr]
    assert "ProviderError" in tool_msg.blocks[0].output  # type: ignore[union-attr]
    assert seen.recovery[0].outcome is PlaybookOutcome.HANDED_OFF
    assert seen.recovery[0].remedy is Remedy.RETRY_PROVIDER


async def test_recovery_disabled_tool_failure_unchanged(tmp_path: Path) -> None:
    rt, _ = _runtime_with_tools(
        tmp_path, _make_flaky_tool(failures=99), recovery_runner=_ScriptedRemedies()
    )
    rt.config = RuntimeConfig(recovery=RecoveryConfig(enabled=False))
    seen = _SeenRecovery()
    summary = await rt.run_turn("go", observer=seen)
    assert summary.final_text == "done"
    assert rt.session.messages[2].blocks[0].is_error is True  # type: ignore[union-attr]
    assert seen.recovery == []


def test_recovery_config_defaults() -> None:
    config = RuntimeConfig()
    assert config.recovery.enabled is True
    assert config.recovery.max_attempts == 1
    assert config.hooks.post_tool_use_failure == []
    with pytest.raises(ValueError):
        RecoveryConfig(max_attempts=-1)
