"""WorkerSupervisor と worker ツール層のテスト。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from satsumaestro.cli.app import _orchestration_flags
from satsumaestro.config.models import (
    OrchestrationConfig,
    RuntimeConfig,
    WorkerConfig,
)
from satsumaestro.engine.subagent import CHILD_EXCLUDED_TOOLS
from satsumaestro.orchestration.worker import TaskReceipt, WorkerProblem, WorkerState
from satsumaestro.orchestration.worker_supervisor import WorkerSupervisor
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.orchestration_tools import ORCHESTRATION_TOOLS
from satsumaestro.tools.builtins.worker_tools import (
    WORKER_MUTATING_TOOL_NAMES,
    WORKER_TOOL_NAMES,
    WORKER_TOOLS,
    ApproveWorkerTrust,
    CheckWorkerReady,
    CreateWorker,
    GetWorker,
    SendWorkerPrompt,
)
from satsumaestro.tools.registry import ToolRegistry
from satsumaestro.util.errors import OrchestrationError

_READY = (
    'json.dumps({"event": "worker.ready", "protocol": 1, "session_id": "sess-stub",'
    ' "cwd": os.getcwd(), "pid": os.getpid(),'
    ' "receipt": json.loads(os.environ.get("SATSUMA_WORKER_RECEIPT") or "null"),'
    ' "emitted_at_ms": 0})'
)
_RESULT = (
    'json.dumps({"event": "worker.result", "final_text": "stub done",'
    ' "stop_reason": "end_turn", "iterations": 1,'
    ' "usage": {"input_tokens": 3, "output_tokens": 7}, "session_id": "sess-stub",'
    ' "emitted_at_ms": 0})'
)
_OK_CHILD = f"import json, os\nprint({_READY}, flush=True)\nprint({_RESULT}, flush=True)\n"
_EXIT_EARLY_CHILD = f"import json, os\nprint({_READY}, flush=True)\n"
_GARBAGE_CHILD = 'print("zsh: command not found: Implement", flush=True)\n'
_HANG_CHILD = f"import json, os, time\nprint({_READY}, flush=True)\ntime.sleep(60)\n"


def _command(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def _supervisor(tmp_path: Path, script: str | None = None, **overrides: object) -> WorkerSupervisor:
    cfg = WorkerConfig(
        enabled=True,
        trusted_roots=[str(tmp_path)],
        command=_command(script) if script is not None else [],
        terminate_grace_s=2.0,
        **overrides,  # type: ignore[arg-type]
    )
    return WorkerSupervisor(config=cfg, cwd=tmp_path)


async def test_send_prompt_spawns_child_and_finishes_with_output(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path, _OK_CHILD)
    worker = sup.create()
    assert worker.state is WorkerState.READY

    sent = await sup.dispatch(worker.worker_id, "Run tests")
    assert sent.state in (WorkerState.STARTING, WorkerState.RUNNING)

    done = await sup.wait(worker.worker_id, timeout_s=15.0)
    assert done.state is WorkerState.FINISHED
    assert done.output == "stub done"
    assert done.session_id == "sess-stub"
    assert done.error is None


async def test_receipt_env_reaches_child_verbatim(tmp_path: Path) -> None:
    script = (
        "import json, os\n"
        f"print({_READY}, flush=True)\n"
        'echo = os.environ.get("SATSUMA_WORKER_RECEIPT", "")\n'
        'print(json.dumps({"event": "worker.result", "final_text": echo,'
        ' "stop_reason": "end_turn", "iterations": 1,'
        ' "usage": {"output_tokens": 1}, "emitted_at_ms": 0}), flush=True)\n'
    )
    sup = _supervisor(tmp_path, script)
    worker = sup.create()
    receipt = TaskReceipt(
        repo="satsuma-demo",
        task_kind="bugfix",
        objective_preview="Fix the date parser",
        source_surface="cli",
    )
    await sup.dispatch(worker.worker_id, "go", receipt)
    done = await sup.wait(worker.worker_id, timeout_s=15.0)
    assert done.state is WorkerState.FINISHED
    assert json.loads(done.output) == receipt.model_dump()


async def test_garbage_first_line_is_shell_misdelivery_with_replay_armed(
    tmp_path: Path,
) -> None:
    sup = _supervisor(tmp_path, _GARBAGE_CHILD)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    done = await sup.wait(worker.worker_id, timeout_s=15.0)
    assert done.state is WorkerState.READY
    assert done.replay == "go"
    assert done.dispatches == 2
    assert done.error is not None
    assert done.error.problem is WorkerProblem.MISDELIVERED
    assert "other than a satsumaestro worker" in done.error.message


async def test_exit_without_terminal_event_records_protocol_failure(
    tmp_path: Path,
) -> None:
    sup = _supervisor(tmp_path, _EXIT_EARLY_CHILD)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    done = await sup.wait(worker.worker_id, timeout_s=15.0)
    assert done.state is WorkerState.FAILED
    assert done.error is not None
    assert "before reporting a result" in done.error.message


async def test_spawn_failure_records_startup_no_evidence(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path)
    sup.config.command = [str(tmp_path / "no-such-binary")]
    worker = sup.create()
    failed = await sup.dispatch(worker.worker_id, "go")
    assert failed.state is WorkerState.FAILED
    assert failed.error is not None
    assert failed.error.problem is WorkerProblem.NO_HANDSHAKE


async def test_terminate_kills_hung_child(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path, _HANG_CHILD)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    for _ in range(500):
        if sup.get(worker.worker_id).state is WorkerState.RUNNING:
            break
        await asyncio.sleep(0.01)
    finished = await sup.terminate(worker.worker_id)
    assert finished.state is WorkerState.FINISHED
    assert worker.worker_id not in sup._procs
    events_after = len(sup.get(worker.worker_id).log)
    await asyncio.sleep(0.05)
    assert len(sup.get(worker.worker_id).log) == events_after


async def test_aclose_reaps_all_children(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path, _HANG_CHILD)
    w1 = sup.create()
    w2 = sup.create()
    await sup.dispatch(w1.worker_id, "a")
    await sup.dispatch(w2.worker_id, "b")
    await sup.aclose()
    assert not sup._procs and not sup._readers


async def test_create_enforces_max_workers_and_frees_on_terminal(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path, _OK_CHILD, max_workers=1)
    worker = sup.create()
    with pytest.raises(OrchestrationError, match="too many live workers"):
        sup.create()
    await sup.dispatch(worker.worker_id, "go")
    await sup.wait(worker.worker_id, timeout_s=15.0)
    assert sup.create().state is WorkerState.READY


async def test_resume_argv_uses_session_id(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path, _OK_CHILD)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "first")
    await sup.wait(worker.worker_id, timeout_s=15.0)
    argv = sup._command_for(sup.get(worker.worker_id))
    assert "--resume" in argv and "sess-stub" in argv


def _ctx_with(tmp_path: Path, sup: WorkerSupervisor | None) -> ToolContext:
    from satsumaestro.orchestration.runner import Orchestrator
    from satsumaestro.security.jail import WorkspaceJail

    jail = WorkspaceJail(tmp_path)
    orch = Orchestrator(
        factory=object(),  # type: ignore[arg-type]
        config=OrchestrationConfig(enabled=True),
        cwd=tmp_path,
        workers=sup,
    )
    return ToolContext(
        workspace_root=jail.root, jail=jail, config=RuntimeConfig(), orchestrator=orch
    )


async def test_worker_tools_degrade_without_supervisor(
    tool_ctx: ToolContext, tmp_path: Path
) -> None:
    inputs: dict[str, dict[str, object]] = {
        "worker_create": {},
        "worker_get": {"worker_id": "x"},
        "worker_await_ready": {"worker_id": "x"},
        "worker_resolve_trust": {"worker_id": "x"},
        "worker_send_prompt": {"worker_id": "x", "prompt": "p"},
        "worker_restart": {"worker_id": "x"},
        "worker_terminate": {"worker_id": "x"},
    }
    assert set(inputs) == WORKER_TOOL_NAMES
    for ctx in (tool_ctx, _ctx_with(tmp_path, None)):
        for tool in WORKER_TOOLS:
            res = await tool().run(tool.Input(**inputs[tool.spec.name]), ctx)
            assert res.is_error and "not available" in res.output, tool.spec.name


async def test_worker_tools_full_flow_without_subprocess(tmp_path: Path) -> None:
    sup = WorkerSupervisor(config=WorkerConfig(enabled=True), cwd=tmp_path)
    ctx = _ctx_with(tmp_path, sup)

    created = await CreateWorker().run(CreateWorker.Input(), ctx)
    assert not created.is_error
    worker_id = json.loads(created.output)["worker_id"]
    assert json.loads(created.output)["state"] == "awaiting_trust"

    blocked = await CheckWorkerReady().run(CheckWorkerReady.Input(worker_id=worker_id), ctx)
    assert json.loads(blocked.output)["blocked"] is True

    resolved = await ApproveWorkerTrust().run(ApproveWorkerTrust.Input(worker_id=worker_id), ctx)
    assert json.loads(resolved.output)["state"] == "ready"

    got = await GetWorker().run(GetWorker.Input(worker_id=worker_id), ctx)
    assert json.loads(got.output)["trust_cleared"] is True


async def test_worker_send_prompt_receipt_all_or_none(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path, _OK_CHILD)
    ctx = _ctx_with(tmp_path, sup)
    worker = sup.create()
    res = await SendWorkerPrompt().run(
        SendWorkerPrompt.Input(worker_id=worker.worker_id, prompt="p", repo="SATSUMAESTRO"),
        ctx,
    )
    assert res.is_error and "must be set together" in res.output
    assert sup.get(worker.worker_id).state is WorkerState.READY


async def test_worker_not_found_lists_known_ids(tmp_path: Path) -> None:
    sup = WorkerSupervisor(config=WorkerConfig(enabled=True), cwd=tmp_path)
    ctx = _ctx_with(tmp_path, sup)
    known = sup.create(str(tmp_path), trusted_roots=[str(tmp_path)])
    res = await GetWorker().run(GetWorker.Input(worker_id="worker_missing"), ctx)
    assert res.is_error
    assert "known worker_ids" in res.output and known.worker_id in res.output


def test_worker_tools_schema_lint_and_modes() -> None:
    registry = ToolRegistry(strict_schema=True)
    for tool in WORKER_TOOLS:
        registry.register(tool)
    read_only = {"worker_get", "worker_await_ready"}
    for tool in WORKER_TOOLS:
        if tool.spec.name in read_only:
            assert tool.spec.required_mode is PermissionMode.READ_ONLY and tool.spec.parallel_safe
        else:
            assert tool.spec.required_mode is PermissionMode.DANGER_FULL_ACCESS
            assert not tool.spec.parallel_safe
        assert len(tool.spec.input_schema["properties"]) <= 8
    assert WORKER_MUTATING_TOOL_NAMES == WORKER_TOOL_NAMES - read_only


def test_default_registry_workers_flag() -> None:
    base = [s["name"] for s in default_registry(orchestration=True).json_schemas()]
    assert not (set(base) & WORKER_TOOL_NAMES)
    full = [s["name"] for s in default_registry(orchestration=True, workers=True).json_schemas()]
    assert full[: len(base)] == base
    assert full[len(base) :] == [t.spec.name for t in WORKER_TOOLS]
    assert set(base[-len(ORCHESTRATION_TOOLS) :]) == {t.spec.name for t in ORCHESTRATION_TOOLS}


def test_child_excluded_tools_cover_worker_tools() -> None:
    assert WORKER_TOOL_NAMES <= CHILD_EXCLUDED_TOOLS


def test_orchestration_flags_fork_bomb_guard() -> None:
    cfg = RuntimeConfig(
        orchestration=OrchestrationConfig(enabled=True, workers=WorkerConfig(enabled=True))
    )
    assert _orchestration_flags(cfg, {}) == (True, True)
    assert _orchestration_flags(cfg, {"SATSUMA_WORKER_CHILD": "1"}) == (False, False)
    cfg2 = RuntimeConfig(orchestration=OrchestrationConfig(enabled=True))
    assert _orchestration_flags(cfg2, {}) == (True, False)
    cfg3 = RuntimeConfig(
        orchestration=OrchestrationConfig(enabled=False, workers=WorkerConfig(enabled=True))
    )
    assert _orchestration_flags(cfg3, {}) == (False, False)


async def test_snapshot_includes_workers_key(tmp_path: Path) -> None:
    from satsumaestro.orchestration.runner import Orchestrator

    sup = WorkerSupervisor(config=WorkerConfig(enabled=True), cwd=tmp_path)
    orch = Orchestrator(
        factory=object(),  # type: ignore[arg-type]
        config=OrchestrationConfig(enabled=True),
        cwd=tmp_path,
        workers=sup,
    )
    worker = sup.create(str(tmp_path), trusted_roots=[str(tmp_path)])
    snap = orch.snapshot()
    assert snap["version"] == 1
    assert [w["worker_id"] for w in snap["workers"]] == [worker.worker_id]

    without = Orchestrator(
        factory=object(),  # type: ignore[arg-type]
        config=OrchestrationConfig(enabled=True),
        cwd=tmp_path,
    )
    assert without.snapshot()["workers"] == []
