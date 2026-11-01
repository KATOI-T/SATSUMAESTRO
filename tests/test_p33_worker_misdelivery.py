"""誤配送検出・replay・evidence・実行イベントの E2E テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from satsumaestro.config.models import WorkerConfig
from satsumaestro.orchestration.run_events import FailureCause, RunEventKind
from satsumaestro.orchestration.worker import (
    Misdelivery,
    StallReason,
    TaskReceipt,
    WorkerLogKind,
    WorkerProblem,
    WorkerState,
)
from satsumaestro.orchestration.worker_supervisor import WorkerSupervisor

_STUB = Path(__file__).parent / "fixtures" / "worker_stub.py"

RECEIPT = TaskReceipt(
    repo="satsuma-demo",
    task_kind="docs",
    objective_preview="Document the CLI flags",
    source_surface="cli",
    expected_artifacts=["README.md"],
)


def _supervisor(
    tmp_path: Path,
    mode: str,
    *,
    monkeypatch: pytest.MonkeyPatch,
    mode_file: Path | None = None,
    **overrides: object,
) -> WorkerSupervisor:
    if mode_file is not None:
        mode_file.write_text(mode, encoding="utf-8")
        monkeypatch.setenv("STUB_MODE_FILE", str(mode_file))
        monkeypatch.delenv("STUB_MODE", raising=False)
    else:
        monkeypatch.setenv("STUB_MODE", mode)
        monkeypatch.delenv("STUB_MODE_FILE", raising=False)
    cfg = WorkerConfig(
        enabled=True,
        trusted_roots=[str(tmp_path)],
        command=[sys.executable, str(_STUB)],
        terminate_grace_s=2.0,
        **overrides,  # type: ignore[arg-type]
    )
    return WorkerSupervisor(config=cfg, cwd=tmp_path)


def _misdeliveries(sup: WorkerSupervisor, worker_id: str) -> list[Misdelivery]:
    return [
        e.delivery.misdelivery
        for e in sup.get(worker_id).log
        if e.kind is WorkerLogKind.MISDELIVERED and e.delivery is not None
    ]


async def test_ok_mode_finishes_and_emits_a_ready_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(tmp_path, "ok", monkeypatch=monkeypatch)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go", RECEIPT)
    done = await sup.wait(worker.worker_id, timeout_s=15.0)
    assert done.state is WorkerState.FINISHED
    assert done.output.startswith("stub done")
    names = [e.kind for e in sup.events(worker.worker_id)]
    assert RunEventKind.READY in names and RunEventKind.FINISHED in names


async def test_wrong_cwd_detected_as_wrong_target_and_replay_escalates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(tmp_path, "wrong_cwd", monkeypatch=monkeypatch)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "Write the changelog")
    done = await sup.wait(worker.worker_id, timeout_s=20.0)

    assert done.dispatches == 2
    assert done.state is WorkerState.READY
    assert done.replay == "Write the changelog"
    assert _misdeliveries(sup, worker.worker_id) == [
        Misdelivery.WRONG_DIRECTORY,
        Misdelivery.WRONG_DIRECTORY,
    ]
    names = [e.kind for e in sup.events(worker.worker_id)]
    assert names.count(RunEventKind.PROMPT_MISDELIVERED) == 2
    blocked = [
        e
        for e in sup.events(worker.worker_id)
        if e.kind is RunEventKind.BLOCKED and e.cause is FailureCause.PROMPT_NOT_DELIVERED
    ]
    assert blocked and "resend was used up" in (blocked[-1].detail or "")


async def test_wrong_receipt_detected_as_wrong_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(tmp_path, "wrong_receipt", monkeypatch=monkeypatch)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go", RECEIPT)
    done = await sup.wait(worker.worker_id, timeout_s=20.0)
    assert Misdelivery.RECEIPT_MISMATCH in _misdeliveries(sup, worker.worker_id)
    assert done.error is not None
    assert done.error.problem is WorkerProblem.MISDELIVERED
    assert "receipt for a different task" in done.error.message


async def test_garbage_first_line_is_not_a_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(tmp_path, "garbage_first_line", monkeypatch=monkeypatch)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    await sup.wait(worker.worker_id, timeout_s=20.0)
    assert Misdelivery.NOT_A_HANDSHAKE in _misdeliveries(sup, worker.worker_id)


async def test_exit_early_detected_as_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(tmp_path, "exit_early", monkeypatch=monkeypatch)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    await sup.wait(worker.worker_id, timeout_s=20.0)
    assert Misdelivery.NO_OUTPUT in _misdeliveries(sup, worker.worker_id)


async def test_replay_recovers_when_second_attempt_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode_file = tmp_path / "stub-mode.txt"
    sup = _supervisor(tmp_path, "wrong_receipt\nok", monkeypatch=monkeypatch, mode_file=mode_file)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go", RECEIPT)
    done = await sup.wait(worker.worker_id, timeout_s=20.0)
    assert done.state is WorkerState.FINISHED
    assert done.dispatches == 2
    assert any(e.kind is WorkerLogKind.REPLAY_ARMED for e in done.log)
    names = [e.kind for e in sup.events(worker.worker_id)]
    assert RunEventKind.PROMPT_MISDELIVERED in names and RunEventKind.FINISHED in names


async def test_auto_recover_off_fails_as_prompt_not_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(
        tmp_path,
        "wrong_receipt",
        monkeypatch=monkeypatch,
        auto_recover_prompt_misdelivery=False,
    )
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go", RECEIPT)
    done = await sup.wait(worker.worker_id, timeout_s=20.0)
    assert done.state is WorkerState.FAILED
    assert done.replay is None
    failed = [
        e
        for e in sup.events(worker.worker_id)
        if e.kind is RunEventKind.FAILED and e.cause is FailureCause.PROMPT_NOT_DELIVERED
    ]
    assert failed


async def test_manual_replay_after_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mode_file = tmp_path / "stub-mode.txt"
    sup = _supervisor(
        tmp_path, "wrong_cwd\nwrong_cwd\nok", monkeypatch=monkeypatch, mode_file=mode_file
    )
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    escalated = await sup.wait(worker.worker_id, timeout_s=20.0)
    assert escalated.state is WorkerState.READY
    await sup.dispatch(worker.worker_id, None)
    done = await sup.wait(worker.worker_id, timeout_s=15.0)
    assert done.state is WorkerState.FINISHED
    assert done.dispatches == 3


async def test_hang_before_ready_times_out_with_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(tmp_path, "hang_before_ready", monkeypatch=monkeypatch, ready_timeout_s=0.5)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    done = await sup.wait(worker.worker_id, timeout_s=20.0)
    assert done.state is WorkerState.FAILED
    assert done.error is not None
    assert done.error.problem is WorkerProblem.NO_HANDSHAKE
    stalled = next(e for e in done.log if e.kind is WorkerLogKind.STALLED)
    assert stalled.stall is not None
    assert stalled.stall.reason is StallReason.NO_HANDSHAKE and stalled.stall.child_alive
    assert stalled.stall.dispatched_at is not None
    assert stalled.stall.argv[0] == sys.executable
    failed_events = [
        e
        for e in sup.events(worker.worker_id)
        if e.kind is RunEventKind.FAILED and e.cause is FailureCause.PROMPT_NOT_DELIVERED
    ]
    assert failed_events
    assert worker.worker_id not in sup._procs


async def test_hang_after_ready_hits_result_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(tmp_path, "hang_after_ready", monkeypatch=monkeypatch, result_timeout_s=0.5)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    done = await sup.wait(worker.worker_id, timeout_s=20.0)
    assert done.state is WorkerState.FAILED
    assert done.error is not None
    assert "no result from the worker" in done.error.message
    assert worker.worker_id not in sup._procs


async def test_finish_unknown_zero_tokens_classified_as_provider_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sup = _supervisor(tmp_path, "finish_unknown_zero_tokens", monkeypatch=monkeypatch)
    worker = sup.create()
    await sup.dispatch(worker.worker_id, "go")
    done = await sup.wait(worker.worker_id, timeout_s=15.0)
    assert done.state is WorkerState.FAILED
    assert done.error is not None
    assert done.error.problem is WorkerProblem.PROVIDER
    assert "the provider may be unhealthy" in done.error.message


async def test_untrusted_create_emits_a_blocked_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_MODE", "ok")
    sup = WorkerSupervisor(
        config=WorkerConfig(enabled=True, command=[sys.executable, str(_STUB)]),
        cwd=tmp_path,
    )
    worker = sup.create()
    blocked = [
        e
        for e in sup.events(worker.worker_id)
        if e.kind is RunEventKind.BLOCKED and e.cause is FailureCause.TRUST_PENDING
    ]
    assert blocked
