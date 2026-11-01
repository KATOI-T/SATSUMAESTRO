"""worker の台帳（状態の決まり方）と純関数のテスト。"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.orchestration.ids import UnknownIdError
from satsumaestro.orchestration.worker import (
    Misdelivery,
    PreflightIssue,
    StallReason,
    TaskReceipt,
    TrustSource,
    WorkerLedger,
    WorkerLogKind,
    WorkerProblem,
    WorkerState,
    _paths_in,
    combine_roots,
    excerpt,
    is_within,
    preflight_warnings,
    receipt_mismatches,
    stall_reason,
)
from satsumaestro.util.errors import OrchestrationError

DOCS = TaskReceipt(
    repo="satsuma-demo",
    task_kind="docs",
    objective_preview="Document the CLI flags",
    source_surface="cli",
    expected_artifacts=["README.md"],
)


def _trusted(tmp_path: Path, ledger: WorkerLedger, name: str = "alpha") -> str:
    """信頼済みのルート（tmp_path/trusted）の下に作った worker の ID。"""
    cwd = tmp_path / "trusted" / name
    cwd.mkdir(parents=True, exist_ok=True)
    return ledger.create(str(cwd), trusted_roots=[str(tmp_path / "trusted")]).worker_id


def _kinds(ledger: WorkerLedger, worker_id: str) -> list[WorkerLogKind]:
    w = ledger.get(worker_id)
    assert w is not None
    return [e.kind for e in w.log]


def test_worker_inside_a_trusted_root_is_ready_at_once(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    w = ledger.get(_trusted(tmp_path, ledger))
    assert w is not None
    assert w.state is WorkerState.READY
    assert w.trusted_by_root and w.trust_cleared and w.error is None
    assert w.worker_id.startswith("worker_1_")
    assert [(e.kind, e.trusted_by) for e in w.log] == [(WorkerLogKind.TRUSTED, TrustSource.ROOT)]


def test_worker_outside_the_roots_waits_for_approval(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    w = ledger.create(str(tmp_path / "elsewhere"), trusted_roots=[str(tmp_path / "trusted")])
    assert w.state is WorkerState.AWAITING_TRUST
    assert not w.trusted_by_root and not w.trust_cleared
    assert w.error is not None and w.error.problem is WorkerProblem.UNTRUSTED_CWD
    assert [e.kind for e in w.log] == [WorkerLogKind.GATED]
    with pytest.raises(OrchestrationError, match="cannot take a prompt now"):
        ledger.dispatch(w.worker_id, "Write the changelog")

    approved = ledger.approve_trust(w.worker_id)
    assert approved.state is WorkerState.READY
    assert approved.trust_cleared and approved.error is None
    assert approved.log[-1].kind is WorkerLogKind.TRUSTED
    assert approved.log[-1].trusted_by is TrustSource.APPROVED
    with pytest.raises(OrchestrationError, match="not awaiting trust approval"):
        ledger.approve_trust(w.worker_id)


def test_restart_judges_trust_again_and_drops_a_manual_approval(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    inside = _trusted(tmp_path, ledger)
    ledger.dispatch(inside, "Write the changelog", DOCS)
    ledger.accept_handshake(inside, "sess-a")
    ledger.record_result(inside, text="done", stop_reason="end_turn", output_tokens=12)
    again = ledger.restart(inside)
    assert again.state is WorkerState.READY
    assert (again.dispatches, again.prompt, again.receipt, again.session_id, again.output) == (
        0,
        None,
        None,
        None,
        "",
    )
    assert not again.awaiting_handshake and again.dispatched_at is None
    assert again.log[-1].kind is WorkerLogKind.RESTARTED
    assert again.log[-1].trusted_by is TrustSource.ROOT

    outside = ledger.create(str(tmp_path / "elsewhere"))
    ledger.approve_trust(outside.worker_id)
    gated = ledger.restart(outside.worker_id)
    assert gated.state is WorkerState.AWAITING_TRUST
    assert gated.error is not None and gated.error.problem is WorkerProblem.UNTRUSTED_CWD


def test_dispatch_records_the_prompt_and_waits_for_a_handshake(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    w = ledger.dispatch(wid, "  Write the changelog  ", DOCS)
    assert w.state is WorkerState.STARTING
    assert (w.dispatches, w.awaiting_handshake, w.prompt, w.receipt) == (
        1,
        True,
        "Write the changelog",
        DOCS,
    )
    assert w.dispatched_at is not None
    assert w.log[-1].kind is WorkerLogKind.DISPATCHED
    assert "Write the changelog" in (w.log[-1].text or "")
    running = ledger.accept_handshake(wid, "sess-42")
    assert running.state is WorkerState.RUNNING
    assert running.session_id == "sess-42" and not running.awaiting_handshake
    with pytest.raises(OrchestrationError, match="not waiting for a handshake"):
        ledger.accept_handshake(wid, "sess-43")


def test_dispatch_needs_a_prompt_or_an_armed_replay(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    for empty in (None, "", "   "):
        with pytest.raises(OrchestrationError, match="neither a new prompt nor an armed replay"):
            ledger.dispatch(wid, empty)


@pytest.mark.parametrize(
    ("stop_reason", "tokens", "state"),
    [
        ("end_turn", 30, WorkerState.FINISHED),
        ("max_tokens", 0, WorkerState.FINISHED),
        ("unknown", 5, WorkerState.FINISHED),
        ("unknown", 0, WorkerState.FAILED),
        ("error", 30, WorkerState.FAILED),
    ],
)
def test_result_is_a_failure_only_when_the_provider_failed(
    tmp_path: Path, stop_reason: str, tokens: int, state: WorkerState
) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    ledger.dispatch(wid, "Summarise docs/guide.md")
    ledger.accept_handshake(wid, "sess-1")
    w = ledger.record_result(wid, text="summary", stop_reason=stop_reason, output_tokens=tokens)
    assert w.state is state and w.output == "summary" and not w.awaiting_handshake
    if state is WorkerState.FAILED:
        assert w.error is not None and w.error.problem is WorkerProblem.PROVIDER
        assert w.log[-1].kind is WorkerLogKind.FAILED
    else:
        assert w.error is None and w.log[-1].kind is WorkerLogKind.FINISHED


def test_failures_are_not_added_after_the_worker_ended(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    ledger.dispatch(wid, "Rename the module")
    failed = ledger.fail(wid, WorkerProblem.BAD_PROTOCOL, "the worker exited early")
    assert failed.state is WorkerState.FAILED
    assert failed.error is not None and failed.error.problem is WorkerProblem.BAD_PROTOCOL
    n = len(failed.log)
    ledger.fail(wid, WorkerProblem.PROVIDER, "late")
    ledger.record_misdelivery(wid, Misdelivery.NO_OUTPUT)
    ledger.record_stall(wid, argv=[], child_alive=False, limit_ms=1)
    after = ledger.get(wid)
    assert after is not None and len(after.log) == n
    assert after.error is not None and after.error.message == "the worker exited early"


def test_terminate_ends_the_worker(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    ledger.dispatch(wid, "Rename the module")
    w = ledger.terminate(wid)
    assert w.state is WorkerState.FINISHED and not w.awaiting_handshake
    assert w.log[-1].kind is WorkerLogKind.TERMINATED


def test_misdelivery_arms_the_same_prompt_and_keeps_the_receipt(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    ledger.dispatch(wid, "Write the changelog", DOCS)
    w = ledger.record_misdelivery(
        wid,
        Misdelivery.RECEIPT_MISMATCH,
        answered_cwd=str(tmp_path),
        echoed_receipt={"repo": "another-repo"},
        detail="the echoed receipt differs in: repo",
    )
    assert w.state is WorkerState.READY and w.replay == "Write the changelog"
    assert w.error is not None and w.error.problem is WorkerProblem.MISDELIVERED
    assert [e.kind for e in w.log[-2:]] == [WorkerLogKind.MISDELIVERED, WorkerLogKind.REPLAY_ARMED]
    missed, armed = w.log[-2], w.log[-1]
    assert missed.state is WorkerState.FAILED and armed.state is WorkerState.READY
    assert missed.delivery is not None and armed.delivery is not None
    assert (missed.delivery.replay_armed, armed.delivery.replay_armed) == (False, True)
    assert missed.delivery.misdelivery is Misdelivery.RECEIPT_MISMATCH
    assert missed.delivery.echoed_receipt == {"repo": "another-repo"}
    assert missed.delivery.sent_receipt == DOCS
    assert missed.text == "the echoed receipt differs in: repo"

    resent = ledger.dispatch(wid, None)
    assert resent.state is WorkerState.STARTING
    assert (resent.prompt, resent.receipt, resent.replay, resent.dispatches) == (
        "Write the changelog",
        DOCS,
        None,
        2,
    )
    ledger.record_misdelivery(wid, Misdelivery.NO_OUTPUT)
    fresh = ledger.dispatch(wid, "Rename the module")
    assert fresh.receipt is None


def test_misdelivery_without_auto_replay_stays_failed(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    cwd = tmp_path / "trusted" / "beta"
    cwd.mkdir(parents=True)
    w = ledger.create(str(cwd), trusted_roots=[str(tmp_path / "trusted")], auto_replay=False)
    ledger.dispatch(w.worker_id, "Write the changelog")
    failed = ledger.record_misdelivery(
        w.worker_id, Misdelivery.WRONG_DIRECTORY, answered_cwd="/srv/other"
    )
    assert failed.state is WorkerState.FAILED and failed.replay is None
    assert failed.log[-1].kind is WorkerLogKind.MISDELIVERED
    assert failed.log[-1].delivery is not None
    assert failed.log[-1].delivery.answered_cwd == "/srv/other"


@pytest.mark.parametrize("kind", list(Misdelivery))
def test_every_misdelivery_has_its_own_message(tmp_path: Path, kind: Misdelivery) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    ledger.dispatch(wid, "Write the changelog")
    w = ledger.record_misdelivery(wid, kind)
    assert w.error is not None and "Write the changelog" in w.error.message
    assert w.log[-2].text == f"misdelivered ({kind.value})"


def test_stall_while_starting_records_the_situation(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    dispatched = ledger.dispatch(wid, "Write the changelog")
    w = ledger.record_stall(wid, argv=["satsumaestro", "prompt"], child_alive=True, limit_ms=500)
    assert w.state is WorkerState.FAILED
    assert w.error is not None and w.error.problem is WorkerProblem.NO_HANDSHAKE
    report = w.log[-1].stall
    assert w.log[-1].kind is WorkerLogKind.STALLED and report is not None
    assert report.reason is StallReason.NO_HANDSHAKE and "no_handshake" in w.error.message
    assert report.state is WorkerState.STARTING and report.child_alive
    assert report.argv == ["satsumaestro", "prompt"] and report.limit_ms == 500
    assert report.dispatched_at == dispatched.dispatched_at
    assert report.created_at <= report.state_since <= report.checked_at
    assert report.waited_ms == report.checked_at - (report.dispatched_at or 0)

    other = _trusted(tmp_path, ledger, "gamma")
    ledger.dispatch(other, "Rename the module")
    gone = ledger.record_stall(other, argv=[], child_alive=False, limit_ms=500)
    assert gone.log[-1].stall is not None
    assert gone.log[-1].stall.reason is StallReason.CHILD_EXITED


@pytest.mark.parametrize(
    ("alive", "state", "dispatched", "reason"),
    [
        (False, WorkerState.STARTING, True, StallReason.CHILD_EXITED),
        (False, WorkerState.READY, False, StallReason.CHILD_EXITED),
        (True, WorkerState.STARTING, True, StallReason.NO_HANDSHAKE),
        (True, WorkerState.STARTING, False, StallReason.UNKNOWN),
        (True, WorkerState.READY, True, StallReason.UNKNOWN),
        (True, WorkerState.AWAITING_TRUST, False, StallReason.UNKNOWN),
    ],
)
def test_stall_reason(
    alive: bool, state: WorkerState, dispatched: bool, reason: StallReason
) -> None:
    assert stall_reason(child_alive=alive, state=state, dispatched=dispatched) is reason


def test_readiness(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    w = ledger.create(str(tmp_path / "elsewhere"))
    gated = ledger.readiness(w.worker_id)
    assert (gated.can_dispatch, gated.blocked, gated.replay_armed) == (False, True, False)
    assert gated.error is not None
    ledger.approve_trust(w.worker_id)
    ready = ledger.readiness(w.worker_id)
    assert (ready.can_dispatch, ready.blocked, ready.error) == (True, False, None)
    ledger.dispatch(w.worker_id, "Write the changelog")
    starting = ledger.readiness(w.worker_id)
    assert (starting.can_dispatch, starting.blocked) == (False, False)
    ledger.record_misdelivery(w.worker_id, Misdelivery.NOT_A_HANDSHAKE)
    armed = ledger.readiness(w.worker_id)
    assert (armed.state, armed.can_dispatch, armed.replay_armed) == (WorkerState.READY, True, True)


def test_ledger_hands_out_copies_and_rejects_unknown_ids(tmp_path: Path) -> None:
    ledger = WorkerLedger()
    wid = _trusted(tmp_path, ledger)
    copy = ledger.get(wid)
    assert copy is not None
    copy.state = WorkerState.FAILED
    copy.log.clear()
    fresh = ledger.get(wid)
    assert fresh is not None and fresh.state is WorkerState.READY and fresh.log
    assert len(ledger) == 1 and [w.worker_id for w in ledger.list()] == [wid]
    assert ledger.get("worker_9_0") is None
    with pytest.raises(UnknownIdError, match="unknown worker id: worker_9_0"):
        ledger.readiness("worker_9_0")


def test_is_within_compares_path_components(tmp_path: Path) -> None:
    (tmp_path / "srv" / "app" / "api").mkdir(parents=True)
    (tmp_path / "srv" / "application").mkdir()
    app = tmp_path / "srv" / "app"
    assert is_within(app / "api", app) and is_within(app, app)
    assert not is_within(tmp_path / "srv" / "application", app)
    assert not is_within(app, app / "api")
    assert is_within(app / "api" / ".." / "api", app)


def test_combine_roots_keeps_the_configured_ones_first() -> None:
    assert combine_roots(["/opt/x", "/opt/y"], ["/opt/z", "/opt/x"]) == [
        "/opt/x",
        "/opt/y",
        "/opt/z",
    ]
    assert combine_roots([], []) == []


def test_receipt_mismatches() -> None:
    echoed = DOCS.model_dump()
    assert receipt_mismatches(DOCS, echoed) == []
    assert receipt_mismatches(DOCS, {**echoed, "note": "extra keys are ignored"}) == []
    assert receipt_mismatches(DOCS, {**echoed, "repo": "x", "source_surface": "api"}) == [
        "repo",
        "source_surface",
    ]
    assert receipt_mismatches(DOCS, {"repo": "satsuma-demo"}) == [
        "expected_artifacts",
        "objective_preview",
        "source_surface",
        "task_kind",
    ]
    assert receipt_mismatches(DOCS, None) == sorted(DOCS.model_dump())


def test_excerpt() -> None:
    assert excerpt("  short  ") == "short"
    long = "word " * 20
    cut = excerpt(long)
    assert cut.endswith("…") and len(cut) <= 49


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)


def test_paths_in_picks_relative_file_paths() -> None:
    prompt = (
        "Update docs/guide.md and (./src/app/main.py), not /etc/hosts, ../up/x.py,"
        " https://example.com/a.html, src/app/ or plain words. Also `docs/guide.md` again."
    )
    assert _paths_in(prompt) == ["docs/guide.md", "src/app/main.py"]


def test_preflight_warns_about_untracked_paths(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
    _git(tmp_path, "add", "docs/guide.md")
    found = preflight_warnings(tmp_path, "Polish docs/guide.md and docs/missing.md")
    assert [(w.issue, w.path) for w in found] == [
        (PreflightIssue.UNTRACKED_PATH, "docs/missing.md")
    ]
    outside = tmp_path / "plain"
    outside.mkdir()
    assert all(
        w.issue is not PreflightIssue.GIT_DIR_READ_ONLY for w in preflight_warnings(outside, "")
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root は権限ビットを無視できる")
def test_preflight_warns_when_the_git_dir_is_read_only(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    dot_git = tmp_path / ".git"
    mode = stat.S_IMODE(dot_git.stat().st_mode)
    dot_git.chmod(0o555)
    try:
        found = preflight_warnings(tmp_path, "Write the changelog")
    finally:
        dot_git.chmod(mode)
    read_only = [w for w in found if w.issue is PreflightIssue.GIT_DIR_READ_ONLY]
    assert len(read_only) == 1
    assert Path(read_only[0].path or "").resolve() == dot_git.resolve()


def test_note_preflight_logs_warnings_without_changing_state(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    ledger = WorkerLedger()
    w = ledger.create(str(tmp_path), trusted_roots=[str(tmp_path)])
    noted = ledger.note_preflight(w.worker_id, "Fix lib/absent.py")
    assert noted.state is WorkerState.READY
    entry = noted.log[-1]
    assert entry.kind is WorkerLogKind.PREFLIGHT_WARNING and entry.state is WorkerState.READY
    assert entry.warning is not None and entry.warning.path == "lib/absent.py"


def test_worker_settings_are_off_by_default() -> None:
    workers = RuntimeConfig().orchestration.workers
    assert (workers.enabled, workers.trusted_roots, workers.command) == (False, [], [])
    assert workers.auto_recover_prompt_misdelivery is True
    assert (workers.ready_timeout_s, workers.result_timeout_s, workers.terminate_grace_s) == (
        45.0,
        0.0,
        5.0,
    )
    assert workers.max_workers == 4
