"""外部の satsumaestro 子プロセス（worker）の台帳と、その状態の決まり方。"""

from __future__ import annotations

import builtins
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from satsumaestro.orchestration.ids import UnknownIdError, make_id
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import OrchestrationError
from satsumaestro.util.paths import is_within


class WorkerState(StrEnum):
    AWAITING_TRUST = "awaiting_trust"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

    @property
    def ended(self) -> bool:
        """これ以上の観測を受け付けない（終わった worker）。"""
        return self in (WorkerState.FINISHED, WorkerState.FAILED)


class WorkerProblem(StrEnum):
    """直近のエラーの種類。"""

    UNTRUSTED_CWD = "untrusted_cwd"
    MISDELIVERED = "misdelivered"
    NO_HANDSHAKE = "no_handshake"
    BAD_PROTOCOL = "bad_protocol"
    PROVIDER = "provider"


class WorkerError(BaseModel):
    problem: WorkerProblem
    message: str
    at: int


class TaskReceipt(BaseModel):
    """子に渡す「何の仕事か」の控え。"""

    repo: str
    task_kind: str
    objective_preview: str
    source_surface: str
    expected_artifacts: list[str] = Field(default_factory=list)


class Misdelivery(StrEnum):
    """handshake から分かった宛先違いの種類（判定は監督）。"""

    NO_OUTPUT = "no_output"
    NOT_A_HANDSHAKE = "not_a_handshake"
    RECEIPT_MISMATCH = "receipt_mismatch"
    WRONG_DIRECTORY = "wrong_directory"


class StallReason(StrEnum):
    """handshake が期限内に来なかった理由の見立て。"""

    CHILD_EXITED = "child_exited"
    NO_HANDSHAKE = "no_handshake"
    UNKNOWN = "unknown"


class StallReport(BaseModel):
    """handshake を待ちきれなかった時点の様子（診断用）。"""

    reason: StallReason
    state: WorkerState
    state_since: int
    argv: list[str]
    created_at: int
    dispatched_at: int | None = None
    checked_at: int
    waited_ms: int
    limit_ms: int
    child_alive: bool


class PreflightIssue(StrEnum):
    UNTRACKED_PATH = "untracked_path"
    GIT_DIR_READ_ONLY = "git_dir_read_only"


class PreflightWarning(BaseModel):
    issue: PreflightIssue
    message: str
    path: str | None = None


class TrustSource(StrEnum):
    ROOT = "root"
    APPROVED = "approved"


class WorkerLogKind(StrEnum):
    GATED = "gated"
    TRUSTED = "trusted"
    RESTARTED = "restarted"
    PREFLIGHT_WARNING = "preflight_warning"
    DISPATCHED = "dispatched"
    HANDSHAKE = "handshake"
    MISDELIVERED = "misdelivered"
    REPLAY_ARMED = "replay_armed"
    STALLED = "stalled"
    FINISHED = "finished"
    FAILED = "failed"
    TERMINATED = "terminated"


class DeliveryNote(BaseModel):
    """宛先違いの記録。"""

    misdelivery: Misdelivery
    prompt: str
    answered_cwd: str | None = None
    echoed_receipt: dict[str, Any] | None = None
    sent_receipt: TaskReceipt | None = None
    replay_armed: bool = False


class WorkerLogEntry(BaseModel):
    """worker の出来事 1 つ。"""

    seq: int
    kind: WorkerLogKind
    state: WorkerState
    at: int
    text: str | None = None
    trusted_by: TrustSource | None = None
    delivery: DeliveryNote | None = None
    stall: StallReport | None = None
    warning: PreflightWarning | None = None


class Worker(BaseModel):
    worker_id: str
    cwd: str
    state: WorkerState
    created_at: int
    updated_at: int
    trusted_by_root: bool
    trust_cleared: bool = False
    auto_replay: bool = True
    dispatches: int = 0
    awaiting_handshake: bool = False
    dispatched_at: int | None = None
    prompt: str | None = None
    receipt: TaskReceipt | None = None
    replay: str | None = None
    session_id: str | None = None
    output: str = ""
    error: WorkerError | None = None
    log: list[WorkerLogEntry] = Field(default_factory=list)


class Readiness(BaseModel):
    """``worker_await_ready`` が返す、いま指示を渡せるかの見立て。"""

    worker_id: str
    state: WorkerState
    can_dispatch: bool
    blocked: bool
    replay_armed: bool
    error: WorkerError | None = None


def combine_roots(configured: Sequence[str], extra: Sequence[str]) -> list[str]:
    """設定のルートを先に、呼び出しで足されたルートを後に、重複を除いて並べる。"""
    return list(dict.fromkeys([*configured, *extra]))


def receipt_mismatches(expected: TaskReceipt, echoed: Mapping[str, Any] | None) -> list[str]:
    """写して返された受領書と渡した受領書で値が違う項目の名前（昇順）。"""
    wanted = expected.model_dump()
    got: Mapping[str, Any] = echoed if echoed is not None else {}
    return sorted(k for k, v in wanted.items() if echoed is None or got.get(k) != v)


def stall_reason(*, child_alive: bool, state: WorkerState, dispatched: bool) -> StallReason:
    """handshake を待ちきれなかったときの見立て。"""
    if not child_alive:
        return StallReason.CHILD_EXITED
    if dispatched and state is WorkerState.STARTING:
        return StallReason.NO_HANDSHAKE
    return StallReason.UNKNOWN


def excerpt(text: str, limit: int = 48) -> str:
    """記録に載せる指示の冒頭（前後の空白を落とし、長ければ ``limit`` 文字で切って … を付ける）。"""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


_RELATIVE_PATH = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+")
_WRAPPING = "`\"'()[]{},;:"


def preflight_warnings(cwd: Path, prompt: str) -> list[PreflightWarning]:
    """子を起動する前に分かる心配ごと（``.git`` に書けない・指示のファイルが追跡外）。"""
    found: list[PreflightWarning] = []
    meta = _metadata_dir(cwd)
    if meta is not None and not _writable_dir(meta):
        found.append(
            PreflightWarning(
                issue=PreflightIssue.GIT_DIR_READ_ONLY,
                message=f"{meta} is read-only; commits and pushes from the worker may fail",
                path=str(meta),
            )
        )
    for rel in _paths_in(prompt):
        if not _git_tracks(cwd, rel):
            found.append(
                PreflightWarning(
                    issue=PreflightIssue.UNTRACKED_PATH,
                    message=f"{rel} appears in the prompt but git does not track it here",
                    path=rel,
                )
            )
    return found


def _paths_in(prompt: str) -> list[str]:
    """指示に出てくる相対ファイルパスを、出てきた順に重複なく。"""
    found: dict[str, None] = {}
    for word in prompt.split():
        token = word.strip(_WRAPPING)
        while token.startswith("./"):
            token = token[2:]
        if _RELATIVE_PATH.fullmatch(token) and ".." not in token and "." in token.rsplit("/", 1)[1]:
            found[token] = None
    return list(found)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)
    except OSError:
        return None


def _git_tracks(cwd: Path, rel: str) -> bool:
    done = _git(cwd, "ls-files", "-z", "--", rel)
    return done is not None and done.returncode == 0 and bool(done.stdout)


def _metadata_dir(cwd: Path) -> Path | None:
    done = _git(cwd, "rev-parse", "--absolute-git-dir")
    if done is None or done.returncode != 0:
        return None
    text = done.stdout.decode("utf-8", errors="replace").strip()
    return Path(text) if text else None


def _writable_dir(path: Path) -> bool:
    folder = path if path.is_dir() else path.parent
    return folder.is_dir() and os.access(folder, os.W_OK | os.X_OK)


class WorkerLedger:
    """worker の台帳。"""

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self._serial = 0

    def __len__(self) -> int:
        return len(self._workers)

    def get(self, worker_id: str) -> Worker | None:
        w = self._workers.get(worker_id)
        return None if w is None else w.model_copy(deep=True)

    def list(self) -> builtins.list[Worker]:
        return [w.model_copy(deep=True) for w in self._workers.values()]

    def readiness(self, worker_id: str) -> Readiness:
        w = self._find(worker_id)
        return Readiness(
            worker_id=w.worker_id,
            state=w.state,
            can_dispatch=w.state is WorkerState.READY,
            blocked=w.state in (WorkerState.AWAITING_TRUST, WorkerState.FAILED),
            replay_armed=w.replay is not None,
            error=None if w.error is None else w.error.model_copy(),
        )

    def create(
        self, cwd: str, *, trusted_roots: Sequence[str] = (), auto_replay: bool = True
    ) -> Worker:
        """worker を登録し、作業ディレクトリの信頼をその場で決める（子は起動しない）。"""
        self._serial += 1
        now = now_ms()
        w = Worker(
            worker_id=make_id("worker", self._serial),
            cwd=cwd,
            state=WorkerState.AWAITING_TRUST,
            created_at=now,
            updated_at=now,
            trusted_by_root=any(is_within(cwd, root) for root in trusted_roots),
            auto_replay=auto_replay,
        )
        self._gate(w, kind=None)
        self._workers[w.worker_id] = w
        return w.model_copy(deep=True)

    def approve_trust(self, worker_id: str) -> Worker:
        """承認待ちの worker の作業ディレクトリを手で承認して ``ready`` にする。"""
        w = self._find(worker_id)
        if w.state is not WorkerState.AWAITING_TRUST:
            raise OrchestrationError(
                f"worker {worker_id} is not awaiting trust approval (state: {w.state.value})"
            )
        w.trust_cleared = True
        w.error = None
        self._log(
            w,
            WorkerLogKind.TRUSTED,
            WorkerState.READY,
            "the working directory was approved by hand",
            trusted_by=TrustSource.APPROVED,
        )
        return w.model_copy(deep=True)

    def restart(self, worker_id: str) -> Worker:
        """指示まわりを空にして、信頼の判定からやり直す（手での承認は失効する）。"""
        w = self._find(worker_id)
        w.dispatches = 0
        w.awaiting_handshake = False
        w.dispatched_at = None
        w.prompt = None
        w.receipt = None
        w.replay = None
        w.session_id = None
        w.output = ""
        w.error = None
        w.trust_cleared = False
        self._gate(w, kind=WorkerLogKind.RESTARTED)
        return w.model_copy(deep=True)

    def dispatch(
        self, worker_id: str, prompt: str | None = None, receipt: TaskReceipt | None = None
    ) -> Worker:
        """指示を渡したことを記録して ``starting`` にする（子の起動は監督がこの後で行う）。"""
        w = self._find(worker_id)
        if w.state is not WorkerState.READY:
            raise OrchestrationError(
                f"worker {worker_id} cannot take a prompt now (state: {w.state.value})"
            )
        fresh = (prompt or "").strip()
        text = fresh or w.replay
        if text is None:
            raise OrchestrationError(
                f"worker {worker_id} has neither a new prompt nor an armed replay"
            )
        if fresh or receipt is not None:
            w.receipt = receipt
        w.dispatches += 1
        w.awaiting_handshake = True
        w.dispatched_at = now_ms()
        w.prompt = text
        w.replay = None
        w.error = None
        self._log(w, WorkerLogKind.DISPATCHED, WorkerState.STARTING, f"prompt: {excerpt(text)}")
        return w.model_copy(deep=True)

    def note_preflight(self, worker_id: str, prompt: str) -> Worker:
        """起動前の心配ごとを記録する（状態は変えない）。"""
        w = self._find(worker_id)
        for warning in preflight_warnings(Path(w.cwd), prompt):
            self._log(w, WorkerLogKind.PREFLIGHT_WARNING, w.state, warning.message, warning=warning)
        return w.model_copy(deep=True)

    def accept_handshake(self, worker_id: str, session_id: str) -> Worker:
        """宛先の確認を通った handshake を受け付けて ``running`` にする（確認そのものは監督）。"""
        w = self._find(worker_id)
        if w.state is not WorkerState.STARTING:
            raise OrchestrationError(
                f"worker {worker_id} is not waiting for a handshake (state: {w.state.value})"
            )
        w.session_id = session_id
        w.awaiting_handshake = False
        w.error = None
        self._log(
            w, WorkerLogKind.HANDSHAKE, WorkerState.RUNNING, f"handshake from session {session_id}"
        )
        return w.model_copy(deep=True)

    def record_result(
        self, worker_id: str, *, text: str, stop_reason: str, output_tokens: int
    ) -> Worker:
        """子の結果を記録する。"""
        w = self._find(worker_id)
        w.output = text
        w.awaiting_handshake = False
        if stop_reason == "error":
            message = f"the worker's provider reported an error (stop_reason={stop_reason!r})"
        elif stop_reason == "unknown" and output_tokens == 0:
            message = (
                "the worker stopped for an unknown reason without any output"
                " (the provider may be unhealthy, or the context ran out)"
            )
        else:
            w.error = None
            self._log(
                w,
                WorkerLogKind.FINISHED,
                WorkerState.FINISHED,
                f"stop_reason={stop_reason}, output_tokens={output_tokens}",
            )
            return w.model_copy(deep=True)
        w.error = WorkerError(problem=WorkerProblem.PROVIDER, message=message, at=now_ms())
        self._log(w, WorkerLogKind.FAILED, WorkerState.FAILED, message)
        return w.model_copy(deep=True)

    def fail(self, worker_id: str, problem: WorkerProblem, message: str) -> Worker:
        """監督が見た失敗（起動できない・結果を書かずに終わった・子のエラー）を記録する。"""
        w = self._find(worker_id)
        if w.state.ended:
            return w.model_copy(deep=True)
        w.error = WorkerError(problem=problem, message=message, at=now_ms())
        w.awaiting_handshake = False
        self._log(w, WorkerLogKind.FAILED, WorkerState.FAILED, message)
        return w.model_copy(deep=True)

    def record_misdelivery(
        self,
        worker_id: str,
        misdelivery: Misdelivery,
        *,
        answered_cwd: str | None = None,
        echoed_receipt: Mapping[str, Any] | None = None,
        detail: str | None = None,
    ) -> Worker:
        """宛先違いを記録して ``failed`` にする。"""
        w = self._find(worker_id)
        if w.state.ended:
            return w.model_copy(deep=True)
        head = excerpt(w.prompt or "")
        w.error = WorkerError(
            problem=WorkerProblem.MISDELIVERED,
            message=_MISDELIVERY_TEXT[misdelivery].format(cwd=w.cwd, prompt=head),
            at=now_ms(),
        )
        w.awaiting_handshake = False
        note = DeliveryNote(
            misdelivery=misdelivery,
            prompt=head,
            answered_cwd=answered_cwd,
            echoed_receipt=None if echoed_receipt is None else dict(echoed_receipt),
            sent_receipt=w.receipt,
        )
        self._log(
            w,
            WorkerLogKind.MISDELIVERED,
            WorkerState.FAILED,
            detail or f"misdelivered ({misdelivery.value})",
            delivery=note,
        )
        if w.auto_replay:
            w.replay = w.prompt
            self._log(
                w,
                WorkerLogKind.REPLAY_ARMED,
                WorkerState.READY,
                "the same prompt is armed to be sent again",
                delivery=note.model_copy(update={"replay_armed": True}),
            )
        return w.model_copy(deep=True)

    def record_stall(
        self, worker_id: str, *, argv: Sequence[str], child_alive: bool, limit_ms: int
    ) -> Worker:
        """handshake を待ちきれなかったことを記録して ``failed`` にする。"""
        w = self._find(worker_id)
        if w.state.ended:
            return w.model_copy(deep=True)
        now = now_ms()
        since = w.dispatched_at if w.dispatched_at is not None else w.created_at
        report = StallReport(
            reason=stall_reason(
                child_alive=child_alive, state=w.state, dispatched=w.dispatched_at is not None
            ),
            state=w.state,
            state_since=w.updated_at,
            argv=list(argv),
            created_at=w.created_at,
            dispatched_at=w.dispatched_at,
            checked_at=now,
            waited_ms=max(0, now - since),
            limit_ms=limit_ms,
            child_alive=child_alive,
        )
        message = (
            f"no handshake from the worker after {report.waited_ms} ms ({report.reason.value})"
        )
        w.error = WorkerError(problem=WorkerProblem.NO_HANDSHAKE, message=message, at=now)
        w.awaiting_handshake = False
        self._log(w, WorkerLogKind.STALLED, WorkerState.FAILED, message, stall=report)
        return w.model_copy(deep=True)

    def terminate(self, worker_id: str) -> Worker:
        """外から止めたことを記録して ``finished`` にする（子を止めるのは監督）。"""
        w = self._find(worker_id)
        w.awaiting_handshake = False
        self._log(w, WorkerLogKind.TERMINATED, WorkerState.FINISHED, "stopped by the supervisor")
        return w.model_copy(deep=True)

    def _find(self, worker_id: str) -> Worker:
        w = self._workers.get(worker_id)
        if w is None:
            raise UnknownIdError("worker", worker_id)
        return w

    def _gate(self, w: Worker, *, kind: WorkerLogKind | None) -> None:
        """信頼の判定（作成と再起動で共通）。"""
        if w.trusted_by_root:
            w.trust_cleared = True
            self._log(
                w,
                kind or WorkerLogKind.TRUSTED,
                WorkerState.READY,
                "the working directory is inside a trusted root",
                trusted_by=TrustSource.ROOT,
            )
            return
        w.error = WorkerError(
            problem=WorkerProblem.UNTRUSTED_CWD,
            message=f"{w.cwd} is outside the trusted roots; approve it with worker_resolve_trust",
            at=now_ms(),
        )
        self._log(
            w,
            kind or WorkerLogKind.GATED,
            WorkerState.AWAITING_TRUST,
            "waiting for the working directory to be approved",
        )

    @staticmethod
    def _log(
        w: Worker,
        kind: WorkerLogKind,
        state: WorkerState,
        text: str | None = None,
        *,
        trusted_by: TrustSource | None = None,
        delivery: DeliveryNote | None = None,
        stall: StallReport | None = None,
        warning: PreflightWarning | None = None,
    ) -> None:
        """出来事を 1 つ足し、状態と更新時刻を進める。"""
        at = now_ms()
        w.state = state
        w.updated_at = at
        w.log.append(
            WorkerLogEntry(
                seq=len(w.log) + 1,
                kind=kind,
                state=state,
                at=at,
                text=text,
                trusted_by=trusted_by,
                delivery=delivery,
                stall=stall,
                warning=warning,
            )
        )


_MISDELIVERY_TEXT: dict[Misdelivery, str] = {
    Misdelivery.NO_OUTPUT: "the worker exited before writing anything: {prompt}",
    Misdelivery.NOT_A_HANDSHAKE: (
        "the prompt reached something other than a satsumaestro worker"
        " (its first line was not a handshake): {prompt}"
    ),
    Misdelivery.RECEIPT_MISMATCH: (
        "the worker in {cwd} echoed a receipt for a different task: {prompt}"
    ),
    Misdelivery.WRONG_DIRECTORY: "the worker answered from outside {cwd}: {prompt}",
}
