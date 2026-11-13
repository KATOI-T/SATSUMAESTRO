"""worker の子プロセスを起動し、stdout のプロトコル行を台帳の記録に写す。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from satsumaestro.config.models import WorkerConfig
from satsumaestro.engine.recovery import (
    Fault,
    PlaybookOutcome,
    RecoveryLedger,
    Remedy,
    run_playbook,
)
from satsumaestro.orchestration.ids import UnknownIdError
from satsumaestro.orchestration.run_events import FailureCause, RunEvent
from satsumaestro.orchestration.worker import (
    Misdelivery,
    Readiness,
    StallReason,
    TaskReceipt,
    Worker,
    WorkerLedger,
    WorkerProblem,
    WorkerState,
    combine_roots,
    receipt_mismatches,
)
from satsumaestro.providers.choice import HOST_ENV
from satsumaestro.security.secret_env import PROVIDER_KEY_VARS
from satsumaestro.util.environ import ENVIRON, WORKER_DOTENV_ENV
from satsumaestro.util.errors import OrchestrationError
from satsumaestro.util.proc import terminate_graceful

_log = logging.getLogger(__name__)

RECEIPT_ENV = "SATSUMA_WORKER_RECEIPT"
WORKER_CHILD_ENV = "SATSUMA_WORKER_CHILD"

_STDERR_TAIL_LINES = 20
_STREAM_LIMIT = 8 * 1024 * 1024
_CLOSE_TIMEOUT_S = 10.0


def _default_command() -> list[str]:
    """既定の子: この satsumaestro 自身（``python -m satsumaestro.cli.app``）。"""
    return [sys.executable, "-m", "satsumaestro.cli.app"]


def _same_directory(a: str, b: str) -> bool:
    """同じディレクトリか（``resolve`` して比べる。macOS の /tmp → /private/tmp も同じと見る）。"""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return a == b


_HARNESS_NAMES: tuple[str, ...] = (
    *PROVIDER_KEY_VARS,
    *HOST_ENV.values(),
    "OPENAI_BASE_URL",
    "OLLAMA_*",
    "SATSUMA_*",
)


def worker_environment() -> dict[str, str]:
    """worker の環境（持っている ``.env`` のうちハーネスが読む名前は、別の 1 変数で渡す）。"""
    env = dict(os.environ)
    held = {
        name: value
        for name, value in ENVIRON.items()
        if env.get(name) != value and any(fnmatchcase(name, p) for p in _HARNESS_NAMES)
    }
    if held:
        env[WORKER_DOTENV_ENV] = json.dumps(held)
    return env


@dataclass(frozen=True)
class _HandshakeProblem:
    """1 行目の確認で見つかった宛先違い。"""

    misdelivery: Misdelivery
    answered_cwd: str | None
    echoed_receipt: dict[str, Any] | None
    detail: str


class _WorkerRemedies:
    """手当ての手順書を worker に当てる ``RemedyRunner``。"""

    def __init__(self, sup: WorkerSupervisor, worker_id: str) -> None:
        self._sup = sup
        self._target_worker = worker_id

    async def apply(self, remedy: Remedy, fault: Fault) -> bool:
        if remedy is not Remedy.RESEND_PROMPT:
            return False
        worker = await self._sup.dispatch(self._target_worker, None)
        return worker.state is WorkerState.STARTING


class WorkerSupervisor:
    """worker の台帳と子プロセスをまとめる（Orchestrator が持ち、worker ツールが操作する）。"""

    def __init__(self, *, config: WorkerConfig, cwd: Path) -> None:
        self.ledger = WorkerLedger()
        self.config = config
        self.cwd = cwd
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._readers: dict[str, asyncio.Task[None]] = {}
        self._stderr_tails: dict[str, deque[str]] = {}
        self._argvs: dict[str, list[str]] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._seq = 0
        self._resends: dict[str, RecoveryLedger] = {}

    def get(self, worker_id: str) -> Worker:
        w = self.ledger.get(worker_id)
        if w is None:
            raise UnknownIdError("worker", worker_id)
        return w

    def readiness(self, worker_id: str) -> Readiness:
        return self.ledger.readiness(worker_id)

    def events(self, worker_id: str) -> list[RunEvent]:
        """worker の実行イベント（ready / prompt_misdelivered など）。"""
        return list(self._events.get(worker_id, []))

    def events_by_worker(self) -> dict[str, list[RunEvent]]:
        return {wid: list(evs) for wid, evs in self._events.items()}

    def create(self, cwd: str = "", trusted_roots: Sequence[str] = ()) -> Worker:
        """worker を登録する（子はまだ起動しない）。"""
        live = sum(1 for w in self.ledger.list() if not w.state.ended)
        if live >= self.config.max_workers:
            raise OrchestrationError(
                f"too many live workers (limit {self.config.max_workers},"
                " orchestration.workers.max_workers)"
            )
        worker = self.ledger.create(
            cwd or str(self.cwd),
            trusted_roots=combine_roots(self.config.trusted_roots, trusted_roots),
            auto_replay=self.config.auto_recover_prompt_misdelivery,
        )
        if worker.state is WorkerState.AWAITING_TRUST:
            self._emit(
                worker.worker_id,
                RunEvent.blocked(
                    self._next_seq(),
                    cause=FailureCause.TRUST_PENDING,
                    detail="the working directory is waiting for approval",
                ),
            )
        return worker

    def approve_trust(self, worker_id: str) -> Worker:
        return self.ledger.approve_trust(worker_id)

    async def dispatch(
        self, worker_id: str, prompt: str | None = None, receipt: TaskReceipt | None = None
    ) -> Worker:
        """台帳に指示を記録し（渡せる状態かは台帳が確かめる）、子を起動して読み取りを始める。"""
        worker = self.ledger.dispatch(worker_id, prompt, receipt)
        self.ledger.note_preflight(worker_id, worker.prompt or "")

        env = worker_environment()
        env[WORKER_CHILD_ENV] = "1"
        if worker.receipt is not None:
            env[RECEIPT_ENV] = worker.receipt.model_dump_json()
        else:
            env.pop(RECEIPT_ENV, None)

        argv = self._command_for(worker)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=worker.cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=_STREAM_LIMIT,
            )
        except OSError as e:
            return self.ledger.fail(
                worker_id, WorkerProblem.NO_HANDSHAKE, f"could not start the worker process: {e}"
            )
        self._procs[worker_id] = proc
        self._argvs[worker_id] = argv
        self._stderr_tails[worker_id] = deque(maxlen=_STDERR_TAIL_LINES)
        reader = asyncio.create_task(
            self._watch(worker_id, proc), name=f"satsuma-worker:{worker_id}"
        )
        self._readers[worker_id] = reader
        reader.add_done_callback(lambda h: self._on_reader_done(worker_id, h))
        return self.get(worker_id)

    async def wait(self, worker_id: str, timeout_s: float | None = None) -> Worker:
        """子が終わるまで待って、いまの worker を返す。"""
        self.get(worker_id)
        loop = asyncio.get_running_loop()
        deadline = None if timeout_s is None else loop.time() + timeout_s
        while True:
            reader = self._readers.get(worker_id)
            if reader is None or reader.done():
                return self.get(worker_id)
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return self.get(worker_id)
            await asyncio.wait({reader}, timeout=remaining)
            await asyncio.sleep(0)

    async def restart(self, worker_id: str) -> Worker:
        """子が居れば止め、送り直しの回数を戻して、台帳を信頼の判定からやり直す。"""
        self.get(worker_id)
        await self._stop_child(worker_id)
        self._resends.pop(worker_id, None)
        return self.ledger.restart(worker_id)

    async def terminate(self, worker_id: str) -> Worker:
        """子を止め（SIGTERM → 猶予 → SIGKILL）、台帳を finished にする。"""
        self.get(worker_id)
        await self._stop_child(worker_id)
        return self.ledger.terminate(worker_id)

    async def aclose(self) -> None:
        """すべての子と読み取りを回収する（Orchestrator.aclose から呼ばれる）。"""
        ids = set(self._procs) | set(self._readers)
        if not ids:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*(self._stop_child(wid) for wid in ids)),
                timeout=_CLOSE_TIMEOUT_S,
            )
        except TimeoutError:
            _log.warning(
                "worker supervisor: %d worker(s) did not stop within %.0fs",
                len(ids),
                _CLOSE_TIMEOUT_S,
            )

    def _command_for(self, worker: Worker) -> list[str]:
        """子の argv。"""
        argv = [
            *(list(self.config.command) or _default_command()),
            "prompt",
            worker.prompt or "",
            "--output-format",
            "stream-json",
            "--cwd",
            worker.cwd,
        ]
        if worker.session_id:
            argv += ["--resume", worker.session_id]
        if worker.trust_cleared:
            argv.append("--trust")
        return argv

    async def _watch(self, worker_id: str, proc: asyncio.subprocess.Process) -> None:
        """stdout を読んで台帳に写す（stderr は並行して読み捨てる）。"""
        stderr_task = asyncio.create_task(self._drain_stderr(worker_id, proc))
        try:
            await self._follow(worker_id, proc)
        except OrchestrationError:
            _log.debug("worker reader %s: ledger rejected an update", worker_id, exc_info=True)
        finally:
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task

    async def _follow(self, worker_id: str, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            first = await asyncio.wait_for(
                proc.stdout.readline(), timeout=self.config.ready_timeout_s
            )
        except TimeoutError:
            await self._on_stall(worker_id, proc)
            return

        problem = self._check_handshake(self.get(worker_id), first)
        if problem is not None:
            await self._on_misdelivery(worker_id, proc, problem)
            return

        hello = self._parse_line(first) or {}
        session = str(hello.get("session_id") or "")
        self.ledger.accept_handshake(worker_id, session)
        self._emit(worker_id, RunEvent.ready(self._next_seq(), detail=f"session {session}"))

        try:
            if self.config.result_timeout_s > 0:
                got_result = await asyncio.wait_for(
                    self._read_until_result(worker_id, proc), timeout=self.config.result_timeout_s
                )
            else:
                got_result = await self._read_until_result(worker_id, proc)
        except TimeoutError:
            message = f"no result from the worker within {self.config.result_timeout_s:.0f}s"
            self.ledger.fail(worker_id, WorkerProblem.PROVIDER, message)
            self._emit(
                worker_id,
                RunEvent.failed(
                    self._next_seq(), cause=FailureCause.INFRASTRUCTURE, detail=message
                ),
            )
            await terminate_graceful(proc, grace_s=self.config.terminate_grace_s)
            return
        code = await proc.wait()
        if not got_result:
            tail = "; ".join(self._stderr_tails.get(worker_id, ()))
            message = f"the worker exited (code {code}) before reporting a result"
            self.ledger.fail(
                worker_id, WorkerProblem.BAD_PROTOCOL, message + (f": {tail}" if tail else "")
            )
            self._emit(
                worker_id,
                RunEvent.failed(
                    self._next_seq(), cause=FailureCause.INFRASTRUCTURE, detail=message
                ),
            )

    async def _on_stall(self, worker_id: str, proc: asyncio.subprocess.Process) -> None:
        """handshake を待ちきれなかった: 様子を記録して failed にし、子を止める。"""
        stalled = self.ledger.record_stall(
            worker_id,
            argv=self._argvs.get(worker_id, []),
            child_alive=proc.returncode is None,
            limit_ms=int(self.config.ready_timeout_s * 1000),
        )
        last = stalled.log[-1] if stalled.log else None
        reason = last.stall.reason if last is not None and last.stall is not None else None
        cause = (
            FailureCause.PROMPT_NOT_DELIVERED
            if reason is StallReason.NO_HANDSHAKE
            else FailureCause.INFRASTRUCTURE
        )
        self._emit(
            worker_id,
            RunEvent.failed(
                self._next_seq(),
                cause=cause,
                detail=stalled.error.message if stalled.error else None,
            ),
        )
        await terminate_graceful(proc, grace_s=self.config.terminate_grace_s)

    async def _on_misdelivery(
        self, worker_id: str, proc: asyncio.subprocess.Process, problem: _HandshakeProblem
    ) -> None:
        """宛先違い: 子を止めて記録し、送り直すか親に委ねる。"""
        await terminate_graceful(proc, grace_s=self.config.terminate_grace_s)
        after = self.ledger.record_misdelivery(
            worker_id,
            problem.misdelivery,
            answered_cwd=problem.answered_cwd,
            echoed_receipt=problem.echoed_receipt,
            detail=problem.detail,
        )
        self._emit(worker_id, RunEvent.prompt_misdelivered(self._next_seq(), detail=problem.detail))
        if after.state is WorkerState.FAILED:
            self._emit(
                worker_id,
                RunEvent.failed(
                    self._next_seq(), cause=FailureCause.PROMPT_NOT_DELIVERED, detail=problem.detail
                ),
            )
            return
        await self._resend_once(worker_id)

    async def _read_until_result(self, worker_id: str, proc: asyncio.subprocess.Process) -> bool:
        """handshake の後の stdout を、結果かエラーの行まで読む。"""
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = self._parse_line(raw)
            if line is None:
                continue
            event = line.get("event")
            if event == "worker.result":
                usage = line.get("usage")
                tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0
                done = self.ledger.record_result(
                    worker_id,
                    text=str(line.get("final_text") or ""),
                    stop_reason=str(line.get("stop_reason") or "unknown"),
                    output_tokens=int(tokens),
                )
                if done.state is WorkerState.FINISHED:
                    self._emit(worker_id, RunEvent.finished(self._next_seq()))
                else:
                    self._emit(
                        worker_id,
                        RunEvent.failed(
                            self._next_seq(),
                            cause=FailureCause.INFRASTRUCTURE,
                            detail=done.error.message if done.error else None,
                        ),
                    )
                return True
            if event == "worker.error":
                message = f"{line.get('error_type')}: {line.get('message')}"
                self.ledger.fail(worker_id, WorkerProblem.PROVIDER, message)
                self._emit(
                    worker_id,
                    RunEvent.failed(
                        self._next_seq(), cause=FailureCause.INFRASTRUCTURE, detail=message
                    ),
                )
                return True
        return False

    def _check_handshake(self, worker: Worker, raw: bytes) -> _HandshakeProblem | None:
        """1 行目から宛先違いを見つける（合っていれば None）。"""
        if not raw:
            return _HandshakeProblem(Misdelivery.NO_OUTPUT, None, None, "the worker wrote nothing")
        hello = self._parse_line(raw)
        if hello is None or hello.get("event") != "worker.ready":
            return _HandshakeProblem(
                Misdelivery.NOT_A_HANDSHAKE, None, None, "the first line was not a handshake"
            )
        answered_cwd = str(hello.get("cwd") or "")
        echoed = hello.get("receipt")
        echoed_receipt = echoed if isinstance(echoed, dict) else None
        if worker.receipt is not None:
            differing = receipt_mismatches(worker.receipt, echoed_receipt)
            if differing:
                return _HandshakeProblem(
                    Misdelivery.RECEIPT_MISMATCH,
                    answered_cwd,
                    echoed_receipt,
                    f"the echoed receipt differs in: {', '.join(differing)}",
                )
        if answered_cwd and not _same_directory(answered_cwd, worker.cwd):
            return _HandshakeProblem(
                Misdelivery.WRONG_DIRECTORY,
                answered_cwd,
                echoed_receipt,
                f"the worker runs in {answered_cwd}, not {worker.cwd}",
            )
        return None

    async def _resend_once(self, worker_id: str) -> None:
        """構えてある指示を 1 回だけ自動で送り直す（手当ての手順書を使う）。"""
        ledger = self._resends.setdefault(worker_id, RecoveryLedger(attempt_limit=1))
        outcome = await run_playbook(
            Fault.MISDELIVERED_PROMPT, ledger, runner=_WorkerRemedies(self, worker_id)
        )
        if outcome is not PlaybookOutcome.SUCCEEDED:
            self._emit(
                worker_id,
                RunEvent.blocked(
                    self._next_seq(),
                    cause=FailureCause.PROMPT_NOT_DELIVERED,
                    detail=(
                        "the automatic resend was used up; send it with worker_send_prompt"
                        " or restart the worker"
                    ),
                ),
            )

    async def _drain_stderr(self, worker_id: str, proc: asyncio.subprocess.Process) -> None:
        """stderr を読み捨てて末尾だけ持つ（パイプが詰まって子が止まるのを防ぐ）。"""
        if proc.stderr is None:
            return
        tail = self._stderr_tails.setdefault(worker_id, deque(maxlen=_STDERR_TAIL_LINES))
        async for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                tail.append(line)

    async def _stop_child(self, worker_id: str) -> None:
        """読み取りを先に止めてから子を回収する（止めた後に記録を書かせない順序）。"""
        reader = self._readers.pop(worker_id, None)
        if reader is not None and not reader.done():
            reader.cancel()
        proc = self._procs.pop(worker_id, None)
        if proc is not None:
            await terminate_graceful(proc, grace_s=self.config.terminate_grace_s)
        if reader is not None:
            await asyncio.gather(reader, return_exceptions=True)

    def _on_reader_done(self, worker_id: str, handle: asyncio.Task[None]) -> None:
        if self._readers.get(worker_id) is handle:
            self._readers.pop(worker_id, None)
            self._procs.pop(worker_id, None)
        if handle.cancelled():
            return
        exc = handle.exception()
        if exc is not None:
            _log.error("worker reader %s raised: %r", worker_id, exc)

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, worker_id: str, event: RunEvent) -> None:
        self._events.setdefault(worker_id, []).append(event)

    @staticmethod
    def _parse_line(raw: bytes) -> dict[str, Any] | None:
        if not raw.strip():
            return None
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return obj if isinstance(obj, dict) else None
