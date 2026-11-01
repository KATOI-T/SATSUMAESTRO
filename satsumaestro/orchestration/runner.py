"""バックグラウンド Task 実行器 ``Orchestrator``。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from satsumaestro.config.models import OrchestrationConfig
from satsumaestro.engine.recovery import Fault
from satsumaestro.engine.subagent import RuntimeFactory, resolve_subagent_type
from satsumaestro.engine.trace import (
    ModelReplied,
    RecoveryAttempt,
    ToolDispatched,
    ToolReturned,
    TurnAborted,
    TurnBegan,
    TurnEnded,
    TurnEvent,
)
from satsumaestro.engine.turn import resolve_output
from satsumaestro.orchestration.cron import CronJob, CronLedger
from satsumaestro.orchestration.ids import UnknownIdError
from satsumaestro.orchestration.run_events import FailureCause, RunEvent
from satsumaestro.orchestration.task_ledger import (
    Heartbeat,
    Task,
    TaskBoard,
    TaskLedger,
    TaskStatus,
)
from satsumaestro.orchestration.team import Team, TeamLedger
from satsumaestro.orchestration.worker_supervisor import WorkerSupervisor
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import (
    OrchestrationError,
    PathEscapeError,
    ProviderError,
    SatsumaError,
)
from satsumaestro.util.jsonl import write_text_atomic

_log = logging.getLogger(__name__)

_CLOSE_TIMEOUT_S = 10.0
_TASK_PRESET = "general-purpose"


def _finished(status: TaskStatus | None) -> bool:
    """終わった状態か。"""
    return status is not None and status.finished


_FAULT_CLASS: dict[Fault, FailureCause] = {
    Fault.PROVIDER: FailureCause.INFRASTRUCTURE,
    Fault.MCP_INIT: FailureCause.MCP_INIT,
    Fault.BRANCH_BEHIND: FailureCause.BRANCH_BEHIND,
    Fault.UNHEALTHY_SESSION: FailureCause.TOOL_FAILURE,
}


def _classify_exception(e: BaseException) -> FailureCause:
    if isinstance(e, ProviderError):
        return FailureCause.INFRASTRUCTURE
    if isinstance(e, PathEscapeError):
        return FailureCause.OUTSIDE_WORKSPACE
    return FailureCause.TOOL_FAILURE


class _TaskWatcher:
    """子の ``run_turn`` の出来事を、Task の実行イベントと heartbeat に写す受け手。"""

    def __init__(self, orch: Orchestrator, task_id: str) -> None:
        self._orch = orch
        self._task_id = task_id

    def observe(self, event: TurnEvent) -> None:
        orch, task_id = self._orch, self._task_id
        match event:
            case TurnBegan():
                orch._emit(task_id, RunEvent.started(orch._next_seq()))
                orch._heartbeat(task_id, "running")
            case ModelReplied():
                orch._heartbeat(task_id, f"iteration:{event.iteration + 1}")
            case ToolDispatched():
                orch._heartbeat(task_id, f"tool:{event.tool}")
            case ToolReturned():
                orch._heartbeat(task_id, "running")
            case RecoveryAttempt():
                step = event.step
                remedy = step.remedy.value if step.remedy else "no attempts left"
                orch._emit(
                    task_id,
                    RunEvent.blocked(
                        orch._next_seq(),
                        cause=_FAULT_CLASS.get(step.fault, FailureCause.INFRASTRUCTURE),
                        detail=f"{step.fault.value}: {remedy} -> {step.outcome.value}",
                    ),
                )
            case TurnEnded():
                summary = event.summary
                orch._emit(
                    task_id,
                    RunEvent.finished(
                        orch._next_seq(),
                        data={"iterations": summary.iterations, "tool_calls": summary.tool_calls},
                    ),
                )
            case TurnAborted():
                orch._heartbeat(task_id, f"failed:{type(event.error).__name__}")


class Orchestrator:
    """Task / Team / Cron 登録簿とバックグラウンド実行の統合点（Leader はツール経由で操作）。"""

    def __init__(
        self,
        *,
        factory: RuntimeFactory,
        config: OrchestrationConfig,
        cwd: Path,
        state_path: Path | None = None,
        workers: WorkerSupervisor | None = None,
    ) -> None:
        self.tasks = TaskLedger()
        self.teams = TeamLedger()
        self.crons = CronLedger()
        self.cwd = cwd
        self.config = config
        self.workers = workers
        self._factory = factory
        self._state_file = state_path
        self._sem = asyncio.Semaphore(config.max_concurrency)
        self._running: dict[str, asyncio.Task[None]] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._seq = 0
        self._plans: dict[str, dict[str, Any]] = {}

    async def create_task(
        self,
        prompt: str,
        description: str = "",
        *,
        model: str = "",
        preset: str = _TASK_PRESET,
        plan_id: str | None = None,
    ) -> Task:
        """Task を登録し、バックグラウンド実行を予約して即座に返す（status=created）。"""
        if not prompt.strip():
            raise OrchestrationError("prompt is required")
        task = self.tasks.create(
            prompt, description.strip() or None, model=model, preset=preset, plan_id=plan_id
        )
        task_id = task.task_id
        self._events[task_id] = []
        handle = asyncio.create_task(self._run(task_id), name=f"satsuma-task:{task_id}")
        self._running[task_id] = handle
        handle.add_done_callback(lambda h: self._on_done(task_id, h))
        self._persist()
        return task

    def get_task(self, task_id: str) -> Task:
        t = self.tasks.get(task_id)
        if t is None:
            raise UnknownIdError("task", task_id)
        return t

    def list_tasks(self, status: TaskStatus | None = None) -> list[Task]:
        return self.tasks.list(status)

    def task_output(self, task_id: str) -> str:
        return self.tasks.output(task_id)

    def update_task(self, task_id: str, message: str) -> Task:
        task = self.tasks.note(task_id, message)
        self._persist()
        return task

    def stop_task(self, task_id: str) -> Task:
        """Task を停止する。"""
        return self._stop(task_id, detail="stopped by task_stop")

    def _stop(self, task_id: str, *, detail: str) -> Task:
        """stop の共通経路。"""
        task = self.tasks.stop(task_id)
        handle = self._running.get(task_id)
        if handle is not None:
            handle.cancel()
        self._emit(task_id, RunEvent.closed(self._next_seq(), detail=detail))
        self._persist()
        return task

    async def wait(
        self, task_ids: list[str], timeout_s: float | None = None
    ) -> dict[str, TaskStatus]:
        """指定 Task の終端を待ち、待ち終えた時点の status を ID ごとに返す。"""
        unknown = sorted(tid for tid in task_ids if self.tasks.get(tid) is None)
        if unknown:
            raise UnknownIdError("task", ", ".join(unknown))
        pending = [
            handle
            for tid in task_ids
            if (handle := self._running.get(tid)) is not None and not handle.done()
        ]
        if pending:
            await asyncio.wait(pending, timeout=timeout_s)
        return {tid: self.get_task(tid).status for tid in task_ids}

    def running_count(self) -> int:
        return sum(1 for h in self._running.values() if not h.done())

    def events(self, task_id: str) -> list[RunEvent]:
        return list(self._events.get(task_id, []))

    def board(self) -> TaskBoard:
        return self.tasks.board(self.config.stalled_after_s * 1000)

    def record_plan(self, plan_id: str, payload: dict[str, Any]) -> None:
        """分解計画の記録をキー単位でマージする（JSON 化可能な値のみ渡すこと）。"""
        self._plans.setdefault(plan_id, {}).update(payload)
        self._persist()

    def create_team(self, name: str, task_ids: list[str]) -> Team:
        """既存 Task を束ねる。"""
        for tid in task_ids:
            if self.tasks.get(tid) is None:
                raise UnknownIdError("task", tid)
        team = self.teams.create(name, task_ids)
        for tid in task_ids:
            self.tasks.join_team(tid, team.team_id)
        self._persist()
        return team

    def delete_team(self, team_id: str) -> Team:
        """soft delete ＋未終端メンバーの best-effort 停止（「一括破棄」）。"""
        team = self.teams.delete(team_id)
        for tid in team.members:
            t = self.tasks.get(tid)
            if t is not None and not t.status.finished:
                try:
                    self.stop_task(tid)
                except OrchestrationError:
                    pass
        self._persist()
        return team

    def create_cron(self, schedule: str, prompt: str, description: str | None = None) -> CronJob:
        entry = self.crons.create(schedule, prompt, description)
        self._persist()
        return entry

    def delete_cron(self, cron_id: str) -> CronJob:
        entry = self.crons.delete(cron_id)
        self._persist()
        return entry

    def list_crons(self, enabled_only: bool = False) -> list[CronJob]:
        return self.crons.list(enabled_only)

    async def drain(self, timeout_s: float | None = None) -> None:
        """実行中・待機中の Task が全て終わるまで待つ（cancel はしない）。"""
        pending = [h for h in self._running.values() if not h.done()]
        if pending:
            await asyncio.wait(pending, timeout=timeout_s)

    async def aclose(self) -> None:
        """残っている Task を stop→cancel し回収してから最終スナップショットを書く。"""
        handles = [h for h in self._running.values() if not h.done()]
        for tid, h in list(self._running.items()):
            if h.done():
                continue
            try:
                self._stop(tid, detail="stopped by orchestrator shutdown")
            except OrchestrationError:
                h.cancel()
        if handles:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*handles, return_exceptions=True), timeout=_CLOSE_TIMEOUT_S
                )
            except TimeoutError:
                _log.warning(
                    "orchestrator: %d task(s) did not stop within %.0fs",
                    len(handles),
                    _CLOSE_TIMEOUT_S,
                )
        if self.workers is not None:
            await self.workers.aclose()
        self._persist()

    def snapshot(self) -> dict[str, Any]:
        """観測用スナップショット（外部オブザーバがポーリングする想定。復元には使わない）。"""
        return {
            "version": 1,
            "pid": os.getpid(),
            "generated_at": now_ms(),
            "board": self.board().model_dump(mode="json"),
            "tasks": [t.model_dump(mode="json") for t in self.tasks.list()],
            "teams": [t.model_dump(mode="json") for t in self.teams.list()],
            "crons": [c.model_dump(mode="json") for c in self.crons.list()],
            "events": {
                tid: [e.model_dump(mode="json") for e in evs] for tid, evs in self._events.items()
            },
            "plans": self._plans,
            "workers": (
                [w.model_dump(mode="json") for w in self.workers.ledger.list()]
                if self.workers is not None
                else []
            ),
            "worker_events": (
                {
                    wid: [e.model_dump(mode="json") for e in evs]
                    for wid, evs in self.workers.events_by_worker().items()
                }
                if self.workers is not None
                else {}
            ),
        }

    async def _run(self, task_id: str) -> None:
        try:
            async with self._sem:
                current = self.tasks.get(task_id)
                if current is None or current.status is not TaskStatus.CREATED:
                    return
                self.tasks.move_to(task_id, TaskStatus.RUNNING)
                self._heartbeat(task_id, "starting")
                self._persist()
                child = self._factory.build_child(
                    preset=resolve_subagent_type(current.preset), depth=1, model=current.model
                )
                summary = await child.run_turn(current.prompt, observer=_TaskWatcher(self, task_id))
                self.tasks.record_usage(task_id, summary.usage)
                output, empty = resolve_output(summary)
                self.tasks.add_output(task_id, output)
                if self._status(task_id) is TaskStatus.RUNNING:
                    if empty:
                        self.tasks.move_to(task_id, TaskStatus.FAILED)
                        self._emit(
                            task_id,
                            RunEvent.failed(
                                self._next_seq(),
                                cause=FailureCause.TOOL_FAILURE,
                                detail=output,
                            ),
                        )
                    else:
                        self.tasks.move_to(task_id, TaskStatus.COMPLETED)
        except asyncio.CancelledError:
            if not _finished(self._status(task_id)):
                self.tasks.move_to(task_id, TaskStatus.STOPPED)
            raise
        except SatsumaError as e:
            self._fail(task_id, e)
        except Exception as e:  # noqa: BLE001
            _log.exception("background task %s crashed", task_id)
            self._fail(task_id, e)
        finally:
            self._heartbeat(task_id, "finished", alive=False)
            self._persist()

    def _fail(self, task_id: str, error: BaseException) -> None:
        if _finished(self._status(task_id)):
            return
        message = f"{type(error).__name__}: {error}"
        self.tasks.add_output(task_id, message)
        self.tasks.move_to(task_id, TaskStatus.FAILED)
        self._emit(
            task_id,
            RunEvent.failed(self._next_seq(), cause=_classify_exception(error), detail=message),
        )

    def _on_done(self, task_id: str, handle: asyncio.Task[None]) -> None:
        self._running.pop(task_id, None)
        if handle.cancelled():
            return
        exc = handle.exception()
        if exc is not None:
            _log.error("background task %s raised: %r", task_id, exc)

    def _status(self, task_id: str) -> TaskStatus | None:
        t = self.tasks.get(task_id)
        return t.status if t is not None else None

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(self, task_id: str, event: RunEvent) -> None:
        self._events.setdefault(task_id, []).append(event)
        self._persist()

    def _heartbeat(self, task_id: str, status: str, *, alive: bool = True) -> None:
        try:
            self.tasks.pulse(task_id, Heartbeat(at=now_ms(), connected=alive, note=status))
        except OrchestrationError:
            return
        self._persist()

    def _persist(self) -> None:
        if self._state_file is None:
            return
        try:
            write_text_atomic(
                self._state_file, json.dumps(self.snapshot(), ensure_ascii=False, indent=2)
            )
        except OSError:
            _log.debug("orchestrator: failed to write %s", self._state_file, exc_info=True)
