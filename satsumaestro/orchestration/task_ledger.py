"""Task の台帳と、進み具合の一覧（board）。"""

from __future__ import annotations

import builtins
from enum import StrEnum

from pydantic import BaseModel, Field

from satsumaestro.orchestration.ids import UnknownIdError, make_id
from satsumaestro.session.models import TokenUsage
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import OrchestrationError


class TaskStatus(StrEnum):
    """Task の状態。"""

    CREATED = "created"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

    @property
    def finished(self) -> bool:
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED)


class Freshness(StrEnum):
    """生存報告（heartbeat）から見た Task の鮮度。"""

    HEALTHY = "healthy"
    STALLED = "stalled"
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"
    FINISHED = "finished"


class Heartbeat(BaseModel):
    """実行中の Task からの生存報告。"""

    at: int
    connected: bool = True
    note: str = ""

    def freshness(self, now: int, stall_after_ms: int) -> Freshness:
        """``now`` 時点の鮮度。"""
        if not self.connected:
            return Freshness.DISCONNECTED
        if now - self.at > stall_after_ms:
            return Freshness.STALLED
        return Freshness.HEALTHY


class TaskNote(BaseModel):
    """Task に残した注記（``task_update``）。"""

    author: str
    text: str
    at: int


class Task(BaseModel):
    """1 つのバックグラウンド Task。"""

    task_id: str
    prompt: str
    description: str | None = None
    model: str = ""
    preset: str = "general-purpose"
    plan_id: str | None = None
    team_id: str | None = None
    status: TaskStatus = TaskStatus.CREATED
    created_at: int
    updated_at: int
    started_at: int | None = None
    finished_at: int | None = None
    notes: list[TaskNote] = Field(default_factory=list)
    output: str = ""
    heartbeat: Heartbeat | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)


class BoardRow(BaseModel):
    task_id: str
    status: TaskStatus
    freshness: Freshness
    prompt: str
    team_id: str | None = None
    heartbeat: Heartbeat | None = None


class TaskBoard(BaseModel):
    """Task を active / blocked / finished に分け、鮮度を添えた一覧。"""

    as_of: int
    active: list[BoardRow] = Field(default_factory=list)
    blocked: list[BoardRow] = Field(default_factory=list)
    finished: list[BoardRow] = Field(default_factory=list)


class TaskLedger:
    """Task の台帳。"""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._serial = 0

    def __len__(self) -> int:
        return len(self._tasks)

    def get(self, task_id: str) -> Task | None:
        t = self._tasks.get(task_id)
        return None if t is None else t.model_copy(deep=True)

    def list(self, status: TaskStatus | None = None) -> builtins.list[Task]:
        """登録順。"""
        return [
            t.model_copy(deep=True)
            for t in self._tasks.values()
            if status is None or t.status is status
        ]

    def output(self, task_id: str) -> str:
        return self._find(task_id).output

    def board(self, stall_after_ms: int) -> TaskBoard:
        return self.board_at(now_ms(), stall_after_ms)

    def board_at(self, now: int, stall_after_ms: int) -> TaskBoard:
        """created / running → active、blocked → blocked、終わったもの → finished に分ける。"""
        board = TaskBoard(as_of=now)
        for t in self._tasks.values():
            row = BoardRow(
                task_id=t.task_id,
                status=t.status,
                freshness=self._freshness(t, now, stall_after_ms),
                prompt=t.prompt,
                team_id=t.team_id,
                heartbeat=None if t.heartbeat is None else t.heartbeat.model_copy(),
            )
            if t.status.finished:
                board.finished.append(row)
            elif t.status is TaskStatus.BLOCKED:
                board.blocked.append(row)
            else:
                board.active.append(row)
        return board

    def create(
        self,
        prompt: str,
        description: str | None = None,
        *,
        model: str = "",
        preset: str = "general-purpose",
        plan_id: str | None = None,
    ) -> Task:
        self._serial += 1
        now = now_ms()
        task = Task(
            task_id=make_id("task", self._serial),
            prompt=prompt,
            description=description,
            model=model,
            preset=preset,
            plan_id=plan_id,
            created_at=now,
            updated_at=now,
        )
        self._tasks[task.task_id] = task
        return task.model_copy(deep=True)

    def stop(self, task_id: str) -> Task:
        """created / running / blocked → stopped。"""
        t = self._find(task_id)
        if t.status.finished:
            raise OrchestrationError(
                f"task {task_id} has already finished ({t.status.value}) and cannot be stopped"
            )
        self._touch(t, status=TaskStatus.STOPPED)
        return t.model_copy(deep=True)

    def move_to(self, task_id: str, status: TaskStatus) -> None:
        """状態を無条件に置く（遷移の規則は実行器が守る）。"""
        self._touch(self._find(task_id), status=status)

    def note(self, task_id: str, text: str) -> Task:
        """利用者の注記を残す（走っている子には届けない）。"""
        t = self._find(task_id)
        at = self._touch(t)
        t.notes.append(TaskNote(author="user", text=text, at=at))
        return t.model_copy(deep=True)

    def add_output(self, task_id: str, text: str) -> None:
        t = self._find(task_id)
        t.output += text
        self._touch(t)

    def record_usage(self, task_id: str, usage: TokenUsage) -> None:
        """子の ``TurnSummary.usage`` の控え（親の usage には混ぜない）。"""
        t = self._find(task_id)
        t.usage = usage.model_copy()
        self._touch(t)

    def pulse(self, task_id: str, heartbeat: Heartbeat) -> None:
        t = self._find(task_id)
        t.heartbeat = heartbeat
        self._touch(t)

    def join_team(self, task_id: str, team_id: str) -> None:
        t = self._find(task_id)
        t.team_id = team_id
        self._touch(t)

    def remove(self, task_id: str) -> Task | None:
        return self._tasks.pop(task_id, None)

    def _find(self, task_id: str) -> Task:
        t = self._tasks.get(task_id)
        if t is None:
            raise UnknownIdError("task", task_id)
        return t

    @staticmethod
    def _freshness(t: Task, now: int, stall_after_ms: int) -> Freshness:
        if t.status.finished:
            return Freshness.FINISHED
        if t.heartbeat is None:
            return Freshness.UNKNOWN
        return t.heartbeat.freshness(now, stall_after_ms)

    @staticmethod
    def _touch(t: Task, *, status: TaskStatus | None = None) -> int:
        """更新時刻を進め、``status`` が渡されたら置く。"""
        now = now_ms()
        t.updated_at = now
        if status is not None:
            t.status = status
            if status is TaskStatus.RUNNING and t.started_at is None:
                t.started_at = now
            elif status.finished and t.finished_at is None:
                t.finished_at = now
        return now
