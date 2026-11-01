"""タスク分解計画のデータ型と DAG 検証。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from satsumaestro.session.models import TokenUsage
from satsumaestro.util.errors import OrchestrationError


class SubtaskKind(StrEnum):
    LOOKUP = "lookup"
    ANALYSIS = "analysis"
    EDIT = "edit"
    VERIFY = "verify"


_BACKGROUND_KINDS: frozenset[SubtaskKind] = frozenset({SubtaskKind.LOOKUP, SubtaskKind.ANALYSIS})


class SubtaskSpec(BaseModel):
    id: str
    objective: str
    kind: SubtaskKind
    depends_on: list[str] = Field(default_factory=list)
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    acceptance: str = ""
    est_calls: int = Field(default=2, ge=1, le=8)

    def effective_kind(self) -> SubtaskKind:
        """``writes`` が空でなければ EDIT 扱い（宣言 kind より書き込み宣言を優先）。"""
        if self.writes and self.kind in _BACKGROUND_KINDS:
            return SubtaskKind.EDIT
        return self.kind


class TaskPlan(BaseModel):
    single: bool = False
    subtasks: list[SubtaskSpec] = Field(default_factory=list)
    reduce_hint: str = ""


@dataclass(frozen=True)
class ValidatedPlan:
    """検証済みの証明型。"""

    plan: TaskPlan
    levels: list[list[str]]
    background: frozenset[str]
    inline: frozenset[str]


def paths_collide(first: str, second: str) -> bool:
    """2 つの書き込み先が同じ場所か、片方がもう片方の下にあるか。"""
    a = first.rstrip("/").split("/")
    b = second.rstrip("/").split("/")
    shared = min(len(a), len(b))
    return a[:shared] == b[:shared]


def validate_plan(plan: TaskPlan, *, max_subtasks: int, max_depth: int) -> ValidatedPlan:
    """DAG を検証し、段分け・背景/直列の分類まで済ませた :class:`ValidatedPlan` を返す。"""
    subtasks = plan.subtasks
    if len(subtasks) > max_subtasks:
        raise OrchestrationError(f"too many subtasks: {len(subtasks)} > {max_subtasks}")

    by_id: dict[str, SubtaskSpec] = {}
    for s in subtasks:
        if not s.id.strip():
            raise OrchestrationError("subtask with empty id")
        if not s.objective.strip():
            raise OrchestrationError(f"subtask {s.id} has empty objective")
        if s.id in by_id:
            raise OrchestrationError(f"duplicate subtask id: {s.id}")
        by_id[s.id] = s
    for s in subtasks:
        for dep in s.depends_on:
            if dep not in by_id:
                raise OrchestrationError(f"subtask {s.id} depends on unknown id: {dep}")

    levels = _kahn_levels(subtasks)
    if len(levels) > max_depth:
        raise OrchestrationError(f"plan too deep: {len(levels)} > {max_depth}")

    for level in levels:
        specs = [by_id[i] for i in level]
        for i, first in enumerate(specs):
            for second in specs[i + 1 :]:
                for wa in first.writes:
                    for wb in second.writes:
                        if paths_collide(wa, wb):
                            raise OrchestrationError(
                                f"write conflict in same level: {first.id}:{wa} vs {second.id}:{wb}"
                            )

    background: set[str] = set()
    inline: set[str] = set()
    for level in levels:
        for task_id in level:
            s = by_id[task_id]
            eligible = s.effective_kind() in _BACKGROUND_KINDS
            if eligible and all(dep in background for dep in s.depends_on):
                background.add(task_id)
            else:
                inline.add(task_id)

    return ValidatedPlan(
        plan=plan,
        levels=levels,
        background=frozenset(background),
        inline=frozenset(inline),
    )


def _kahn_levels(subtasks: list[SubtaskSpec]) -> list[list[str]]:
    """Kahn 法でトポロジカル段を作る。"""
    indegree = {s.id: len(set(s.depends_on)) for s in subtasks}
    dependents: dict[str, list[str]] = {s.id: [] for s in subtasks}
    for s in subtasks:
        for dep in set(s.depends_on):
            dependents[dep].append(s.id)

    order = [s.id for s in subtasks]
    levels: list[list[str]] = []
    ready = [i for i in order if indegree[i] == 0]
    seen = 0
    while ready:
        levels.append(ready)
        seen += len(ready)
        next_ready: set[str] = set()
        for task_id in ready:
            for child in dependents[task_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_ready.add(child)
        ready = [i for i in order if i in next_ready]
    if seen != len(subtasks):
        remaining = sorted(i for i in indegree if indegree[i] > 0)
        raise OrchestrationError(f"dependency cycle among subtasks: {remaining}")
    return levels


class DispatchMode(StrEnum):
    SINGLE = "single"
    PARALLEL = "parallel"


class DispatchDecision(BaseModel):
    mode: DispatchMode
    reason: str
    serial_cost: int = 0
    parallel_cost: int = 0
    capacity: int = 1
    width: int = 0


class SubtaskResult(BaseModel):
    id: str
    status: Literal["ok", "empty", "failed", "timeout", "inline"]
    output: str = ""
    evidence: list[str] = Field(default_factory=list)
    elapsed_ms: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)


class PlanResult(BaseModel):
    results: list[SubtaskResult] = Field(default_factory=list)
    fallback_ids: list[str] = Field(default_factory=list)
    wall_ms: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)


class DecompositionRecord(BaseModel):
    """1 ターン分の分解・振り分けの記録（``TurnSummary.decomposition``）。"""

    decision: DispatchDecision
    plan: TaskPlan | None = None
    result: PlanResult | None = None
