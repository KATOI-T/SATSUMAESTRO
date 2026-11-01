"""計画の段実行と統合コンテキスト生成 ``PlanExecutor``。"""

from __future__ import annotations

import math
import re
import time

from satsumaestro.config.models import DecompositionConfig
from satsumaestro.orchestration.ids import make_id
from satsumaestro.orchestration.plan import (
    PlanResult,
    SubtaskResult,
    SubtaskSpec,
    ValidatedPlan,
)
from satsumaestro.orchestration.runner import Orchestrator
from satsumaestro.orchestration.task_ledger import Task, TaskStatus
from satsumaestro.session.models import TokenUsage
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import OrchestrationError

_CHILD_SUFFIX = (
    "\n\n# Report format\n"
    "List every file path / URL you referred to. If you do not know something, write "
    '"unknown"; do not fill the gap with a guess.'
)

_EMPTY_DIAGNOSTIC_PREFIX = "(empty final answer"

_URL = re.compile(r"https?://[^\s)>\"'`]+")
_PATH = re.compile(r"(?<![\w./-])(?:[\w.-]+/)+[\w.-]*\w")
_FILE = re.compile(r"(?<![\w./-])[\w-]+\.(?:py|md|toml|json|rs|ts|js|txt|ya?ml)\b")
_MAX_EVIDENCE = 20


def extract_evidence(text: str) -> list[str]:
    """出力からパス／URL を抽出する（順序保持・重複除去・上限付き）。"""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in (_URL, _PATH, _FILE):
        for match in pattern.findall(text):
            token = match.rstrip(".,、。")
            if token not in seen:
                seen.add(token)
                found.append(token)
            if len(found) >= _MAX_EVIDENCE:
                return found
    return found


class PlanExecutor:
    """段ごとに ``create_task`` → ``wait`` し、検収済みの :class:`PlanResult` を返す。"""

    def __init__(self, orch: Orchestrator, cfg: DecompositionConfig, *, capacity: int = 1) -> None:
        self._orch = orch
        self._cfg = cfg
        self._capacity = max(1, capacity)
        self._counter = 0

    async def run(self, plan: ValidatedPlan, *, plan_id: str | None = None) -> PlanResult:
        if plan_id is None:
            self._counter += 1
            plan_id = make_id("plan", self._counter)
        by_id = {s.id: s for s in plan.plan.subtasks}
        outputs: dict[str, str] = {}
        results: dict[str, SubtaskResult] = {}
        usage = TokenUsage()
        started = time.monotonic()

        for level in plan.levels:
            dispatched: dict[str, str] = {}
            for sid in level:
                spec = by_id[sid]
                if sid in plan.inline:
                    results[sid] = SubtaskResult(id=sid, status="inline")
                    continue
                task = await self._orch.create_task(
                    self._child_prompt(spec, outputs),
                    spec.objective[:80],
                    preset="Explore",
                    plan_id=plan_id,
                )
                dispatched[sid] = task.task_id
            if not dispatched:
                continue
            waves = math.ceil(len(dispatched) / self._capacity)
            budget_s = (
                self._cfg.per_call_budget_s
                * max(by_id[sid].est_calls for sid in dispatched)
                * waves
            )
            await self._orch.wait(list(dispatched.values()), timeout_s=budget_s)
            for sid, tid in dispatched.items():
                task = self._orch.get_task(tid)
                result = self._acceptance(sid, task)
                results[sid] = result
                usage = _add_usage(usage, result.usage)
                if result.status == "ok":
                    outputs[sid] = result.output

        ordered = [results[s.id] for s in plan.plan.subtasks if s.id in results]
        plan_result = PlanResult(
            results=ordered,
            fallback_ids=[r.id for r in ordered if r.status != "ok"],
            wall_ms=int((time.monotonic() - started) * 1000),
            usage=usage,
        )
        self._orch.record_plan(
            plan_id,
            {
                "plan": plan.plan.model_dump(mode="json"),
                "result": plan_result.model_dump(mode="json"),
            },
        )
        return plan_result

    def build_context(self, user_text: str, plan: ValidatedPlan, result: PlanResult) -> str:
        """Leader へ渡す統合コンテキスト。"""
        by_id = {s.id: s for s in plan.plan.subtasks}
        lines = [f"Original question: {user_text}"]
        if plan.plan.reduce_hint:
            lines.append(f"How to combine the results: {plan.plan.reduce_hint}")
        for r in result.results:
            spec = by_id[r.id]
            if r.status == "ok":
                lines.append(f"\n## {r.id}: {spec.objective}")
                if spec.acceptance:
                    lines.append(f"Done when: {spec.acceptance}")
                if r.evidence:
                    lines.append(f"Evidence: {', '.join(r.evidence)}")
                lines.append(r.output)
            else:
                lines.append(
                    f"\n## {r.id}: {spec.objective}\n"
                    f"Not finished ({r.status}): look into this yourself."
                )
        return "\n".join(lines)

    def _child_prompt(self, spec: SubtaskSpec, outputs: dict[str, str]) -> str:
        parts = [spec.objective]
        deps = [d for d in spec.depends_on if d in outputs]
        if deps:
            parts.append("\n# Background (results of earlier subtasks)")
            parts.extend(f"## {d}\n{outputs[d]}" for d in deps)
        return "\n".join(parts) + _CHILD_SUFFIX

    def _acceptance(self, sid: str, task: Task) -> SubtaskResult:
        """終端 Task の検収（文字列規則のみ）。"""
        elapsed = (
            task.finished_at - task.started_at
            if task.finished_at is not None and task.started_at is not None
            else 0
        )
        if not task.status.finished:
            try:
                self._orch.stop_task(task.task_id)
            except OrchestrationError:
                pass
            if task.started_at is not None:
                elapsed = max(elapsed, now_ms() - task.started_at)
            return SubtaskResult(id=sid, status="timeout", elapsed_ms=elapsed, usage=task.usage)
        output = task.output.strip()
        if task.status is not TaskStatus.COMPLETED:
            empty = output.startswith(_EMPTY_DIAGNOSTIC_PREFIX)
            return SubtaskResult(
                id=sid,
                status="empty" if empty else "failed",
                elapsed_ms=elapsed,
                usage=task.usage,
            )
        if not output:
            return SubtaskResult(id=sid, status="empty", elapsed_ms=elapsed, usage=task.usage)
        evidence = extract_evidence(output)
        if not evidence:
            return SubtaskResult(
                id=sid, status="failed", output=output, elapsed_ms=elapsed, usage=task.usage
            )
        return SubtaskResult(
            id=sid,
            status="ok",
            output=output,
            evidence=evidence,
            elapsed_ms=elapsed,
            usage=task.usage,
        )


def _add_usage(a: TokenUsage, b: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read=a.cache_read + b.cache_read,
        cache_write=a.cache_write + b.cache_write,
    )
