"""評価実行器。"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from satsumaestro.evals.agents import EvalAgent
from satsumaestro.evals.checks import grade
from satsumaestro.evals.defects import detect_defects, save_defect_record
from satsumaestro.evals.models import EvalReport, EvalTask, TaskResult
from satsumaestro.providers.detect import reset_capabilities
from satsumaestro.providers.retry import is_connection_failure
from satsumaestro.util.clock import monotonic_ms, now_ms
from satsumaestro.util.errors import SatsumaError

_log = logging.getLogger(__name__)

_SLEEP_MIN_MS = 2_000


def _elapsed(started_wall: int, started_mono: int) -> tuple[int, int]:
    """課題の所要（単調時計の差）と、その間に機械が眠っていた時間（壁時計との差）。"""
    duration = monotonic_ms() - started_mono
    gap = now_ms() - started_wall - duration
    return duration, gap if gap >= _SLEEP_MIN_MS else 0


class EvalRunner:
    """タスク列を隔離実行して :class:`EvalReport` を返す。"""

    def __init__(self, agent: EvalAgent, defects_dir: Path | None = None) -> None:
        self._agent = agent
        self._defects_dir = defects_dir

    async def run(self, tasks: list[EvalTask]) -> EvalReport:
        report = EvalReport(agent=self._agent.name, model=self._agent.model, started_at_ms=now_ms())
        for task in tasks:
            report.results.append(await self._run_task(task))
        return report

    async def _run_task(self, task: EvalTask) -> TaskResult:
        reset_capabilities()
        workspace = Path(tempfile.mkdtemp(prefix=f"satsuma-eval-{task.id}-"))
        try:
            result = await self._execute(task, workspace)
            return self._record_defects(task, result, workspace)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _record_defects(self, task: EvalTask, result: TaskResult, workspace: Path) -> TaskResult:
        """実行例外・経路の異常があれば不具合記録を書き、そのパスを結果へ載せる。"""
        if self._defects_dir is None:
            return result
        defects = detect_defects(result, self._agent.name)
        if not defects:
            return result
        try:
            record = save_defect_record(
                self._defects_dir,
                task,
                result,
                defects,
                agent_name=self._agent.name,
                model=self._agent.model,
                workspace=workspace,
            )
        except OSError as e:
            _log.warning("eval task %s: could not write the defect record: %s", task.id, e)
            return result
        return result.model_copy(update={"defect_record": str(record)})

    async def _execute(self, task: EvalTask, workspace: Path) -> TaskResult:
        started, started_mono = now_ms(), monotonic_ms()
        shutil.copytree(task.fixture, workspace, dirs_exist_ok=True)
        try:
            answer = await self._agent.run(
                task.prompt,
                workspace,
                task.max_iterations,
                permission_mode=task.permission_mode,
            )
        except SatsumaError as e:
            _log.warning("eval task %s (%s) failed: %s", task.id, self._agent.name, e)
            duration, slept = _elapsed(started, started_mono)
            return TaskResult(
                task_id=task.id,
                passed=False,
                category=task.category,
                error=f"{type(e).__name__}: {e}",
                connection_lost=is_connection_failure(e),
                duration_ms=duration,
                slept_ms=slept,
            )
        if answer.error:
            _log.warning("eval task %s (%s) failed: %s", task.id, self._agent.name, answer.error)
        failures = [] if answer.error else grade(answer.final_text, task.checks, workspace)
        duration, slept = _elapsed(started, started_mono)
        return TaskResult(
            task_id=task.id,
            passed=not failures and not answer.error,
            category=task.category,
            failures=failures,
            duration_ms=duration,
            slept_ms=slept,
            **answer.model_dump(),
        )
