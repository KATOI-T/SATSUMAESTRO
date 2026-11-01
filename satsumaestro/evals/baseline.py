"""課題ごとの履歴を基準にした対応付け。"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from satsumaestro.evals.defects import detect_defects
from satsumaestro.evals.models import Comparison, EvalReport, TaskResult
from satsumaestro.evals.suite import PLAN_SUFFIX
from satsumaestro.i18n.text import ui
from satsumaestro.util.errors import ConfigError

STABLE_MIN_RUNS = 5
STABLE_MIN_PASS_RATE = 0.95
DEFAULT_KEEP = 10
MAD_MULTIPLIER = 3.0


class TaskHistory(BaseModel):
    """1 課題の直近 N 本（古い順）。"""

    passed: list[bool] = Field(default_factory=list)
    duration_s: list[float] = Field(default_factory=list)
    iterations: list[int] = Field(default_factory=list)
    tool_calls: list[int] = Field(default_factory=list)
    output_tokens: list[int] = Field(default_factory=list)

    @property
    def runs(self) -> int:
        return len(self.passed)

    @property
    def pass_rate(self) -> float:
        return sum(self.passed) / self.runs if self.runs else 0.0

    @property
    def stable(self) -> bool:
        """門の条件: 履歴 5 本以上・合格率 95% 以上。"""
        return self.runs >= STABLE_MIN_RUNS and self.pass_rate >= STABLE_MIN_PASS_RATE

    @property
    def majority_pass(self) -> bool:
        """履歴では pass 側か（対応付けの「before」）。"""
        return self.runs > 0 and self.pass_rate >= 0.5

    def append(self, result: TaskResult, *, keep: int) -> None:
        self.passed.append(bool(result.passed))
        self.duration_s.append(round(result.duration_ms / 1000, 1))
        self.iterations.append(int(result.iterations))
        self.tool_calls.append(int(result.tool_calls))
        self.output_tokens.append(int(result.usage.output_tokens))
        for name in ("passed", "duration_s", "iterations", "tool_calls", "output_tokens"):
            values = getattr(self, name)
            if len(values) > keep:
                del values[: len(values) - keep]


class Baseline(BaseModel):
    """スイート 1 つぶんの基準（課題 ID → 履歴）。"""

    schema_version: int = 1
    suite: str = ""
    model: str = ""
    prompt_lang: str = ""
    sources: list[str] = Field(default_factory=list)
    tasks: dict[str, TaskHistory] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Baseline:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise ConfigError(
                ui("evals.baseline.cannot_read_the_baseline", path=path, error=e)
            ) from e
        except (ValidationError, ValueError) as e:
            raise ConfigError(ui("evals.baseline.invalid_baseline", path=path, error=e)) from e

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def add_report(self, report: EvalReport, *, keep: int = DEFAULT_KEEP, source: str = "") -> int:
        """レポートの課題を履歴へ積む。"""
        if self.model and report.model and self.model != report.model:
            raise ConfigError(
                ui(
                    "evals.baseline.the_baseline_s_model_differs",
                    model=self.model,
                    model_2=report.model,
                )
            )
        if self.sources and self.prompt_lang != report.prompt_lang:
            original = ui("evals.baseline.original")
            ours, theirs = self.prompt_lang or original, report.prompt_lang or original
            raise ConfigError(
                ui("evals.baseline.the_baseline_s_request_version", ours=ours, theirs=theirs)
            )
        if not self.sources:
            self.prompt_lang = report.prompt_lang
        if report.model and not self.model:
            self.model = report.model
        if report.suite and not self.suite:
            self.suite = report.suite
        added = 0
        for result in report.results:
            if result.connection_lost:
                continue
            self.tasks.setdefault(result.task_id, TaskHistory()).append(result, keep=keep)
            added += 1
        if source:
            self.sources.append(source)
        return added

    def gate_tasks(self) -> list[str]:
        return sorted(task_id for task_id, h in self.tasks.items() if h.stable)


def _agent_name_for(report: EvalReport, task_id: str) -> str:
    """不具合記録の判定に渡すエージェント名（plan 経路の課題は ``+plan`` を含める）。"""
    if task_id.endswith(PLAN_SUFFIX) and PLAN_SUFFIX not in report.agent:
        return report.agent + PLAN_SUFFIX
    return report.agent


def _over_budget(name: str, value: int, history: list[int]) -> str | None:
    """``value`` が履歴の median + 3 MAD を超えていれば説明を返す。"""
    if len(history) < STABLE_MIN_RUNS:
        return None
    med = statistics.median(history)
    mad = statistics.median(abs(v - med) for v in history)
    if mad <= 0:
        return None
    limit = med + MAD_MULTIPLIER * mad
    if value > limit:
        return f"{name} {value} > {med:g} + {MAD_MULTIPLIER:g}×{mad:g}"
    return None


def compare(report: EvalReport, baseline: Baseline, *, path: str = "") -> Comparison:
    """レポートを基準と対応付け、再実行の対象と判定を返す。"""
    comp = Comparison(baseline=path, gate_tasks=baseline.gate_tasks())
    rerun: list[str] = []
    for r in report.results:
        if r.connection_lost:
            comp.connection_lost.append(r.task_id)
            rerun.append(r.task_id)
            continue
        history = baseline.tasks.get(r.task_id)
        if history is None or history.runs == 0:
            comp.new.append(r.task_id)
        elif r.passed:
            (comp.maintained if history.majority_pass else comp.improved).append(r.task_id)
        elif history.majority_pass:
            if history.stable:
                comp.regressed.append(r.task_id)
                rerun.append(r.task_id)
            else:
                comp.flaky_failed.append(r.task_id)
        else:
            comp.still_failing.append(r.task_id)

        for defect in detect_defects(r, _agent_name_for(report, r.task_id)):
            comp.structural_errors.append(f"{r.task_id}: {defect.kind}")
            rerun.append(r.task_id)

        if history is not None:
            for name, value, values in (
                ("iterations", r.iterations, history.iterations),
                ("tool_calls", r.tool_calls, history.tool_calls),
            ):
                over = _over_budget(name, value, values)
                if over is not None:
                    comp.over_budget.append(f"{r.task_id}: {over}")
                    rerun.append(r.task_id)

    comp.rerun_cases = list(dict.fromkeys(rerun))
    comp.decision = "rerun" if comp.rerun_cases else "pass"
    return comp


@dataclass(frozen=True)
class HistoryRow:
    """``eval-baseline show`` の 1 課題 1 行（runs / 合格率 / 門 / 中央値）。"""

    task: str
    runs: int
    pass_rate: float
    gate: bool
    iterations: float
    tool_calls: float
    duration_s: float


def describe_history(baseline: Baseline) -> list[HistoryRow]:
    rows: list[HistoryRow] = []
    for task_id in sorted(baseline.tasks):
        h = baseline.tasks[task_id]
        rows.append(
            HistoryRow(
                task=task_id,
                runs=h.runs,
                pass_rate=h.pass_rate,
                gate=h.stable,
                iterations=float(statistics.median(h.iterations)) if h.iterations else 0.0,
                tool_calls=float(statistics.median(h.tool_calls)) if h.tool_calls else 0.0,
                duration_s=float(statistics.median(h.duration_s)) if h.duration_s else 0.0,
            )
        )
    return rows


def load_report(path: Path) -> EvalReport:
    """レポート JSON を読む（``eval-baseline add`` 用。壊れていれば :class:`ConfigError`）。"""
    try:
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(ui("evals.baseline.cannot_read_the_report", path=path, error=e)) from e
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        raise ConfigError(ui("evals.baseline.invalid_report", path=path, error=e)) from e
