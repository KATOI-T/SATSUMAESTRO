"""評価ハーネスのドメイン型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from satsumaestro.i18n.text import ui
from satsumaestro.session.models import PlanStepRecord, TokenUsage, WriteChunk

EvalPermissionMode = Literal["read-only", "workspace-write"]

TAXONOMY: dict[str, tuple[str, ...]] = {
    "WHAT": ("concept-definition", "architecture-exploration", "dependency-tracing"),
    "WHY": ("purpose-exploration", "design-rationale", "performance"),
    "WHERE": ("feature-location", "data-control-flow", "identifier-location"),
    "HOW": ("system-design", "algorithm-implementation", "api-framework-support"),
}


def is_valid_category(category: str) -> bool:
    """``"WHERE/identifier-location"`` 形式がタクソノミーに存在するか。"""
    level1, _, level2 = category.partition("/")
    return level2 in TAXONOMY.get(level1, ())


class EvalChecks(BaseModel):
    """決定的採点の基準（言語非依存の安定トークンを使う）。"""

    model_config = ConfigDict(extra="forbid")

    contains_all: list[str] = Field(default_factory=list)
    contains_any: list[str] = Field(default_factory=list)
    regex: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    file_min_lines: dict[str, int] = Field(default_factory=dict)


class EvalTask(BaseModel):
    """コード解析タスクの台本。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    prompt: str
    prompt_en: str = ""
    fixture: str
    category: str = ""
    checks: EvalChecks = Field(default_factory=EvalChecks)
    max_iterations: int | None = None
    permission_mode: EvalPermissionMode = "read-only"

    @model_validator(mode="after")
    def _generated_files_need_write(self) -> EvalTask:
        if self.checks.file_min_lines and self.permission_mode == "read-only":
            mode = 'permission_mode = "workspace-write"'
            raise ValueError(ui("evals.models.a_task_that_uses_checks", mode=mode))
        return self


class TaskResult(BaseModel):
    """1 タスクの実行・採点結果。"""

    task_id: str
    passed: bool
    category: str = ""
    failures: list[str] = Field(default_factory=list)
    error: str = ""
    connection_lost: bool = False
    final_text: str = ""
    empty_final: bool = False
    iterations: int = 0
    tool_calls: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_ms: int = 0
    slept_ms: int = 0
    iteration_limit: int = 0
    compacted: bool = False
    tool_call_repairs: int = 0
    empty_tool_args: int = 0
    reminders: int = 0
    decomposition_reason: str = ""
    delegated_usage: TokenUsage = Field(default_factory=TokenUsage)
    plan_exits: int = 0
    language_expected: str = ""
    language_detected: str = ""
    language_repaired: bool = False
    prompt_prefix_breaks: int = 0
    timeout_recoveries: int = 0
    timeout_resumes: int = 0
    tool_call_rewrites: int = 0
    repetition_aborts: int = 0
    syntax_rejects: int = 0
    syntax_warnings: int = 0
    edit_mismatches: int = 0
    edit_reindents: int = 0
    edit_stale_notes: int = 0
    related_code_notes: int = 0
    overwrite_rejects: int = 0
    overwrite_overrides: int = 0
    step_write_rejects: int = 0
    step_write_overrides: int = 0
    plan_exit_not_active: int = 0
    plan_exit_step_done: int = 0
    mkdir_skips: int = 0
    output_delete_blocks: int = 0
    state_write_blocks: int = 0
    secret_read_checks: int = 0
    first_fetch_asks: int = 0
    lsp_start_asks: int = 0
    tool_result_cuts: int = 0
    written_bodies_shrunk: int = 0
    omitted_copies: int = 0
    presend_compactions: int = 0
    context_truncations: int = 0
    prefill_waits: int = 0
    unknown_tool_calls: int = 0
    malformed_tool_calls: int = 0
    args_only_replies: int = 0
    no_change_reports: int = 0
    tool_name_fixes: int = 0
    deferred_autoloads: int = 0
    plan_exit_nudges: int = 0
    plan_steps_run: int = 0
    defect_record: str = ""
    write_chunks: list[WriteChunk] = Field(default_factory=list)
    tokens_per_s: float = 0.0
    plan_step_records: list[PlanStepRecord] = Field(default_factory=list)
    plan_budget_sendbacks: int = 0
    plan_budget_lines: int = 0
    plan_budget_verdicts: list[str] = Field(default_factory=list)
    plan_budget_forced: bool = False


class Comparison(BaseModel):
    """基準（課題ごとの履歴）との対応付けと判定。"""

    baseline: str = ""
    gate_tasks: list[str] = Field(default_factory=list)
    maintained: list[str] = Field(default_factory=list)
    improved: list[str] = Field(default_factory=list)
    regressed: list[str] = Field(default_factory=list)
    still_failing: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    flaky_failed: list[str] = Field(default_factory=list)
    structural_errors: list[str] = Field(default_factory=list)
    over_budget: list[str] = Field(default_factory=list)
    connection_lost: list[str] = Field(default_factory=list)
    rerun_cases: list[str] = Field(default_factory=list)
    decision: Literal["pass", "rerun"] = "pass"


class EvalReport(BaseModel):
    """スイート全体のレポート。"""

    agent: str = "satsuma"
    model: str = ""
    started_at_ms: int
    results: list[TaskResult] = Field(default_factory=list)
    suite: str = ""
    prompt_lang: str = ""
    comparison: Comparison | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.results else 0.0
