"""ターン結果型・プロンプタ / レンダラ Protocol。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from satsumaestro.engine.events import AssistantEvent, ToolUse
from satsumaestro.engine.repetition import RepetitionVerdict
from satsumaestro.orchestration.plan import DecompositionRecord
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionOutcome
from satsumaestro.session.models import PlanStepRecord, TokenUsage, ToolResultBlock, WriteChunk

if TYPE_CHECKING:
    from satsumaestro.engine.plan_budget import BudgetVerdict, PlanBudget
    from satsumaestro.engine.plan_steps import PlanStep
    from satsumaestro.providers.notices import ProviderNotice


class TurnSummary(BaseModel):
    """1 ターンの実行結果サマリ。"""

    final_text: str
    stop_reason: str
    iterations: int
    usage: TokenUsage
    compacted: bool = False
    health_warning: str | None = None
    assistant_messages: int = 0
    tool_calls: int = 0
    delegated_usage: TokenUsage = Field(default_factory=TokenUsage)
    tool_call_repairs: int = 0
    empty_tool_args: int = 0
    reminders: int = 0
    language_expected: str = ""
    language_detected: str = ""
    language_repaired: bool = False
    prompt_prefix_breaks: int = 0
    turn_prefix_break: bool = False
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
    permission_details: int = 0
    unknown_tool_calls: int = 0
    tool_name_fixes: int = 0
    deferred_autoloads: int = 0
    malformed_tool_calls: int = 0
    plan_exit_nudges: int = 0
    args_only_replies: int = 0
    no_change_reports: int = 0
    plan_steps_run: int = 0
    write_chunks: list[WriteChunk] = Field(default_factory=list)
    tokens_per_s: float = 0.0
    plan_step_records: list[PlanStepRecord] = Field(default_factory=list)
    plan_budget_sendbacks: int = 0
    plan_budget_lines: int = 0
    plan_budget_verdicts: list[str] = Field(default_factory=list)
    plan_budget_forced: bool = False
    last_text: str = ""
    last_thinking: str = ""
    decomposition: DecompositionRecord | None = None


def resolve_output(summary: TurnSummary) -> tuple[str, bool]:
    """委譲成果の本文と「空成果か」を返す（Task / agent が共有）。"""
    if summary.final_text:
        return summary.final_text, False
    if summary.last_text:
        return summary.last_text, False
    if summary.last_thinking:
        return f"(answer found only in reasoning)\n{summary.last_thinking}", False
    return (
        f"(empty final answer after {summary.iterations} iterations, "
        f"{summary.tool_calls} tool calls)",
        True,
    )


USER_DENY_ORIGIN = "user"
DETAIL_REQUEST_ORIGIN = "detail"


@dataclass(frozen=True)
class PermissionDetail:
    """``d`` で出す「何をしようとしているか」。"""

    explanation: str | None
    failure: str | None
    model_text: str = ""
    step: tuple[int, int, str] | None = None
    user_request: str = ""


@runtime_checkable
class PermissionPrompter(Protocol):
    """``ask`` を対話解決する境界（実装は cli 側）。"""

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome: ...


class PlanExitVerdict(BaseModel):
    """``exit_plan_mode`` の承認カードの結果。"""

    approved: bool
    feedback: str | None = None
    note: str | None = None


@runtime_checkable
class PlanExitConfirmer(Protocol):
    """モデルの ``exit_plan_mode`` をユーザーに確認する境界。"""

    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict: ...


NONINTERACTIVE_DENY_REASON = "permission required but running non-interactively"


class DenyPrompter:
    """非対話プロンプタ: ``ask`` を常に deny へ落とす（既定）。"""

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        return PermissionOutcome(decision="deny", reason=NONINTERACTIVE_DENY_REASON)

    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        return PlanExitVerdict(approved=False)


@runtime_checkable
class Renderer(Protocol):
    """ストリーム/ツールイベントの描画境界（実装は cli 側）。"""

    def on_assistant_event(self, event: AssistantEvent) -> None: ...
    def on_tool_start(self, call: ToolUse) -> None: ...
    def on_tool_result(self, block: ToolResultBlock) -> None: ...


@runtime_checkable
class ProgressRenderer(Protocol):
    """長時間フェーズの可視化境界。"""

    def on_phase(self, label: str | None) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...


def phase(render: object, label: str | None) -> None:
    """長時間フェーズのラベルを状態行へ出す（未対応レンダラでは何もしない）。"""
    if isinstance(render, ProgressRenderer):
        render.on_phase(label)


@runtime_checkable
class StreamAbortRenderer(Protocol):
    """応答ストリームが途中で断たれたことを受け取る境界。"""

    def on_stream_aborted(self, timeout_recovery_s: float | None = None) -> None: ...


def stream_aborted(render: object, timeout_recovery_s: float | None = None) -> None:
    """途中で断たれた応答の描画を確定・整理する（未対応レンダラでは何もしない）。"""
    if isinstance(render, StreamAbortRenderer):
        render.on_stream_aborted(timeout_recovery_s)


@runtime_checkable
class StepTimeoutResumeRenderer(Protocol):
    """段の実行中に read タイムアウトが続き、小さく分けて再開するよう依頼したことを受け取る境界。"""

    def on_step_timeout_resume(
        self, timeout_s: float, index: int, total: int, attempt: int, limit: int
    ) -> None: ...


def step_timeout_resume(
    render: object, timeout_s: float, index: int, total: int, attempt: int, limit: int
) -> None:
    """段の再開を依頼したことを描画する。"""
    if isinstance(render, StepTimeoutResumeRenderer):
        render.on_step_timeout_resume(timeout_s, index, total, attempt, limit)
    else:
        stream_aborted(render, timeout_s)


@runtime_checkable
class ProviderNoticeRenderer(Protocol):
    """プロバイダ層の出来事（再試行・モデルの切り替え）を 1 行で受け取る境界。"""

    def on_provider_notice(self, notice: ProviderNotice) -> None: ...


def provider_notice(render: object, notice: ProviderNotice) -> None:
    """プロバイダ層の出来事を画面へ（未対応レンダラでは何もしない）。"""
    if isinstance(render, ProviderNoticeRenderer):
        render.on_provider_notice(notice)


@runtime_checkable
class RepetitionAbortRenderer(Protocol):
    """応答の反復を検知して打ち切ったことを受け取る境界。"""

    def on_repetition_aborted(self, verdict: RepetitionVerdict, recovering: bool) -> None: ...


def repetition_aborted(render: object, verdict: RepetitionVerdict, recovering: bool) -> None:
    """反復で打ち切った応答の描画を確定・整理する（未対応レンダラでは何もしない）。"""
    if isinstance(render, RepetitionAbortRenderer):
        render.on_repetition_aborted(verdict, recovering)


@runtime_checkable
class PlanStepRenderer(Protocol):
    """承認済み計画の段が進んだことを受け取る境界。"""

    def on_plan_step(self, index: int, total: int, text: str) -> None: ...


def plan_step(render: object, index: int, total: int, text: str) -> None:
    """いま何段目に入ったかを描画する（未対応レンダラでは何もしない）。"""
    if isinstance(render, PlanStepRenderer):
        render.on_plan_step(index, total, text)


@runtime_checkable
class PlanBudgetRenderer(Protocol):
    """段の予算の出来事を受け取る境界。"""

    def on_plan_sendback(self, summary: str) -> None: ...
    def on_plan_step_done(self, record: PlanStepRecord) -> None: ...


def plan_sendback(render: object, summary: str) -> None:
    """計画を予算超過で差し戻したことを描画する（未対応レンダラでは何もしない）。"""
    if isinstance(render, PlanBudgetRenderer):
        render.on_plan_sendback(summary)


def plan_step_done(render: object, record: PlanStepRecord) -> None:
    """段を終えたときの見積と実績を描画する（未対応レンダラでは何もしない）。"""
    if isinstance(render, PlanBudgetRenderer):
        render.on_plan_step_done(record)


@runtime_checkable
class ContextNoticeRenderer(Protocol):
    """文脈の量に合わせてハーネスが手を入れたことを受け取る境界。"""

    def on_context_notice(self, text: str) -> None: ...


def context_notice(render: object, text: str) -> None:
    """文脈の量に合わせた手当てを描画する（未対応レンダラでは何もしない）。"""
    if isinstance(render, ContextNoticeRenderer):
        render.on_context_notice(text)


@runtime_checkable
class PlanBudgetAware(Protocol):
    """承認カードに予算の根拠を出せるプロンプタの境界。"""

    def set_plan_budget(
        self,
        budget: PlanBudget,
        steps: Sequence[PlanStep],
        verdict: BudgetVerdict | None,
    ) -> None: ...


@runtime_checkable
class PermissionDetailPrompter(Protocol):
    """``d`` の説明を出して y / n / a で受け直せるプロンプタの境界。"""

    def waiting_for_detail(self) -> AbstractContextManager[object]:
        """説明を待つ間の表示（REPL は「考え中」と同じ回転アイコン）。"""
        ...

    def decide_with_detail(
        self,
        call: ToolUse,
        required: PermissionMode,
        warning: str | None,
        detail: PermissionDetail,
    ) -> PermissionOutcome: ...


@contextmanager
def paused(render: object) -> Iterator[None]:
    """対話入力（権限カード）の間だけ Live を畳む（``console.input`` との競合回避）。"""
    if not isinstance(render, ProgressRenderer):
        yield
        return
    render.pause()
    try:
        yield
    finally:
        render.resume()


class NullRenderer:
    """何も描画しないレンダラ（テスト・非対話既定）。"""

    def on_assistant_event(self, event: AssistantEvent) -> None:
        return None

    def on_tool_start(self, call: ToolUse) -> None:
        return None

    def on_tool_result(self, block: ToolResultBlock) -> None:
        return None

    def on_phase(self, label: str | None) -> None:
        return None

    def pause(self) -> None:
        return None

    def resume(self) -> None:
        return None

    def on_stream_aborted(self, timeout_recovery_s: float | None = None) -> None:
        return None

    def on_step_timeout_resume(
        self, timeout_s: float, index: int, total: int, attempt: int, limit: int
    ) -> None:
        return None

    def on_repetition_aborted(self, verdict: RepetitionVerdict, recovering: bool) -> None:
        return None

    def on_plan_step(self, index: int, total: int, text: str) -> None:
        return None

    def on_plan_sendback(self, summary: str) -> None:
        return None

    def on_plan_step_done(self, record: PlanStepRecord) -> None:
        return None
