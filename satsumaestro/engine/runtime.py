"""会話ループエンジン。"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import math
import os
import re
import signal
import statistics
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import urlsplit

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.engine import written_bodies
from satsumaestro.engine.compaction import CompactionLimits, Compactor, limits_for
from satsumaestro.engine.context_budget import (
    ResultCut,
    estimate_upper,
    fit_history_results,
    fit_tool_results,
    input_limit,
    message_floor,
    message_tokens,
    request_floor,
    request_tokens,
    result_budget,
    tool_result_tokens,
)
from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    ToolUse,
    closing_stream,
    fold_events,
)
from satsumaestro.engine.permission_explain import explain_call, explain_question, explain_request
from satsumaestro.engine.plan_approval import is_plan_approval_text
from satsumaestro.engine.plan_budget import (
    MAX_SENDBACKS,
    PlanBudget,
    check_plan_budget,
    compute_plan_budget,
)
from satsumaestro.engine.plan_request import (
    PLAN_FILE_READ_MISSING,
    PLAN_MODE_OFF_NOTE,
    ensure_plan_request,
    plan_left_open,
)
from satsumaestro.engine.plan_steps import (
    STEP_WRITE_REJECT_LIMIT,
    compacted_step_note,
    first_step_note,
    is_blocked_reply,
    next_step_note,
    parse_plan_steps,
    resume_step_note,
    retry_step_note,
    step_write_reject_message,
    timeout_resume_step_note,
    wants_resume,
)
from satsumaestro.engine.prefill import PrefillClock, keep_alive_seconds
from satsumaestro.engine.recovery import (
    Fallback,
    Fault,
    PlaybookOutcome,
    RecoveryLedger,
    Remedy,
    RemedyRunner,
    SessionAbortError,
    fault_of,
    playbook_for,
    run_playbook,
)
from satsumaestro.engine.reminders import (
    PLAN_MODE_DENY_PREFIX,
    ReminderState,
    args_only_reply_note,
    background_done_note,
    empty_final_approved_note,
    empty_final_note,
    empty_final_plan_note,
    format_reminder,
    language_mismatch_note,
    language_note,
    no_change_report_note,
    plan_exit_nudge_note,
    plan_missing_plan_note,
    repetition_note,
    timeout_note,
    tool_call_rewrite_note,
)
from satsumaestro.engine.repetition import RepetitionAbort, RepetitionDetector
from satsumaestro.engine.reply_checks import (
    detect_bare_tool_args,
    no_change_report,
    strip_code_fence,
)
from satsumaestro.engine.tool_call_repair import ToolCallRepairPolicy, ToolCallRepairState
from satsumaestro.engine.tool_check import check_tool_executor
from satsumaestro.engine.trace import (
    AutoAllowed,
    ModelReplied,
    Notify,
    RecoveryAttempt,
    ToolDispatched,
    ToolReturned,
    TurnAborted,
    TurnBegan,
    TurnEnded,
    TurnObserver,
    guarded,
)
from satsumaestro.engine.turn import (
    DETAIL_REQUEST_ORIGIN,
    NONINTERACTIVE_DENY_REASON,
    USER_DENY_ORIGIN,
    DenyPrompter,
    NullRenderer,
    PermissionDetail,
    PermissionDetailPrompter,
    PermissionPrompter,
    PlanBudgetAware,
    PlanExitConfirmer,
    PlanExitVerdict,
    Renderer,
    TurnSummary,
    context_notice,
    paused,
    phase,
    plan_sendback,
    plan_step,
    plan_step_done,
    provider_notice,
    repetition_aborted,
    step_timeout_resume,
    stream_aborted,
)
from satsumaestro.engine.window_breakdown import WindowBreakdown, window_breakdown
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.state import LanguageState
from satsumaestro.i18n.text import ui
from satsumaestro.prompt.builder import SystemPromptBuilder, TurnContext, git_change_note
from satsumaestro.prompt.instructions import instruction_tokens
from satsumaestro.providers import notices, timing
from satsumaestro.providers.base import ApiRequest, Provider, ProviderKind
from satsumaestro.providers.retry import (
    is_context_overflow,
    is_read_timeout,
    is_tool_call_parse_error,
)
from satsumaestro.security import output_guard, state_write_guard
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.auto_allow import SecretMatcher, evaluate_auto_allow, secret_path_hit
from satsumaestro.security.bash_pipeline import review_command
from satsumaestro.security.deletion_guard import (
    PROHIBITED_DELETION_GUIDANCE,
    bash_targets,
    check_bash,
    check_path,
    path_targets,
)
from satsumaestro.security.enforcer import PermissionMode, classify, mode_le
from satsumaestro.security.jail import PATH_ESCAPE_PREFIX, WorkspaceJail
from satsumaestro.security.policy import PermissionOutcome, PermissionPolicy
from satsumaestro.security.self_invocation_guard import (
    PROHIBITED_SELF_INVOCATION_GUIDANCE,
    PROHIBITED_SELF_INVOCATION_REPEAT,
)
from satsumaestro.security.self_invocation_guard import (
    check_bash as check_self_invocation,
)
from satsumaestro.session.models import (
    ConversationMessage,
    MainMode,
    ModeState,
    OmittedBody,
    PlanStepRecord,
    Session,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
    WriteChunk,
    plan_file_relpath,
)
from satsumaestro.session.store import JSONLSessionStore
from satsumaestro.tools import (
    edit_match,
    mkdir_hint,
    overwrite_guard,
    read_ledger,
    related_code,
    syntax_check,
    tool_names,
    unavailable_tools,
)
from satsumaestro.tools.aliases import name_key
from satsumaestro.tools.base import ToolContext, ToolResult
from satsumaestro.tools.builtins.plan_mode import (
    FLAG_EXIT_NOT_ACTIVE,
    FLAG_EXIT_STEP_DONE,
    PLAN_EXIT_APPROVED_IMPLEMENT,
    PLAN_EXIT_APPROVED_NOTE_WHILE_STEPPING,
    PLAN_EXIT_APPROVED_WITH_NOTE,
    PLAN_EXIT_NO_PLAN,
    PLAN_EXIT_NOT_APPROVED,
    PLAN_EXIT_REVISE,
    PLAN_EXIT_STEP_DONE_PREFIX,
)
from satsumaestro.tools.lsp.servers import language_for_path, resolve_server_command
from satsumaestro.tools.read_ledger import ReadLedger
from satsumaestro.tools.registry import ToolRegistry
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import (
    ContextOverflowError,
    ContextTruncatedError,
    MaxIterationsError,
    PathEscapeError,
    PlanExitLoopError,
    ProviderError,
    RepetitionLoopError,
    StateDirError,
    ToolCallingUnsupportedError,
    ToolError,
    UnknownToolLoopError,
)
from satsumaestro.util.lang import (
    UNDETERMINED,
    detect_language,
    requests_other_language,
    strip_non_prose,
    with_language_directive,
)
from satsumaestro.util.proc import BackgroundProcessRegistry
from satsumaestro.util.tokens import estimate_tokens

if TYPE_CHECKING:
    from satsumaestro.engine.context_fit import ContextFitResult
    from satsumaestro.engine.subagent import RuntimeFactory
    from satsumaestro.mcp.manager import MCPManager
    from satsumaestro.orchestration.plan import DecompositionRecord
    from satsumaestro.rag.service import RAGService

_log = logging.getLogger(__name__)


class DecompositionHook(Protocol):
    """自動タスク分解の前段フックの境界（具象は orchestration 側 ``DecompositionCoordinator``）。"""

    async def before_turn(self, user_text: str) -> str | None: ...

    def take_record(self) -> DecompositionRecord | None: ...


_UNKNOWN_REQUIRED_MODE = PermissionMode.DANGER_FULL_ACCESS

_PLAN_MODE_TOOLS = frozenset({"enter_plan_mode", "exit_plan_mode"})
PLAN_EXIT_LOOP_LIMIT = 3
_CHUNKED_WRITE_TOOLS = frozenset({"write_file", "append_file"})
_SECRET_FILE_TOOLS = {
    "read_file": "path",
    "edit_file": "path",
    "write_file": "path",
    "append_file": "path",
    "notebook_edit": "notebook_path",
}
_FETCH_NONINTERACTIVE = (
    f"{NONINTERACTIVE_DENY_REASON}: web_fetch asks the user once per session before its first "
    "fetch. Only the user can turn this off, with the setting web_fetch.confirm "
    "(not a tool argument)."
)
_LSP_NONINTERACTIVE = (
    f"{NONINTERACTIVE_DENY_REASON}: lsp asks the user once per session before it starts a "
    "language server for a language. Only the user can turn this off, with the setting "
    "lsp.confirm_start (not a tool argument)."
)


@dataclass
class _TurnCounters:
    """ターン寿命の計測カウンタ。"""

    empty_tool_args: int = 0
    response_s: float = 0.0
    tokens_per_s: list[float] = field(default_factory=list)
    write_chunks: list[WriteChunk] = field(default_factory=list)
    edit_lines: list[int] = field(default_factory=list)
    plan_step_records: list[PlanStepRecord] = field(default_factory=list)
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
    stream_timing: timing.StreamTiming | None = None
    plan_exit_not_active: int = 0
    plan_exit_step_done: int = 0
    unknown_tool_calls: int = 0
    tool_name_fixes: int = 0
    deferred_autoloads: int = 0
    malformed_tool_calls: int = 0
    unknown_streak: int = 0
    unknown_name: str = ""
    unknown_loop: str = ""
    self_invocation_blocks: int = 0
    args_only_replies: int = 0
    mutating_attempted: int = 0
    no_change_reports: int = 0
    user_denied: str | None = None
    permission_details: int = 0


_WINDOW_MARGIN = 1_000
_TRUNCATION_SLACK = 500
_SENT_FLOOR_SHARE = 0.95


def _cut_results(
    messages: list[ConversationMessage], start: int, reduce_by: int
) -> list[ResultCut]:
    """``start`` から後ろのツールの結果を、見積もりで ``reduce_by`` だけ減るように切る。"""
    tail = messages[start:]
    budget = tool_result_tokens(tail) - reduce_by
    if budget <= 0:
        return []
    fitted, cuts = fit_history_results(tail, budget)
    if cuts:
        messages[start:] = fitted
    return cuts


@dataclass(frozen=True)
class _SentMark:
    """直近の応答を生んだ要求の大きさ。"""

    count: int
    last: ConversationMessage
    input_tokens: int
    digest: str = ""
    model: str = ""
    at: float = 0.0
    floor: bool = True


@dataclass
class _PlanExitState:
    """plan の承認へ繋ぐ差し戻しの判定材料（ターン寿命）。"""

    wrote_plan: bool = False
    write_denied: bool = False
    nudged: bool = False
    approved: bool = False


class _RuntimeRemedies:
    """本番の :class:`RemedyRunner`。"""

    async def apply(self, remedy: Remedy, fault: Fault) -> bool:
        return False


class ConversationRuntime:
    """1 セッション分の会話ループ実行器。"""

    def __init__(
        self,
        *,
        session: Session,
        provider: Provider,
        tools: ToolRegistry,
        policy: PermissionPolicy | PermissionResolver,
        prompt_builder: SystemPromptBuilder,
        store: JSONLSessionStore,
        hooks: HookRunner,
        config: RuntimeConfig,
        compactor: Compactor | None = None,
        mcp: MCPManager | None = None,
        rag: RAGService | None = None,
        recovery_runner: RemedyRunner | None = None,
        runtime_factory: RuntimeFactory | None = None,
        depth: int = 0,
        orchestrator: object | None = None,
        decomposition: DecompositionHook | None = None,
        background: BackgroundProcessRegistry | None = None,
        lsp: object | None = None,
        language: LanguageState | None = None,
    ) -> None:
        self.session = session
        self.provider = provider
        if language is None:
            language = (
                runtime_factory.language
                if runtime_factory is not None
                else LanguageState.from_config(config.language)
            )
        self.language = language
        self.language.bind(session)
        self.tools = tools
        self.policy = policy
        self.prompt_builder = prompt_builder
        self.store = store
        self.hooks = hooks
        self.config = config
        self._secret_matcher = SecretMatcher.from_config(config.permission)
        self._first_use_local: set[str] = set()
        self.compactor = compactor or Compactor()
        self.mcp = mcp
        self.rag = rag
        self.orchestrator = orchestrator
        self.decomposition = decomposition
        self.background = background
        self.lsp = lsp
        self.recovery_runner: RemedyRunner = recovery_runner or _RuntimeRemedies()
        self._recovery_lock = asyncio.Lock()
        self._expanded: set[str] = set(session.expanded_tools)
        self._last_request: ApiRequest | None = None
        self._missing_plan_nudges = 0
        self._mkdir_hinted: set[Path] = set()
        self._session_outputs = output_guard.SessionOutputs()
        self._rate_samples: list[tuple[int, float]] = []
        self.last_plan_budget: PlanBudget | None = None
        self._sent: _SentMark | None = None
        self._estimate_ratio = 1.0
        self._fixed_tokens = 0
        self.window_unconfirmed = False
        self.fit_base: tuple[int, int] | None = None
        self._window_notes: list[str] = []
        self._prefill = PrefillClock()
        self.aborted_summary: TurnSummary | None = None
        self._turn_snapshot: Callable[[], TurnSummary] | None = None
        self._plan_budget_sendbacks = 0
        self._step_write_rejects: dict[int, int] = {}
        self._plan_budget_verdicts: list[str] = []
        self._plan_budget_sent: list[str] = []
        self._plan_budget_forced = False
        self._iteration_limit = 0
        self._runtime_factory = runtime_factory
        self.cwd = Path(session.workspace_root)
        self._ctx = ToolContext(
            workspace_root=self.cwd,
            jail=WorkspaceJail(
                self.cwd, visible_state=[self.cwd / plan_file_relpath(session.session_id)]
            ),
            config=config,
            base_mode=self._base_mode(),
            session=session,
            mcp=mcp,
            rag=rag,
            runtime_factory=runtime_factory,
            depth=depth,
            orchestrator=orchestrator,
            background=background,
            registry=tools,
            lsp=lsp,
            reads=ReadLedger(),
        )
        self.session.mode = self.session.mode.model_copy(update={"sub": self._base_mode()})

    def set_mode(
        self, main: MainMode | None = None, sub: PermissionMode | None = None
    ) -> ModeState:
        """ユーザー操作（/mode・Shift+Tab・--mode）からの唯一の遷移口。"""
        cur = self.session.mode
        self.session.mode = ModeState(main=main or cur.main, sub=sub or cur.sub)
        if main == "plan" and cur.main != "plan":
            self._clear_plan_steps()
        self._apply_mode_effects()
        return self.session.mode

    def _apply_mode_effects(self) -> None:
        """``session.mode.sub`` を policy の基準のモードと ToolContext へ反映する。"""
        base = self.policy.base if isinstance(self.policy, PermissionResolver) else self.policy
        base.mode = self.session.mode.sub
        self._ctx.base_mode = self.session.mode.sub

    def plan_file_relpath(self) -> str:
        """plan mode 中にモデルが書く計画ファイル（workspace 相対）。"""
        return plan_file_relpath(self.session.session_id)

    def approve_plan_from_text(self, user_text: str) -> bool:
        """ユーザー発話による plan 承認。"""
        if not self._plan_mode_active():
            return False
        if not (self.cwd / self.plan_file_relpath()).is_file():
            return False
        if not is_plan_approval_text(user_text):
            return False
        self.set_mode(main="code")
        self._stage_plan(self._read_plan_file())
        return True

    def _step_budget_enabled(self) -> bool:
        """段の予算が有効か。"""
        return self.config.guidance.plan_steps and self.config.guidance.step_budget

    def _plan_budget(self) -> PlanBudget:
        """いまの標本から 1 段の予算を出す（釘付け設定があればそれが勝つ）。"""
        return compute_plan_budget(
            self.config.read_timeout_s(),
            self._rate_samples,
            self.config.guidance.step_budget_lines,
            self.config.compaction.context_window,
        )

    def _stage_plan(self, plan_text: str | None) -> bool:
        """承認された計画を段へ割って session へ載せる（2 段以上のときだけ True）。"""
        if not self.config.guidance.plan_steps:
            return False
        steps = parse_plan_steps(plan_text)
        if len(steps) < 2:
            return False
        self.session.plan_steps = [s.text for s in steps]
        self.session.plan_step_index = 0
        self._step_write_rejects.clear()
        if self._step_budget_enabled():
            budget = self.last_plan_budget or self._plan_budget()
            self.session.plan_step_estimates = [s.estimated_lines for s in steps]
            self.session.plan_budget_lines = budget.lines
        else:
            self.session.plan_step_estimates = []
            self.session.plan_budget_lines = 0
        return True

    def _clear_plan_steps(self) -> None:
        """段の状態を捨てる（最終段を終えた・別の計画を立て直す・/clear）。"""
        self.session.plan_steps = []
        self.session.plan_step_index = 0
        self.session.plan_step_estimates = []
        self.session.plan_budget_lines = 0
        self._step_write_rejects.clear()

    async def _permission_detail(
        self, call: ToolUse, input_dict: dict[str, Any], warning: str | None
    ) -> PermissionDetail:
        """権限カードの d で出す説明と、説明が取れないときの事実を集める。"""
        last = self.session.messages[-1] if self.session.messages else None
        model_text = _text_of(last) if last is not None and last.role == "assistant" else ""
        step = self._current_step()
        history = self.session.prompt_history
        user_request = history[-1] if history else ""

        def detail(explanation: str | None, failure: str | None) -> PermissionDetail:
            return PermissionDetail(
                explanation=explanation,
                failure=failure,
                model_text=model_text,
                step=step,
                user_request=user_request,
            )

        base = self._last_request
        if not self.config.guidance.permission_explain or base is None or last is None:
            return detail(None, "off")
        question = explain_question(
            call,
            input_dict,
            warning=warning,
            step=step,
            model_text=model_text,
            language=self.language.ui_language,
        )
        req = explain_request(base, list(self.session.messages[:-1]), question)
        task = asyncio.ensure_future(explain_call(self.provider, req))
        return detail(*await _interruptible(task))

    def _current_step(self) -> tuple[int, int, str] | None:
        """作業中の段 ``(番号 1..n, 総数, 本文)``。"""
        steps = self.session.plan_steps
        index = self.session.plan_step_index
        if not steps or not 1 <= index <= len(steps):
            return None
        return index, len(steps), steps[index - 1]

    def compaction_limits(self) -> CompactionLimits:
        """窓から決めた圧縮の量（直近として残す量・LLM 要約に渡す字数）。"""
        return limits_for(
            self.config.compaction.context_window,
            self.config.compact_token_threshold,
            self._fixed_tokens,
            self.config.compaction.llm_summary_max_tokens,
        )

    def _rehand_folded_step(self, expected_language: str) -> None:
        """段の途中で畳んで、段を渡した文が要約に入ったら、いまの段を渡し直す。"""
        current = self._current_step()
        if current is None:
            return
        index, total, text = current
        kept = self.session.messages[1:]
        if any(text in _message_text(m) for m in kept):
            return
        note = compacted_step_note(
            index,
            total,
            text,
            budget_lines=self.session.plan_budget_lines or None,
            estimated_lines=self._step_estimate(index),
        )
        self.session.messages.append(
            _user_msg(format_reminder(_with_language(note, expected_language)))
        )

    def _step_estimate(self, index: int) -> int | None:
        """段 ``index``（1..n）の計画の見積もり ``(~N lines)``。"""
        estimates = self.session.plan_step_estimates
        return estimates[index - 1] if 0 <= index - 1 < len(estimates) else None

    def _sync_git_context(self, turn: TurnContext) -> str | None:
        """凍結する git 節を用意し、前にモデルへ伝えた状態から変わっていれば通知文を返す。"""
        if not self.config.guidance.freeze_git_context:
            return None
        s = self.session
        marker = _compaction_marker(s)
        if s.git_context is None or s.git_context_compaction != marker:
            s.git_context = turn.git
            s.git_context_compaction = marker
            s.git_seen = turn.git_state
            return None
        if turn.git_state == s.git_seen:
            return None
        s.git_seen = turn.git_state
        return git_change_note(turn.git_state)

    def _frozen_git(self, turn: TurnContext, *, rebaseline: bool = False) -> TurnContext:
        """system に使う :class:`TurnContext`（git 節を凍結した値へ差し替える）。"""
        if not self.config.guidance.freeze_git_context:
            return turn
        s = self.session
        if rebaseline or s.git_context is None:
            s.git_context = turn.git
            s.git_context_compaction = _compaction_marker(s)
            s.git_seen = turn.git_state
        return replace(turn, git=s.git_context)

    def _hand_first_step(self) -> str | None:
        """段 1 を渡す定型文を返し、index を 1 にする（未着手のときだけ）。"""
        if not self.session.plan_steps or self.session.plan_step_index:
            return None
        self.session.plan_step_index = 1
        return first_step_note(
            len(self.session.plan_steps),
            self.session.plan_steps[0],
            self.plan_file_relpath(),
            budget_lines=self.session.plan_budget_lines or None,
            estimated_lines=self._step_estimate(1),
        )

    def _read_plan_file(self) -> str | None:
        """承認カードに載せる計画本文（計画ファイル。無ければ直前のアシスタント本文）。"""
        path = self.cwd / self.plan_file_relpath()
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    return text
        except OSError:
            pass
        for m in reversed(self.session.messages):
            if m.role == "assistant":
                text = "\n".join(b.text for b in m.blocks if isinstance(b, TextBlock)).strip()
                if text:
                    return text if _looks_like_plan(text) else None
        return None

    def _write_chunks(
        self,
        calls: list[ToolUseBlock],
        results: list[ToolResultBlock],
        output_tokens: int,
        response_s: float,
        iteration: int,
    ) -> list[WriteChunk]:
        """``write_file`` / ``append_file`` の 1 回ごとの分量。"""
        ok = {r.tool_use_id: not r.is_error for r in results}
        chunks: list[WriteChunk] = []
        for call in calls:
            if call.name not in _CHUNKED_WRITE_TOOLS:
                continue
            path = call.input.get("path")
            content = call.input.get("content")
            path = path if isinstance(path, str) else ""
            content = content if isinstance(content, str) else ""
            if path and self._targets_plan_file(call.name, SimpleNamespace(path=path)):
                continue
            chunks.append(
                WriteChunk(
                    tool=call.name,
                    path=path,
                    ok=ok.get(call.id, False),
                    chars=len(content),
                    lines=len(content.splitlines()),
                    bytes=len(content.encode("utf-8")),
                    output_tokens=output_tokens,
                    response_s=round(response_s, 2),
                    iteration=iteration,
                )
            )
        return chunks

    def _edit_lines(self, calls: list[ToolUseBlock], results: list[ToolResultBlock]) -> list[int]:
        """通った ``edit_file`` の 1 回ごとの、置き換えた後の行数（``new`` の行数）。"""
        ok = {r.tool_use_id: not r.is_error for r in results}
        lines: list[int] = []
        for call in calls:
            if call.name != "edit_file" or not ok.get(call.id, False):
                continue
            path = call.input.get("path")
            if isinstance(path, str) and self._targets_plan_file(
                call.name, SimpleNamespace(path=path)
            ):
                continue
            new = call.input.get("new")
            lines.append(len(new.splitlines()) if isinstance(new, str) else 0)
        return lines

    def _note_plan_exit_state(
        self, state: _PlanExitState, calls: list[ToolUseBlock], results: list[ToolResultBlock]
    ) -> None:
        """この反復のツールを**呼び出し順に**観測して :class:`_PlanExitState` を更新する。"""
        ok = {r.tool_use_id: not r.is_error for r in results}
        for call in calls:
            if call.name == "exit_plan_mode":
                state.wrote_plan = False
                step_done = any(
                    r.tool_use_id == call.id and r.output.startswith(PLAN_EXIT_STEP_DONE_PREFIX)
                    for r in results
                )
                if ok.get(call.id, False) and not self._plan_mode_active() and not step_done:
                    state.approved = True
            elif self._targets_plan_file(
                call.name, SimpleNamespace(path=call.input.get("path"))
            ) and ok.get(call.id, False):
                state.wrote_plan = True
        if any(r.is_error and r.output.startswith(PLAN_MODE_DENY_PREFIX) for r in results):
            state.write_denied = True

    def _plan_file_ready(self) -> bool:
        """計画ファイルが存在し、中身が空でないか（過去のターンで書いた分も含む）。"""
        try:
            path = self.cwd / self.plan_file_relpath()
            return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
        except OSError:
            return False

    def _plan_exit_note(self, state: _PlanExitState, final_text: str) -> str | None:
        """差し戻す定型文。"""
        plan_file = self.plan_file_relpath()
        if state.wrote_plan:
            return plan_exit_nudge_note(plan_file)
        if state.write_denied or _promises_a_plan(final_text, plan_file):
            if self._plan_file_ready():
                return plan_exit_nudge_note(plan_file)
            return plan_missing_plan_note(plan_file)
        return None

    def _targets_plan_file(self, name: str, inp: object) -> bool:
        """ファイル書込ツールの対象が計画ファイルそのものか（jail 解決後の一致）。"""
        if name not in ("write_file", "edit_file", "append_file"):
            return False
        return self._is_plan_file_path(getattr(inp, "path", None))

    def _reads_missing_plan_file(self, name: str, inp: object) -> bool:
        """``read_file`` の対象が**自分の**計画ファイルで、まだ存在しないか。"""
        if name != "read_file" or not self._plan_mode_active():
            return False
        if not self._is_plan_file_path(getattr(inp, "path", None)):
            return False
        return not (self.cwd / self.plan_file_relpath()).is_file()

    def _is_plan_file_path(self, raw: object) -> bool:
        """与えられた path 入力が計画ファイルを指すか（jail 解決後の一致）。"""
        if not isinstance(raw, str) or not raw:
            return False
        try:
            jail = self._ctx.jail
            return jail.resolve(raw) == jail.resolve(self.plan_file_relpath())
        except (PathEscapeError, ValueError):
            return False

    def _has_budget(self, iteration: int) -> bool:
        """この反復の後にもう 1 回まわす余地があるか（段階実行では段の窓で数える）。"""
        return iteration + 1 < self._iteration_limit

    def _open_step_budget(self, iteration: int) -> None:
        """段を渡した。"""
        self._iteration_limit = iteration + 1 + self.config.max_iterations

    def _step_position(self, stepping: bool) -> tuple[int, int] | None:
        """段階実行中なら (いまの段, 段数)。"""
        index, total = self.session.plan_step_index, len(self.session.plan_steps)
        return (index, total) if stepping and 0 < index <= total else None

    async def run_turn(
        self,
        user_text: str,
        prompter: PermissionPrompter | None = None,
        render: Renderer | None = None,
        observer: TurnObserver | None = None,
    ) -> TurnSummary:
        """1 ユーザーターンを実行し :class:`TurnSummary` を返す。"""
        self.aborted_summary = None
        try:
            return await self._run_turn(user_text, prompter, render, observer)
        except Exception:
            snapshot = self._turn_snapshot
            if snapshot is not None:
                try:
                    self.aborted_summary = snapshot()
                except Exception:
                    _log.warning(
                        "could not build the summary of a turn that ended in an exception",
                        exc_info=True,
                    )
            raise
        finally:
            self._turn_snapshot = None

    async def _run_turn(
        self,
        user_text: str,
        prompter: PermissionPrompter | None,
        render: Renderer | None,
        observer: TurnObserver | None,
    ) -> TurnSummary:
        prompter = prompter or DenyPrompter()
        render = render or NullRenderer()
        notify = guarded(observer)

        self.session.prompt_history.append(user_text)
        self._missing_plan_nudges = 0
        self._plan_budget_sendbacks = 0
        self._plan_budget_verdicts = []
        self._plan_budget_sent = []
        self._plan_budget_forced = False
        self.last_plan_budget = None
        self._iteration_limit = self.config.max_iterations
        expected_language = self.language.begin_turn(user_text)
        turn_context = await self.prompt_builder.snapshot(self.cwd)
        git_note = self._sync_git_context(turn_context)
        turn_context = self._frozen_git(turn_context)
        if self._plan_mode_active():
            model_text = ensure_plan_request(
                user_text,
                self.plan_file_relpath(),
                steps=self.config.guidance.plan_steps,
                budget=self.config.guidance.step_budget,
                exists=(self.cwd / self.plan_file_relpath()).is_file(),
            )
        elif plan_left_open(self.session.messages):
            off_note = PLAN_MODE_OFF_NOTE.format(sub=self.session.mode.sub.value)
            model_text = f"{user_text}\n\n{format_reminder([off_note])}"
        else:
            model_text = user_text
        if git_note is not None:
            model_text = f"{model_text}\n\n{format_reminder([git_note])}"
        stepping = False
        step_note = self._hand_first_step() if self.session.plan_step_index == 0 else None
        if step_note is None:
            resuming = self._current_step()
            if resuming is not None and wants_resume(user_text):
                step_note = resume_step_note(
                    *resuming,
                    budget_lines=self.session.plan_budget_lines or None,
                    estimated_lines=self._step_estimate(resuming[0]),
                )
        if step_note is not None:
            stepping = True
            model_text = f"{model_text}\n\n{format_reminder([step_note])}"
            handed = self._current_step()
            if handed is not None:
                plan_step(render, *handed)
        turn_message = _user_msg(with_language_directive(model_text, expected_language))
        self.session.messages.append(turn_message)
        notify(TurnBegan(len(user_text)))
        total_usage = TokenUsage()
        delegated_usage = TokenUsage()
        decomposition_record: DecompositionRecord | None = None
        if self.decomposition is not None and not stepping:
            phase(render, ui("engine.runtime.researching"))
            background_context = await self.decomposition.before_turn(user_text)
            phase(render, None)
            decomposition_record = self.decomposition.take_record()
            if background_context is not None:
                self.session.messages.append(
                    _user_msg(
                        with_language_directive(
                            f"[background research results]\n{background_context}",
                            expected_language,
                        )
                    )
                )
            if decomposition_record is not None and decomposition_record.result is not None:
                delegated_usage = _add(delegated_usage, decomposition_record.result.usage)
        prompt_prefix_breaks = 0
        turn_prefix_break = False
        previous_prefix = ""
        compacted = False
        truncation_retried = False
        health_warning: str | None = None
        total_tool_calls = 0
        last_text = ""
        exit_noop_streak = 0
        last_thinking = ""
        empty_final_retried = False
        args_reply_latched_at: int | None = None
        no_change_retried = False
        turn_started_at = now_ms() / 1000.0
        language_repaired = False
        detected_language = ""
        step_timeouts = 0
        timeout_recoveries = 0
        timeout_resumes = 0
        step_rewrites = 0
        tool_call_rewrites = 0
        repetition_recovered = False
        repetition_aborts = 0
        plan_exit = _PlanExitState()
        plan_exit_nudges = 0
        plan_steps_run = 1 if stepping else 0
        tools_while_stepping = 0
        step_retried: set[int] = set()
        repair = ToolCallRepairState(ToolCallRepairPolicy.from_config(self.config))
        counters = _TurnCounters()
        step_open: tuple[int, int, int, int] | None = (0, 0, 0, 0) if stepping else None
        budget_lines_seen = self.session.plan_budget_lines if stepping else 0

        def close_step(index: int, at_iteration: int) -> None:
            nonlocal step_open
            if step_open is None:
                return
            if not self._step_budget_enabled():
                step_open = None
                return
            start_iteration, start_chunks, start_edits, start_timeouts = step_open
            step_open = None
            chunks = counters.write_chunks[start_chunks:]
            edits = counters.edit_lines[start_edits:]
            record = PlanStepRecord(
                index=index,
                total=len(self.session.plan_steps),
                estimated_lines=self._step_estimate(index),
                budget_lines=self.session.plan_budget_lines,
                actual_lines=sum(c.lines for c in chunks if c.ok),
                max_write_lines=max((c.lines for c in chunks), default=0),
                writes=len(chunks),
                max_response_s=max((c.response_s for c in chunks), default=0.0),
                timeouts=timeout_recoveries - start_timeouts,
                iterations=at_iteration + 1 - start_iteration,
                edits=len(edits),
                edited_lines=sum(edits),
            )
            counters.plan_step_records.append(record)
            plan_step_done(render, record)

        def advance_step(iteration: int) -> Literal["retried", "next", "finished"]:
            """いまの段を終えたとして段を進める（段の実行中の exit_plan_mode からも）。"""
            nonlocal plan_steps_run, step_open, step_timeouts, step_rewrites
            index = self.session.plan_step_index
            total = len(self.session.plan_steps)
            budget = self._has_budget(iteration)
            idle = not tools_while_stepping and index not in step_retried
            if index < total and budget and idle:
                step_retried.add(index)
                self.session.messages.append(
                    _user_msg(
                        format_reminder(
                            _with_language(
                                retry_step_note(index, total, self.session.plan_steps[index - 1]),
                                expected_language,
                            )
                        )
                    )
                )
                self.store.save(self.session)
                return "retried"
            if index < total:
                close_step(index, iteration)
                text = self.session.plan_steps[index]
                self.session.plan_step_index = index + 1
                self._open_step_budget(iteration)
                plan_steps_run += 1
                step_open = (
                    iteration + 1,
                    len(counters.write_chunks),
                    len(counters.edit_lines),
                    timeout_recoveries,
                )
                step_timeouts = 0
                step_rewrites = 0
                repair.new_section()
                self.session.messages.append(
                    _user_msg(
                        format_reminder(
                            _with_language(
                                next_step_note(
                                    index + 1,
                                    total,
                                    text,
                                    budget_lines=self.session.plan_budget_lines or None,
                                    estimated_lines=self._step_estimate(index + 1),
                                ),
                                expected_language,
                            )
                        )
                    )
                )
                plan_step(render, index + 1, total, text)
                self.store.save(self.session)
                return "next"
            close_step(index, iteration)
            self._clear_plan_steps()
            self.store.save(self.session)
            return "finished"

        reminders = ReminderState() if self.config.guidance.reminders else None
        recovery_ctx = (
            RecoveryLedger(attempt_limit=self.config.recovery.max_attempts)
            if self.config.recovery.enabled
            else None
        )

        def summarize(final_text: str, stop_reason: str, iterations: int) -> TurnSummary:
            """ここまでの集計からターンの要約を作る（知らせは出さない）。"""
            return TurnSummary(
                final_text=final_text,
                stop_reason=stop_reason,
                iterations=iterations,
                usage=total_usage,
                compacted=compacted,
                health_warning=health_warning,
                assistant_messages=iterations,
                tool_calls=total_tool_calls,
                delegated_usage=delegated_usage,
                tool_call_repairs=repair.repairs,
                empty_tool_args=counters.empty_tool_args,
                reminders=reminders.emitted if reminders is not None else 0,
                last_text=last_text,
                last_thinking=last_thinking,
                decomposition=decomposition_record,
                language_expected=expected_language,
                language_detected=detected_language,
                language_repaired=language_repaired,
                prompt_prefix_breaks=prompt_prefix_breaks,
                turn_prefix_break=turn_prefix_break,
                timeout_recoveries=timeout_recoveries,
                timeout_resumes=timeout_resumes,
                tool_call_rewrites=tool_call_rewrites,
                repetition_aborts=repetition_aborts,
                syntax_rejects=counters.syntax_rejects,
                syntax_warnings=counters.syntax_warnings,
                edit_mismatches=counters.edit_mismatches,
                edit_reindents=counters.edit_reindents,
                edit_stale_notes=counters.edit_stale_notes,
                related_code_notes=counters.related_code_notes,
                overwrite_rejects=counters.overwrite_rejects,
                overwrite_overrides=counters.overwrite_overrides,
                step_write_rejects=counters.step_write_rejects,
                step_write_overrides=counters.step_write_overrides,
                plan_exit_not_active=counters.plan_exit_not_active,
                plan_exit_step_done=counters.plan_exit_step_done,
                mkdir_skips=counters.mkdir_skips,
                output_delete_blocks=counters.output_delete_blocks,
                state_write_blocks=counters.state_write_blocks,
                secret_read_checks=counters.secret_read_checks,
                first_fetch_asks=counters.first_fetch_asks,
                lsp_start_asks=counters.lsp_start_asks,
                tool_result_cuts=counters.tool_result_cuts,
                written_bodies_shrunk=counters.written_bodies_shrunk,
                omitted_copies=counters.omitted_copies,
                presend_compactions=counters.presend_compactions,
                context_truncations=counters.context_truncations,
                prefill_waits=counters.prefill_waits,
                permission_details=counters.permission_details,
                unknown_tool_calls=counters.unknown_tool_calls,
                malformed_tool_calls=counters.malformed_tool_calls,
                tool_name_fixes=counters.tool_name_fixes,
                deferred_autoloads=counters.deferred_autoloads,
                args_only_replies=counters.args_only_replies,
                no_change_reports=counters.no_change_reports,
                plan_exit_nudges=plan_exit_nudges,
                plan_steps_run=plan_steps_run,
                write_chunks=counters.write_chunks,
                tokens_per_s=(
                    round(statistics.median(counters.tokens_per_s), 1)
                    if counters.tokens_per_s
                    else 0.0
                ),
                plan_step_records=counters.plan_step_records,
                plan_budget_sendbacks=self._plan_budget_sendbacks,
                plan_budget_verdicts=list(self._plan_budget_verdicts),
                plan_budget_forced=self._plan_budget_forced,
                plan_budget_lines=budget_lines_seen
                or (self.last_plan_budget.lines if self.last_plan_budget else 0),
            )

        def finish(final_text: str, stop_reason: str, iteration: int) -> TurnSummary:
            """ターンの要約を作って記録する（返答で終えた / ユーザーの拒否で止めた）。"""
            summary = summarize(final_text, stop_reason, iteration + 1)
            notify(TurnEnded(summary))
            return summary

        rounds = 0
        self._turn_snapshot = lambda: summarize("", "aborted", rounds)

        async def compact_history(iteration: int) -> bool:
            """履歴を畳む。"""
            nonlocal compacted, turn_context, health_warning
            phase(
                render,
                ui("engine.runtime.summarizing_the_history")
                if not compacted
                else ui("engine.runtime.folding_the_history"),
            )
            outcome = await self.compactor.acompact(
                self.session, allow_llm=not compacted, limits=self.compaction_limits()
            )
            phase(render, None)
            if outcome is None:
                return False
            compacted = True
            self._sent = None
            if reminders is not None:
                reminders.note_compaction()
            if self.config.guidance.freeze_git_context:
                turn_context = self._frozen_git(
                    await self.prompt_builder.snapshot(self.cwd), rebaseline=True
                )
            _log.info(
                "compacted %d message(s) (llm=%s): %s",
                outcome.dropped_messages,
                outcome.used_llm,
                outcome.condensed.describe(),
            )
            phase(render, ui("engine.runtime.checking_the_session"))
            health_warning = await check_tool_executor(self)
            phase(render, None)
            if health_warning is not None:
                _log.warning("tool check after compaction failed: %s", health_warning)
                if recovery_ctx is not None:
                    await self._abort_unhealthy_session(
                        health_warning, recovery_ctx, notify, iteration
                    )
            return True

        for iteration in itertools.count():
            if iteration >= self._iteration_limit:
                break
            rounds = iteration + 1
            projected = self._projected_tokens()
            if (
                projected is not None
                and self.compactor.needs_compaction(TokenUsage(input_tokens=projected), self.config)
                and not self._shrink_before_compaction(counters)
                and await compact_history(iteration)
            ):
                counters.presend_compactions += 1
                _log.info("compacted before sending: projected ~%d tokens", projected)
                self._rehand_folded_step(expected_language)
            system = self.prompt_builder.compose(
                self.session.model, self.config, self.language, turn_context
            )
            schemas = self._all_schemas()
            tools_json = json.dumps(schemas, sort_keys=True, ensure_ascii=False)
            self._fixed_tokens = estimate_upper(system.joined()) + estimate_upper(tools_json)
            prefix = f"{system.joined()}\n{tools_json}"
            if iteration and prefix != previous_prefix:
                prompt_prefix_breaks += 1
            digest = hashlib.sha256(prefix.encode()).hexdigest()
            if not iteration and self.session.last_system_digest not in ("", digest):
                turn_prefix_break = True
            previous_prefix = prefix
            self.session.last_system_digest = digest
            try:
                sent_estimate = self._fit_window(
                    system.joined(),
                    tools_json,
                    render,
                    counters,
                    drop_from=turn_message if not iteration else None,
                )
            except ContextOverflowError as overflow:
                notify(TurnAborted(iteration, overflow))
                raise
            sent_floor = self._sent_floor(system.joined(), tools_json, digest)
            cached = self._cached_tokens(digest)
            expected = self._projected_tokens() or int(sent_estimate * self._estimate_ratio)
            first_byte = self._first_byte_wait(cached, expected, render, counters)
            req = ApiRequest(
                model=self.session.model,
                system=system.static,
                system_dynamic=system.dynamic,
                messages=self.session.messages,
                tools=schemas,
                sampling=self.config.provider.sampling,
                strict_tools=self.config.tool_schema.strict_tools,
                first_byte_timeout_s=first_byte,
            )
            self._last_request = req
            sent = len(self.session.messages)

            try:
                assistant = await self._request_reply(req, render, repair, counters)
            except ProviderError as e:
                window = (
                    await self._refit_window()
                    if is_context_overflow(e)
                    and not truncation_retried
                    and self._has_budget(iteration)
                    else None
                )
                if window is not None:
                    truncation_retried = True
                    self._sent = None
                    _log.warning("the server rejected the request as over its window: %s", e)
                    context_notice(
                        render,
                        ui(
                            "engine.runtime.the_server_rejected_the_request",
                            window_text=f"{window:,}",
                        ),
                    )
                    continue
                if (
                    is_tool_call_parse_error(e)
                    and step_rewrites < 1
                    and self.config.guidance.tool_call_rewrite
                    and self._has_budget(iteration)
                ):
                    step_rewrites += 1
                    tool_call_rewrites += 1
                    _log.debug("tool_call parse retries ran out; asking for a rewrite: %s", e)
                    provider_notice(render, notices.ProviderNotice(kind="rewrite"))
                    if reminders is not None:
                        reminders.note_tool_call_rewrite()
                    rewrite_notes = _with_language(tool_call_rewrite_note(), expected_language)
                    self.session.messages.append(_user_msg(format_reminder(rewrite_notes)))
                    self.store.save(self.session)
                    continue
                read_s = self._waited_s(req, counters)
                limit = self.config.guidance.step_timeout_recoveries if stepping else 1
                recovering = (
                    is_read_timeout(e)
                    and step_timeouts < limit
                    and self.config.guidance.timeout_recovery
                    and self._has_budget(iteration)
                )
                resuming = (
                    self._current_step() if recovering and stepping and step_timeouts else None
                )
                if resuming is not None:
                    step_timeout_resume(
                        render, read_s, resuming[0], resuming[1], step_timeouts + 1, limit
                    )
                else:
                    stream_aborted(render, read_s if recovering else None)
                if not recovering:
                    e.step = self._step_position(stepping)
                    e.timeout_recoveries = timeout_recoveries
                    if is_read_timeout(e):
                        e.timeout_s = read_s
                    notify(TurnAborted(iteration, e))
                    raise
                step_timeouts += 1
                timeout_recoveries += 1
                _log.debug(
                    "asking for a smaller resend after a read timeout (%d/%d in this stretch): %s",
                    step_timeouts,
                    limit,
                    e,
                )
                if reminders is not None:
                    reminders.note_timeout_recovery()
                budget_lines = self.session.plan_budget_lines or None
                if resuming is not None:
                    timeout_resumes += 1
                    budget = self._plan_budget()
                    note = timeout_resume_step_note(
                        *resuming,
                        seconds=read_s,
                        attempt=step_timeouts,
                        rate_tok_s=budget.rate_tok_s if budget.samples else None,
                        budget_lines=budget_lines,
                        estimated_lines=self._step_estimate(resuming[0]),
                    )
                    timeout_notes = _with_language(note, expected_language)
                else:
                    timeout_notes = [timeout_note(read_s, lines=budget_lines)]
                    if language_note(expected_language):
                        timeout_notes.append(language_note(expected_language))
                self.session.messages.append(_user_msg(format_reminder(timeout_notes)))
                self.store.save(self.session)
                continue
            except RepetitionAbort as e:
                recovering = not repetition_recovered and self._has_budget(iteration)
                repetition_aborted(render, e.verdict, recovering)
                if not recovering:
                    loop_error = RepetitionLoopError(
                        e.verdict.repeats, e.verdict.snippet, recovered_before=repetition_recovered
                    )
                    notify(TurnAborted(iteration, loop_error))
                    raise loop_error from None
                repetition_recovered = True
                repetition_aborts += 1
                _log.debug(
                    "stopped on repetition (%s %d/%d seen, at most %d times, %d chars): %s",
                    e.verdict.unit,
                    e.verdict.seen,
                    e.verdict.window,
                    e.verdict.repeats,
                    e.verdict.chars,
                    e.verdict.snippet,
                )
                if reminders is not None:
                    reminders.note_repetition_abort()
                repetition_notes = [repetition_note(e.verdict)]
                if language_note(expected_language):
                    repetition_notes.append(language_note(expected_language))
                self.session.messages.append(_user_msg(format_reminder(repetition_notes)))
                self.store.save(self.session)
                continue
            total_usage = _add(total_usage, assistant.usage)
            read = self._truncated_read(assistant.usage, sent_floor)
            if read is not None:
                counters.context_truncations += 1
                _log.warning(
                    "the server truncated the request: sent >= ~%d, read %d", sent_floor, read
                )
                window = (
                    await self._refit_window()
                    if not truncation_retried and self._has_budget(iteration)
                    else None
                )
                if window is not None:
                    truncation_retried = True
                    self._sent = None
                    sent_text, read_text, window_text = (
                        f"{sent_floor:,}",
                        f"{read:,}",
                        f"{window:,}",
                    )
                    context_notice(
                        render,
                        ui(
                            "engine.runtime.the_server_had_truncated_the",
                            sent_text=sent_text,
                            read_text=read_text,
                            window_text=window_text,
                        ),
                    )
                    continue
                truncated = ContextTruncatedError(sent_floor, read)
                notify(TurnAborted(iteration, truncated))
                raise truncated
            self._note_sent(sent, assistant.usage, sent_estimate, digest)
            self._note_prefill(cached, assistant.usage, counters.stream_timing)
            if self.window_unconfirmed:
                self.window_unconfirmed = False
                await self._recheck_window()
            output_tokens = assistant.usage.output_tokens if assistant.usage else 0
            if output_tokens and counters.response_s > 0:
                counters.tokens_per_s.append(output_tokens / counters.response_s)
                self._rate_samples.append((output_tokens, counters.response_s))
            last_text = _text_of(assistant) or last_text
            last_thinking = _thinking_of(assistant) or last_thinking
            self.session.messages.append(assistant)

            tool_calls = [b for b in assistant.blocks if isinstance(b, ToolUseBlock)]
            context_usage = assistant.usage or TokenUsage(
                input_tokens=estimate_tokens(
                    "".join(m.model_dump_json() for m in self.session.messages)
                )
            )
            if (
                not (tool_calls and self._sent is not None)
                and self.compactor.needs_compaction(context_usage, self.config)
                and not self._shrink_before_compaction(counters)
            ):
                await compact_history(iteration)
            self.store.save(self.session)

            total_tool_calls += len(tool_calls)
            if stepping:
                tools_while_stepping += sum(1 for c in tool_calls if c.name != "exit_plan_mode")
            notify(ModelReplied(iteration, len(assistant.blocks), len(tool_calls)))
            if not tool_calls:
                mid_step = stepping and 0 < self.session.plan_step_index < len(
                    self.session.plan_steps
                )
                leaked = _detect_text_tool_call(_text_of(assistant), self._known_tool_names())
                if leaked is not None:
                    raise ToolCallingUnsupportedError(self.session.model, leaked)
                bare = detect_bare_tool_args(
                    _text_of(assistant),
                    self._all_schemas(),
                    read_only=lambda n: mode_le(self._spec_mode(n), PermissionMode.READ_ONLY),
                )
                if bare is not None:
                    counters.args_only_replies += 1
                    if (
                        args_reply_latched_at != self.session.plan_step_index
                        and self.config.guidance.args_only_retry
                        and self._has_budget(iteration)
                    ):
                        args_reply_latched_at = self.session.plan_step_index
                        if reminders is not None:
                            reminders.note_args_only_reply()
                        note = args_only_reply_note(bare.keys, bare.tools)
                        self.session.messages.append(
                            _user_msg(format_reminder(_with_language(note, expected_language)))
                        )
                        self.store.save(self.session)
                        continue
                if (
                    self._plan_mode_active()
                    and not plan_exit.nudged
                    and self.config.guidance.plan_exit_nudge
                    and self._has_budget(iteration)
                ):
                    plan_note = self._plan_exit_note(plan_exit, _text_of(assistant))
                    if plan_note is not None:
                        plan_exit.nudged = True
                        plan_exit_nudges += 1
                        if reminders is not None:
                            reminders.note_plan_exit_nudge()
                        self.session.messages.append(
                            _user_msg(format_reminder(_with_language(plan_note, expected_language)))
                        )
                        self.store.save(self.session)
                        continue
                if (
                    not _text_of(assistant)
                    and not empty_final_retried
                    and not mid_step
                    and self.config.guidance.empty_final_retry
                    and self._has_budget(iteration)
                ):
                    empty_final_retried = True
                    if reminders is not None:
                        reminders.note_empty_final_retry()
                    if self._plan_mode_active():
                        first_note = empty_final_plan_note(self.plan_file_relpath())
                    elif plan_exit.approved:
                        first_note = empty_final_approved_note()
                    else:
                        first_note = empty_final_note()
                    retry_notes = [first_note]
                    if language_note(expected_language):
                        retry_notes.append(language_note(expected_language))
                    self.session.messages.append(_user_msg(format_reminder(retry_notes)))
                    self.store.save(self.session)
                    continue
                verdict = None
                if (
                    not self._plan_mode_active()
                    and not mid_step
                    and counters.mutating_attempted == 0
                    and not is_blocked_reply(_text_of(assistant))
                ):
                    verdict = no_change_report(
                        _text_of(assistant),
                        cwd=self.cwd,
                        written=self._ctx.reads.written if self._ctx.reads is not None else (),
                        since=turn_started_at,
                    )
                if verdict is not None:
                    counters.no_change_reports += 1
                    if (
                        not no_change_retried
                        and self.config.guidance.no_change_report_retry
                        and self._has_budget(iteration)
                    ):
                        no_change_retried = True
                        if reminders is not None:
                            reminders.note_no_change_report()
                        note = no_change_report_note(verdict.paths, verdict.missing)
                        self.session.messages.append(
                            _user_msg(format_reminder(_with_language(note, expected_language)))
                        )
                        self.store.save(self.session)
                        continue
                final_text = _text_of(assistant)
                detected_language = (
                    detect_language(strip_non_prose(final_text), min_chars=self.language.min_chars)
                    if expected_language and final_text
                    else ""
                )
                if (
                    detected_language not in ("", UNDETERMINED)
                    and detected_language != expected_language
                    and self.language.repair
                    and not language_repaired
                    and not mid_step
                    and bare is None
                    and not requests_other_language(user_text)
                    and self._has_budget(iteration)
                ):
                    language_repaired = True
                    if reminders is not None:
                        reminders.note_language_mismatch()
                    self.session.messages.append(
                        _user_msg(format_reminder([language_mismatch_note(expected_language)]))
                    )
                    self.store.save(self.session)
                    continue
                if stepping and self.session.plan_steps and not is_blocked_reply(final_text):
                    if advance_step(iteration) in ("retried", "next"):
                        continue
                return finish(final_text, "end_turn", iteration)

            step_index_before = self.session.plan_step_index
            exit_noop_before = counters.plan_exit_not_active
            step_done_before = counters.plan_exit_step_done
            try:
                results, tool_usage = await self._run_tools(
                    tool_calls, prompter, render, notify, iteration, recovery_ctx, counters
                )
            except SessionAbortError as abort:
                notify(TurnAborted(iteration, abort))
                raise
            delegated_usage = _add(delegated_usage, tool_usage)
            self._note_plan_exit_state(plan_exit, tool_calls, results)
            exit_noop = counters.plan_exit_not_active - exit_noop_before == len(tool_calls)
            exit_noop_streak = exit_noop_streak + 1 if exit_noop else 0
            counters.write_chunks.extend(
                self._write_chunks(
                    tool_calls, results, output_tokens, counters.response_s, iteration + 1
                )
            )
            counters.edit_lines.extend(self._edit_lines(tool_calls, results))
            results = self._fit_results(tool_calls, results, render, counters)
            if (
                self.session.plan_steps
                and self.session.plan_step_index == 1
                and step_index_before != 1
            ):
                stepping = True
                self._open_step_budget(iteration)
                plan_steps_run += 1
                tools_while_stepping = 0
                step_open = (
                    iteration + 1,
                    len(counters.write_chunks),
                    len(counters.edit_lines),
                    timeout_recoveries,
                )
                step_timeouts = 0
                step_rewrites = 0
                repair.new_section()
                budget_lines_seen = self.session.plan_budget_lines
                plan_step(render, 1, len(self.session.plan_steps), self.session.plan_steps[0])
            elif not self.session.plan_steps and self.session.plan_step_index != step_index_before:
                stepping = False
                step_open = None
                step_timeouts = 0
                step_rewrites = 0
                repair.new_section()
            notes: list[str] = []
            if self.background is not None:
                notes.extend(
                    background_done_note(t.task_id, t.proc.returncode, t.command_preview)
                    for t in self.background.take_completed_unnotified()
                )
            if reminders is not None:
                reminders.emitted += len(notes)
                notes.extend(reminders.collect(tool_calls, results, self.session))
            if notes:
                results[-1] = results[-1].model_copy(
                    update={"output": results[-1].output + "\n\n" + format_reminder(notes)}
                )
            self._shrink_written(assistant, results, counters)
            self.session.messages.append(_tool_result_msg(results))
            self.store.save(self.session)
            if counters.user_denied is not None:
                return finish("", "user_denied", iteration)
            if counters.unknown_loop:
                unknown_loop = UnknownToolLoopError(
                    counters.unknown_loop, counters.unknown_streak, last_text
                )
                notify(TurnAborted(iteration, unknown_loop))
                raise unknown_loop
            if counters.plan_exit_step_done > step_done_before and self.session.plan_steps:
                if not stepping:
                    stepping = True
                    step_open = None
                advance_step(iteration)
                continue
            if exit_noop_streak >= PLAN_EXIT_LOOP_LIMIT:
                exit_loop = PlanExitLoopError(exit_noop_streak, last_text)
                notify(TurnAborted(iteration, exit_loop))
                raise exit_loop

        error = MaxIterationsError(
            self.config.max_iterations, last_text=last_text, step=self._step_position(stepping)
        )
        notify(TurnAborted(iteration, error))
        raise error

    async def _request_reply(
        self,
        req: ApiRequest,
        render: Renderer,
        repair: ToolCallRepairState,
        counters: _TurnCounters | None = None,
    ) -> ConversationMessage:
        """1 回分のアシスタント応答を畳む。"""
        while True:
            try:
                async with closing_stream(self._stream_render(req, render, counters)) as events:
                    return await fold_events(events)
            except Exception as e:
                if not repair.should_repair(e):
                    raise
                _log.warning("tool_call parse failure; retrying once with a repair hint: %s", e)
                provider_notice(
                    render,
                    notices.ProviderNotice(kind="retry", reason="tool_parse"),
                )
                req = self._apply_repair_role(repair.repaired_request(req))

    def _apply_repair_role(self, req: ApiRequest) -> ApiRequest:
        """修復の投げ直しへ repair ロールのモデル・サンプリングを適用する。"""
        from satsumaestro.providers.registry import expand_model_name

        roles = self.config.provider.roles
        update: dict[str, Any] = {}
        if roles.repair:
            update["model"] = expand_model_name(roles.repair)
        override = roles.sampling_overrides.get("repair")
        if override is not None:
            update["sampling"] = override
        return req.model_copy(update=update) if update else req

    async def _stream_render(
        self, req: ApiRequest, render: Renderer, counters: _TurnCounters | None = None
    ) -> AsyncIterator[AssistantEvent]:
        started = time.monotonic()
        if counters is not None:
            counters.response_s = 0.0
        detector = RepetitionDetector() if self.config.guidance.repetition_abort else None
        stream_timing = timing.StreamTiming()
        if counters is not None:
            counters.stream_timing = stream_timing
        with (
            notices.bind(lambda notice: provider_notice(render, notice)),
            timing.bind(stream_timing),
        ):
            inner = self._stream_events(req, render, counters, started, detector)
            async with closing_stream(inner) as events:
                async for ev in events:
                    yield ev

    async def _stream_events(
        self,
        req: ApiRequest,
        render: Renderer,
        counters: _TurnCounters | None,
        started: float,
        detector: RepetitionDetector | None,
    ) -> AsyncIterator[AssistantEvent]:
        async with closing_stream(self.provider.stream(req)) as events:
            async for ev in events:
                if counters is not None and isinstance(ev, ToolUse) and ev.input_fallback:
                    counters.empty_tool_args += 1
                if counters is not None and isinstance(ev, MessageStop):
                    counters.response_s = time.monotonic() - started
                render.on_assistant_event(ev)
                if detector is not None and isinstance(ev, TextDelta):
                    verdict = detector.feed(ev.text)
                    if verdict is not None:
                        raise RepetitionAbort(verdict)
                yield ev

    def _sent_count(self) -> int | None:
        """直近の応答を生んだ要求に載せた発言の数。"""
        mark = self._sent
        messages = self.session.messages
        if mark is None or mark.count > len(messages) or messages[mark.count - 1] is not mark.last:
            return None
        return mark.count

    def _projected_tokens(self) -> int | None:
        """次の要求の量の見積もり（前の要求の実測＋その後に積んだ発言）。"""
        count = self._sent_count()
        if count is None or self._sent is None:
            return None
        tail = self.session.messages[count:]
        return self._sent.input_tokens + sum(message_tokens(m) for m in tail)

    def _note_sent(
        self, count: int, usage: TokenUsage | None, estimate: int, digest: str = ""
    ) -> None:
        """応答の usage を、次の見積もりの起点と、見積もりを実測に寄せる比として残す。"""
        if usage is None or not usage.input_tokens or not count:
            self._sent = None
            return
        self._sent = _SentMark(
            count,
            self.session.messages[count - 1],
            usage.input_tokens,
            digest,
            self.session.model,
            time.monotonic(),
        )
        if estimate > 0:
            self._estimate_ratio = min(max(usage.input_tokens / estimate, 0.3), 1.5)

    def _cached_tokens(self, digest: str) -> int:
        """サーバの接頭辞キャッシュに残っていそうな量（前の要求の実測）。"""
        count = self._sent_count()
        mark = self._sent
        if count is None or mark is None or mark.digest != digest:
            return 0
        if mark.model != self.session.model:
            return 0
        idle = keep_alive_seconds(self.config.provider.ollama.keep_alive)
        if idle is not None and time.monotonic() - mark.at > idle:
            return 0
        return mark.input_tokens

    def _first_byte_wait(
        self, cached: int, expected: int, render: Renderer, counters: _TurnCounters
    ) -> float | None:
        """読み直しが長い見込みなら、最初の中身までに待つ秒を返す（延ばさないなら None）。"""
        if not self.config.provider.retry.extend_first_wait:
            return None
        silence = self.config.read_timeout_s()
        wait = self._prefill.first_byte_timeout(
            cached, expected, silence, max(self.config.api_timeout_s, silence)
        )
        if wait is None:
            return None
        counters.prefill_waits += 1
        tokens = max(expected - cached, 0)
        seconds = self._prefill.seconds(cached, expected)
        if seconds > silence:
            secs, size, limit = f"{seconds:.0f}", f"{tokens:,}", f"{wait:.0f}"
            if self._prefill.measured:
                notice = ui(
                    "engine.runtime.reading_the_conversation_is_expected",
                    secs=secs,
                    size=size,
                    limit=limit,
                )
            else:
                notice = ui(
                    "engine.runtime.reading_the_conversation_tokens_is", size=size, limit=limit
                )
            context_notice(render, notice)
        _log.info(
            "extended the first-byte wait to %.0f s: ~%d uncached of ~%d tokens (rate %.0f)",
            wait,
            tokens,
            expected,
            self._prefill.rate,
        )
        return wait

    def _note_prefill(
        self, cached: int, usage: TokenUsage | None, stream: timing.StreamTiming | None
    ) -> None:
        """最初の中身までの時間を、読み込みの速さの標本にする。"""
        if usage is None or stream is None or stream.first_chunk_at is None:
            return
        self._prefill.note(cached, usage.input_tokens, stream.first_chunk_at - stream.started_at)

    def _waited_s(self, req: ApiRequest, counters: _TurnCounters) -> float:
        """打ち切りまでに待った無音の秒（最初の中身が届く前なら、延ばした待ち）。"""
        stream = counters.stream_timing
        if req.first_byte_timeout_s is not None and (
            stream is None or stream.first_chunk_at is None
        ):
            return req.first_byte_timeout_s
        return self.config.read_timeout_s()

    def _sent_floor(self, system: str, tools_json: str, digest: str) -> int:
        """これから送る要求の量の下限（少なめに出る側）。"""
        messages = self.session.messages
        floor = request_floor(system, tools_json, messages)
        count = self._sent_count()
        mark = self._sent
        if count is not None and mark is not None and mark.digest == digest and mark.floor:
            tail = sum(message_floor(m) for m in messages[count:])
            floor = max(floor, int(mark.input_tokens * _SENT_FLOOR_SHARE) + tail)
        return floor

    def _truncated_read(self, usage: TokenUsage | None, floor: int) -> int | None:
        """usage が送った量の下限より大きく少なければ、読まれた量を返す（切り詰めの印）。"""
        if usage is None or not usage.input_tokens or self.provider.kind is ProviderKind.ANTHROPIC:
            return None
        if usage.input_tokens + _TRUNCATION_SLACK >= floor:
            return None
        return usage.input_tokens

    async def window_breakdown(self) -> WindowBreakdown | None:
        """窓の内訳（起動時と測り直しの後に出す）。"""
        window = self.config.compaction.context_window
        if window <= 0:
            return None
        turn = await self.prompt_builder.snapshot(self.cwd)
        system = self.prompt_builder.compose(self.session.model, self.config, self.language, turn)
        tools_json = json.dumps(self._all_schemas(), sort_keys=True, ensure_ascii=False)
        return window_breakdown(
            window,
            self.config.compact_token_threshold,
            system.joined(),
            tools_json,
            turn.instructions,
        )

    async def _recheck_window(self) -> None:
        """起動時に分からなかった窓を、読み込んだ後の値で測り直す。"""
        from satsumaestro.engine.context_fit import loaded_source, local_server_for, source_label

        before = self.config.compaction.context_window
        window = await self._refit_window()
        if window is None:
            return
        notes = [ui("engine.runtime.window_rechecked", window=f"{window:,}", before=f"{before:,}")]
        breakdown = await self.window_breakdown()
        if breakdown is not None:
            server = local_server_for(self.config, self.session.model, os.environ)
            kind = server.kind if server is not None else "ollama"
            notes.append(breakdown.describe(source_label(loaded_source(kind))))
            if breakdown.tight:
                notes.append(breakdown.advice(kind, self.session.model))
        self._window_notes = notes

    async def refit_for_model(self) -> ContextFitResult | None:
        """モデルを替えた後に、窓・圧縮の閾値・指示ファイルの量を新しいモデルで決め直す。"""
        from satsumaestro.engine.context_fit import apply_context_fit

        if self.fit_base is None:
            return None
        threshold, window = self.fit_base
        self.config.compact_token_threshold = threshold
        self.config.compaction.context_window = window
        self._sent = None
        self.window_unconfirmed = False
        try:
            fit = await apply_context_fit(self.config, self.session.model, os.environ)
        except Exception:
            _log.debug("could not fit the context window to %s", self.session.model, exc_info=True)
            fit = None
        self.prompt_builder.use_instruction_tokens(
            instruction_tokens(self.config.compaction.context_window)
        )
        return fit

    def take_window_notes(self) -> list[str]:
        """測り直した窓の知らせを取り出す（REPL がターンの末尾の行の後に出す）。"""
        notes, self._window_notes = self._window_notes, []
        return notes

    async def _refit_window(self) -> int | None:
        """載っているモデルの実際の窓を取り直し、思っていたより小さければ合わせる。"""
        from satsumaestro.engine.context_fit import refit_window

        try:
            return await refit_window(self.config, self.session.model, os.environ)
        except Exception:
            _log.debug("could not refit the context window", exc_info=True)
            return None

    def _fit_window(
        self,
        system: str,
        tools_json: str,
        render: Renderer,
        counters: _TurnCounters,
        *,
        drop_from: ConversationMessage | None,
    ) -> int:
        """送る前に、要求が窓に収まるかを見積もる。"""
        messages = self.session.messages
        whole = request_tokens(system, tools_json, messages)
        limit = input_limit(self.config.compaction.context_window)
        if limit is None:
            return whole
        ratio = self._estimate_ratio
        projected = self._projected_tokens()
        if (
            projected if projected is not None else int(whole * ratio)
        ) >= limit and self._shrink_history(counters):
            self._sent = None
            whole = request_tokens(system, tools_json, messages)
        cuts: list[ResultCut] = []
        count = self._sent_count()
        projected = self._projected_tokens()
        if count is not None and projected is not None:
            expected = projected
            if expected >= limit:
                cuts += _cut_results(messages, count, expected - limit + _WINDOW_MARGIN)
                expected = self._projected_tokens() or expected
        else:
            expected = int(whole * ratio)
        if expected >= limit:
            excess = math.ceil((int(whole * ratio) - limit + _WINDOW_MARGIN) / ratio)
            whole_cuts = _cut_results(messages, 0, excess)
            if whole_cuts:
                self._sent = None
            cuts += whole_cuts
            expected = int(request_tokens(system, tools_json, messages) * ratio)
        if cuts:
            for cut in cuts:
                counters.tool_result_cuts += 1
                context_notice(render, cut.describe())
            _log.info(
                "cut %d tool result(s) to fit the window: ~%d tokens (limit %d)",
                len(cuts),
                expected,
                limit,
            )
            self.store.save(self.session)
            whole = request_tokens(system, tools_json, messages)
        if expected < limit:
            return whole
        dropped = False
        if drop_from is not None:
            index = next((i for i, m in enumerate(messages) if m is drop_from), None)
            if index is not None:
                del messages[index:]
                dropped = True
        self.store.save(self.session)
        raise ContextOverflowError(expected, limit, dropped_request=dropped)

    def _keeps_written_body(self, tool: str, path: str) -> bool:
        """縮めずに残す書き込みか。"""
        return bool(path) and self._targets_plan_file(tool, SimpleNamespace(path=path))

    def _shrinks_written(self) -> bool:
        """書き込みの本文を縮めるか。"""
        return self.provider.kind is not ProviderKind.ANTHROPIC

    def _shrink_written(
        self,
        assistant: ConversationMessage,
        results: list[ToolResultBlock],
        counters: _TurnCounters,
    ) -> None:
        """いまの応答に書き込みがあれば、それより前の大きい書き込みの本文を縮める。"""
        if not written_bodies.writes_text(assistant, {r.tool_use_id: r for r in results}):
            return
        messages = self.session.messages
        index = next((i for i, m in reversed(list(enumerate(messages))) if m is assistant), None)
        if index is not None:
            self._shrink_bodies(counters, upto=index)

    def _shrink_before_compaction(self, counters: _TurnCounters) -> bool:
        """畳む前に、残していた書き込みの本文を縮める。"""
        if not self._shrink_bodies(counters):
            return False
        projected = self._projected_tokens()
        return projected is not None and not self.compactor.needs_compaction(
            TokenUsage(input_tokens=projected), self.config
        )

    def _shrink_bodies(self, counters: _TurnCounters, *, upto: int | None = None) -> bool:
        """窓に比べて大きい書き込みの本文を縮める（``upto`` より前の発言だけ）。"""
        threshold = written_bodies.shrink_threshold(self.config.compaction.context_window)
        if threshold is None or not self._shrinks_written():
            return False
        messages = self.session.messages
        count = self._sent_count()
        before = list(messages)
        omitted = written_bodies.shrink_history(
            messages, keep=self._keeps_written_body, threshold=threshold, upto=upto
        )
        if not omitted:
            return False
        self._note_omitted(omitted, counters)
        mark = self._sent
        if mark is not None and count is not None:
            saved = sum(
                message_tokens(old) - message_tokens(new)
                for old, new in zip(before[:count], messages[:count], strict=True)
                if old is not new
            )
            reduced = mark.input_tokens - int(saved * self._estimate_ratio)
            self._sent = replace(mark, input_tokens=max(1, reduced), floor=False)
        return True

    def _shrink_history(self, counters: _TurnCounters) -> int:
        """会話にあるすべての書き込みの本文を、計画ファイルも含めて縮める。"""
        if not self._shrinks_written():
            return 0
        omitted = written_bodies.shrink_history(self.session.messages, keep=_keep_none)
        if omitted:
            self._note_omitted(omitted, counters)
            _log.info("shrank %d written body(ies) to fit the window", len(omitted))
        return len(omitted)

    def _note_omitted(self, omitted: list[OmittedBody], counters: _TurnCounters) -> None:
        counters.written_bodies_shrunk += len(omitted)
        self.session.omitted_bodies.extend(omitted)

    def _fit_results(
        self,
        calls: list[ToolUseBlock],
        results: list[ToolResultBlock],
        render: Renderer,
        counters: _TurnCounters,
    ) -> list[ToolResultBlock]:
        """1 回の応答のツールの結果を、圧縮の閾値から決めた上限に収める。"""
        budget = result_budget(
            self.config.compact_token_threshold, self.config.compaction.max_tool_result_share
        )
        if budget is None:
            return results
        fitted, cuts = fit_tool_results(results, calls, budget)
        for cut in cuts:
            counters.tool_result_cuts += 1
            context_notice(render, cut.describe())
            _log.info(
                "cut a %s result to fit the context: kept ~%d of ~%d tokens (budget %d)",
                cut.tool,
                cut.kept_tokens,
                cut.total_tokens,
                cut.budget,
            )
        return fitted

    def _expand(self, names: Iterable[str]) -> None:
        """遅延ツールを展開済みにし、セッションにも残す（--resume で同じツール一覧を送るため）。"""
        self._expanded |= set(names)
        self.session.expanded_tools = sorted(self._expanded)

    def _all_schemas(self) -> list[dict[str, object]]:
        if self.config.tools.deferred_tool_loading:
            return [*self.tools.schemas_for(self._expanded), *self._mcp_schemas()]
        return [*self.tools.json_schemas(), *self._mcp_schemas()]

    def _deferred_unexposed(self, name: str) -> bool:
        """``name`` が「遅延ツールかつ未展開」か（遅延ロードのガード判定）。"""
        if name == "enter_plan_mode" and self._plan_mode_active():
            return False
        return (
            self.config.tools.deferred_tool_loading
            and self.tools.is_deferred(name)
            and name not in self._expanded
        )

    def _unknown_tool_block(
        self, call: ToolUseBlock, counters: _TurnCounters | None
    ) -> ToolResultBlock:
        """存在しないツール名への返答。"""
        off = unavailable_tools(self.config)
        hit = next((name for name in off if name_key(name) == name_key(call.name)), None)
        if hit is not None:
            if counters is not None:
                counters.unknown_tool_calls += 1
            return _err_block(call, off[hit])
        loading = self.config.tools.deferred_tool_loading
        exposed = (
            set(self.tools.core_names()) | self._expanded
            if loading
            else {*self.tools.core_names(), *self.tools.deferred_names()}
        )
        available = [
            n for n in [*self.tools.core_names(), *self.tools.deferred_names()] if n in exposed
        ]
        if tool_names.malformed_name(call.name):
            if counters is not None:
                counters.malformed_tool_calls += 1
            return _err_block(
                call, tool_names.malformed_call_message(call.name, self.tools, available)
            )
        if counters is not None:
            key = name_key(call.name) or call.name.strip()
            same = counters.unknown_name == key
            counters.unknown_streak = counters.unknown_streak + 1 if same else 1
            counters.unknown_name = key
            streak = counters.unknown_streak
            if streak >= tool_names.UNKNOWN_TOOL_LOOP_LIMIT:
                counters.unknown_tool_calls += 1
                counters.unknown_loop = call.name
                return _err_block(call, tool_names.unknown_tool_stop_message(call.name, streak))
            if streak >= 2 or tool_names.echoes_its_name(call.name, dict(call.input or {})):
                counters.unknown_tool_calls += 1
                return _err_block(
                    call,
                    tool_names.unknown_tool_insist_message(
                        call.name, available, repeated=streak >= 2
                    ),
                )
        suggestion = tool_names.suggest(call.name, dict(call.input or {}), self.tools, exposed)
        loaded: list[str] = []
        top = suggestion.candidates[0] if suggestion.candidates else None
        if loading and top is not None and top.deferred and top.name not in tool_names.NO_AUTOLOAD:
            self._expand([top.name])
            loaded.append(top.name)
            if counters is not None:
                counters.deferred_autoloads += 1
        if counters is not None:
            counters.unknown_tool_calls += 1
        return _err_block(call, tool_names.unknown_tool_message(call.name, suggestion, loaded))

    def _mcp_schemas(self) -> list[dict[str, object]]:
        return []

    def _known_tool_names(self) -> set[str]:
        return {str(s["name"]) for s in self._all_schemas() if "name" in s}

    @property
    def tool_context(self) -> ToolContext:
        """ツール実行文脈（圧縮の直後の確かめなど、runtime の外から何も変えずに呼ぶ処理が参照）。"""
        return self._ctx

    async def _run_recovery(
        self, fault: Fault, ledger: RecoveryLedger, notify: Notify
    ) -> PlaybookOutcome:
        """手順書を使い、台帳に増えた行を 1 行ずつ知らせる。"""
        async with self._recovery_lock:
            before = len(ledger.steps())
            outcome = await run_playbook(fault, ledger, runner=self.recovery_runner)
            for step in ledger.steps()[before:]:
                notify(RecoveryAttempt(step))
            return outcome

    async def _abort_unhealthy_session(
        self, reason: str, ledger: RecoveryLedger, notify: Notify, iteration: int
    ) -> None:
        """圧縮の直後の確かめが通らなかったことを台帳へ記録し、ターンを中止する。"""
        await self._run_recovery(Fault.UNHEALTHY_SESSION, ledger, notify)
        error = SessionAbortError(
            ui("engine.runtime.the_tools_stopped_working_after", reason=reason)
        )
        notify(TurnAborted(iteration, error))
        raise error

    async def _try_recover_tool_failure(
        self,
        name: str,
        effective_input: object,
        exc: Exception,
        res: ToolResult,
        ledger: RecoveryLedger | None,
        notify: Notify,
        *,
        auto_allowed: bool = False,
    ) -> ToolResult:
        """既知のツール失敗に 1 回だけ手当てを試みる。"""
        if ledger is None:
            return res
        fault = fault_of(exc)
        if fault is None:
            return res
        outcome = await self._run_recovery(fault, ledger, notify)
        if outcome is PlaybookOutcome.SUCCEEDED:
            try:
                return await self._dispatch(name, effective_input, auto_allowed=auto_allowed)
            except Exception as e:
                _log.warning("tool %s failed again after recovery: %s", name, e)
                return ToolResult(output=f"{type(e).__name__}: {e}", is_error=True)
        if playbook_for(fault).fallback is Fallback.ABORT_TURN:
            failed = f"recovery failed for {fault.value}: {res.output}."
            raise SessionAbortError(
                ui("engine.runtime.the_session_may_be_inconsistent", failed=failed)
            ) from exc
        return res

    def _mkdir_not_needed(self, inp: object) -> str | None:
        """作業ディレクトリの中へ mkdir するだけなら「要らない」の文を返す。"""
        if not self.config.tools.mkdir_hint:
            return None
        targets = mkdir_hint.pure_mkdir_targets(str(getattr(inp, "command", "")), self.cwd)
        if targets is None or set(targets) <= self._mkdir_hinted:
            return None
        self._mkdir_hinted.update(targets)
        return mkdir_hint.not_needed_message(targets, self.cwd)

    def _hold_secret_file(
        self, tool_name: str, inp: object, outcome: PermissionOutcome
    ) -> tuple[PermissionOutcome, str | None] | None:
        """秘密ファイルを指すファイルツールの判定を、確認（か拒否）に替える。"""
        field = _SECRET_FILE_TOOLS.get(tool_name)
        if field is None or outcome.decision == "deny" or outcome.scoped:
            return None
        path = getattr(inp, field, None)
        if not isinstance(path, str):
            return None
        try:
            hit = secret_path_hit(path, jail=self._ctx.jail, secrets=self._secret_matcher)
        except Exception as e:
            _log.warning("secret file check could not read the path %r: %s", path, e)
            return None
        if hit is None:
            return None
        if self.config.permission.secret_access == "deny":
            denied = PermissionOutcome(
                decision="deny",
                reason=f"secret file access denied (permission.secret_access=deny): {hit}",
                origin="secret-deny",
            )
            return denied, None
        asked = PermissionOutcome(
            decision="ask", reason=f"secret file: {hit}", origin="secret-file"
        )
        return asked, f"secret file: {hit}"

    def _hold_config_write(
        self, tool_name: str, inp: object, outcome: PermissionOutcome
    ) -> tuple[PermissionOutcome, str] | None:
        """エージェントの設定ファイル（.satsuma.json・.env）を書き換える呼び出しを、必ず確認にする。"""
        if outcome.decision == "deny":
            return None
        try:
            if tool_name == "bash":
                command = str(getattr(inp, "command", ""))
                target = state_write_guard.config_file_bash(command, self.cwd)
            else:
                field = state_write_guard.tool_path_field(tool_name)
                if field is None:
                    return None
                path = str(getattr(inp, field, "") or "")
                target = state_write_guard.config_file_tool(tool_name, path, self.cwd)
        except Exception as e:
            _log.warning("settings file check could not read the input of %s: %s", tool_name, e)
            note = "agent settings file: the target could not be checked"
            return PermissionOutcome(decision="ask", reason=note, origin="agent-config"), note
        if target is None:
            return None
        workspace = self.cwd.resolve()
        shown = target.relative_to(workspace) if target.is_relative_to(workspace) else target
        note = f"agent settings file: {shown.as_posix()}"
        return PermissionOutcome(decision="ask", reason=note, origin="agent-config"), note

    def _first_use_cleared(self) -> set[str]:
        """初回の確認を済ませたツール（PermissionResolver なら親子で共有する入れ物）。"""
        if isinstance(self.policy, PermissionResolver):
            return self.policy.session.first_use_cleared
        return self._first_use_local

    def _hold_first_fetch(
        self, tool_name: str, inp: object, outcome: PermissionOutcome
    ) -> tuple[PermissionOutcome, str] | None:
        """確認なしで許可された web_fetch を、このセッションの最初の 1 回だけ確認に替える。"""
        mode = self.config.web_fetch.confirm
        if tool_name != "web_fetch" or mode == "never" or not self.config.web_fetch.enabled:
            return None
        if outcome.decision != "allow" or outcome.scoped:
            return None
        if mode == "first-use" and "web_fetch" in self._first_use_cleared():
            return None
        url = str(getattr(inp, "url", ""))
        host = urlsplit(url).hostname or url
        if mode == "first-use":
            note = f"first web_fetch this session: {host}"
        else:
            note = f"web_fetch: {host}"
        return PermissionOutcome(decision="ask", reason=note, origin="web-fetch-confirm"), note

    def _hold_lsp_start(
        self, tool_name: str, inp: object, outcome: PermissionOutcome
    ) -> tuple[PermissionOutcome, str, str] | None:
        """確認なしで許可された lsp が言語サーバを初めて起動するなら、言語ごとに 1 回確認にする。"""
        cfg = self.config.lsp
        if tool_name != "lsp" or not cfg.confirm_start or self._ctx.lsp is None:
            return None
        if outcome.decision != "allow" or outcome.scoped:
            return None
        path = getattr(inp, "path", None)
        if not path:
            return None
        language = language_for_path(Path(str(path)))
        if language is None:
            return None
        key = f"lsp:{language}"
        if key in self._first_use_cleared():
            return None
        command = resolve_server_command(language, cfg.servers)
        if command is None:
            return None
        note = f"first {language} language server this session: {command[0]}"
        asked = PermissionOutcome(decision="ask", reason=note, origin="lsp-start-confirm")
        return asked, note, key

    def _unreadable_input(self, label: str, e: Exception) -> str:
        """カード前の禁止判定そのものが落ちたときの拒否文。"""
        _log.warning("%s guard could not read the tool input: %s", label, e)
        return (
            f"invalid input: the {label} check could not interpret this tool input "
            f"({type(e).__name__}: {e}). Fix the path or command and call the tool again."
        )

    def _prohibited_self_invocation(self, name: str, inp: object) -> str | None:
        """bash が SATSUMAESTRO 自身を入れ子起動するなら理由を返す。"""
        if name != "bash":
            return None
        return check_self_invocation(str(getattr(inp, "command", "")))

    def _prohibited_deletion(self, name: str, inp: object) -> str | None:
        """絶対禁止の削除なら理由を返す。"""
        if name == "delete_path":
            return check_path(str(getattr(inp, "path", "")), self.cwd)
        if name == "bash":
            return check_bash(str(getattr(inp, "command", "")), self.cwd)
        return None

    def _prohibited_state_write(self, name: str, inp: object) -> str | None:
        """状態ディレクトリを書き換える呼び出しなら理由を返す。"""
        visible = [self.cwd / plan_file_relpath(self.session.session_id)]
        if name == "bash":
            return state_write_guard.check_bash(
                str(getattr(inp, "command", "")), self.cwd, visible=visible
            )
        field = state_write_guard.tool_path_field(name)
        if field is None:
            return None
        return state_write_guard.check_tool(
            name, str(getattr(inp, field, "") or ""), self.cwd, visible=visible
        )

    def _state_write_guidance(self, name: str, inp: object) -> str:
        """状態ディレクトリへの書き込みを断るときの補足。"""
        field = state_write_guard.tool_path_field(name)
        if field is not None:
            guidance = state_write_guard.plan_dir_guidance(
                name,
                str(getattr(inp, field, "") or ""),
                self.cwd,
                plan_file_relpath(self.session.session_id),
            )
            if guidance is not None:
                return guidance
        return state_write_guard.PROHIBITED_STATE_WRITE_GUIDANCE

    def _deletion_targets(self, name: str, inp: object) -> list[Path]:
        """この呼び出しが消すパス（削除でなければ空。bash はベストエフォート）。"""
        if name == "delete_path":
            return path_targets(str(getattr(inp, "path", "")), self.cwd)
        if name == "bash":
            return bash_targets(str(getattr(inp, "command", "")), self.cwd)
        return []

    def _over_step_limit(self, name: str, inp: object) -> int:
        """段の実行中の write_file / append_file が段の上限を超えていれば、その行数（他は 0）。"""
        if (
            name not in _CHUNKED_WRITE_TOOLS
            or not self.session.plan_steps
            or self.session.plan_step_index < 1
            or not self.session.plan_budget_lines
            or not self._step_budget_enabled()
            or self._targets_plan_file(name, inp)
        ):
            return 0
        lines = len(str(getattr(inp, "content", "") or "").splitlines())
        return lines if lines > self.session.plan_budget_lines else 0

    def _new_output_target(self, name: str, inp: object) -> Path | None:
        """write_file / append_file がまだ無いファイルを作るなら、その絶対パス。"""
        if name not in _CHUNKED_WRITE_TOOLS:
            return None
        try:
            target = self._ctx.jail.resolve(str(getattr(inp, "path", "")))
        except (PathEscapeError, OSError, ValueError):
            return None
        return None if target.exists() else target

    async def _run_tools(
        self,
        calls: list[ToolUseBlock],
        prompter: PermissionPrompter,
        render: Renderer,
        notify: Notify,
        iteration: int,
        recovery_ctx: RecoveryLedger | None = None,
        counters: _TurnCounters | None = None,
    ) -> tuple[list[ToolResultBlock], TokenUsage]:
        """ツール群を実行し ``(結果ブロック列, ツール内部で消費した usage 合計)`` を返す。"""
        delegated = TokenUsage()

        async def one(call: ToolUseBlock) -> ToolResultBlock:
            if counters is not None and counters.user_denied is not None:
                return _not_run_block(call, counters.user_denied)
            written = call.name
            canonical = (
                tool_names.resolve_variant(written, self.tools)
                if self.tools.get(written) is None
                else None
            )
            if canonical is not None:
                call = call.model_copy(update={"name": canonical})
            if counters is not None and self.tools.get(call.name) is not None:
                counters.unknown_streak, counters.unknown_name = 0, ""
            notify(ToolDispatched(iteration, call.name))
            quiet = self._plan_clamp_predicted(call)
            block = await execute(call, quiet=quiet)
            if canonical is not None:
                if counters is not None:
                    counters.tool_name_fixes += 1
                block = block.model_copy(
                    update={
                        "output": block.output
                        + f"\nnote: `{written}` is not a tool name; ran {canonical}."
                    }
                )
            notify(ToolReturned(iteration, call.name, block.is_error))
            if quiet:
                if block.is_error and block.output.startswith(PLAN_MODE_DENY_PREFIX):
                    return block
                render.on_tool_start(ToolUse(id=call.id, name=call.name, input=call.input))
            render.on_tool_result(block)
            return block

        async def execute(call: ToolUseBlock, *, quiet: bool = False) -> ToolResultBlock:
            nonlocal delegated
            tool_use = ToolUse(id=call.id, name=call.name, input=call.input)
            if not quiet:
                render.on_tool_start(tool_use)

            if self.tools.get(call.name) is None:
                return self._unknown_tool_block(call, counters)

            if self._deferred_unexposed(call.name):
                if call.name in tool_names.NO_AUTOLOAD:
                    return _err_block(
                        call,
                        f"tool {call.name} is a deferred tool and is not loaded yet. Call "
                        f'tool_search first (e.g. query "select:{call.name}") to make it '
                        "callable, then retry.",
                    )
                self._expand([call.name])
                if counters is not None:
                    counters.deferred_autoloads += 1
                if not self.config.tools.deferred_run_on_first_call:
                    return _err_block(call, tool_names.autoload_message(call.name, self.tools))

            copied = written_bodies.copied_omission(call.name, call.input)
            if copied is not None:
                if counters is not None:
                    counters.omitted_copies += 1
                return _err_block(call, copied)

            try:
                inp = self.tools.validate_input(tool_use)
            except ToolError as e:
                return _err_block(call, f"invalid input: {e}")

            hook = await self.hooks.pre_tool_use(tool_use, inp)
            if hook.deny:
                return _err_block(call, hook.reason or "denied by hook")

            effective_input = inp
            input_dict = call.input
            if hook.input is not None:
                input_dict = hook.input
                tool_use = ToolUse(id=call.id, name=call.name, input=hook.input)
                try:
                    effective_input = self.tools.validate_input(tool_use)
                except ToolError as e:
                    return _err_block(call, f"invalid hook-rewritten input: {e}")

            try:
                prohibited = self._prohibited_deletion(call.name, effective_input)
            except Exception as e:
                return _err_block(call, self._unreadable_input("deletion", e))
            if prohibited is not None:
                return _err_block(call, f"{prohibited}\n{PROHIBITED_DELETION_GUIDANCE}")

            try:
                state_write = self._prohibited_state_write(call.name, effective_input)
            except Exception as e:
                return _err_block(call, self._unreadable_input("state-write", e))
            if state_write is not None:
                if counters is not None:
                    counters.state_write_blocks += 1
                return _err_block(
                    call, f"{state_write}\n{self._state_write_guidance(call.name, effective_input)}"
                )

            try:
                nested = self._prohibited_self_invocation(call.name, effective_input)
            except Exception as e:
                return _err_block(call, self._unreadable_input("self-invocation", e))
            if nested is not None:
                repeat = ""
                if counters is not None:
                    counters.self_invocation_blocks += 1
                    if counters.self_invocation_blocks >= 2:
                        repeat = f"{PROHIBITED_SELF_INVOCATION_REPEAT}\n"
                return _err_block(call, f"{nested}\n{repeat}{PROHIBITED_SELF_INVOCATION_GUIDANCE}")

            if self._reads_missing_plan_file(call.name, effective_input):
                return _result_block(
                    call,
                    ToolResult(
                        output=PLAN_FILE_READ_MISSING.format(plan_file=self.plan_file_relpath())
                    ),
                )

            required = classify(call.name, effective_input, self._spec_mode(call.name))
            if self._plan_mode_active() and self._targets_plan_file(call.name, effective_input):
                required = PermissionMode.READ_ONLY
            outcome = self.policy.authorize(call.name, input_dict, required, override=hook.override)
            if (
                outcome.decision != "deny"
                and self._plan_mode_active()
                and not mode_le(required, PermissionMode.READ_ONLY)
            ):
                return _err_block(
                    call,
                    f"{PLAN_MODE_DENY_PREFIX} {call.name} requires {required.value}, but only "
                    "read-only tools are allowed while planning. Write the plan to "
                    f"{self.plan_file_relpath()} with write_file (its directory already "
                    "exists) and call exit_plan_mode to ask the user for approval.",
                )
            if (
                counters is not None
                and call.name != "bash"
                and not mode_le(required, PermissionMode.READ_ONLY)
            ):
                counters.mutating_attempted += 1
            try:
                deleting = self._deletion_targets(call.name, effective_input)
            except Exception as e:
                return _err_block(call, self._unreadable_input("deletion-target", e))
            outputs_note: str | None = None
            if deleting and outcome.decision != "deny":
                lost = self._session_outputs.affected(deleting)
                if lost and not output_guard.requests_deletion(self.session.prompt_history[-2:]):
                    if counters is not None:
                        counters.output_delete_blocks += 1
                    return _err_block(call, output_guard.blocked_message(lost, self.cwd.resolve()))
                if lost:
                    outputs_note = output_guard.card_note(lost, self.cwd.resolve())
            warning: str | None = None
            held = self._hold_secret_file(call.name, effective_input, outcome)
            if held is not None:
                outcome, warning = held
                if counters is not None:
                    counters.secret_read_checks += 1
            config_write = self._hold_config_write(call.name, effective_input, outcome)
            if config_write is not None:
                outcome, note = config_write
                warning = note if warning is None else f"{warning}; {note}"
            first_fetch = self._hold_first_fetch(call.name, effective_input, outcome)
            if first_fetch is not None:
                outcome, warning = first_fetch
                if counters is not None:
                    counters.first_fetch_asks += 1
            lsp_start_key: str | None = None
            lsp_start = self._hold_lsp_start(call.name, effective_input, outcome)
            if lsp_start is not None:
                outcome, warning, lsp_start_key = lsp_start
                if counters is not None:
                    counters.lsp_start_asks += 1
            if (
                call.name == "bash"
                and outcome.decision == "ask"
                and outcome.origin == "mode-escalation"
                and self.config.permission.bash_auto_allow == "read-only"
            ):
                command = str(getattr(effective_input, "command", ""))
                verdict_aa = evaluate_auto_allow(
                    command, jail=self._ctx.jail, secrets=self._secret_matcher
                )
                if verdict_aa.kind == "allow":
                    outcome = PermissionOutcome(
                        decision="allow",
                        reason="auto-allow: provably read-only",
                        origin="auto-allow",
                    )
                    notify(AutoAllowed(iteration, call.name, command))
                elif verdict_aa.kind == "secret":
                    paths = ", ".join(verdict_aa.paths)
                    if self.config.permission.secret_access == "deny":
                        outcome = PermissionOutcome(
                            decision="deny",
                            reason=(
                                "secret file access denied "
                                f"(permission.secret_access=deny): {paths}"
                            ),
                            origin="secret-deny",
                        )
                    else:
                        warning = f"secret file: {paths}"
            if counters is not None and call.name == "bash" and outcome.origin != "auto-allow":
                counters.mutating_attempted += 1
            over = (
                self._over_step_limit(call.name, effective_input)
                if outcome.decision != "deny"
                else 0
            )
            if over:
                step = self.session.plan_step_index
                if self._step_write_rejects.get(step, 0) < STEP_WRITE_REJECT_LIMIT:
                    self._step_write_rejects[step] = self._step_write_rejects.get(step, 0) + 1
                    if counters is not None:
                        counters.step_write_rejects += 1
                    message = step_write_reject_message(
                        call.name,
                        str(getattr(effective_input, "path", "")),
                        over,
                        self.session.plan_budget_lines,
                    )
                    return _err_block(call, message)
                if counters is not None:
                    counters.step_write_overrides += 1
            if call.name == "bash" and outcome.decision != "deny":
                verdict = review_command(
                    str(getattr(effective_input, "command", "")),
                    self._base_mode(),
                    self.cwd,
                )
                if verdict.kind == "block":
                    return _err_block(call, f"blocked by bash validation: {verdict.message}")
                if verdict.kind == "warn":
                    warning = verdict.message
                    if outcome.decision == "allow":
                        outcome = PermissionOutcome(decision="ask", reason=verdict.message)
            if outcome.decision == "ask" and call.name == "bash":
                skipped = self._mkdir_not_needed(effective_input)
                if skipped is not None:
                    if counters is not None:
                        counters.mkdir_skips += 1
                    return _result_block(call, ToolResult(output=skipped))
            if outcome.decision == "ask":
                if outputs_note is not None:
                    warning = outputs_note if warning is None else f"{outputs_note}\n{warning}"
                if counters is not None and counters.user_denied is not None:
                    return _not_run_block(call, counters.user_denied)
                with paused(render):
                    outcome = prompter.decide(tool_use, required, warning=warning)
                    if outcome.origin == DETAIL_REQUEST_ORIGIN and isinstance(
                        prompter, PermissionDetailPrompter
                    ):
                        if counters is not None:
                            counters.permission_details += 1
                        with prompter.waiting_for_detail():
                            detail = await self._permission_detail(tool_use, input_dict, warning)
                        outcome = prompter.decide_with_detail(tool_use, required, warning, detail)
                if outcome.decision == "ask":
                    outcome = PermissionOutcome(
                        decision="deny", reason="permission prompt returned no decision"
                    )
                if outcome.remember and isinstance(self.policy, PermissionResolver):
                    self.policy.record(call.name, input_dict)
                if call.name == "web_fetch" and outcome.decision == "allow":
                    self._first_use_cleared().add("web_fetch")
                elif first_fetch is not None and outcome.reason == NONINTERACTIVE_DENY_REASON:
                    outcome = outcome.model_copy(update={"reason": _FETCH_NONINTERACTIVE})
                if lsp_start_key is not None:
                    if outcome.decision == "allow":
                        self._first_use_cleared().add(lsp_start_key)
                    elif outcome.reason == NONINTERACTIVE_DENY_REASON:
                        outcome = outcome.model_copy(update={"reason": _LSP_NONINTERACTIVE})
                if (
                    outcome.decision == "deny"
                    and outcome.origin == USER_DENY_ORIGIN
                    and counters is not None
                ):
                    counters.user_denied = call.name
            if outcome.decision == "deny":
                return _err_block(call, outcome.reason)

            plan_note: str | None = None
            plan_text: str | None = None
            if call.name == "exit_plan_mode" and self._plan_mode_active():
                plan_text = self._read_plan_file()
                if plan_text is None and not self._missing_plan_nudges:
                    self._missing_plan_nudges += 1
                    return _result_block(
                        call,
                        ToolResult(
                            output=PLAN_EXIT_NO_PLAN.format(plan_file=self.plan_file_relpath())
                        ),
                    )
                if plan_text is not None and self._step_budget_enabled():
                    budget = self._plan_budget()
                    self.last_plan_budget = budget
                    plan_steps = parse_plan_steps(plan_text)
                    budget_verdict = check_plan_budget(plan_steps, budget)
                    if budget_verdict is not None:
                        self._plan_budget_verdicts.append(budget_verdict.kind)
                    if (
                        budget_verdict is not None
                        and budget_verdict.sends_back
                        and self._plan_budget_sent.count(budget_verdict.kind)
                        < budget_verdict.sendback_limit
                        and self._plan_budget_sendbacks < MAX_SENDBACKS
                        and self._has_budget(iteration)
                    ):
                        self._plan_budget_sendbacks += 1
                        self._plan_budget_sent.append(budget_verdict.kind)
                        plan_sendback(render, budget_verdict.summary())
                        return _result_block(
                            call,
                            ToolResult(output=budget_verdict.note(self.plan_file_relpath())),
                        )
                    if budget_verdict is not None and budget_verdict.sends_back:
                        self._plan_budget_forced = True
                    if isinstance(prompter, PlanBudgetAware):
                        prompter.set_plan_budget(budget, plan_steps, budget_verdict)
                with paused(render):
                    plan_verdict = (
                        prompter.confirm_plan_exit(plan_text)
                        if isinstance(prompter, PlanExitConfirmer)
                        else PlanExitVerdict(approved=False)
                    )
                if not plan_verdict.approved:
                    if plan_verdict.feedback:
                        text = (
                            PLAN_EXIT_REVISE.format(plan_file=self.plan_file_relpath())
                            + f"\n\nUser feedback:\n{plan_verdict.feedback}"
                        )
                    else:
                        text = PLAN_EXIT_NOT_APPROVED
                    return _result_block(call, ToolResult(output=text))
                plan_note = plan_verdict.note or None

            mode_before = self.session.mode
            new_output = self._new_output_target(call.name, effective_input)
            auto_allowed = outcome.origin == "auto-allow"
            try:
                res = await self._dispatch(call.name, effective_input, auto_allowed=auto_allowed)
            except StateDirError as e:
                res = ToolResult(output=str(e), is_error=True)
            except PathEscapeError as e:
                res = ToolResult(output=f"{PATH_ESCAPE_PREFIX}{e}", is_error=True)
            except Exception as e:
                _log.warning("tool %s failed: %s", call.name, e)
                res = ToolResult(output=f"{type(e).__name__}: {e}", is_error=True)
                res = await self._try_recover_tool_failure(
                    call.name,
                    effective_input,
                    e,
                    res,
                    recovery_ctx,
                    notify,
                    auto_allowed=auto_allowed,
                )
            if counters is not None:
                if syntax_check.FLAG_REJECT in res.flags:
                    counters.syntax_rejects += 1
                if syntax_check.FLAG_WARNING in res.flags:
                    counters.syntax_warnings += 1
                if edit_match.FLAG_MISMATCH in res.flags:
                    counters.edit_mismatches += 1
                if edit_match.FLAG_REINDENT in res.flags:
                    counters.edit_reindents += 1
                if read_ledger.FLAG_STALE in res.flags:
                    counters.edit_stale_notes += 1
                if related_code.FLAG_RELATED in res.flags:
                    counters.related_code_notes += 1
                if overwrite_guard.FLAG_REJECT in res.flags:
                    counters.overwrite_rejects += 1
                if overwrite_guard.FLAG_OVERRIDE in res.flags:
                    counters.overwrite_overrides += 1
                if FLAG_EXIT_NOT_ACTIVE in res.flags:
                    counters.plan_exit_not_active += 1
                if FLAG_EXIT_STEP_DONE in res.flags:
                    counters.plan_exit_step_done += 1
            if not res.is_error:
                if new_output is not None and new_output.exists():
                    self._session_outputs.add(new_output)
                if deleting:
                    self._session_outputs.forget(deleting)
            res = await self.hooks.post_tool_use_failure(tool_use, res)
            res = await self.hooks.post_tool_use(tool_use, res)
            if call.name == "tool_search" and not res.is_error:
                self._expand(
                    self.tools.search(
                        str(getattr(effective_input, "query", "")),
                        int(getattr(effective_input, "max_results", 5)),
                    )
                )
            if call.name in _PLAN_MODE_TOOLS:
                if res.is_error:
                    self.session.mode = mode_before
                else:
                    self._apply_mode_effects()
                    extra: list[str] = []
                    if call.name == "exit_plan_mode" and self._stage_plan(plan_text):
                        first = self._hand_first_step()
                        if first is not None:
                            extra.append(first)
                    if plan_note is not None:
                        template = (
                            PLAN_EXIT_APPROVED_NOTE_WHILE_STEPPING
                            if extra
                            else PLAN_EXIT_APPROVED_WITH_NOTE
                        )
                        extra.append(template.format(note=plan_note))
                    elif (
                        call.name == "exit_plan_mode"
                        and not extra
                        and mode_before.main == "plan"
                        and self.session.mode.main == "code"
                    ):
                        extra.append(
                            PLAN_EXIT_APPROVED_IMPLEMENT.format(plan_file=self.plan_file_relpath())
                        )
                    if extra:
                        res = res.model_copy(update={"output": "\n\n".join([res.output, *extra])})
            if res.usage is not None:
                delegated = _add(delegated, res.usage)
            return _result_block(call, res)

        blocks = await self._scheduled_gather(calls, one)
        return blocks, delegated

    async def _dispatch(self, name: str, inp: object, *, auto_allowed: bool = False) -> ToolResult:
        """ツールを実行する（MCP ツールも registry 登録済みのため同一経路）。"""
        from pydantic import BaseModel

        assert isinstance(inp, BaseModel)
        ctx = self._ctx.model_copy(update={"auto_allowed": True}) if auto_allowed else self._ctx
        return await self.tools.run(name, inp, ctx)

    async def _scheduled_gather(
        self,
        calls: list[ToolUseBlock],
        fn: Callable[[ToolUseBlock], Awaitable[ToolResultBlock]],
    ) -> list[ToolResultBlock]:
        """``parallel_safe`` の連続区間は並行、それ以外は直列。"""
        out: list[ToolResultBlock] = []
        i, n = 0, len(calls)
        while i < n:
            if self._is_parallel_safe(calls[i]):
                j = i
                while j < n and self._is_parallel_safe(calls[j]):
                    j += 1
                batch = await asyncio.gather(*(fn(c) for c in calls[i:j]))
                out.extend(batch)
                i = j
            else:
                out.append(await fn(calls[i]))
                i += 1
        return out

    def _is_parallel_safe(self, call: ToolUseBlock) -> bool:
        """並列区間に乗せてよいか。"""
        name = call.name
        cls = self.tools.get(name)
        if cls is not None:
            if not cls.spec.parallel_safe:
                return False
            if name == "lsp":
                raw = call.input.get("action", "") if isinstance(call.input, dict) else ""
                return str(raw).strip().lower() not in ("format", "formatting")
            return True
        for schema in self._mcp_schemas():
            if schema.get("name") == name:
                return bool(schema.get("parallel_safe", False))
        return False

    def _plan_mode_active(self) -> bool:
        return self.session.mode.main == "plan"

    def _plan_clamp_predicted(self, call: ToolUseBlock) -> bool:
        """この呼び出しが plan クランプ（read-only 超過の自動 deny）で差し戻される見込みか。"""
        if not self._plan_mode_active():
            return False
        try:
            inp = self.tools.validate_input(ToolUse(id=call.id, name=call.name, input=call.input))
        except ToolError:
            return False
        if self._targets_plan_file(call.name, inp):
            return False
        required = classify(call.name, inp, self._spec_mode(call.name))
        return not mode_le(required, PermissionMode.READ_ONLY)

    def _base_mode(self) -> PermissionMode:
        """policy / resolver の基準のモード（= code のサブモード）。"""
        base = self.policy.base if isinstance(self.policy, PermissionResolver) else self.policy
        return base.mode

    def _spec_mode(self, name: str) -> PermissionMode:
        """classify に渡す既定必要モード（required_modes 上書き → spec → 未知既定）。"""
        if name in self.policy.required_modes:
            return self.policy.required_modes[name]
        cls = self.tools.get(name)
        if cls is not None:
            return cls.spec.required_mode
        return _UNKNOWN_REQUIRED_MODE


_PROSE_PLAN_MIN_CHARS = 200
_PLAN_SHAPE_RE = re.compile(r"^(?:#{1,6}[ \t]|[-*+][ \t]|\d+[.)][ \t])", re.MULTILINE)


def _looks_like_plan(text: str) -> bool:
    """本文が承認カードに出せる計画か（Markdown の見出し・箇条書き、または十分な長さ）。"""
    return len(_PLAN_SHAPE_RE.findall(text)) >= 2 or len(text) >= _PROSE_PLAN_MIN_CHARS


def _user_msg(text: str) -> ConversationMessage:
    return ConversationMessage(role="user", blocks=[TextBlock(text=text)])


def _with_language(note: str, language: str) -> list[str]:
    """定型文に短い言語指示を併記する（注入はユーザー文より後ろ＝より直近のため）。"""
    extra = language_note(language)
    return [note, extra] if extra else [note]


def _keep_none(tool: str, path: str) -> bool:
    """書き込みの本文をどれも残さない（窓に収める処理の最後の手）。"""
    return False


def _tool_result_msg(results: list[ToolResultBlock]) -> ConversationMessage:
    return ConversationMessage(role="tool", blocks=list(results))


def _message_text(msg: ConversationMessage) -> str:
    """発言の本文とツールの結果（段 1 は exit_plan_mode の結果で渡る）。"""
    return "\n".join(
        b.text if isinstance(b, TextBlock) else b.output
        for b in msg.blocks
        if isinstance(b, TextBlock | ToolResultBlock)
    )


def _text_of(msg: ConversationMessage) -> str:
    return "".join(b.text for b in msg.blocks if isinstance(b, TextBlock))


def _thinking_of(msg: ConversationMessage) -> str:
    return "\n".join(b.thinking for b in msg.blocks if isinstance(b, ThinkingBlock) and b.thinking)


def _compaction_marker(session: Session) -> int:
    """最後の圧縮の時刻（圧縮前は 0）。"""
    return session.compaction.compacted_at_ms if session.compaction is not None else 0


async def _interruptible(
    task: asyncio.Future[tuple[str | None, str | None]],
) -> tuple[str | None, str | None]:
    """説明の Task を待つ。"""
    loop = asyncio.get_running_loop()
    previous: Any = None
    swapped = False
    if threading.current_thread() is threading.main_thread():
        try:
            previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, lambda *_: loop.call_soon_threadsafe(task.cancel))
            swapped = True
        except (ValueError, OSError):
            swapped = False
    try:
        return await task
    except asyncio.CancelledError:
        if not task.cancelled():
            raise
        return None, "interrupted"
    finally:
        if swapped:
            signal.signal(signal.SIGINT, previous)


def _not_run_block(call: ToolUseBlock, denied: str) -> ToolResultBlock:
    """ユーザーの拒否でターンを止めたため、実行しなかった呼び出しの結果。"""
    return _err_block(
        call,
        f"not run: the user declined `{denied}` earlier in this response, "
        "so the turn was stopped before this call.",
    )


def _err_block(call: ToolUseBlock, reason: str) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=call.id, tool_name=call.name, output=reason, is_error=True)


def _result_block(call: ToolUseBlock, res: ToolResult) -> ToolResultBlock:
    return ToolResultBlock(
        tool_use_id=call.id,
        tool_name=call.name,
        output=res.output,
        is_error=res.is_error,
    )


def _promises_a_plan(text: str, plan_file: str) -> bool:
    """本文が**承認ツールか計画ファイル**に触れているか（＝言うだけで作らなかった）。"""
    return "exit_plan_mode" in text or plan_file in text


_TEXT_CALL_HEAD_LINES = 3


def _detect_text_tool_call(text: str, tool_names: set[str]) -> str | None:
    """本文テキストがツール呼び出しの JSON なら、そのツール名を返す（保守的判定）。"""
    if not tool_names:
        return None
    s = strip_code_fence(text.strip())
    if s.startswith("{"):
        return _tool_call_in(s, tool_names)
    lines = s.splitlines()
    seen = 0
    for i, line in enumerate(lines):
        head = line.strip()
        if not head or head.startswith("```"):
            continue
        seen += 1
        if seen > _TEXT_CALL_HEAD_LINES:
            break
        if head.startswith("{"):
            name = _tool_call_in("\n".join(lines[i:]).lstrip(), tool_names)
            if name is not None:
                return name
    return None


def _tool_call_in(s: str, tool_names: set[str]) -> str | None:
    """``{`` で始まる ``s`` の先頭の JSON が、登録済みのツールの呼び出しならその名前。"""
    obj = _parse_leading_json(s)
    if not isinstance(obj, dict):
        return None
    calls = obj.get("tool_calls")
    if isinstance(calls, list):
        if len(calls) == 1 and isinstance(calls[0], dict):
            inner = calls[0]
            fn = inner.get("function")
            return _match_tool_call(fn if isinstance(fn, dict) else inner, tool_names)
        return None
    return _match_tool_call(obj, tool_names)


def _match_tool_call(obj: dict[str, Any], tool_names: set[str]) -> str | None:
    """単一オブジェクトが「登録済みツール名 + 引数キー」を持つならツール名を返す。"""
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    has_args = any(k in obj for k in ("arguments", "parameters", "input", "args"))
    if isinstance(name, str) and name in tool_names and has_args:
        return name
    return None


def _parse_leading_json(s: str) -> Any:
    """先頭の JSON 値をパースする（末尾に余分な文字があっても許容）。"""
    try:
        obj, _end = json.JSONDecoder().raw_decode(s)
        return obj
    except ValueError:
        return None


def _add(a: TokenUsage, b: TokenUsage | None) -> TokenUsage:
    if b is None:
        return a
    return TokenUsage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read=a.cache_read + b.cache_read,
        cache_write=a.cache_write + b.cache_write,
    )
