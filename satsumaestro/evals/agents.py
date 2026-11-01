"""評価対象エージェントのアダプタ。"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import RuntimeFactory
from satsumaestro.engine.turn import DenyPrompter, PlanExitVerdict, TurnSummary, resolve_output
from satsumaestro.evals.models import EvalPermissionMode
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.state import LanguageState
from satsumaestro.i18n.text import ui
from satsumaestro.orchestration.hook import DecompositionCoordinator
from satsumaestro.orchestration.runner import Orchestrator
from satsumaestro.prompt.builder import SystemPromptBuilder, format_tool_catalog
from satsumaestro.prompt.instructions import instruction_tokens
from satsumaestro.providers.base import Provider
from satsumaestro.providers.retry import is_connection_failure
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ModeState, PlanStepRecord, TokenUsage, WriteChunk
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, unavailable_tools
from satsumaestro.util.errors import ConfigError, EvalAgentError, SatsumaError
from satsumaestro.util.lang import UNDETERMINED, detect_language, strip_non_prose
from satsumaestro.util.proc import terminate

_PROMPT_PLACEHOLDER = "{prompt}"
DEFAULT_AGENT_TIMEOUT_S = 600.0


class AgentAnswer(BaseModel):
    """1 タスクに対するエージェントの回答と計測値。"""

    final_text: str = ""
    empty_final: bool = False
    iterations: int = 0
    tool_calls: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
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
    write_chunks: list[WriteChunk] = Field(default_factory=list)
    tokens_per_s: float = 0.0
    iteration_limit: int = 0
    error: str = ""
    connection_lost: bool = False
    plan_step_records: list[PlanStepRecord] = Field(default_factory=list)
    plan_budget_sendbacks: int = 0
    plan_budget_lines: int = 0
    plan_budget_verdicts: list[str] = Field(default_factory=list)
    plan_budget_forced: bool = False


def _answer_from(
    summary: TurnSummary,
    *,
    plan_exits: int,
    iteration_limit: int,
    error: str = "",
    connection_lost: bool = False,
) -> AgentAnswer:
    """ターンの集計から回答を作る（例外で終えたターンは ``error`` 付きで、その時点まで）。"""
    final_text, _ = resolve_output(summary)
    return AgentAnswer(
        final_text=final_text,
        empty_final=not summary.final_text and not error,
        iterations=summary.iterations,
        iteration_limit=iteration_limit,
        tool_calls=summary.tool_calls,
        usage=summary.usage,
        compacted=summary.compacted,
        tool_call_repairs=summary.tool_call_repairs,
        empty_tool_args=summary.empty_tool_args,
        reminders=summary.reminders,
        decomposition_reason=(
            summary.decomposition.decision.reason if summary.decomposition else ""
        ),
        delegated_usage=summary.delegated_usage,
        plan_exits=plan_exits,
        language_expected=summary.language_expected,
        language_detected=summary.language_detected,
        language_repaired=summary.language_repaired,
        prompt_prefix_breaks=summary.prompt_prefix_breaks,
        timeout_recoveries=summary.timeout_recoveries,
        timeout_resumes=summary.timeout_resumes,
        tool_call_rewrites=summary.tool_call_rewrites,
        repetition_aborts=summary.repetition_aborts,
        syntax_rejects=summary.syntax_rejects,
        syntax_warnings=summary.syntax_warnings,
        edit_mismatches=summary.edit_mismatches,
        edit_reindents=summary.edit_reindents,
        edit_stale_notes=summary.edit_stale_notes,
        related_code_notes=summary.related_code_notes,
        overwrite_rejects=summary.overwrite_rejects,
        overwrite_overrides=summary.overwrite_overrides,
        step_write_rejects=summary.step_write_rejects,
        step_write_overrides=summary.step_write_overrides,
        plan_exit_not_active=summary.plan_exit_not_active,
        plan_exit_step_done=summary.plan_exit_step_done,
        mkdir_skips=summary.mkdir_skips,
        output_delete_blocks=summary.output_delete_blocks,
        state_write_blocks=summary.state_write_blocks,
        secret_read_checks=summary.secret_read_checks,
        first_fetch_asks=summary.first_fetch_asks,
        lsp_start_asks=summary.lsp_start_asks,
        tool_result_cuts=summary.tool_result_cuts,
        written_bodies_shrunk=summary.written_bodies_shrunk,
        omitted_copies=summary.omitted_copies,
        presend_compactions=summary.presend_compactions,
        context_truncations=summary.context_truncations,
        prefill_waits=summary.prefill_waits,
        unknown_tool_calls=summary.unknown_tool_calls,
        malformed_tool_calls=summary.malformed_tool_calls,
        args_only_replies=summary.args_only_replies,
        no_change_reports=summary.no_change_reports,
        tool_name_fixes=summary.tool_name_fixes,
        deferred_autoloads=summary.deferred_autoloads,
        plan_exit_nudges=summary.plan_exit_nudges,
        plan_steps_run=summary.plan_steps_run,
        write_chunks=summary.write_chunks,
        tokens_per_s=summary.tokens_per_s,
        plan_step_records=summary.plan_step_records,
        plan_budget_sendbacks=summary.plan_budget_sendbacks,
        plan_budget_lines=summary.plan_budget_lines,
        plan_budget_verdicts=summary.plan_budget_verdicts,
        plan_budget_forced=summary.plan_budget_forced,
        error=error,
        connection_lost=connection_lost,
    )


@runtime_checkable
class EvalAgent(Protocol):
    """評価対象エージェントの境界。"""

    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    async def run(
        self,
        prompt: str,
        workspace: Path,
        max_iterations: int | None = None,
        *,
        permission_mode: EvalPermissionMode = "read-only",
    ) -> AgentAnswer: ...


class AutoApprovePlanPrompter(DenyPrompter):
    """評価専用: ``exit_plan_mode`` の承認カードだけ自動承認する。"""

    def __init__(self) -> None:
        self.approvals = 0

    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        self.approvals += 1
        return PlanExitVerdict(approved=True)


_SATSUMA_SPEC_SUFFIXES = ("+decompose", "+plan")

PLAN_EVAL_PROTOCOL = (
    "\n\n(Plan mode protocol for this task: first investigate with read-only tools and "
    "write the plan to the plan file as instructed above, then call exit_plan_mode. "
    "Give the final answer only after the plan is approved, by carrying out that plan.)"
)


def _strip_satsuma_suffixes(text: str) -> tuple[str, set[str]]:
    """末尾の ``+decompose`` / ``+plan`` を（順不同・重複可で）剥がし、見つけた旗名を返す。"""
    found: set[str] = set()
    changed = True
    while changed:
        changed = False
        for suffix in _SATSUMA_SPEC_SUFFIXES:
            if text.endswith(suffix):
                found.add(suffix[1:])
                text = text[: -len(suffix)]
                changed = True
    return text, found


def parse_satsuma_agent_spec(spec: str) -> tuple[str, bool, bool]:
    """``satsuma[:<model>][+decompose][+plan]`` を ``(model, decompose, plan)`` に分解する。"""
    base, _, model = spec.partition(":")
    _, base_flags = _strip_satsuma_suffixes(base)
    model, model_flags = _strip_satsuma_suffixes(model)
    flags = base_flags | model_flags
    return model, "decompose" in flags, "plan" in flags


def is_satsuma_agent_spec(spec: str) -> bool:
    """``--agent`` 指定が satsuma（インプロセス）系か。"""
    return _strip_satsuma_suffixes(spec.partition(":")[0])[0] == "satsuma"


class SatsumaAgent:
    """SATSUMAESTRO 自身をインプロセスで実行する（既定経路）。"""

    def __init__(
        self,
        provider: Provider,
        config: RuntimeConfig,
        model: str,
        *,
        decompose: bool = False,
        plan: bool = False,
        provider_builder: Callable[[str], Provider] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._model = model
        self._decompose = decompose
        self._plan = plan
        self._provider_builder = provider_builder or (lambda _model: provider)
        self._env = env or {}

    @property
    def name(self) -> str:
        name = "satsuma"
        if self._decompose:
            name += "+decompose"
        if self._plan:
            name += "+plan"
        return name

    @property
    def model(self) -> str:
        return self._model

    async def run(
        self,
        prompt: str,
        workspace: Path,
        max_iterations: int | None = None,
        *,
        permission_mode: EvalPermissionMode = "read-only",
    ) -> AgentAnswer:
        config = self._config
        if max_iterations is not None:
            config = config.model_copy(update={"max_iterations": max_iterations})
        if self._decompose:
            config = config.model_copy(
                update={
                    "orchestration": config.orchestration.model_copy(update={"enabled": True}),
                    "decomposition": config.decomposition.model_copy(update={"enabled": True}),
                }
            )
        tools = default_registry(
            orchestration=self._decompose,
            core_override=config.tools.core,
            planning_guidance=config.planning_guidance,
            unavailable=unavailable_tools(config),
        )
        mode = PermissionMode(permission_mode)
        policy = default_policy(mode)
        hooks = HookRunner(HookConfig(), workspace)
        session = new_session(workspace, self._model)
        language = LanguageState.from_config(config.language)
        prompter: AutoApprovePlanPrompter | None = None
        if self._plan:
            session.mode = ModeState(main="plan", sub=mode)
            prompter = AutoApprovePlanPrompter()
            task_language = detect_language(strip_non_prose(prompt), min_chars=language.min_chars)
            if language.mode == "auto" and task_language != UNDETERMINED:
                language.set_fixed(task_language)
            prompt = prompt + PLAN_EVAL_PROTOCOL
        orchestrator = None
        decomposition = None
        factory: RuntimeFactory | None = None
        if self._decompose:
            factory = RuntimeFactory(
                provider=self._provider,
                model=self._model,
                config=config,
                tools=tools,
                policy=policy,
                hooks=hooks,
                cwd=workspace,
                provider_builder=self._provider_builder,
                language=language,
            )
            orchestrator = Orchestrator(factory=factory, config=config.orchestration, cwd=workspace)
            decomposition = DecompositionCoordinator(
                orchestrator=orchestrator,
                config=config,
                provider=self._provider,
                model=self._model,
                session=session,
                cwd=workspace,
                env=self._env,
                provider_builder=self._provider_builder,
            )
        runtime = ConversationRuntime(
            session=session,
            provider=self._provider,
            tools=tools,
            policy=policy,
            prompt_builder=SystemPromptBuilder(
                instruction_tokens=instruction_tokens(config.compaction.context_window),
                tool_catalog=(
                    format_tool_catalog(
                        tools.deferred_catalog(),
                        run_on_first_call=config.tools.deferred_run_on_first_call,
                    )
                    if config.tools.deferred_tool_loading
                    else ""
                ),
            ),
            store=JSONLSessionStore(),
            hooks=hooks,
            config=config,
            compactor=Compactor(),
            orchestrator=orchestrator,
            decomposition=decomposition,
            runtime_factory=factory,
            language=language,
        )
        try:
            summary = await runtime.run_turn(prompt, prompter=prompter)
        except SatsumaError as e:
            aborted = runtime.aborted_summary
            if aborted is None:
                raise
            return _answer_from(
                aborted,
                plan_exits=prompter.approvals if prompter is not None else 0,
                iteration_limit=config.max_iterations,
                error=f"{type(e).__name__}: {e}",
                connection_lost=is_connection_failure(e),
            )
        finally:
            if orchestrator is not None:
                await orchestrator.aclose()
        return _answer_from(
            summary,
            plan_exits=prompter.approvals if prompter is not None else 0,
            iteration_limit=config.max_iterations,
        )


class CliAgent:
    """外部 CLI エージェント（claude / 任意のコマンド）をサブプロセスで実行する。"""

    def __init__(
        self,
        name: str,
        argv: list[str],
        *,
        model: str = "",
        parse_json: bool = False,
        timeout_s: float = DEFAULT_AGENT_TIMEOUT_S,
    ) -> None:
        if not argv:
            raise ConfigError(ui("evals.agents.agent_the_command_is_empty", name=name))
        if not any(_PROMPT_PLACEHOLDER in a for a in argv):
            raise ConfigError(ui("evals.agents.agent_the_command_needs_a", name=name))
        self._name = name
        self._argv = argv
        self._model = model
        self._parse_json = parse_json
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    async def run(
        self,
        prompt: str,
        workspace: Path,
        max_iterations: int | None = None,
        *,
        permission_mode: EvalPermissionMode = "read-only",
    ) -> AgentAnswer:
        argv = [a.replace(_PROMPT_PLACEHOLDER, prompt) for a in self._argv]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(workspace),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as e:
            raise EvalAgentError(
                ui("evals.agents.agent_cannot_start", name=self._name, error=e)
            ) from e
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout_s)
        except TimeoutError as e:
            await terminate(proc)
            limit = f"{self._timeout_s:.0f}s"
            raise EvalAgentError(
                ui("evals.agents.agent_timed_out_after", name=self._name, limit=limit)
            ) from e
        if proc.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            if not detail:
                output = stdout.decode(errors="replace").strip()
                try:
                    error_obj = json.loads(output)
                except json.JSONDecodeError:
                    detail = output
                else:
                    detail = (
                        str(error_obj.get("error") or error_obj.get("message") or output)
                        if isinstance(error_obj, dict)
                        else output
                    )
            detail = detail[:200]
            raise EvalAgentError(f"agent {self._name}: exit {proc.returncode}: {detail}")
        text = stdout.decode(errors="replace").strip()
        if self._parse_json:
            return _parse_agent_json(text)
        return AgentAnswer(final_text=text, empty_final=not text)


def _parse_agent_json(stdout: str) -> AgentAnswer:
    """外部 CLI の JSON 出力を防御的にパースする（JSON でなければ生テキストへ degrade）。"""
    try:
        obj = json.loads(stdout)
    except json.JSONDecodeError:
        return AgentAnswer(final_text=stdout, empty_final=not stdout)
    if not isinstance(obj, dict):
        return AgentAnswer(final_text=stdout, empty_final=not stdout)
    text = obj.get("result") or obj.get("message")
    if not isinstance(text, str) or not text:
        return AgentAnswer(final_text=stdout, empty_final=not stdout)
    iterations = obj.get("num_turns") or obj.get("iterations") or 0
    raw_usage = obj.get("usage")
    usage = TokenUsage()
    if isinstance(raw_usage, dict):
        usage = TokenUsage(
            input_tokens=int(raw_usage.get("input_tokens") or 0),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
        )
    return AgentAnswer(
        final_text=text,
        iterations=int(iterations) if isinstance(iterations, int) else 0,
        usage=usage,
    )


def build_cli_agent(spec: str, *, timeout_s: float = DEFAULT_AGENT_TIMEOUT_S) -> CliAgent:
    """``--agent`` の外部エージェント指定からアダプタを構築する。"""
    if "=" in spec:
        name, _, command = spec.partition("=")
        name, command = name.strip(), command.strip()
        if not name or not command:
            raise ConfigError(ui("evals.agents.invalid_agent_name_command", spec=spec))
        return CliAgent(name, shlex.split(command), parse_json=True, timeout_s=timeout_s)

    kind, _, model = spec.partition(":")
    if kind == "claude":
        binary = shutil.which("claude")
        if binary is None:
            raise ConfigError(ui("evals.agents.the_claude_cli_was_not"))
        argv = [binary, "-p", "--output-format", "json"]
        if model:
            argv += ["--model", model]
        argv.append(_PROMPT_PLACEHOLDER)
        return CliAgent("claude", argv, model=model, parse_json=True, timeout_s=timeout_s)
    forms = "satsuma | claude[:model] | name=command …{prompt}…"
    raise ConfigError(ui("evals.agents.unknown_agent", spec=spec, forms=forms))
