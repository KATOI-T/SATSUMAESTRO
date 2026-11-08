"""Typer CLI・サブコマンド・ランタイム配線。"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sqlite3
import statistics
import sys
from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from satsumaestro.__about__ import __version__
from satsumaestro.cli import language_gate
from satsumaestro.cli.banner import print_banner
from satsumaestro.cli.load_watch import load_probe
from satsumaestro.cli.render import RichRenderer
from satsumaestro.cli.repl import InteractivePrompter, run_repl
from satsumaestro.cli.trust_gate import (
    QUESTION,
    TrustBasis,
    TrustQuit,
    decide_trust,
    question_lines,
    untrusted_notice,
)
from satsumaestro.config.loader import ConfigLoader, save_screen_language
from satsumaestro.config.models import ROLE_NAMES, RuntimeConfig
from satsumaestro.engine.context_fit import (
    ContextFitResult,
    MissingModelNotice,
    apply_context_fit,
    check_model,
    local_server_for,
    source_label,
)
from satsumaestro.engine.provider_switch import (
    MissingModelError,
    build_compactor,
    connect,
    unloaded_text,
)
from satsumaestro.engine.recovery import SessionAbortError
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import RuntimeFactory
from satsumaestro.engine.turn import DenyPrompter, NullRenderer, PermissionPrompter, TurnSummary
from satsumaestro.engine.window_breakdown import WindowBreakdown, window_breakdown
from satsumaestro.evals.agents import (
    EvalAgent,
    SatsumaAgent,
    build_cli_agent,
    is_satsuma_agent_spec,
    parse_satsuma_agent_spec,
)
from satsumaestro.evals.baseline import (
    DEFAULT_KEEP,
    Baseline,
    compare,
    describe_history,
    load_report,
)
from satsumaestro.evals.defects import REPEAT_THRESHOLD, language_leaked, repeated_defects
from satsumaestro.evals.index import render_markdown, scan_reports
from satsumaestro.evals.loader import load_tasks, localize_tasks
from satsumaestro.evals.models import Comparison, EvalReport, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.evals.suite import (
    PLAN_SUFFIX,
    SuiteTasks,
    is_suite_manifest,
    load_suite,
    select_tasks,
)
from satsumaestro.evals.web_routing import run_web_routing, summarize
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.schema import catalog
from satsumaestro.i18n.state import LanguageState, locale_language
from satsumaestro.i18n.text import ui, unknown_language, use_language, use_screen_language
from satsumaestro.mcp.manager import MCPManager
from satsumaestro.orchestration.hook import DecompositionCoordinator
from satsumaestro.orchestration.runner import Orchestrator
from satsumaestro.orchestration.worker_supervisor import WORKER_CHILD_ENV, WorkerSupervisor
from satsumaestro.prompt.builder import (
    SystemPromptBuilder,
    format_skill_catalog,
    format_tool_catalog,
)
from satsumaestro.prompt.instructions import instruction_tokens
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.choice import (
    CHOICE_LABELS,
    HOST_ENV,
    OPENAI_API_BASE_URL,
    current_choice,
    is_choice,
    resolve_endpoint,
)
from satsumaestro.providers.detect import (
    LocalKind,
    ModelContextInfo,
    choose_provider_kind,
    fetch_model_context,
    ollama_root,
)
from satsumaestro.providers.factory import (
    build_provider,
    openai_endpoint,
)
from satsumaestro.providers.local import (
    LlamaCppServer,
    LMStudioServer,
    LocalServer,
    OllamaServer,
    looks_local,
    resolve_server,
    server_for_endpoint,
    unload_used,
    unreached_cause,
)
from satsumaestro.providers.registry import MODEL_SHORTHANDS, expand_model_name
from satsumaestro.rag.embed import DEFAULT_EMBED_ENDPOINT as DEFAULT_EMBED_BASE_URL
from satsumaestro.rag.models import IngestReport
from satsumaestro.rag.service import RAGService
from satsumaestro.security.allowlist import MemoryLayer, PermissionResolver
from satsumaestro.security.auto_allow import SecretMatcher
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.security.policy_markdown import ParsedPolicy, load_policies
from satsumaestro.security.trust import TrustStore, trust_root
from satsumaestro.session.models import WriteChunk, resolve_initial_mode
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.skill.resolver import SkillMeta, list_skills
from satsumaestro.telemetry.log import TraceLog
from satsumaestro.telemetry.recorder import TraceRecorder
from satsumaestro.tools import default_registry, unavailable_tools
from satsumaestro.tools.builtins.rag_search import RagSearch
from satsumaestro.tools.builtins.skill import Skill, UntrustedSkill
from satsumaestro.tools.lsp.registry import LanguageServers
from satsumaestro.tools.lsp.servers import DEFAULT_SERVERS, resolve_server_command
from satsumaestro.tools.mcp_bridge import register_mcp_tools
from satsumaestro.tools.registry import ToolRegistry
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import (
    ConfigError,
    MaxIterationsError,
    ProviderError,
    RagError,
    SatsumaError,
    ToolCallingUnsupportedError,
    ToolError,
)
from satsumaestro.util.logging_setup import configure_logging, wants_stderr
from satsumaestro.util.paths import (
    AGENT_DIRNAME,
    default_config_home,
    ensure_state_gitignore,
    find_project_file,
    project_dirs,
)
from satsumaestro.util.proc import BackgroundProcessRegistry


class OutputFormat(StrEnum):
    """``prompt --output-format`` の許容値（未知値は Typer が拒否）。"""

    text = "text"
    json = "json"
    stream_json = "stream-json"


def _user_screen_language() -> str | None:
    """ユーザーの設定（``~/.satsuma/settings.json``）で選んだ画面の言語。"""
    try:
        return ConfigLoader(Path.cwd(), trusted=False).load_language().chosen_ui
    except (ConfigError, OSError):
        return None


use_screen_language(_user_screen_language())
app = typer.Typer(
    name="satsumaestro",
    help=ui("cli.app.satsumaestro_an_agentic_coding"),
    add_completion=False,
    no_args_is_help=True,
)
_console = Console()
_err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help=ui("cli.app.show_the_version_and_exit"),
        callback=_version_callback,
        is_eager=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help=ui("cli.app.also_show_developer_logs_warning"),
    ),
) -> None:
    """SATSUMAESTRO ルートコマンド。"""
    configure_logging(stderr=wants_stderr(debug))


def _orchestration_flags(cfg: RuntimeConfig, env: Mapping[str, str]) -> tuple[bool, bool]:
    """(orchestration 有効, workers 有効) を返す（fork-bomb ガード込み）。"""
    if env.get(WORKER_CHILD_ENV) == "1":
        return False, False
    orch = cfg.orchestration.enabled
    return orch, orch and cfg.orchestration.workers.enabled


def missing_model_text(notice: MissingModelNotice, *, prefix: str = "") -> Text:
    """存在しないモデルの案内を色分けする（見出しだけ赤、補足は見出し語と取得コマンドを黄色）。"""
    out = Text().append(prefix + notice.headline, style="red")
    if notice.close:
        out.append(f"\n{notice.close_label}: ", style="yellow").append(", ".join(notice.close))
    if notice.shown:
        out.append(f"\n{notice.shown_label}: ", style="yellow").append(", ".join(notice.shown))
        out.append(notice.more)
    before, _, after = notice.pull_hint(notice.pull_command).partition(notice.pull_command)
    out.append("\n" + before).append(notice.pull_command, style="yellow").append(after)
    return out


def _print_error(e: SatsumaError) -> None:
    """``prompt`` / ``repl`` / ``session`` の終了前のエラー表示（存在しないモデルは色分け）。"""
    if isinstance(e, MissingModelError):
        _console.print(missing_model_text(e.notice, prefix="error: "))
    else:
        _console.print(f"[red]error: {e}[/red]")


_TRUST_HELP = ui("cli.app.trust_this_directory_s_settings")
_LANG_HELP = ui("cli.app.fix_the_reply_language_for")
_MODEL_HELP = ui("cli.app.model_shorthands_work")
_ALLOWED_TOOLS_HELP = ui("cli.app.tool_names_to_allow_comma")


def _check_lang(value: str | None) -> str | None:
    """``--lang`` の値を言語の定義と照らす。"""
    if value is not None and catalog().get(value) is None:
        raise typer.BadParameter(unknown_language(value))
    return value


_CWD_HELP = ui("cli.app.working_directory_the_base_for")


def _ask_trust(root: Path) -> str:
    """初めて開くディレクトリの確認。"""
    for line in question_lines(root):
        _console.print(line)
    while True:
        try:
            answer = _console.input(QUESTION, markup=False).strip().lower()
        except EOFError:
            return "n"
        except KeyboardInterrupt:
            return "q"
        if answer in ("y", "n", "q"):
            return answer


def _load_config(cwd: Path, *, trusted: bool) -> RuntimeConfig:
    """設定を読み、会話の持ち主の居ない所の画面の言語を、設定の ``language.ui`` に合わせる。"""
    cfg = ConfigLoader(cwd, trusted=trusted).load()
    use_screen_language(cfg.language.chosen_ui)
    return cfg


def _stdin_is_terminal() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _choose_screen_language(cwd: Path, *, trust: bool) -> None:
    """画面の言語を決める。"""
    trusted = trust or TrustStore().is_trusted(trust_root(cwd))
    try:
        language = ConfigLoader(cwd, trusted=trusted).load_language()
    except ConfigError:
        return
    if language.ui is None and _stdin_is_terminal():
        use_screen_language(_ask_screen_language())
    else:
        use_screen_language(language.chosen_ui)


def _ask_screen_language() -> str | None:
    """初回の画面の言語の問い。"""
    default = locale_language(os.environ)
    for line in language_gate.question_lines():
        _console.print(line)
    while True:
        try:
            answer = _console.input(language_gate.question(default), markup=False)
        except EOFError:
            return None
        except KeyboardInterrupt:
            raise typer.Exit(0) from None
        code = language_gate.parse_answer(answer, default)
        if code is not None:
            break
    chosen = None if code == language_gate.AUTO else code
    use_screen_language(chosen)
    try:
        path = save_screen_language(code)
    except (ConfigError, OSError) as e:
        _console.print(Text(ui("cli.app.screen_language_not_saved", error=str(e)), style="yellow"))
        return None
    name = language_gate.native_name(code)
    _console.print(Text(ui("cli.app.screen_language_saved", name=name, path=path), style="dim"))
    return chosen


def _trusted_line(root: Path, record: Path) -> str:
    return ui("cli.app.trusted_recorded_in", root=root, record=record)


def _decide_trust(cwd: Path, *, trust: bool, interactive: bool) -> bool:
    """開いたディレクトリの設定を今回読むか。"""
    try:
        decision = decide_trust(cwd, trust_flag=trust, ask=_ask_trust if interactive else None)
    except TrustQuit:
        raise typer.Exit(0) from None
    if decision.basis is TrustBasis.ANSWERED:
        _console.print(Text(_trusted_line(decision.root, TrustStore().path), style="dim"))
    elif decision.basis is TrustBasis.DECLINED:
        _console.print(
            Text(
                ui("cli.app.untrusted_continuing_without_the_settings", root=decision.root),
                style="yellow",
            )
        )
    elif decision.basis is TrustBasis.UNTRUSTED:
        typer.echo(untrusted_notice(decision.root), err=True)
    return decision.trusted


async def _assemble_runtime(
    model: str | None,
    cwd: Path,
    interactive: bool,
    resume_path: Path | None = None,
    allowed_tools: str | None = None,
    mode_spec: str | None = None,
    *,
    trusted: bool,
    lang: str | None = None,
) -> tuple[ConversationRuntime, PermissionPrompter, RichRenderer]:
    """設定を読み、ランタイムとプロンプタ・レンダラを組み立てる。"""
    loader = ConfigLoader(cwd, trusted=trusted)
    cfg = loader.load()
    use_screen_language(cfg.language.chosen_ui)
    resolved = expand_model_name(model or cfg.provider.default_model)
    connection = await connect(cfg, resolved, os.environ, build=build_provider)
    provider = connection.provider
    for note in connection.notes:
        _err_console.print(Text(note, style="yellow"))
    fit_base = (cfg.compact_token_threshold, cfg.compaction.context_window)
    fit = await apply_context_fit(cfg, resolved, os.environ)
    orch_enabled, workers_enabled = _orchestration_flags(cfg, os.environ)
    tools = _tool_registry(cfg, orchestration=orch_enabled, workers=workers_enabled)
    mcp_manager = await MCPManager.start(cfg.mcp_servers)
    rag: RAGService | None = None
    try:
        register_mcp_tools(tools, mcp_manager)
        rag = _build_rag(cfg, cwd)
        if rag is not None:
            tools.register(RagSearch)
        skills = list_skills(cwd, state_dir=trusted)
        if skills:
            tools.register(Skill if trusted else UntrustedSkill)
        if allowed_tools:
            try:
                tools = tools.filtered(tools.resolve_allowed_tools([allowed_tools]))
            except ToolError as e:
                raise ConfigError(str(e)) from e
        default_mode = PermissionMode(cfg.permission.default_mode)
        parsed = (
            load_policies(_policy_files(cwd, cfg.permission.policy_file), default_mode)
            if trusted
            else ParsedPolicy(policy=default_policy(default_mode))
        )
        store = JSONLSessionStore()
        store.prune(cwd)
        session = store.load(resume_path) if resume_path else new_session(cwd, resolved)
        if resume_path and session.model != resolved:
            saved = await check_model(cfg, session.model, os.environ)
            if saved is not None and saved.missing:
                notice = ui(
                    "cli.app.the_saved_model_was_not", model=session.model, resolved=resolved
                )
                _err_console.print(f"[yellow]{notice}[/yellow]")
                session.model = resolved
        try:
            initial_mode = resolve_initial_mode(
                session.mode if resume_path else None, mode_spec, parsed.policy.mode
            )
        except ValueError as e:
            raise ConfigError(ui("cli.app.invalid_mode_plan_code_code", error=e)) from e
        session.mode = initial_mode
        parsed.policy.mode = initial_mode.sub

        resolver = PermissionResolver(
            parsed.policy, _memory_layers(cwd, cfg.permission.memory_file, project=trusted)
        )
        hooks = HookRunner(cfg.hooks, cwd, sandbox=cfg.sandbox)
        lsp_registry = (
            LanguageServers(cwd, cfg.lsp, sandbox=cfg.sandbox) if cfg.lsp.enabled else None
        )
        background = BackgroundProcessRegistry(
            cwd / AGENT_DIRNAME / "background" / session.session_id
        )
        language = LanguageState.from_config(cfg.language)
        language.watch(loader.load_language)
        if lang is not None:
            language.set_fixed(lang)
        use_language(language)
        factory = RuntimeFactory(
            provider=provider,
            model=resolved,
            config=cfg,
            tools=tools,
            policy=resolver,
            hooks=hooks,
            cwd=cwd,
            provider_builder=lambda m: build_provider(cfg, m, os.environ),
            mcp=mcp_manager,
            rag=rag,
            background=background,
            lsp=lsp_registry,
            language=language,
        )
        orchestrator: Orchestrator | None = None
        if orch_enabled:
            orchestrator = Orchestrator(
                factory=factory,
                config=cfg.orchestration,
                cwd=cwd,
                state_path=cwd / cfg.orchestration.state_file,
                workers=(
                    WorkerSupervisor(config=cfg.orchestration.workers, cwd=cwd)
                    if workers_enabled
                    else None
                ),
            )
        decomposition: DecompositionCoordinator | None = None
        if orchestrator is not None and cfg.decomposition.enabled:
            decomposition = DecompositionCoordinator(
                orchestrator=orchestrator,
                config=cfg,
                provider=provider,
                model=resolved,
                session=session,
                cwd=cwd,
                env=os.environ,
                provider_builder=lambda m: build_provider(cfg, m, os.environ),
            )
        rt = ConversationRuntime(
            session=session,
            provider=provider,
            tools=tools,
            policy=resolver,
            prompt_builder=_prompt_builder(cfg, tools, skills, interactive=interactive),
            store=store,
            hooks=hooks,
            config=cfg,
            compactor=build_compactor(cfg, provider, resolved, os.environ, build=build_provider),
            mcp=mcp_manager,
            rag=rag,
            runtime_factory=factory,
            orchestrator=orchestrator,
            decomposition=decomposition,
            background=background,
            lsp=lsp_registry,
            language=language,
        )
        rt.fit_base = fit_base
        prompter: PermissionPrompter = (
            InteractivePrompter(_console, plan_steps=cfg.guidance.plan_steps)
            if interactive
            else DenyPrompter()
        )
        if interactive:
            await _print_window(rt, fit)
    except BaseException:
        await _shutdown(mcp_manager, rag)
        raise
    return (
        rt,
        prompter,
        RichRenderer(
            _console,
            show_thinking=cfg.show_thinking,
            stream_tool_args=cfg.stream_tool_args,
            loading_model=_unloaded_model(fit),
            local_server=_local_kind(cfg, rt.session.model),
            cwd=cwd,
            context_limit=lambda: cfg.compact_token_threshold,
        ),
    )


def _tool_registry(cfg: RuntimeConfig, *, orchestration: bool, workers: bool) -> ToolRegistry:
    """起動時のツールの一覧（MCP とスキルは呼び側で足す）。"""
    return default_registry(
        strict_schema=cfg.tool_schema.strict_schema,
        orchestration=orchestration,
        workers=workers,
        core_override=cfg.tools.core,
        planning_guidance=cfg.planning_guidance,
        unavailable=unavailable_tools(cfg),
    )


def _prompt_builder(
    cfg: RuntimeConfig, tools: ToolRegistry, skills: list[SkillMeta], *, interactive: bool
) -> SystemPromptBuilder:
    """起動時の system の組み立て。"""
    return SystemPromptBuilder(
        instruction_tokens=instruction_tokens(cfg.compaction.context_window),
        tool_catalog=(
            format_tool_catalog(
                tools.deferred_catalog(),
                run_on_first_call=cfg.tools.deferred_run_on_first_call,
            )
            if cfg.tools.deferred_tool_loading
            else ""
        ),
        skill_catalog=format_skill_catalog([(m.name, m.description) for m in skills]),
        surface="repl" if interactive else "prompt",
    )


async def _print_window(
    rt: ConversationRuntime,
    fit: ContextFitResult | None,
    *,
    console: Console = _console,
    breakdown_line: bool = False,
) -> None:
    """起動時の窓の確かめ。"""
    if fit is None or fit.context_length is None:
        return
    breakdown = await rt.window_breakdown()
    if breakdown is None:
        return
    if breakdown_line:
        console.print(Text(breakdown.describe(source_label(fit.info.source)), style="dim"))
    if breakdown.tight:
        advice = breakdown.advice(fit.info.server, fit.model, fit.info.slots)
        console.print(Text(advice, style="yellow"))
    if fit.info.unconfirmed:
        rt.window_unconfirmed = True
        if breakdown_line and fit.info.server == "lmstudio":
            console.print(Text(ui("cli.app.window_not_loaded_yet_lmstudio"), style="dim"))
        elif breakdown_line and fit.info.server == "llamacpp":
            console.print(Text(ui("cli.app.window_not_loaded_yet_llamacpp"), style="dim"))
        elif breakdown_line:
            console.print(Text(ui("cli.app.window_not_loaded_yet"), style="dim"))


def _local_kind(cfg: RuntimeConfig, model: str) -> LocalKind:
    """表示の文を選ぶためのローカルのサーバの種類（ローカルでなければ Ollama の文）。"""
    server = local_server_for(cfg, model, os.environ)
    return server.kind if server is not None else "ollama"


def _unloaded_model(fit: ContextFitResult | None) -> str | None:
    """起動時の ``/api/ps`` で未ロードと**確認できた**モデル名。"""
    if fit is None or not fit.info.ps_ok or fit.info.loaded:
        return None
    return fit.model


def _build_telemetry(rt: ConversationRuntime) -> tuple[TraceRecorder | None, TraceLog | None]:
    """テレメトリの記録帳と、そこへ書く受け手を組む。"""
    tcfg = rt.config.telemetry
    if not tcfg.enabled:
        return None, None
    file = None
    if tcfg.jsonl_file:
        file = Path(tcfg.jsonl_file)
        file = file if file.is_absolute() else rt.cwd / file
    log = TraceLog(file)
    return TraceRecorder(rt.session.session_id, log), log


def _rag_db_path(cfg: RuntimeConfig, cwd: Path) -> Path:
    """設定の ``rag.db_path`` をワークスペース基準の絶対パスへ解決する。"""
    db_path = Path(cfg.rag.db_path)
    return db_path if db_path.is_absolute() else cwd / db_path


def _build_rag(cfg: RuntimeConfig, cwd: Path) -> RAGService | None:
    """索引済みワークスペースに限り :class:`RAGService` を構築する。"""
    if not cfg.rag.enabled or not _rag_db_path(cfg, cwd).is_file():
        return None
    try:
        return RAGService(cfg.rag, cwd)
    except (OSError, sqlite3.Error):
        return None


async def _shutdown(
    mcp_manager: MCPManager | None,
    rag: RAGService | None,
    orchestrator: object | None = None,
    background: BackgroundProcessRegistry | None = None,
    lsp: object | None = None,
) -> None:
    """ランタイムが握る外部資源（背景 Task・MCP 子プロセス・RAG の DB/HTTP）を確実に閉じる。"""
    try:
        if isinstance(orchestrator, Orchestrator):
            await orchestrator.aclose()
    finally:
        try:
            if isinstance(lsp, LanguageServers):
                await lsp.aclose()
            if background is not None:
                await background.aclose()
        finally:
            try:
                if mcp_manager is not None:
                    await mcp_manager.aclose()
            finally:
                if rag is not None:
                    await rag.aclose()


async def _unload_on_exit(cfg: RuntimeConfig) -> None:
    """REPL を閉じるとき、このプロセスがローカルのサーバで使ったモデルを降ろす。"""
    for unloaded in await unload_used(cfg.provider):
        _console.print(Text(unloaded_text(unloaded), style="dim"))


async def _drain_background(rt: ConversationRuntime) -> list[dict[str, str]]:
    """一発 ``prompt`` の終了前に背景 Task を扱う（``orchestration.drain_on_exit``）。"""
    orch = rt.orchestrator
    if not isinstance(orch, Orchestrator) or orch.running_count() == 0:
        return []
    if rt.config.orchestration.drain_on_exit:
        n = orch.running_count()
        waiting = ui("cli.app.waiting_for_background_tasks_to", n=n)
        _err_console.print(f"[dim]{waiting}[/dim]")
        await orch.drain()
    return [{"task_id": t.task_id, "status": t.status.value} for t in orch.list_tasks()]


def _policy_files(cwd: Path, policy_file: str) -> list[Path]:
    """git の根から作業ディレクトリまでの policy.md の候補を、根に近い順に返す（重ねる順）。"""
    if Path(policy_file).is_absolute():
        return [Path(policy_file)]
    return [directory / policy_file for directory in reversed(project_dirs(cwd))]


def _memory_layers(cwd: Path, project_memory_file: str, *, project: bool) -> list[MemoryLayer]:
    """memory レイヤを優先順（project → global）で構成する。"""
    project_path = find_project_file(cwd, project_memory_file) or cwd / project_memory_file
    layers = [MemoryLayer(project_path, writable=True)] if project else []
    global_path = default_config_home() / "memory.md"
    if global_path.is_file():
        layers.append(MemoryLayer(global_path, writable=False))
    return layers


async def _run_once(
    text: str,
    model: str | None,
    output_format: str,
    cwd: Path,
    resume_path: Path | None = None,
    allowed_tools: str | None = None,
    mode_spec: str | None = None,
    *,
    trusted: bool,
    lang: str | None = None,
) -> None:
    if output_format == "stream-json":
        await _run_stream(
            text, model, cwd, resume_path, allowed_tools, mode_spec, trusted=trusted, lang=lang
        )
        return
    rt, prompter, render = await _assemble_runtime(
        model,
        cwd,
        interactive=False,
        resume_path=resume_path,
        allowed_tools=allowed_tools,
        mode_spec=mode_spec,
        trusted=trusted,
        lang=lang,
    )
    try:
        recorder, _log = _build_telemetry(rt)
        if output_format == "json":
            summary = await rt.run_turn(text, prompter, NullRenderer(), observer=recorder)
            background = await _drain_background(rt)
            payload = {
                "final_text": summary.final_text,
                "stop_reason": summary.stop_reason,
                "iterations": summary.iterations,
                "usage": summary.usage.model_dump(),
                "delegated_usage": summary.delegated_usage.model_dump(),
                "compacted": summary.compacted,
                "health_warning": summary.health_warning,
                "args_only_replies": summary.args_only_replies,
                "no_change_reports": summary.no_change_reports,
                "background_tasks": background,
                "session_id": rt.session.session_id,
            }
            typer.echo(json.dumps(payload, ensure_ascii=False))
        else:
            render.start_turn()
            done: TurnSummary | None = None
            try:
                done = await rt.run_turn(text, prompter, render, observer=recorder)
            finally:
                render.end_turn(delegated=done.delegated_usage if done else None)
            await _drain_background(rt)
    finally:
        await _shutdown(rt.mcp, rt.rag, rt.orchestrator, rt.background, rt.lsp)


async def _run_stream(
    text: str,
    model: str | None,
    cwd: Path,
    resume_path: Path | None,
    allowed_tools: str | None = None,
    mode_spec: str | None = None,
    *,
    trusted: bool,
    lang: str | None = None,
) -> None:
    """Worker 子 CLI プロトコルで 1 ターン実行する。"""
    from satsumaestro.cli.stream import ProgressLines, emit_error, emit_ready, emit_result

    try:
        rt, prompter, _render = await _assemble_runtime(
            model,
            cwd,
            interactive=False,
            resume_path=resume_path,
            allowed_tools=allowed_tools,
            mode_spec=mode_spec,
            trusted=trusted,
            lang=lang,
        )
    except SatsumaError as e:
        emit_error(e, None)
        raise
    session_id = rt.session.session_id
    try:
        emit_ready(session_id, cwd)
        try:
            summary = await rt.run_turn(text, prompter, NullRenderer(), observer=ProgressLines())
        except SatsumaError as e:
            emit_error(e, session_id)
            raise
        background = await _drain_background(rt)
        emit_result(summary, session_id, background)
    finally:
        await _shutdown(rt.mcp, rt.rag, rt.orchestrator, rt.background, rt.lsp)


async def _repl_session(
    model: str | None,
    cwd: Path,
    allowed_tools: str | None = None,
    mode_spec: str | None = None,
    *,
    trusted: bool,
    lang: str | None = None,
) -> None:
    print_banner(_console)
    rt, prompter, render = await _assemble_runtime(
        model,
        cwd,
        interactive=True,
        allowed_tools=allowed_tools,
        mode_spec=mode_spec,
        trusted=trusted,
        lang=lang,
    )
    assert isinstance(prompter, InteractivePrompter)
    recorder, trace_log = _build_telemetry(rt)
    try:
        await run_repl(
            rt,
            prompter,
            render,
            _console,
            observer=recorder,
            trace_log=trace_log,
            trusted=trusted,
            load_probe=load_probe(rt, os.environ),
        )
    finally:
        try:
            await _shutdown(rt.mcp, rt.rag, rt.orchestrator, rt.background, rt.lsp)
        finally:
            await _unload_on_exit(rt.config)


@app.command(help=ui("cli.app.run_a_single_prompt_non"))
def prompt(
    text: str = typer.Argument(..., help=ui("cli.app.the_prompt_text")),
    model: str | None = typer.Option(None, "--model", "-m", help=_MODEL_HELP),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--output-format", help="text | json | stream-json"
    ),
    cwd: Path | None = typer.Option(None, "--cwd", help=_CWD_HELP),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help=ui("cli.app.session_id_to_continue_multi"),
    ),
    allowed_tools: str | None = typer.Option(None, "--allowed-tools", help=_ALLOWED_TOOLS_HELP),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help=ui("cli.app.run_mode_plan_code_code"),
    ),
    trust: bool = typer.Option(False, "--trust", help=_TRUST_HELP),
    lang: str | None = typer.Option(None, "--lang", help=_LANG_HELP, callback=_check_lang),
) -> None:
    workspace = cwd or Path.cwd()
    if output_format is OutputFormat.stream_json:
        configure_logging(stderr=True)
    try:
        resume_path: Path | None = None
        if resume:
            resume_path = _sessions_folder(workspace) / f"{resume}.jsonl"
            if not resume_path.is_file():
                raise ConfigError(ui("cli.app.resume_session_not_found", resume=resume))
        asyncio.run(
            _run_once(
                text,
                model,
                output_format.value,
                workspace,
                resume_path,
                allowed_tools,
                mode,
                trusted=_decide_trust(workspace, trust=trust, interactive=False),
                lang=lang,
            )
        )
    except (
        ProviderError,
        ConfigError,
        ToolCallingUnsupportedError,
        SessionAbortError,
        MaxIterationsError,
    ) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


@app.command(help=ui("cli.app.interactive_repl"))
def repl(
    model: str | None = typer.Option(None, "--model", "-m", help=_MODEL_HELP),
    cwd: Path | None = typer.Option(None, "--cwd", help=_CWD_HELP),
    allowed_tools: str | None = typer.Option(None, "--allowed-tools", help=_ALLOWED_TOOLS_HELP),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help=ui("cli.app.run_mode_at_startup_plan"),
    ),
    trust: bool = typer.Option(False, "--trust", help=_TRUST_HELP),
    lang: str | None = typer.Option(None, "--lang", help=_LANG_HELP, callback=_check_lang),
) -> None:
    try:
        workspace = cwd or Path.cwd()
        _choose_screen_language(workspace, trust=trust)
        trusted = _decide_trust(workspace, trust=trust, interactive=True)
        asyncio.run(
            _repl_session(model, workspace, allowed_tools, mode, trusted=trusted, lang=lang)
        )
    except (ProviderError, ConfigError) as e:
        _print_error(e)
        raise typer.Exit(1) from e


@app.command(help=ui("cli.app.check_the_health_of_the"))
def doctor(
    cwd: Path | None = typer.Option(None, "--cwd", help=_CWD_HELP),
    probe: bool = typer.Option(
        False,
        "--probe",
        help=ui("cli.app.ask_ollama_about_the_context"),
    ),
    trust: bool = typer.Option(False, "--trust", help=_TRUST_HELP),
) -> None:
    _doctor(cwd or Path.cwd(), probe=probe, trust=trust)


@app.command(help=ui("cli.app.show_the_merged_settings_show"))
def config(
    action: str = typer.Argument("show"),
    cwd: Path | None = typer.Option(None, "--cwd", help=_CWD_HELP),
    trust: bool = typer.Option(False, "--trust", help=_TRUST_HELP),
) -> None:
    workspace = cwd or Path.cwd()
    try:
        trusted = _decide_trust(workspace, trust=trust, interactive=False)
        cfg = _load_config(workspace, trusted=trusted)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    if action == "show":
        data = _redact_config_dump(cfg.model_dump(mode="json"))
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        typer.echo(
            ui("cli.app.unknown_action_use_choices", action=action, choices="show"), err=True
        )
        raise typer.Exit(1)


def _redact_config_dump(data: dict[str, Any]) -> dict[str, Any]:
    """表示用の設定から、MCP サーバへ渡す環境変数の値を伏せる（鍵が入る場所）。"""
    servers = data.get("mcp_servers")
    if isinstance(servers, dict):
        for server in servers.values():
            env = server.get("env") if isinstance(server, dict) else None
            if isinstance(env, dict):
                server["env"] = dict.fromkeys(env, "***")
    return data


@app.command(help=ui("cli.app.list_the_model_shorthands"))
def models() -> None:
    table = Table(title="model shorthands")
    table.add_column("shorthand", style="cyan")
    table.add_column("→ model")
    for shorthand, target in sorted(MODEL_SHORTHANDS.items()):
        table.add_row(shorthand, target)
    _console.print(table)


@app.command(
    "languages",
    help=ui("cli.app.list_the_languages_for_replies"),
)
def languages_cmd(
    as_json: bool = typer.Option(
        False,
        "--json",
        help=ui("cli.app.output_as_json_for_choices"),
    ),
) -> None:
    specs = catalog().languages
    if as_json:
        listed = [
            {"code": s.code, "name": s.name, "native_name": s.native_name, "ui": s.ui}
            for s in specs
        ]
        typer.echo(json.dumps(listed, ensure_ascii=False))
        return
    for s in specs:
        screen = "" if s.ui else "\t" + ui("cli.app.screen_in_english")
        name = s.native_name if s.native_name == s.name else f"{s.native_name} ({s.name})"
        typer.echo(f"{s.code}\t{name}{screen}")


@app.command(
    "trust",
    help=ui("cli.app.trust_a_directory_add_record"),
)
def trust_cmd(
    action: str = typer.Argument("add", help="add | list | remove"),
    cwd: Path | None = typer.Option(None, "--cwd", help=_CWD_HELP),
) -> None:
    store = TrustStore()
    root = trust_root(cwd or Path.cwd())
    if action == "add":
        store.trust(root)
        typer.echo(_trusted_line(root, store.path))
    elif action == "list":
        roots = store.roots()
        for r in roots:
            typer.echo(str(r))
        if not roots:
            typer.echo(ui("cli.app.no_trusted_directories"))
    elif action == "remove":
        if not store.untrust(root):
            typer.echo(ui("cli.app.not_recorded", root=root), err=True)
            raise typer.Exit(1)
        typer.echo(ui("cli.app.removed", root=root))
    else:
        typer.echo(
            ui("cli.app.unknown_action_use_choices", action=action, choices="add | list | remove"),
            err=True,
        )
        raise typer.Exit(1)


@app.command("session", help=ui("cli.app.list_or_resume_sessions"))
def session_cmd(
    action: str = typer.Argument("list", help="list | resume"),
    session_id: str | None = typer.Argument(None),
    model: str | None = typer.Option(None, "--model", "-m", help=_MODEL_HELP),
    cwd: Path | None = typer.Option(None, "--cwd", help=_CWD_HELP),
    trust: bool = typer.Option(False, "--trust", help=_TRUST_HELP),
) -> None:
    cwd = cwd or Path.cwd()
    if action == "list":
        _session_list(cwd)
    elif action == "resume":
        if not session_id:
            typer.echo(
                ui("cli.app.resume_needs_a_session_id"),
                err=True,
            )
            raise typer.Exit(1)
        path = _sessions_folder(cwd) / f"{session_id}.jsonl"
        if not path.is_file():
            typer.echo(
                ui("cli.app.session_not_found", session_id=session_id),
                err=True,
            )
            raise typer.Exit(1)
        try:
            _choose_screen_language(cwd, trust=trust)
            trusted = _decide_trust(cwd, trust=trust, interactive=True)
            asyncio.run(_run_repl_resume(model, cwd, path, trusted=trusted))
        except (ProviderError, ConfigError) as e:
            _print_error(e)
            raise typer.Exit(1) from e
    else:
        typer.echo(
            ui("cli.app.unknown_action_use_choices", action=action, choices="list | resume"),
            err=True,
        )
        raise typer.Exit(1)


@app.command(
    "mcp",
    help=ui("cli.app.manage_mcp_servers_list_connect"),
)
def mcp_cmd(
    action: str = typer.Argument("list", help="list"),
    cwd: Path | None = typer.Option(None, "--cwd", help=_CWD_HELP),
    trust: bool = typer.Option(False, "--trust", help=_TRUST_HELP),
) -> None:
    if action != "list":
        typer.echo(
            ui("cli.app.unknown_action_use_choices", action=action, choices="list"), err=True
        )
        raise typer.Exit(1)
    workspace = cwd or Path.cwd()
    trusted = _decide_trust(workspace, trust=trust, interactive=False)
    try:
        asyncio.run(_mcp_list(workspace, trusted=trusted))
    except (ConfigError, ProviderError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


async def _mcp_list(cwd: Path, *, trusted: bool) -> None:
    cfg = _load_config(cwd, trusted=trusted)
    if not cfg.mcp_servers:
        none = ui("cli.app.no_mcp_servers_are_configured")
        _console.print(f"[dim]{none}[/dim]")
        return
    mgr = await MCPManager.start(cfg.mcp_servers)
    try:
        table = Table(title="MCP servers")
        table.add_column("server", style="cyan")
        table.add_column("transport")
        table.add_column("status")
        table.add_column("tools", justify="right")
        table.add_column("detail")
        by_name = {h.name: h for h in mgr.handles()}
        for name, scfg in cfg.mcp_servers.items():
            h = by_name.get(name)
            status = "connected" if h and h.connected else "degraded"
            tools = str(len(h.tools)) if h else "0"
            table.add_row(name, scfg.transport, status, tools, (h.error if h else "") or "")
        _console.print(table)
    finally:
        await mgr.aclose()


@app.command(
    "rag",
    help=ui("cli.app.manage_the_workspace_index_ingest"),
)
def rag_cmd(
    action: str = typer.Argument("stats", help="ingest | search | stats"),
    target: str | None = typer.Argument(
        None,
        help=ui("cli.app.ingest_the_path_default_cwd"),
    ),
    k: int = typer.Option(
        8,
        "--k",
        help=ui("cli.app.number_of_chunks_search_returns"),
    ),
    cwd: Path | None = typer.Option(None, "--cwd", help=_CWD_HELP),
    trust: bool = typer.Option(False, "--trust", help=_TRUST_HELP),
) -> None:
    workspace = cwd or Path.cwd()
    if action not in {"ingest", "search", "stats"}:
        typer.echo(
            ui(
                "cli.app.unknown_action_use_choices",
                action=action,
                choices="ingest | search | stats",
            ),
            err=True,
        )
        raise typer.Exit(1)
    if action == "search" and not target:
        typer.echo(ui("cli.app.search_needs_a_search_query"), err=True)
        raise typer.Exit(1)
    trusted = _decide_trust(workspace, trust=trust, interactive=False)
    try:
        asyncio.run(_rag_action(action, target, k, workspace, trusted=trusted))
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


async def _rag_action(action: str, target: str | None, k: int, cwd: Path, *, trusted: bool) -> None:
    cfg = _load_config(cwd, trusted=trusted)
    if not cfg.rag.enabled:
        disabled = ui("cli.app.rag_is_disabled_in_the")
        _console.print(f"[yellow]{disabled}[/yellow]")
        raise typer.Exit(1)
    if action != "ingest" and not _rag_db_path(cfg, cwd).is_file():
        missing = ui("cli.app.there_is_no_index_yet")
        _console.print(f"[yellow]{missing}[/yellow]")
        raise typer.Exit(1)
    service = RAGService(cfg.rag, cwd, secrets=SecretMatcher.from_config(cfg.permission))
    try:
        if action == "ingest":
            root = Path(target).expanduser() if target else cwd
            if not root.is_dir():
                not_found = ui("cli.app.directory_not_found", root=root)
                _console.print(f"[red]{not_found}[/red]")
                raise typer.Exit(1)
            embed_at = cfg.rag.embed_base_url or DEFAULT_EMBED_BASE_URL
            embed = f"{cfg.rag.embed_model} @ {embed_at}"
            ingesting = ui("cli.app.ingesting_embeddings", root=root, embed=embed)
            _console.print(f"[dim]{ingesting}[/dim]")
            report = await service.ingest(root)
            _render_ingest_report(report)
        elif action == "search":
            assert target is not None
            try:
                _console.print(await service.query(target, k))
            except RagError as e:
                failed = ui("cli.app.cannot_search", error=e)
                _console.print(f"[red]{failed}[/red]")
                raise typer.Exit(1) from e
        else:
            _render_rag_stats(await service.stats())
    finally:
        await service.aclose()


def _render_ingest_report(report: IngestReport) -> None:
    table = Table(title="rag ingest")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("files scanned", str(report.files_scanned))
    table.add_row("files indexed", str(report.files_indexed))
    table.add_row("files removed", str(report.files_removed))
    table.add_row("chunks added", str(report.chunks_added))
    table.add_row("skipped (unchanged)", str(report.chunks_skipped))
    table.add_row("skipped (secret files)", str(report.files_secret_skipped))
    table.add_row("errors", str(len(report.errors)))
    _console.print(table)
    for line in report.errors[:10]:
        _console.print(f"[yellow]warn:[/yellow] {line}")
    if len(report.errors) > 10:
        rest = len(report.errors) - 10
        more = ui("cli.app.and_more", rest=rest)
        _console.print(f"[dim]{more}[/dim]")


def _render_rag_stats(stats: dict[str, object]) -> None:
    table = Table(title="rag index")
    table.add_column("key", style="cyan")
    table.add_column("value")
    for key in ("chunks", "files", "backend", "embed_model", "dim"):
        table.add_row(key, str(stats.get(key, "")))
    _console.print(table)


@app.command(
    "eval",
    help=ui("cli.app.run_and_compare_the_code"),
)
def eval_cmd(
    target: Path | None = typer.Argument(
        None,
        help=ui("cli.app.task_toml_directory_or_suite"),
    ),
    agents: list[str] = typer.Option(
        ["satsuma"],
        "--agent",
        "-a",
        help=ui("cli.app.agents_to_evaluate_repeatable_satsuma"),
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help=ui("cli.app.model_for_the_satsuma_agent"),
    ),
    agent_timeout: float = typer.Option(
        600.0,
        "--agent-timeout",
        help=ui("cli.app.time_limit_in_seconds_per"),
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=ui("cli.app.where_to_write_the_report"),
    ),
    defects: bool = typer.Option(
        True,
        "--defects/--no-defects",
        help=ui("cli.app.keep_defect_records_of_tasks"),
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help=ui("cli.app.exit_with_code_1_if"),
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help=ui("cli.app.workspace_the_base_for_resolving"),
    ),
    runs: int = typer.Option(
        1,
        "--runs",
        min=1,
        help=ui("cli.app.how_many_times_to_run"),
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help=ui("cli.app.task_ids_to_run_comma"),
    ),
    baseline: Path | None = typer.Option(
        None,
        "--baseline",
        help=ui("cli.app.compare_with_the_per_task"),
    ),
    web_routing: bool = typer.Option(
        False,
        "--web-routing",
        help=ui("cli.app.run_the_web_routing_suite"),
    ),
    prompt_lang: str | None = typer.Option(
        None,
        "--prompt-lang",
        help=ui("cli.app.en_run_with_the_english"),
    ),
    trust: bool = typer.Option(False, "--trust", help=_TRUST_HELP),
) -> None:
    workspace = cwd or Path.cwd()
    defects_dir = (
        ((out.parent if out is not None else workspace / AGENT_DIRNAME / "evals") / "defects")
        if defects
        else None
    )
    only_ids = _parse_only(only)
    if target is None:
        if not web_routing:
            raise typer.BadParameter(ui("cli.app.a_path_to_the_eval"))
        target = workspace / "evals" / "web_routing"
    failed = False
    trusted = _decide_trust(workspace, trust=trust, interactive=False)
    try:
        base = Baseline.load(baseline) if baseline is not None else None
        if base is not None and base.prompt_lang != (prompt_lang or ""):
            original = ui("cli.app.original")
            theirs, ours = base.prompt_lang or original, prompt_lang or original
            raise ConfigError(
                ui("cli.app.the_baseline_s_request_version", theirs=theirs, ours=ours)
            )
        for run in range(1, runs + 1):
            reports = asyncio.run(
                _run_eval(
                    target,
                    agents,
                    model,
                    agent_timeout,
                    workspace,
                    defects_dir,
                    only_ids,
                    web_routing=web_routing,
                    prompt_lang=prompt_lang,
                    trusted=trusted,
                )
            )
            for report in reports:
                if base is not None and baseline is not None:
                    report.comparison = compare(report, base, path=str(baseline))
                _render_eval_report(report)
                saved = _save_eval_report(
                    report, _out_for_run(out, run, runs) if len(reports) == 1 else None, workspace
                )
                _console.print(f"[dim]{ui('cli.app.report')}: {saved}[/dim]")
                if report.comparison is not None:
                    _render_comparison(report.comparison)
                failed = failed or report.passed < report.total
            if len(reports) > 1:
                _render_eval_comparison(reports)
    except (ConfigError, ProviderError) as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    if defects_dir is not None:
        _render_repeated_defects(defects_dir)
    if strict and failed:
        raise typer.Exit(1)


def _parse_only(only: str | None) -> set[str] | None:
    if only is None:
        return None
    ids = {part.strip() for part in only.split(",") if part.strip()}
    return ids or None


def _out_for_run(out: Path | None, run: int, runs: int) -> Path | None:
    """``--runs 2`` のときは ``<stem>-1.json`` / ``<stem>-2.json`` に分けて保存する。"""
    if out is None or runs == 1:
        return out
    return out.with_name(f"{out.stem}-{run}{out.suffix}")


async def _run_eval(
    target: Path,
    agent_specs: list[str],
    model: str | None,
    agent_timeout: float,
    cwd: Path,
    defects_dir: Path | None = None,
    only: set[str] | None = None,
    *,
    web_routing: bool = False,
    prompt_lang: str | None = None,
    trusted: bool,
) -> list[EvalReport]:
    reports: list[EvalReport] = []
    if web_routing:
        if prompt_lang is not None:
            raise ConfigError(ui("cli.app.web_routing_without_prompt_lang"))
        if agent_specs != ["satsuma"]:
            raise ConfigError(ui("cli.app.web_routing_satsuma_only"))
        routing_model = model or _load_config(cwd, trusted=trusted).provider.default_model
        if not routing_model:
            raise ConfigError(ui("cli.app.web_routing_needs_a_model"))
        report, outcomes = await run_web_routing(target, routing_model, only=only)
        _console.print(f"[dim]{summarize(outcomes)}[/dim]")
        return [report]
    if is_suite_manifest(target):
        suite = load_suite(target)
        for spec in agent_specs:
            reports.append(
                await _run_suite(
                    suite,
                    spec,
                    model,
                    agent_timeout,
                    cwd,
                    defects_dir,
                    only,
                    prompt_lang=prompt_lang,
                    trusted=trusted,
                )
            )
        return _with_prompt_lang(reports, prompt_lang)
    tasks = localize_tasks(select_tasks(load_tasks(target), only), prompt_lang)
    for spec in agent_specs:
        agent = await _build_eval_agent(spec, model, agent_timeout, cwd, trusted=trusted)
        reports.append(await EvalRunner(agent, defects_dir).run(tasks))
    return _with_prompt_lang(reports, prompt_lang)


def _with_prompt_lang(reports: list[EvalReport], prompt_lang: str | None) -> list[EvalReport]:
    """どの版の依頼で回したかをレポートに残す（基準に積むときと比べるときに見る）。"""
    for report in reports:
        report.prompt_lang = prompt_lang or ""
    return reports


async def _run_suite(
    suite: SuiteTasks,
    spec: str,
    model: str | None,
    agent_timeout: float,
    cwd: Path,
    defects_dir: Path | None,
    only: set[str] | None,
    *,
    prompt_lang: str | None = None,
    trusted: bool,
) -> EvalReport:
    """manifest の課題を通常経路と plan 経路に分けて回し、1 本のレポートに畳む。"""
    if not is_satsuma_agent_spec(spec):
        raise ConfigError(ui("cli.app.a_manifest_can_be_run", spec=spec))
    if parse_satsuma_agent_spec(spec)[2]:
        raise ConfigError(ui("cli.app.with_a_manifest_do_not"))
    plain, plan = (localize_tasks(tasks, prompt_lang) for tasks in suite.select(only))
    by_id: dict[str, TaskResult] = {}
    agent_name = ""
    model_name = ""
    started = now_ms()
    if plain:
        agent = await _build_eval_agent(spec, model, agent_timeout, cwd, trusted=trusted)
        report = await EvalRunner(agent, defects_dir).run(plain)
        agent_name, model_name = report.agent, report.model
        by_id.update({r.task_id: r for r in report.results})
    if plan:
        agent = await _build_eval_agent(
            spec + PLAN_SUFFIX, model, agent_timeout, cwd, trusted=trusted
        )
        report = await EvalRunner(agent, defects_dir).run(plan)
        agent_name = agent_name or report.agent.removesuffix(PLAN_SUFFIX)
        model_name = model_name or report.model
        by_id.update({r.task_id: r for r in report.results})
    return EvalReport(
        agent=agent_name or "satsuma",
        model=model_name,
        started_at_ms=started,
        suite=suite.name,
        results=[by_id[task_id] for task_id in suite.order if task_id in by_id],
    )


def _render_comparison(comp: Comparison) -> None:
    """基準との対応付けと判定。"""
    gate, kept, better = len(comp.gate_tasks), len(comp.maintained), len(comp.improved)
    worse, still, new = len(comp.regressed), len(comp.still_failing), len(comp.new)
    flaky, lost = len(comp.flaky_failed), len(comp.connection_lost)
    line = ui(
        "cli.app.baseline_gate_tasks_kept_improved",
        baseline=comp.baseline,
        gate=gate,
        kept=kept,
        better=better,
        worse=worse,
        still=still,
        new=new,
    )
    if flaky:
        line += ui("cli.app.failures_of_flaky_tasks_observed", flaky=flaky)
    if lost:
        line += ui("cli.app.connection_failures_item", lost=lost)
    _console.print(line)
    for label, items in (
        (ui("cli.app.regression_candidates"), comp.regressed),
        (ui("cli.app.structural_errors"), comp.structural_errors),
        (ui("cli.app.over_the_iteration_or_tool"), comp.over_budget),
        (ui("cli.app.connection_failures_label"), comp.connection_lost),
    ):
        if items:
            _console.print(f"[yellow]  {label}: {', '.join(items)}[/yellow]")
    if comp.decision == "rerun":
        cases = ",".join(comp.rerun_cases)
        rerun = ui("cli.app.verdict_rerun_run_2_more", cases=cases)
        _console.print(f"[bold yellow]{rerun}[/bold yellow]")
    else:
        _console.print(f"[green]{ui('cli.app.verdict_pass')}[/green]")


@app.command(
    "eval-index",
    help=ui("cli.app.turn_the_list_of_eval"),
)
def eval_index_cmd(
    root: Path | None = typer.Option(
        None,
        "--root",
        help=ui("cli.app.directory_to_scan_default_cwd"),
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help=ui("cli.app.where_to_write_the_markdown"),
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        min=1,
        help=ui("cli.app.number_of_reports_in_the"),
    ),
    cwd: Path | None = typer.Option(
        None,
        "--cwd",
        help=ui("cli.app.workspace_the_base_for_satsuma"),
    ),
) -> None:
    workspace = cwd or Path.cwd()
    scan_root = root or workspace / AGENT_DIRNAME / "evals"
    if not scan_root.is_dir():
        typer.echo(
            ui("cli.app.error_the_report_directory_does", scan_root=scan_root),
            err=True,
        )
        raise typer.Exit(1)
    rows, skipped = scan_reports(scan_root, workspace / "evals")
    text = render_markdown(rows, limit=limit, root=str(scan_root))
    if out is None:
        _console.print(text, markup=False, highlight=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        n = len(rows)
        _console.print(
            ui("cli.app.index_reports_unreadable_json_files", out=out, n=n, skipped=skipped)
        )


@app.command(
    "eval-baseline",
    help=ui("cli.app.add_to_or_show_the"),
)
def eval_baseline_cmd(
    action: str = typer.Argument(
        ...,
        help=ui("cli.app.add_add_reports_to_the"),
    ),
    reports: list[Path] | None = typer.Argument(
        None,
        help=ui("cli.app.add_report_jsons_to_add"),
    ),
    into: Path = typer.Option(
        ...,
        "--into",
        help=ui("cli.app.baseline_file_satsuma_evals_baselines"),
    ),
    keep: int = typer.Option(
        DEFAULT_KEEP,
        "--keep",
        min=1,
        help=ui("cli.app.how_many_recent_runs_to"),
    ),
) -> None:
    try:
        if action == "add":
            if not reports:
                raise ConfigError(ui("cli.app.add_needs_one_or_more"))
            base = Baseline.load(into) if into.exists() else Baseline()
            for path in reports:
                report = load_report(path)
                added = base.add_report(report, keep=keep, source=path.name)
                skipped = len(report.results) - added
                stacked = (
                    ui(
                        "cli.app.added_tasks_skipped_tasks_with",
                        name=path.name,
                        added=added,
                        skipped=skipped,
                    )
                    if skipped
                    else ui("cli.app.added_tasks", name=path.name, added=added)
                )
                _console.print(f"[dim]{stacked}[/dim]")
            base.save(into)
            tasks, gate, sources = len(base.tasks), len(base.gate_tasks()), len(base.sources)
            _console.print(
                ui(
                    "cli.app.baseline_tasks_gate_sources",
                    into=into,
                    tasks=tasks,
                    gate=gate,
                    sources=sources,
                )
            )
        elif action == "show":
            base = Baseline.load(into)
            table = Table(title=f"baseline ({base.suite or '-'} / {base.model or '-'})")
            for column in ("task", "runs", "pass%", "gate", "iters", "tools", "sec"):
                table.add_column(column, justify="right" if column != "task" else "left")
            for row in describe_history(base):
                table.add_row(
                    row.task,
                    str(row.runs),
                    f"{row.pass_rate * 100:.0f}",
                    "[green]yes[/green]" if row.gate else "-",
                    f"{row.iterations:g}",
                    f"{row.tool_calls:g}",
                    f"{row.duration_s:.1f}",
                )
            _console.print(table)
        else:
            raise ConfigError(ui("cli.app.unknown_action_add_show", action=action))
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e


async def _build_eval_agent(
    spec: str, model: str | None, timeout_s: float, cwd: Path, *, trusted: bool
) -> EvalAgent:
    if is_satsuma_agent_spec(spec):
        cfg = _load_config(cwd, trusted=trusted)
        spec_model, decompose, plan = parse_satsuma_agent_spec(spec)
        resolved = expand_model_name(spec_model or model or cfg.provider.default_model)
        provider = (await connect(cfg, resolved, os.environ, build=build_provider)).provider
        await apply_context_fit(cfg, resolved, os.environ)
        return SatsumaAgent(
            provider,
            cfg,
            resolved,
            decompose=decompose,
            plan=plan,
            provider_builder=lambda m: build_provider(cfg, m, os.environ),
            env=os.environ,
        )
    return build_cli_agent(spec, timeout_s=timeout_s)


def _language_leaked(r: TaskResult) -> bool:
    """修復後も期待言語と違うまま終わったか（ユーザーが目にする英語化）。"""
    return language_leaked(r)


def _language_drifted(r: TaskResult) -> bool:
    """修復前に期待言語を外していたか。"""
    return _language_leaked(r) or bool(r.language_expected and r.language_repaired)


def _render_eval_report(report: EvalReport) -> None:
    label = report.agent + (f" / {report.model}" if report.model else "")
    if report.suite:
        label = f"{report.suite} · {label}"
    if report.prompt_lang:
        label += f" · prompt {report.prompt_lang}"
    table = Table(title=f"eval ({label})")
    table.add_column("task", style="cyan")
    table.add_column("category", style="dim")
    table.add_column("result")
    table.add_column("iters", justify="right")
    table.add_column("tools", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("sec", justify="right")
    table.add_column("detail")
    for r in report.results:
        status = "[green]pass[/green]" if r.passed else "[red]fail[/red]"
        if r.connection_lost:
            status = "[yellow]conn[/yellow]"
        tokens = r.usage.input_tokens + r.usage.output_tokens
        detail = r.error or "; ".join(r.failures[:2])
        marks = [
            "empty_final" if r.empty_final else "",
            f"repairs={r.tool_call_repairs}" if r.tool_call_repairs else "",
            f"empty_args={r.empty_tool_args}" if r.empty_tool_args else "",
            "compacted" if r.compacted else "",
            f"reminders={r.reminders}" if r.reminders else "",
            f"decomp={r.decomposition_reason}" if r.decomposition_reason else "",
            f"plan={r.plan_exits}" if r.plan_exits else "",
            f"lang={r.language_detected}" if _language_leaked(r) else "",
            "lang_repaired" if r.language_repaired else "",
            f"prefix_breaks={r.prompt_prefix_breaks}" if r.prompt_prefix_breaks else "",
            f"timeout_recovery={r.timeout_recoveries}" if r.timeout_recoveries else "",
            f"timeout_resume={r.timeout_resumes}" if r.timeout_resumes else "",
            f"rewrite={r.tool_call_rewrites}" if r.tool_call_rewrites else "",
            f"rep_abort={r.repetition_aborts}" if r.repetition_aborts else "",
            f"syntax_rej={r.syntax_rejects}" if r.syntax_rejects else "",
            f"syntax_warn={r.syntax_warnings}" if r.syntax_warnings else "",
            f"edit_miss={r.edit_mismatches}" if r.edit_mismatches else "",
            f"reindent={r.edit_reindents}" if r.edit_reindents else "",
            f"stale={r.edit_stale_notes}" if r.edit_stale_notes else "",
            f"related={r.related_code_notes}" if r.related_code_notes else "",
            f"ow_rej={r.overwrite_rejects}" if r.overwrite_rejects else "",
            f"ow_over={r.overwrite_overrides}" if r.overwrite_overrides else "",
            f"step_rej={r.step_write_rejects}" if r.step_write_rejects else "",
            f"step_over={r.step_write_overrides}" if r.step_write_overrides else "",
            f"exit_noop={r.plan_exit_not_active}" if r.plan_exit_not_active else "",
            f"exit_step_done={r.plan_exit_step_done}" if r.plan_exit_step_done else "",
            f"mkdir_skip={r.mkdir_skips}" if r.mkdir_skips else "",
            f"output_delete_block={r.output_delete_blocks}" if r.output_delete_blocks else "",
            f"state_write_block={r.state_write_blocks}" if r.state_write_blocks else "",
            f"secret_read={r.secret_read_checks}" if r.secret_read_checks else "",
            f"first_fetch={r.first_fetch_asks}" if r.first_fetch_asks else "",
            f"lsp_start={r.lsp_start_asks}" if r.lsp_start_asks else "",
            f"result_cut={r.tool_result_cuts}" if r.tool_result_cuts else "",
            f"body_shrunk={r.written_bodies_shrunk}" if r.written_bodies_shrunk else "",
            f"omitted_copy={r.omitted_copies}" if r.omitted_copies else "",
            f"presend_compact={r.presend_compactions}" if r.presend_compactions else "",
            f"truncated={r.context_truncations}" if r.context_truncations else "",
            f"prefill_wait={r.prefill_waits}" if r.prefill_waits else "",
            f"unknown_tool={r.unknown_tool_calls}" if r.unknown_tool_calls else "",
            f"malformed_call={r.malformed_tool_calls}" if r.malformed_tool_calls else "",
            f"args_json={r.args_only_replies}" if r.args_only_replies else "",
            f"no_change={r.no_change_reports}" if r.no_change_reports else "",
            f"name_fix={r.tool_name_fixes}" if r.tool_name_fixes else "",
            f"autoload={r.deferred_autoloads}" if r.deferred_autoloads else "",
            f"plan_nudge={r.plan_exit_nudges}" if r.plan_exit_nudges else "",
            f"steps={r.plan_steps_run}" if r.plan_steps_run else "",
            f"budget={r.plan_budget_lines}" if r.plan_budget_lines else "",
            f"over={_over_budget(r)}" if _over_budget(r) else "",
            f"sendback={r.plan_budget_sendbacks}" if r.plan_budget_sendbacks else "",
            "forced" if r.plan_budget_forced else "",
            f"verdicts={','.join(r.plan_budget_verdicts)}" if r.plan_budget_verdicts else "",
            _write_chunk_mark(r.write_chunks),
            f"tok/s={r.tokens_per_s:.1f}" if r.tokens_per_s else "",
            f"slept={r.slept_ms / 1000:.0f}s" if r.slept_ms else "",
        ]
        if any(marks):
            detail = " ".join(filter(None, [detail, *marks]))
        table.add_row(
            r.task_id,
            r.category or "-",
            status,
            str(r.iterations) if r.iterations else "-",
            str(r.tool_calls) if r.tool_calls else "-",
            str(tokens) if tokens else "-",
            f"{r.duration_ms / 1000:.1f}",
            detail,
        )
    _console.print(table)
    _console.print(_summary_line(report))
    records = [Path(r.defect_record) for r in report.results if r.defect_record]
    if records:
        n, where = len(records), records[0].parent
        saved = ui("cli.app.defect_records", n=n, where=where)
        _console.print(f"[yellow]{saved}[/yellow]")
    _render_taxonomy_coverage(report)


def _summary_line(report: EvalReport) -> str:
    """表の下の 1 行（合格の数と、合否と独立に数える出来事。0 の項目は出さない）。"""
    results = report.results

    def total(field: str) -> int:
        return sum(int(getattr(r, field)) for r in results)

    rate = f"{report.pass_rate * 100:.0f}%"
    parts = [ui("cli.app.passed", passed=report.passed, total=report.total, rate=rate)]
    speeds = [r.tokens_per_s for r in results if r.tokens_per_s]
    if speeds:
        speed = f"{statistics.median(speeds):.1f}"
        parts.append(ui("cli.app.generation_tok_s_median", speed=speed))
    if slept := total("slept_ms"):
        secs = f"{slept / 1000:.0f}"
        parts.append(ui("cli.app.the_machine_slept_s_not", secs=secs))
    if lost := sum(1 for r in results if r.connection_lost):
        parts.append(ui("cli.app.connection_failures_counted_as_failures", lost=lost))
    if empties := sum(1 for r in results if r.empty_final):
        parts.append(ui("cli.app.empty_final_replies", empties=empties))
    if drifted := sum(1 for r in results if _language_drifted(r)):
        repaired = sum(1 for r in results if r.language_repaired)
        parts.append(ui("cli.app.language_drift_repaired", drifted=drifted, repaired=repaired))
    if breaks := total("prompt_prefix_breaks"):
        iterations = total("iterations")
        per = f"{breaks / iterations if iterations else 0.0:.2f}"
        parts.append(ui("cli.app.prefix_breaks_iteration", breaks=breaks, per=per))
    for fields, sentence in _event_counts():
        counts = [total(field) for field in fields]
        if any(counts):
            parts.append(sentence(*counts))
    if nudges := total("plan_exit_nudges"):
        exits = total("plan_exits")
        parts.append(ui("cli.app.sent_back_to_approval_reached", nudges=nudges, exits=exits))
    if steps_run := total("plan_steps_run"):
        parts.append(ui("cli.app.plan_steps", steps_run=steps_run))
    if over := sum(_over_budget(r) for r in results):
        parts.append(ui("cli.app.steps_over_budget", over=over))
    if sendbacks := total("plan_budget_sendbacks"):
        parts.append(ui("cli.app.plan_send_backs", sendbacks=sendbacks))
    return ui("cli.app.list_separator").join(parts)


def _event_counts() -> list[tuple[tuple[str, ...], Callable[..., str]]]:
    """``(レポートの欄, 文を作る関数)``。"""
    return [
        (
            ("timeout_recoveries",),
            lambda count: ui("cli.app.timeout_recoveries", count=count),
        ),
        (
            ("timeout_resumes",),
            lambda count: ui("cli.app.step_resumes", count=count),
        ),
        (
            ("tool_call_rewrites",),
            lambda count: ui("cli.app.broken_calls_rewritten", count=count),
        ),
        (
            ("repetition_aborts",),
            lambda count: ui("cli.app.repetition_aborts", count=count),
        ),
        (
            ("syntax_rejects", "syntax_warnings"),
            lambda rejected, written: ui(
                "cli.app.syntax_breaking_edits_sent_back", rejected=rejected, written=written
            ),
        ),
        (
            ("overwrite_rejects", "overwrite_overrides"),
            lambda rejected, written: ui(
                "cli.app.partial_overwrites_sent_back_written", rejected=rejected, written=written
            ),
        ),
        (
            ("step_write_rejects", "step_write_overrides"),
            lambda rejected, written: ui(
                "cli.app.writes_over_the_step_limit", rejected=rejected, written=written
            ),
        ),
        (
            ("secret_read_checks",),
            lambda count: ui("cli.app.secret_file_reads_sent_to", count=count),
        ),
        (
            ("first_fetch_asks",),
            lambda count: ui("cli.app.first_web_fetch_sent_to", count=count),
        ),
        (
            ("lsp_start_asks",),
            lambda count: ui("cli.app.language_server_starts_sent_to", count=count),
        ),
        (
            ("tool_result_cuts",),
            lambda count: ui("cli.app.oversized_tool_results_cut", count=count),
        ),
        (
            ("written_bodies_shrunk",),
            lambda count: ui("cli.app.written_bodies_shrunk", count=count),
        ),
        (
            ("omitted_copies",),
            lambda count: ui("cli.app.omitted_writes_copied", count=count),
        ),
        (
            ("presend_compactions",),
            lambda count: ui("cli.app.history_folded_before_sending", count=count),
        ),
        (
            ("context_truncations",),
            lambda count: ui("cli.app.requests_truncated_by_the_server", count=count),
        ),
        (
            ("prefill_waits",),
            lambda count: ui("cli.app.first_reply_waits_extended_for", count=count),
        ),
        (
            ("edit_mismatches",),
            lambda count: ui("cli.app.edit_file_mismatches", count=count),
        ),
        (
            ("edit_reindents",),
            lambda count: ui("cli.app.applied_with_matched_indentation", count=count),
        ),
        (
            ("args_only_replies",),
            lambda count: ui("cli.app.arguments_only_json_replies", count=count),
        ),
        (
            ("no_change_reports",),
            lambda count: ui("cli.app.completion_reports_without_edits", count=count),
        ),
        (
            ("unknown_tool_calls",),
            lambda count: ui("cli.app.unknown_tool_names", count=count),
        ),
        (
            ("malformed_tool_calls",),
            lambda count: ui("cli.app.malformed_calls", count=count),
        ),
    ]


def _over_budget(r: TaskResult) -> int:
    """1 回の書き込みが段の予算を超えた段の数。"""
    return sum(1 for s in r.plan_step_records if s.over_budget)


def _write_chunk_mark(chunks: list[WriteChunk]) -> str:
    """``writes=35 med=1.2k max=4.8k``（1 回あたりの文字数。書き込みが無ければ空）。"""
    if not chunks:
        return ""
    sizes = sorted(c.chars for c in chunks)
    return (
        f"writes={len(sizes)} med={statistics.median(sizes) / 1000:.1f}k "
        f"max={sizes[-1] / 1000:.1f}k"
    )


def _render_taxonomy_coverage(report: EvalReport) -> None:
    """SWE-QA タクソノミーの Level1 別合否と、スイート未カバーの分類を表示する。"""
    from satsumaestro.evals.models import TAXONOMY

    tagged = [r for r in report.results if r.category]
    if not tagged:
        return
    by_level1: dict[str, list[bool]] = {}
    seen_level2: set[str] = set()
    for r in tagged:
        level1 = r.category.partition("/")[0]
        by_level1.setdefault(level1, []).append(r.passed)
        seen_level2.add(r.category)
    parts = [
        f"{level1} {sum(flags)}/{len(flags)}"
        for level1, flags in sorted(by_level1.items())
        if flags
    ]
    by_category = ui("cli.app.list_separator").join(parts)
    _console.print(f"[dim]{ui('cli.app.by_category')}: {by_category}[/dim]")
    missing = [
        f"{level1}/{level2}"
        for level1, level2s in TAXONOMY.items()
        for level2 in level2s
        if f"{level1}/{level2}" not in seen_level2
    ]
    if missing:
        names = ", ".join(missing)
        not_covered = ui("cli.app.categories_not_covered", missing_count=len(missing), names=names)
        _console.print(f"[dim]{not_covered}[/dim]")


def _render_eval_comparison(reports: list[EvalReport]) -> None:
    """複数エージェントのタスク別合否を並べる（行 = タスク、列 = エージェント）。"""
    table = Table(title="agents comparison")
    table.add_column("task", style="cyan")
    for report in reports:
        table.add_column(report.agent, justify="center")
    by_agent = [{r.task_id: r for r in report.results} for report in reports]
    task_ids = [r.task_id for r in reports[0].results]
    for task_id in task_ids:
        cells = []
        for results in by_agent:
            r = results.get(task_id)
            if r is None:
                cells.append("-")
            elif r.passed:
                cells.append("[green]pass[/green]")
            else:
                cells.append("[red]err[/red]" if r.error else "[red]fail[/red]")
        table.add_row(task_id, *cells)
    table.add_row(
        f"[bold]{ui('cli.app.pass_rate')}[/bold]",
        *(f"{r.passed}/{r.total}" for r in reports),
    )
    _console.print(table)


def _render_repeated_defects(defects_dir: Path) -> None:
    """同じ種別の未起票の不具合記録が 5 件以上たまっていれば知らせる。"""
    if not defects_dir.is_dir():
        return
    repeated = repeated_defects(defects_dir)
    if not repeated:
        return
    piled = ui("cli.app.the_same_defect_has_piled", repeat_threshold=REPEAT_THRESHOLD)
    _console.print(f"[bold yellow]{piled}[/bold yellow]")
    for r in repeated:
        latest = r.records[-1].name
        line = ui("cli.app.latest", kind=r.kind, count=r.count, latest=latest)
        _console.print(f"[yellow]{line}[/yellow]")
    _console.print(f"[dim]  {ui('cli.app.records')}: {defects_dir}[/dim]")


def _save_eval_report(report: EvalReport, out: Path | None, cwd: Path) -> Path:
    if out is None:
        stamp = datetime.datetime.fromtimestamp(
            report.started_at_ms / 1000, tz=datetime.UTC
        ).strftime("%Y%m%dT%H%M%SZ")
        out = cwd / AGENT_DIRNAME / "evals" / f"{stamp}-{report.agent}.json"
        ensure_state_gitignore(cwd)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return out


async def _run_repl_resume(model: str | None, cwd: Path, path: Path, *, trusted: bool) -> None:
    print_banner(_console)
    rt, prompter, render = await _assemble_runtime(
        model, cwd, interactive=True, resume_path=path, trusted=trusted
    )
    assert isinstance(prompter, InteractivePrompter)
    try:
        await run_repl(
            rt,
            prompter,
            render,
            _console,
            trusted=trusted,
            load_probe=load_probe(rt, os.environ),
        )
    finally:
        try:
            await _shutdown(rt.mcp, rt.rag, rt.orchestrator, rt.background, rt.lsp)
        finally:
            await _unload_on_exit(rt.config)


def _trust_row(basis: TrustBasis, root: Path) -> tuple[str, str]:
    """doctor の trust 行（状態・詳細）。"""
    if basis is TrustBasis.FLAG:
        return "ok", ui("cli.app.trust_this_run_only_not", root=root)
    if basis is TrustBasis.RECORDED:
        record = TrustStore().path
        return "ok", ui("cli.app.recorded", record=record, root=root)
    return "warn", ui("cli.app.untrusted_the_settings_are_not", root=root)


def _screen_language_row(cfg: RuntimeConfig) -> tuple[str, str]:
    """doctor の「画面の言語」の行（状態・説明）。"""
    chosen = cfg.language.chosen_ui
    if chosen is not None:
        return "ok", ui("cli.app.screen_language_chosen", name=language_gate.native_name(chosen))
    name = language_gate.native_name(locale_language(os.environ))
    if cfg.language.ui == "auto":
        return "ok", ui("cli.app.screen_language_auto", name=name)
    return "-", ui("cli.app.screen_language_not_set", name=name)


def _doctor(cwd: Path, *, probe: bool = False, trust: bool = False) -> None:
    table = Table(title="SATSUMAESTRO doctor")
    table.add_column("check", style="cyan")
    table.add_column("status")
    table.add_column("detail")

    decision = decide_trust(cwd, trust_flag=trust, ask=None)
    table.add_row("trust", *_trust_row(decision.basis, decision.root))
    try:
        cfg = _load_config(cwd, trusted=decision.trusted)
    except ConfigError as e:
        table.add_row("config", "ERROR", str(e))
        _console.print(table)
        raise typer.Exit(1) from e
    table.add_row(
        "config",
        "ok",
        ui("cli.app.loaded_user_and_project_settings")
        if decision.trusted
        else ui("cli.app.loaded_user_settings_and_environment"),
    )
    table.add_row("screen language", *_screen_language_row(cfg))

    model = expand_model_name(cfg.provider.default_model or "")
    table.add_row(
        "model",
        "ok" if model else "warn",
        model or ui("cli.app.not_set_set_provider_default"),
    )
    kind = choose_provider_kind(model, os.environ, cfg.provider.base_url, choice=cfg.provider.kind)
    table.add_row("provider", "ok", _provider_label(cfg, model, os.environ))
    table.add_row("api key", *_key_status(cfg, model, os.environ))

    default_mode = PermissionMode(cfg.permission.default_mode)
    parsed = (
        load_policies(_policy_files(cwd, cfg.permission.policy_file), default_mode)
        if decision.trusted
        else ParsedPolicy(policy=default_policy(default_mode))
    )
    if parsed.warnings:
        table.add_row("policy", "warn", "; ".join(parsed.warnings[:3]))
    else:
        table.add_row("policy", "ok", f"mode={parsed.policy.mode.value}")

    if cfg.mcp_servers:
        kinds = ", ".join(f"{n}({c.transport})" for n, c in cfg.mcp_servers.items())
        servers = ui("cli.app.mcp_servers_count", n=len(cfg.mcp_servers), kinds=kinds)
        detail = ui("cli.app.status_with_mcp_list", servers=servers)
        table.add_row("mcp", "ok", detail)
    else:
        table.add_row("mcp", "-", ui("cli.app.not_set"))
    status, detail = _rag_status(cfg, cwd)
    table.add_row("rag", status, detail)

    sampling = cfg.provider.sampling
    table.add_row(
        "sampling",
        "-" if sampling.is_unset() else "ok",
        ui("cli.app.not_set_provider_defaults")
        if sampling.is_unset()
        else ", ".join(f"{k}={v}" for k, v in sampling.model_dump().items() if v is not None),
    )
    roles = {r: getattr(cfg.provider.roles, r) for r in ROLE_NAMES}
    configured_roles = {k: v for k, v in roles.items() if v}
    table.add_row(
        "model roles",
        "ok" if configured_roles else "-",
        ", ".join(f"{k}={v}" for k, v in configured_roles.items())
        if configured_roles
        else ui("cli.app.not_set_the_main_model"),
    )
    strict = cfg.tool_schema.strict_tools
    table.add_row(
        "strict tools",
        "ok" if strict else "-",
        ui("cli.app.on_turned_off_automatically_on")
        if strict
        else ui("cli.app.off_ollama_ignores_strict_an"),
    )
    for row in _auto_allow_rows(cfg):
        table.add_row(*row)
    for row in _web_search_rows(cfg, probe=probe):
        table.add_row(*row)
    table.add_row(*_web_fetch_row(cfg))
    for row in _lsp_rows(cfg):
        table.add_row(*row)
    for row in _local_rows(cfg, model, kind, probe=probe, cwd=cwd, trusted=decision.trusted):
        table.add_row(*row)
    _console.print(table)


def _auto_allow_rows(cfg: RuntimeConfig) -> list[tuple[str, str, str]]:
    """bash 読み取り自動許可の状態（字句解析ベースの限界を明示する）。"""
    perm = cfg.permission
    if perm.bash_auto_allow == "off":
        off = ui("cli.app.off_read_only_bash_is")
        return [("bash auto-allow", "-", off)]
    patterns = len(SecretMatcher.from_config(perm).patterns)
    access = f"secret_access={perm.secret_access}"
    detail = ui("cli.app.read_only_lexical_analysis_deny", access=access, patterns=patterns)
    return [("bash auto-allow", "ok", detail)]


def _lsp_rows(cfg: RuntimeConfig) -> list[tuple[str, str, str]]:
    """LSP 統合の検出状況（ネットワーク不要 — PATH 検出のみでサーバは起動しない）。"""
    if not cfg.lsp.enabled:
        return [("lsp", "-", ui("cli.app.disabled_lsp_enabled_false"))]
    languages = sorted(set(DEFAULT_SERVERS) | set(cfg.lsp.servers))
    found: list[str] = []
    missing: list[str] = []
    for lang in languages:
        command = resolve_server_command(lang, cfg.lsp.servers)
        if command is None:
            missing.append(lang)
        else:
            found.append(f"{lang}={command[0]}")
    if not found:
        none = ui("cli.app.no_language_server_found_put")
        return [("lsp", "warn", none)]
    detail = ", ".join(found)
    if missing:
        names = ", ".join(missing)
        detail += ui("cli.app.not_installed", names=names)
    return [("lsp", "ok", detail)]


def _web_fetch_row(cfg: RuntimeConfig) -> tuple[str, str, str]:
    """web_fetch の制限の doctor 行（設定だけを読む。ネットワークは使わない）。"""
    wf = cfg.web_fetch
    if not wf.enabled:
        disabled = ui("cli.app.disabled_web_fetch_enabled_satsuma")
        return ("web fetch", "-", disabled)
    if wf.confirm == "first-use":
        confirm = ui("cli.app.confirmed_only_the_first_time")
    elif wf.confirm == "always":
        confirm = ui("cli.app.confirmed_every_time")
    else:
        confirm = ui("cli.app.not_confirmed")
    if wf.allow_private_hosts:
        return (
            "web fetch",
            "warn",
            ui("cli.app.internal_destinations_allowed_too_allow", confirm=confirm),
        )
    limit = f"{wf.max_bytes:,}"
    return (
        "web fetch",
        "ok",
        ui("cli.app.internal_destinations_private_loopback_metadata", confirm=confirm, limit=limit),
    )


def _web_search_rows(cfg: RuntimeConfig, *, probe: bool) -> list[tuple[str, str, str]]:
    """web_search プロバイダの設定・疎通の doctor 行。"""
    from satsumaestro.tools.builtins.web_search import _query_url

    ws = cfg.web_search
    url = _query_url(ws, "satsumaestro")
    if url is None:
        if ws.provider is None:
            disabled = ui("cli.app.disabled_web_search_provider_not")
            return [("web search", "-", disabled)]
        idle = ui("cli.app.not_running_because_base_url", provider=ws.provider)
        return [("web search", "warn", idle)]
    if not probe:
        hint = (
            ui("cli.app.the_unofficial_html_version_sometimes")
            if ws.provider == "duckduckgo"
            else ws.base_url or ""
        )
        shown = f"{ws.provider} / {hint}"
        return [
            (
                "web search",
                "ok",
                ui("cli.app.probe_to_check_that_it", shown=shown),
            )
        ]
    status, detail = asyncio.run(_probe_web_search(ws.provider, ws.timeout_s, url))
    return [("web search", status, f"{ws.provider} / {detail}")]


async def _probe_web_search(provider: str | None, timeout_s: float, url: str) -> tuple[str, str]:
    """実クエリ 1 発で web_search バックエンドの疎通を確認する（--probe 時のみ）。"""
    import httpx

    from satsumaestro.tools.builtins import web_search as web_search_mod

    try:
        async with web_search_mod._make_client(min(timeout_s, 5.0)) as client:
            resp = await client.get(url)
    except httpx.HTTPError as e:
        return "warn", ui("cli.app.cannot_connect", error=e)
    if provider == "duckduckgo":
        if resp.status_code == 202:
            return "warn", ui("cli.app.bot_detection_in_progress_http")
        if resp.status_code >= 400:
            return "warn", f"HTTP {resp.status_code}"
        return "ok", ui("cli.app.reachable")
    if resp.status_code >= 400:
        code = resp.status_code
        return "warn", ui("cli.app.http_searxng_needs_json_in", code=code)
    try:
        payload = resp.json()
    except ValueError:
        return "warn", ui("cli.app.it_does_not_answer_in")
    results = payload.get("results") if isinstance(payload, dict) else None
    count = len(results) if isinstance(results, list) else 0
    failed = web_search_mod.failed_engines(payload)
    reached = ui("cli.app.reachable_results", n=count)
    if not failed:
        return "ok", reached
    engines = web_search_mod.engines_text(failed)
    stopped = ui("cli.app.engines_not_answering", engines=engines)
    if count == 0:
        return "warn", ui("cli.app.t_0_results", stopped=stopped)
    return "ok", ui("cli.app.reached_and_stopped_engines", reached=reached, stopped=stopped)


def _local_rows(
    cfg: RuntimeConfig,
    model: str,
    kind: ProviderKind,
    *,
    probe: bool,
    cwd: Path | None = None,
    trusted: bool = False,
) -> list[tuple[str, str, str]]:
    """ローカルのサーバの種類ごとの doctor 行（--probe のときだけサーバの種類を問い合わせる）。"""
    if kind is not ProviderKind.OPENAI:
        return []
    endpoint = openai_endpoint(cfg, os.environ)
    server: LocalServer | None = None
    if endpoint and probe:
        server = asyncio.run(resolve_server(endpoint, os.environ, cfg.provider))
    elif endpoint:
        server = server_for_endpoint(endpoint, os.environ, cfg.provider)
    if isinstance(server, LMStudioServer):
        return _lmstudio_rows(cfg, model, server, probe=probe, cwd=cwd, trusted=trusted)
    if isinstance(server, LlamaCppServer):
        return _llamacpp_rows(cfg, model, server, probe=probe, cwd=cwd, trusted=trusted)
    if server is not None:
        return _ollama_rows(cfg, model, kind, probe=probe, cwd=cwd, trusted=trusted)
    cause = unreached_cause(endpoint) if endpoint and probe else None
    if endpoint and cause is not None:
        root = ollama_root(endpoint)
        unreached = ui("engine.context_fit.cannot_reach_local_server", root=root, cause=cause)
        return [("local server", "warn", escape(unreached))]
    undecided = cfg.provider.local.server == "auto" and cfg.provider.ollama.detect == "auto"
    if endpoint and not probe and undecided and looks_local(endpoint):
        return [("local server", "-", ui("cli.app.not_ollama_probe_checks_for"))]
    return [("local server", "-", ui("cli.app.not_a_local_server_can_be"))]


def _ollama_rows(
    cfg: RuntimeConfig,
    model: str,
    kind: ProviderKind,
    *,
    probe: bool,
    cwd: Path | None = None,
    trusted: bool = False,
) -> list[tuple[str, str, str]]:
    """Ollama 判定・コンテキスト長整合・窓の内訳・常駐状態の doctor 行。"""
    if kind is not ProviderKind.OPENAI:
        return []
    endpoint = openai_endpoint(cfg, os.environ)
    server = server_for_endpoint(endpoint, os.environ, cfg.provider) if endpoint else None
    if not endpoint or not isinstance(server, OllamaServer):
        not_ollama = ui("cli.app.not_ollama_can_be_fixed")
        return [("ollama", "-", not_ollama)]
    keep = cfg.provider.ollama.keep_alive or ui("cli.app.off")
    fit_flag = "on" if cfg.compaction.auto_context_fit else "off"
    unload_flag = "on" if cfg.provider.ollama.unload_on_exit else "off"
    if not probe:
        flags = (
            f"{ollama_root(endpoint)} / keep_alive={keep} / auto_context_fit={fit_flag}"
            f" / unload_on_exit={unload_flag}"
        )
        detail = ui("cli.app.probe_to_query_the_context", flags=flags)
        return [("ollama", "ok", detail)]
    if not model:
        no_model = ui("cli.app.cannot_query_because_no_model")
        return [("ollama", "warn", no_model)]
    fit = asyncio.run(_probe_ollama(cfg, model))
    rows: list[tuple[str, str, str]] = []
    info = fit.info
    if info.source == "none":
        root = ollama_root(endpoint)
        unreachable = ui("cli.app.cannot_reach_ollama", root=root)
        rows.append(("ollama ctx", "warn", unreachable))
        return rows
    status = "ok"
    detail = fit.describe()
    if not cfg.compaction.auto_context_fit and fit.clamped:
        status = "warn"
        detail += ui("cli.app.not_applied_because_auto_context")
    if info.effective is None:
        status = "warn"
        detail += ui("cli.app.effective_num_ctx_unknown_not")
    rows.append(("ollama ctx", status, escape(detail)))
    rows.extend(_window_rows(cfg, model, fit, cwd, trusted))
    loaded = _loaded_label(info.expires_at) if info.loaded else ui("cli.app.not_loaded")
    rows.append(("ollama keep_alive", "ok", f"config={keep} / {loaded}"))
    rows.extend(_role_residency_rows(cfg, model, ollama_root(endpoint)))
    return rows


def _window_rows(
    cfg: RuntimeConfig, model: str, fit: ContextFitResult, cwd: Path | None, trusted: bool
) -> list[tuple[str, str, str]]:
    """doctor の窓の内訳の行（作業場所が分かるときだけ）。"""
    if cwd is None or not fit.context_length:
        return []
    breakdown = asyncio.run(_doctor_breakdown(cfg, model, cwd, trusted, fit))
    if breakdown is None:
        return []
    text = breakdown.describe(source_label(fit.info.source)) + ui("cli.app.mcp_tools_not_counted")
    if breakdown.tight:
        text += "\n" + breakdown.advice(fit.info.server, model, fit.info.slots)
    return [("context window", "warn" if breakdown.tight else "ok", escape(text))]


def _lmstudio_rows(
    cfg: RuntimeConfig,
    model: str,
    server: LMStudioServer,
    *,
    probe: bool,
    cwd: Path | None,
    trusted: bool,
) -> list[tuple[str, str, str]]:
    """LM Studio の窓・内訳・常駐状態の doctor 行。"""
    local = cfg.provider.local
    ttl = local.ttl or "-"
    if not probe:
        fit_flag = "on" if cfg.compaction.auto_context_fit else "off"
        unload_flag = "on" if local.unload_on_exit else "off"
        flags = (
            f"{server.root} / ttl={ttl} / auto_context_fit={fit_flag}"
            f" / unload_on_exit={unload_flag}"
        )
        return [("lmstudio", "ok", ui("cli.app.probe_to_query_the_context", flags=flags))]
    if not model:
        return [("lmstudio", "warn", ui("cli.app.cannot_query_because_no_model"))]
    fit = asyncio.run(_probe_local(cfg, model, server))
    info = fit.info
    if not info.ps_ok:
        return [("lmstudio ctx", "warn", ui("cli.app.cannot_reach_lmstudio", root=server.root))]
    if info.source == "none":
        missing = ui(
            "engine.context_fit.model_was_not_found_on_lmstudio", model=model, root=server.root
        )
        return [("lmstudio ctx", "warn", escape(missing))]
    status = "ok"
    detail = fit.describe()
    if not cfg.compaction.auto_context_fit and fit.clamped:
        status = "warn"
        detail += ui("cli.app.not_applied_because_auto_context")
    if not info.loaded:
        status = "warn"
        detail += ui("cli.app.lmstudio_window_unknown_not")
    rows = [("lmstudio ctx", status, escape(detail))]
    rows.extend(_window_rows(cfg, model, fit, cwd, trusted))
    loaded = _loaded_label(info.expires_at) if info.loaded else ui("cli.app.not_loaded")
    rows.append(("lmstudio loaded", "ok", f"ttl={ttl} / {loaded}"))
    for name in _role_models(cfg, model):
        role = asyncio.run(server.context_info(name))
        if role.loaded:
            rows.append(("lmstudio role", "ok", f"{name}: {_loaded_label(role.expires_at)}"))
        else:
            unloaded = ui("cli.app.lmstudio_role_not_loaded", name=name)
            rows.append(("lmstudio role", "warn", unloaded))
    return rows


def _llamacpp_rows(
    cfg: RuntimeConfig,
    model: str,
    server: LlamaCppServer,
    *,
    probe: bool,
    cwd: Path | None,
    trusted: bool,
) -> list[tuple[str, str, str]]:
    """llama.cpp のサーバの形・窓・内訳の doctor 行。"""
    if not probe:
        fit_flag = "on" if cfg.compaction.auto_context_fit else "off"
        unload_flag = "on" if cfg.provider.local.unload_on_exit else "off"
        flags = f"{server.root} / auto_context_fit={fit_flag} / unload_on_exit={unload_flag}"
        return [("llamacpp", "ok", ui("cli.app.probe_to_query_the_context", flags=flags))]
    props = asyncio.run(server.props())
    if props is None:
        return [("llamacpp", "warn", ui("cli.app.cannot_reach_llamacpp", root=server.root))]
    if props.router:
        rows = [("llamacpp", "ok", ui("cli.app.llamacpp_router_mode", root=server.root))]
    else:
        shape = ui(
            "cli.app.llamacpp_single_model",
            model=props.model or "-",
            slots=props.slots or "-",
            root=server.root,
        )
        rows = [("llamacpp", "ok", escape(shape))]
        if props.tools is False:
            name = props.model or model or "-"
            no_tools = ui("engine.provider_switch.llamacpp_template_has_no_tools", model=name)
            rows.append(("llamacpp tools", "warn", escape(no_tools)))
    if not model:
        return [*rows, ("llamacpp ctx", "warn", ui("cli.app.cannot_query_because_no_model"))]
    fit = asyncio.run(_probe_local(cfg, model, server))
    info = fit.info
    if props.router and not asyncio.run(server.has_model(model)):
        missing = ui(
            "engine.context_fit.model_was_not_found_on_llamacpp", model=model, root=server.root
        )
        return [*rows, ("llamacpp ctx", "warn", escape(missing))]
    status = "ok"
    detail = fit.describe() if fit.context_length else ui("cli.app.llamacpp_window_unknown")
    if not cfg.compaction.auto_context_fit and fit.clamped:
        status = "warn"
        detail += ui("cli.app.not_applied_because_auto_context")
    if info.unconfirmed or not fit.context_length:
        status = "warn"
        detail += ui("cli.app.llamacpp_window_after_load")
    rows.append(("llamacpp ctx", status, escape(detail)))
    rows.extend(_window_rows(cfg, model, fit, cwd, trusted))
    return rows


def _role_models(cfg: RuntimeConfig, main_model: str) -> list[str]:
    """メインと別のモデルを使う役割のモデル。"""
    return sorted(
        {expand_model_name(m) for m in (getattr(cfg.provider.roles, r) for r in ROLE_NAMES) if m}
        - {main_model}
    )


def _role_residency_rows(
    cfg: RuntimeConfig, main_model: str, root: str
) -> list[tuple[str, str, str]]:
    """役割モデルの常駐状況（VRAM 競合チェック。--probe 時のみ）。"""
    role_models = _role_models(cfg, main_model)
    if not role_models:
        return []
    infos = asyncio.run(_fetch_role_contexts(root, role_models))
    rows: list[tuple[str, str, str]] = []
    for name, info in zip(role_models, infos, strict=True):
        if info.loaded:
            rows.append(("ollama role", "ok", f"{name}: {_loaded_label(info.expires_at)}"))
        else:
            unloaded = ui("cli.app.not_loaded_it_is_loaded", name=name)
            rows.append(("ollama role", "warn", unloaded))
    return rows


def _loaded_label(expires_at: object) -> str:
    return ui("cli.app.loaded_expires_at", expires_at=expires_at)


async def _fetch_role_contexts(root: str, models: list[str]) -> list[ModelContextInfo]:
    return [await fetch_model_context(root, m, timeout_s=1.5) for m in models]


async def _doctor_breakdown(
    cfg: RuntimeConfig, model: str, cwd: Path, trusted: bool, fit: ContextFitResult
) -> WindowBreakdown | None:
    """doctor 用: 起動時と同じ組み立てで、窓の内訳を数える（cfg は書き換えない）。"""
    probe_cfg = cfg.model_copy(deep=True)
    if probe_cfg.compaction.context_window <= 0 and fit.context_length:
        probe_cfg.compaction.context_window = fit.context_length
    probe_cfg.compact_token_threshold = fit.effective_threshold
    orchestration, workers = _orchestration_flags(probe_cfg, os.environ)
    tools = _tool_registry(probe_cfg, orchestration=orchestration, workers=workers)
    skills = list_skills(cwd, state_dir=trusted)
    if skills:
        tools.register(Skill if trusted else UntrustedSkill)
    builder = _prompt_builder(probe_cfg, tools, skills, interactive=True)
    turn = await builder.snapshot(cwd)
    system = builder.compose(model, probe_cfg, None, turn)
    schemas = (
        tools.schemas_for(set()) if probe_cfg.tools.deferred_tool_loading else tools.json_schemas()
    )
    return window_breakdown(
        probe_cfg.compaction.context_window,
        probe_cfg.compact_token_threshold,
        system.joined(),
        json.dumps(schemas, sort_keys=True, ensure_ascii=False),
        turn.instructions,
    )


async def _probe_local(cfg: RuntimeConfig, model: str, server: LocalServer) -> ContextFitResult:
    """doctor 用: cfg を書き換えずに、ローカルのサーバの窓で整合結果だけ計算する。"""
    from satsumaestro.engine.context_fit import fit_compaction_threshold

    info = await server.context_info(model)
    fitted = fit_compaction_threshold(cfg.compact_token_threshold, info.context_length)
    return ContextFitResult(
        model=model,
        info=info,
        original_threshold=cfg.compact_token_threshold,
        effective_threshold=fitted if fitted is not None else cfg.compact_token_threshold,
    )


async def _probe_ollama(cfg: RuntimeConfig, model: str) -> ContextFitResult:
    """doctor 用: cfg を書き換えずに整合結果だけ計算する。"""
    from satsumaestro.engine.context_fit import fit_compaction_threshold

    endpoint = openai_endpoint(cfg, os.environ) or ""
    info = await fetch_model_context(ollama_root(endpoint), model, timeout_s=1.5)
    fitted = fit_compaction_threshold(cfg.compact_token_threshold, info.context_length)
    return ContextFitResult(
        model=model,
        info=info,
        original_threshold=cfg.compact_token_threshold,
        effective_threshold=fitted if fitted is not None else cfg.compact_token_threshold,
    )


async def _rag_stats(cfg: RuntimeConfig, cwd: Path) -> dict[str, object]:
    """索引サマリを取得して確実に接続を閉じる（doctor 用）。"""
    service = RAGService(cfg.rag, cwd)
    try:
        return await service.stats()
    finally:
        await service.aclose()


def _rag_status(cfg: RuntimeConfig, cwd: Path) -> tuple[str, str]:
    """索引の有無・件数・埋め込み設定を点検する（オフライン。到達性は問い合わせない）。"""
    if not cfg.rag.enabled:
        return "-", ui("cli.app.disabled_rag_enabled_false")
    embed = f"{cfg.rag.embed_model} @ {cfg.rag.embed_base_url or DEFAULT_EMBED_BASE_URL}"
    if not _rag_db_path(cfg, cwd).is_file():
        return "warn", ui("cli.app.no_index_run_rag_ingest", embed=embed)
    try:
        stats = asyncio.run(_rag_stats(cfg, cwd))
    except (OSError, sqlite3.Error) as e:
        return "warn", ui("cli.app.cannot_open_the_index", error=e)
    detail = (
        f"{stats['chunks']} chunks / {stats['files']} files, "
        f"backend={stats['backend']}, embed={embed}"
    )
    if cfg.rag.auto_inject:
        detail += ui("cli.app.auto_inject_is_not_supported")
    return "ok", detail


def _provider_label(cfg: RuntimeConfig, model: str, env: Mapping[str, str]) -> str:
    """doctor の provider の行（``auto`` なら推測した種類を添える）。"""
    provider = cfg.provider
    if provider.kind != "auto":
        return provider.kind
    actual = current_choice(provider, model, env)
    if actual is None or not is_choice(actual):
        actual = choose_provider_kind(model, env, provider.base_url).value
    return f"auto ({actual})"


def _key_status(cfg: RuntimeConfig, model: str, env: Mapping[str, str]) -> tuple[str, str]:
    """doctor の api key の行（実際に使う接続先か、クラウドの鍵）。"""
    provider = cfg.provider
    kind = choose_provider_kind(model, env, provider.base_url, choice=provider.kind)
    if kind is ProviderKind.ANTHROPIC:
        ok = bool(env.get("ANTHROPIC_API_KEY"))
        if ok:
            return "ok", "ANTHROPIC_API_KEY set"
        return "warn", ui("cli.app.anthropic_api_key_not_set")
    if kind is ProviderKind.XAI:
        ok = bool(env.get("XAI_API_KEY"))
        if ok:
            return "ok", "XAI_API_KEY set"
        return "warn", ui("cli.app.xai_api_key_not_set")
    endpoint = resolve_endpoint(provider, env)
    if provider.kind == "openai":
        if env.get("OPENAI_API_KEY"):
            return "ok", f"OPENAI_API_KEY set ({OPENAI_API_BASE_URL})"
        return "warn", ui("providers.choice.key_is_not_set", name="OPENAI_API_KEY")
    if endpoint is not None:
        label = "provider.base_url" if endpoint.source == "settings" else endpoint.source
        return "ok", f"{label} ({endpoint.url})"
    if provider.kind in HOST_ENV:
        name, var = CHOICE_LABELS[provider.kind], HOST_ENV[provider.kind]
        return "warn", ui("providers.factory.no_endpoint_for", name=name, var=var)
    if env.get("OPENAI_API_KEY"):
        return ("ok", "OPENAI_API_KEY set")
    names = "(OPENAI_API_KEY / OLLAMA_HOST / OPENAI_BASE_URL / provider.base_url)"
    return "warn", ui("cli.app.no_openai_compatible_endpoint_or", names=names)


def _sessions_folder(cwd: Path) -> Path:
    return JSONLSessionStore().path_for(cwd, "_").parent


def _session_list(cwd: Path) -> None:
    sdir = _sessions_folder(cwd)
    files = sorted(sdir.glob("*.jsonl")) if sdir.is_dir() else []
    if not files:
        _console.print(f"[dim]{ui('cli.app.no_sessions')}[/dim]")
        return
    table = Table(title=f"sessions ({cwd})")
    table.add_column("session_id", style="cyan")
    table.add_column("model")
    table.add_column("messages", justify="right")
    store = JSONLSessionStore()
    for f in files:
        try:
            s = store.load(f)
            table.add_row(s.session_id, s.model, str(len(s.messages)))
        except Exception:
            table.add_row(f.stem, "(load error)", "-")
    _console.print(table)


def main() -> None:
    """``[project.scripts] satsumaestro`` のエントリポイント。"""
    if sys.platform == "win32":
        _err_console.print(ui("cli.app.windows_is_not_supported"), markup=False)
        raise SystemExit(1)
    app()


if __name__ == "__main__":
    main()
