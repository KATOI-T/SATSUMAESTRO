"""対話 REPL と権限プロンプタ。"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from satsumaestro.cli.commands import (
    CommandOutcome,
    SlashDispatcher,
    cycle_main_mode,
    parse_slash,
    provider_notes,
)
from satsumaestro.cli.completion import ModelNames, SlashCompleter
from satsumaestro.cli.completion_menu import MENU_ROWS, install_menu
from satsumaestro.cli.errors_ui import turn_failure
from satsumaestro.cli.input_box import InputBox
from satsumaestro.cli.interrupts import run_cancellable
from satsumaestro.cli.line_input import ask_line
from satsumaestro.cli.load_watch import LoadProbe, watching_model_load
from satsumaestro.cli.modes import mode_hint_fragments, mode_markup, mode_prompt_fragments
from satsumaestro.cli.render import RichRenderer, action_preview
from satsumaestro.cli.updates import UpdateCheck
from satsumaestro.engine.events import ToolUse
from satsumaestro.engine.plan_budget import BudgetVerdict, PlanBudget
from satsumaestro.engine.plan_steps import PlanStep, parse_steps
from satsumaestro.engine.provider_switch import MissingModelError, reconnect
from satsumaestro.engine.recovery import SessionAbortError
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.trace import TurnObserver
from satsumaestro.engine.turn import (
    DETAIL_REQUEST_ORIGIN,
    USER_DENY_ORIGIN,
    PermissionDetail,
    PlanExitVerdict,
    TurnSummary,
    context_notice,
)
from satsumaestro.i18n.text import ui
from satsumaestro.orchestration.runner import Orchestrator
from satsumaestro.providers import factory
from satsumaestro.providers.key_check import KeyCheck, KeyChecks
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionOutcome
from satsumaestro.security.remember_guard import blocker_note, remember_blocker
from satsumaestro.session.redact import redact_pieces, secret_values, shown_input_pieces
from satsumaestro.telemetry.log import TraceLog
from satsumaestro.util.environ import ENVIRON
from satsumaestro.util.errors import (
    SatsumaError,
    ToolCallingUnsupportedError,
)

_PLAN_APPROVE_WITH_NOTE_ANSWERS = frozenset({"a", "y+", "yes+", "note"})
_ALWAYS_ANSWERS = frozenset({"a", "always"})
_FULL_DETAIL_TOOLS = frozenset({"edit_file", "write_file", "append_file"})
_INTERRUPTED = "\x00interrupted"
_CARD_QUESTIONS = {"許可しますか？": "許可しますか?", "Allow this?": "Allow this ?"}


def _permission_prompt(*, detail: bool, always: bool) -> str:
    """権限カードの問い。"""
    if detail and always:
        return ui("cli.repl.permission_prompt_always_detail")
    if always:
        return ui("cli.repl.permission_prompt_always")
    if detail:
        return ui("cli.repl.permission_prompt_detail")
    return ui("cli.repl.permission_prompt")


def _plan_prompt() -> str:
    return ui("cli.repl.approve_the_plan_and_start")


def _failure_label(failure: str | None) -> str:
    """説明を取れなかった理由。"""
    if failure == "timeout":
        return ui("cli.repl.stopped_after_30_s")
    if failure == "interrupted":
        return ui("cli.repl.interrupted")
    if failure == "off":
        return ui("cli.repl.explanations_are_turned_off")
    return ui("cli.repl.failed")


def _outcome(answer: str) -> PermissionOutcome:
    """y / a / n（とそれ以外）を判定に変える。"""
    if answer in ("y", "yes"):
        return PermissionOutcome(decision="allow", reason="approved by user")
    if answer in _ALWAYS_ANSWERS:
        return PermissionOutcome(decision="allow", reason="approved and remembered", remember=True)
    if answer == _INTERRUPTED:
        return PermissionOutcome(
            decision="deny", reason="denied by user (interrupt)", origin=USER_DENY_ORIGIN
        )
    return PermissionOutcome(decision="deny", reason="denied by user", origin=USER_DENY_ORIGIN)


def _as_is(text: str) -> str:
    return text


def _hider(call: ToolUse) -> Callable[[str], str]:
    """カードで伏せた入力の値と秘密を、説明の文から伏せる関数。"""
    values = secret_values()
    pieces = shown_input_pieces(call.name, call.input, values=values)
    return lambda text: redact_pieces(text, pieces, values=values)


def _detail_panel(detail: PermissionDetail, hide: Callable[[str], str] = _as_is) -> Panel:
    """d で出すパネル: モデルの説明（``hide`` で伏せる）。"""
    title = ui("cli.repl.what_it_is_about_to")
    if detail.explanation is not None:
        text = hide(detail.explanation)
        for question, broken in _CARD_QUESTIONS.items():
            text = text.replace(question, broken)
        return Panel(Text(text), title=title, border_style="cyan", expand=False)
    label = _failure_label(detail.failure)
    body = Text(
        ui("cli.repl.could_not_get_an_explanation", label=label),
        style="dim",
    )
    if detail.model_text.strip():
        said = ui("cli.repl.what_the_model_said_just")
        body.append(said, style="bold").append(hide(detail.model_text.strip()))
    if detail.step is not None:
        index, total, step_text = detail.step
        step = ui("cli.repl.current_step", index=index, total=total)
        body.append(step, style="bold").append(hide(step_text))
    if detail.user_request.strip():
        request = hide(detail.user_request.strip())
        body.append(ui("cli.repl.request"), style="bold").append(request)
    return Panel(body, title=title, border_style="cyan", expand=False)


class InteractivePrompter:
    """``ask`` を実行内容カード＋``[y/n/a/d]`` で解決する対話プロンプタ。"""

    def __init__(
        self, console: Console | None = None, *, plan_steps: bool = False, redact: bool = False
    ) -> None:
        self.console = console or Console()
        self._plan_steps = plan_steps
        self._redact = redact
        self._plan_budget: tuple[PlanBudget, Sequence[PlanStep], BudgetVerdict | None] | None = None

    def set_plan_budget(
        self,
        budget: PlanBudget,
        steps: Sequence[PlanStep],
        verdict: BudgetVerdict | None,
    ) -> None:
        self._plan_budget = (budget, steps, verdict)

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.console.print(
            action_preview(call, required.value, warning=warning, redact=self._redact)
        )
        blocker = remember_blocker(call.name, call.input)
        if blocker is not None:
            self.console.print(Text(blocker_note(blocker), style="dim"))
        answer = self._ask_choice(_permission_prompt(detail=True, always=blocker is None), blocker)
        if answer in ("d", "detail"):
            return PermissionOutcome(decision="ask", origin=DETAIL_REQUEST_ORIGIN)
        return _outcome(answer)

    def waiting_for_detail(self) -> Status:
        """説明を待つ間、「考え中」と同じ回転アイコン（cyan の dots）を左に出す。"""
        waiting = ui("cli.repl.asking_the_model_what_it")
        return self.console.status(Text(waiting, style="dim"), spinner="dots", spinner_style="cyan")

    def decide_with_detail(
        self,
        call: ToolUse,
        required: PermissionMode,
        warning: str | None,
        detail: PermissionDetail,
    ) -> PermissionOutcome:
        """d の後: 説明（取れなければ代わりの事実）と詳細を出し、y / n / a で受ける。"""
        self.console.print(_detail_panel(detail, _hider(call) if self._redact else _as_is))
        if call.name in _FULL_DETAIL_TOOLS:
            card = action_preview(
                call, required.value, detailed=True, warning=warning, redact=self._redact
            )
            self.console.print(card)
        blocker = remember_blocker(call.name, call.input)
        prompt = _permission_prompt(detail=False, always=blocker is None)
        while True:
            answer = self._ask_choice(prompt, blocker)
            if answer in ("d", "detail"):
                shown = ui("cli.repl.details_are_already_shown")
                self.console.print(f"[dim]{shown}[/dim]")
                continue
            return _outcome(answer)

    def _ask_choice(self, prompt: str, blocker: str | None) -> str:
        """選択肢を 1 つ受け取る。"""
        while True:
            answer = self._ask(prompt)
            if blocker is None or answer not in _ALWAYS_ANSWERS:
                return answer
            retry = ui("cli.repl.this_action_cannot_be_remembered")
            self.console.print(Text(retry, style="dim"))

    def _ask(self, prompt: str) -> str:
        """1 行受け取る。"""
        try:
            return ask_line(self.console, prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.console.print(f"[dim]{ui('cli.repl.denied')}[/dim]")
            return _INTERRUPTED

    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        """モデルの ``exit_plan_mode`` を承認カードで確認する。"""
        body = Markdown(plan_text) if plan_text else ui("cli.repl.no_plan_file")
        title = ui("cli.repl.plan_approval")
        self.console.print(Panel(body, title=title, border_style="yellow"))
        steps = parse_steps(plan_text) if self._plan_steps else []
        if steps:
            n = len(steps)
            runs = ui("cli.repl.runs_in_steps_one_at", n=n)
            self.console.print(f"[dim]{runs}[/dim]", markup=True)
        if self._plan_budget is not None:
            budget, plan_steps, verdict = self._plan_budget
            self._plan_budget = None
            largest = max(
                ((i, s.estimated_lines) for i, s in enumerate(plan_steps, 1) if s.estimated_lines),
                key=lambda pair: pair[1],
                default=None,
            )
            line = budget.explain_card()
            if largest is not None:
                index, lines = largest
                line += ui("cli.repl.largest_step_lines", index=index, lines=lines)
            self.console.print(
                f"[dim]{ui('cli.repl.line_in_parentheses', line=line)}[/dim]", markup=True
            )
            if verdict is not None:
                style = "yellow" if verdict.sends_back else "dim"
                card = verdict.card()
                self.console.print(
                    f"[{style}]{ui('cli.repl.card_in_parentheses', card=card)}[/{style}]",
                    markup=True,
                )
        try:
            answer = ask_line(self.console, _plan_prompt()).strip().lower()
        except (EOFError, KeyboardInterrupt):
            staying = ui("cli.repl.not_approved_staying_in_plan")
            self.console.print(f"[dim]{staying}[/dim]")
            return PlanExitVerdict(approved=False)
        if answer in ("y", "yes"):
            return PlanExitVerdict(approved=True)
        if answer in _PLAN_APPROVE_WITH_NOTE_ANSWERS:
            try:
                note = ask_line(self.console, ui("cli.repl.a_note_to_apply_while")).strip()
            except (EOFError, KeyboardInterrupt):
                note = ""
            return PlanExitVerdict(approved=True, note=note or None)
        try:
            feedback = ask_line(self.console, ui("cli.repl.what_to_change_empty_to")).strip()
        except (EOFError, KeyboardInterrupt):
            feedback = ""
        return PlanExitVerdict(approved=False, feedback=feedback or None)


def _prompt_session(
    rt: ConversationRuntime,
    bindings: KeyBindings,
    *,
    key_checks: KeyChecks | None = None,
    **io: Any,
) -> PromptSession[str]:
    """端末でないときの 1 行入力（行が ``/`` で始まる間は、打ちながら候補を出す）。"""
    session: PromptSession[str] = PromptSession(
        key_bindings=bindings,
        completer=SlashCompleter(
            ModelNames(rt),
            providers=lambda: provider_notes(
                rt.config.provider, rt.session.model, ENVIRON, _rejected(key_checks)
            ),
            rejected=lambda: _rejected(key_checks),
        ),
        complete_while_typing=Condition(lambda: session.default_buffer.text.startswith("/")),
        reserve_space_for_menu=MENU_ROWS + 1,
        **io,
    )
    install_menu(session)
    return session


def _rejected(checks: KeyChecks | None) -> dict[str, KeyCheck]:
    return checks.rejected() if checks is not None else {}


async def _reconnected(rt: ConversationRuntime, console: Console) -> bool:
    """接続できないまま起動したランタイムを、送る前につなぎ直す（だめなら理由と案内を出す）。"""
    from satsumaestro.cli.app import missing_model_text

    try:
        connection = await reconnect(rt, ENVIRON, build=factory.build_provider)
    except MissingModelError as exc:
        console.print(missing_model_text(exc.notice))
        return False
    if connection.problem is not None:
        console.print(Text(connection.problem, style="yellow"))
        console.print(Text(ui("cli.repl.not_connected_nothing_sent"), style="yellow"))
        return False
    for note in connection.notes:
        console.print(Text(note, style="yellow"))
    return True


# Windows でも読み込めるように（起動の入口が WSL2 を案内して止める）
_STOP_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)


@contextmanager
def _closing_on_signals(session: InputBox, received: list[int]) -> Iterator[None]:
    """入力欄を待つ間だけ、SIGTERM・SIGHUP で入力欄を閉じる（端末のモードは入力欄が戻す）。"""
    loop = asyncio.get_running_loop()

    def stop(signum: int) -> None:
        received.append(signum)
        app = session.app
        if app.is_running and app.future is not None and not app.future.done():
            app.exit(exception=EOFError())

    installed: list[int] = []
    for signum in _STOP_SIGNALS:
        if signal.getsignal(signum) is not signal.SIG_DFL:
            continue
        try:
            loop.add_signal_handler(signum, stop, signum)
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        installed.append(signum)
    try:
        yield
    finally:
        for signum in installed:
            loop.remove_signal_handler(signum)


async def run_repl(
    rt: ConversationRuntime,
    prompter: InteractivePrompter,
    render: RichRenderer,
    console: Console,
    observer: TurnObserver | None = None,
    trace_log: TraceLog | None = None,
    *,
    trusted: bool = False,
    load_probe: LoadProbe | None = None,
    key_checks: KeyChecks | None = None,
    updates: UpdateCheck | None = None,
) -> int | None:
    """行入力ごとに ``run_turn`` を回す対話ループ（入力欄で受けて終わったシグナルを返す）。"""
    bindings = KeyBindings()

    @bindings.add(Keys.BackTab)
    def _cycle_mode(event: KeyPressEvent) -> None:
        cycle_main_mode(rt)
        event.app.invalidate()

    session = InputBox(
        console=console,
        hint=lambda: mode_hint_fragments(rt.session.mode),
        status=lambda: rt.session.model,
        fallback_prompt=lambda: mode_prompt_fragments(rt.session.mode),
        key_bindings=bindings,
        fallback=lambda: _prompt_session(rt, bindings, key_checks=key_checks),
    )
    dispatcher = SlashDispatcher(
        rt, console, trace_log=trace_log, trusted=trusted, render=render, key_checks=key_checks
    )
    banner = ui("cli.repl.satsumaestro_repl_help_for")
    console.print(f"[dim]{banner}[/dim]")
    workdir = escape(str(rt.cwd))
    workdir_line = ui("cli.repl.working_directory", workdir=workdir)
    console.print(f"[dim]{workdir_line}[/dim]")
    newer = updates.notice() if updates else None
    if newer:
        console.print(Text(newer, style="yellow"))
    # 結果は /provider の一覧と候補にだけ出す
    checking = asyncio.create_task(key_checks.run(ENVIRON)) if key_checks else None
    # 取り直した結果は次の起動で知らせる
    refreshing = asyncio.create_task(updates.refresh()) if updates else None
    received: list[int] = []
    while True:
        try:
            with _closing_on_signals(session, received):
                line = await session.prompt_async()
        except EOFError:
            break
        except KeyboardInterrupt:
            if received:
                break
            continue
        if received:
            break

        line = line.strip()
        if not line:
            continue
        cmd = parse_slash(line)
        if cmd is not None:
            try:
                if await dispatcher.dispatch(cmd) is CommandOutcome.EXIT:
                    break
            except Exception as e:
                failed = ui("cli.repl.the_command_failed", error=e)
                console.print(f"[red]{failed}[/red]")
            continue

        if rt.unconnected is not None and not await _reconnected(rt, console):
            continue
        if not rt.session.model.strip():
            console.print(Text(ui("cli.repl.choose_a_model_first"), style="yellow"))
            continue
        if rt.approve_plan_from_text(line):
            mode = mode_markup(rt.session.mode)
            approved = ui("cli.repl.taken_as_an_instruction_to", mode=mode)
            console.print(f"[dim]{approved}[/dim]")
        render.start_turn()
        summary: TurnSummary | None = None
        try:
            async with watching_model_load(render, load_probe):
                summary = await run_cancellable(
                    rt.run_turn(line, prompter, render, observer=observer)
                )
            if summary.stop_reason == "user_denied":
                stopped = ui("cli.repl.denied_so_this_turn_was")
                console.print(f"[dim]{stopped}[/dim]")
            if summary.health_warning is not None:
                warning = summary.health_warning
                health = ui("cli.repl.session_health_warning_starting_a", warning=warning)
                console.print(f"[yellow]{health}[/yellow]")
        except KeyboardInterrupt:
            render.pause()
            console.print(f"[dim]{ui('cli.repl.interrupted_notice')}[/dim]")
        except SessionAbortError as e:
            console.print(f"[red]⚠ {e}[/red]")
        except ToolCallingUnsupportedError as e:
            console.print(f"[red]⚠ {e}[/red]")
        except SatsumaError as e:
            failure = turn_failure(e)
            console.print(Text(failure.headline, style="red"))
            if failure.next_step and not rt.session.plan_steps:
                console.print(Text(f"  {failure.next_step}", style="dim"))
        finally:
            render.end_turn(delegated=summary.delegated_usage if summary else None)
            for note in rt.take_window_notes():
                context_notice(render, note)
            pending = rt.session.plan_steps
            if pending:
                step = f"{rt.session.plan_step_index}/{len(pending)}"
                unfinished = ui("cli.repl.plan_step_is_not_finished", step=step)
                console.print(f"[dim]{unfinished}[/dim]")

    if checking is not None:
        checking.cancel()
    if refreshing is not None:
        refreshing.cancel()
    orch = rt.orchestrator
    if isinstance(orch, Orchestrator) and orch.running_count() > 0:
        n = orch.running_count()
        stopping = ui("cli.repl.stopping_background_tasks", n=n)
        console.print(f"[dim]{stopping}[/dim]")
    return received[0] if received else None
