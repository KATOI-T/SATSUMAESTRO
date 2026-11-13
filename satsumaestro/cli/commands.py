"""REPL スラッシュコマンド。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from satsumaestro.cli import language_gate
from satsumaestro.cli.modes import mode_markup
from satsumaestro.config.loader import save_model, save_screen_language
from satsumaestro.config.models import ROLE_NAMES, ProviderConfig, ReasoningEffort
from satsumaestro.i18n.text import screen_languages, ui, use_screen_language
from satsumaestro.providers.choice import (
    CHOICE_LABELS,
    PROVIDER_CHOICES,
    ProviderStatus,
    current_choice,
    is_choice,
    provider_statuses,
    resolve_endpoint,
)
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import ModeState
from satsumaestro.telemetry.log import TraceLog
from satsumaestro.telemetry.recorder import insights_from
from satsumaestro.util.environ import ENVIRON
from satsumaestro.util.errors import ConfigError, SatsumaError
from satsumaestro.util.paths import default_config_home

if TYPE_CHECKING:
    from satsumaestro.cli.render import RichRenderer
    from satsumaestro.engine.runtime import ConversationRuntime

_EFFORT_VALUES = (*get_args(ReasoningEffort), "default")
_SCREEN_VALUES = (*screen_languages(), language_gate.AUTO)
_SUB_VALUES = tuple(m.value for m in PermissionMode)


def _mode_hint() -> str:
    subs = "|".join(_SUB_VALUES)
    return ui("cli.commands.use_mode_plan_mode_code", subs=subs)


def cycle_main_mode(rt: ConversationRuntime) -> ModeState:
    """Shift+Tab: plan ⇄ code を循環する（sub は維持・カード無し・保存あり）。"""
    state = rt.set_mode(main="code" if rt.session.mode.main == "plan" else "plan")
    rt.store.save(rt.session)
    return state


@dataclass(frozen=True)
class ReplCommand:
    """1 つのコマンド。"""

    name: str
    about: Callable[[], str]
    usage: str = ""
    also: tuple[str, ...] = ()


SLASH_COMMANDS: tuple[ReplCommand, ...] = (
    ReplCommand("help", lambda: ui("cli.commands.show_the_commands")),
    ReplCommand(
        "compact",
        lambda: ui("cli.commands.fold_the_conversation_into_a"),
    ),
    ReplCommand(
        "effort",
        lambda: ui("cli.commands.switch_the_reasoning_effort"),
        usage="low|medium|high|default",
    ),
    ReplCommand(
        "language",
        lambda: ui("cli.commands.show_switch_the_screen_language"),
        usage=f"[{'|'.join(_SCREEN_VALUES)}]",
    ),
    ReplCommand(
        "fast",
        lambda: ui("cli.commands.turn_fast_mode_on_or_off"),
    ),
    ReplCommand(
        "model",
        lambda: ui("cli.commands.show_switch_the_current_model"),
        usage="[model]",
    ),
    ReplCommand(
        "provider",
        lambda: ui("cli.commands.show_switch_the_provider"),
        usage="[name|number] [URL] [model]",
    ),
    ReplCommand(
        "session",
        lambda: ui("cli.commands.show_the_sessions_and_the"),
    ),
    ReplCommand(
        "doctor",
        lambda: ui("cli.commands.check_the_settings_and_the"),
    ),
    ReplCommand(
        "mode",
        lambda: ui("cli.commands.show_switch_the_main_mode"),
        usage="[plan|code [" + "|".join(_SUB_VALUES) + "]]",
    ),
    ReplCommand(
        "plan",
        lambda: ui("cli.commands.switch_plan_mode_alias_of"),
        usage="on|off",
    ),
    ReplCommand(
        "sub-mode",
        lambda: ui("cli.commands.show_switch_the_permission_sub"),
        usage="[" + "|".join(_SUB_VALUES) + "]",
        also=("submode",),
    ),
    ReplCommand(
        "clear",
        lambda: ui("cli.commands.clear_the_conversation_todos_and"),
    ),
    ReplCommand(
        "insights",
        lambda: ui("cli.commands.show_telemetry_turns_runs_and"),
    ),
    ReplCommand("exit", lambda: ui("cli.commands.exit_the_repl"), also=("quit", "q")),
)

_BY_NAME: dict[str, ReplCommand] = {
    name: command for command in SLASH_COMMANDS for name in (command.name, *command.also)
}


def command_named(name: str) -> ReplCommand | None:
    """名前か別名のコマンド（大文字と小文字は区別しない）。"""
    return _BY_NAME.get(name.lower())


_SUB_NOTES: dict[str, Callable[[], str]] = {
    "read-only": lambda: ui("cli.commands.choice_sub_read_only"),
    "workspace-write": lambda: ui("cli.commands.choice_sub_workspace_write"),
    "danger-full-access": lambda: ui("cli.commands.choice_sub_danger_full_access"),
    "prompt": lambda: ui("cli.commands.choice_sub_prompt"),
    "allow": lambda: ui("cli.commands.choice_sub_allow"),
}
_VALUE_NOTES: dict[str, dict[str, Callable[[], str]]] = {
    "effort": {
        "none": lambda: ui("cli.commands.choice_effort_none"),
        "minimal": lambda: ui("cli.commands.choice_effort_minimal"),
        "low": lambda: ui("cli.commands.choice_effort_low"),
        "medium": lambda: ui("cli.commands.choice_effort_medium"),
        "high": lambda: ui("cli.commands.choice_effort_high"),
        "xhigh": lambda: ui("cli.commands.choice_effort_xhigh"),
        "max": lambda: ui("cli.commands.choice_effort_max"),
        "default": lambda: ui("cli.commands.choice_effort_default"),
    },
    "mode": {
        "plan": lambda: ui("cli.commands.choice_mode_plan"),
        "code": lambda: ui("cli.commands.choice_mode_code"),
    },
    "sub-mode": _SUB_NOTES,
    "plan": {
        "on": lambda: ui("cli.commands.choice_plan_on"),
        "off": lambda: ui("cli.commands.choice_plan_off"),
    },
}


def _noted(name: str, values: Sequence[str]) -> tuple[tuple[str, str], ...]:
    notes = _VALUE_NOTES.get(name, {})
    return tuple((value, note() if (note := notes.get(value)) else "") for value in values)


def argument_choices(
    name: str, given: Sequence[str], *, providers: Mapping[str, str] | None = None
) -> tuple[tuple[str, str], ...]:
    """``/<name>`` で ``given`` の次の引数に取れる決まった値と、その 1 行の説明。"""
    if name == "mode" and [arg.lower() for arg in given] == ["code"]:
        return _noted("sub-mode", _SUB_VALUES)
    if given:
        return ()
    if name == "language":
        auto = ui("cli.commands.choice_language_auto")
        named = language_gate.native_name
        return tuple((v, auto if v == language_gate.AUTO else named(v)) for v in _SCREEN_VALUES)
    if name == "provider":
        notes = providers or {}
        return tuple((c, notes.get(c) or CHOICE_LABELS[c]) for c in PROVIDER_CHOICES)
    values = {
        "effort": _EFFORT_VALUES,
        "mode": ("plan", "code"),
        "sub-mode": _SUB_VALUES,
        "plan": ("on", "off"),
    }.get(name, ())
    return _noted(name, values)


def provider_notes(provider: ProviderConfig, model: str, env: Mapping[str, str]) -> dict[str, str]:
    """``/provider`` の候補に添える、名前と接続先か鍵の有無（今の種類に ✓）。"""
    notes: dict[str, str] = {}
    for status in provider_statuses(provider, model, env):
        label = CHOICE_LABELS[status.choice] + (" ✓" if status.current else "")
        notes[status.choice] = f"{label} · {_status_text(status)}"
    return notes


@dataclass
class ParsedCommand:
    name: str
    args: list[str]
    raw: str = ""


class CommandOutcome(StrEnum):
    HANDLED = "handled"
    EXIT = "exit"


def parse_slash(line: str) -> ParsedCommand | None:
    """行頭 ``/`` のみコマンド化する。"""
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped[1:].split()
    if not parts:
        return ParsedCommand(name="help", args=[])
    head = parts[0]
    name = head.lower()
    if name not in _BY_NAME and _reads_as_path(head):
        return None
    return ParsedCommand(name=name, args=parts[1:], raw=head)


def _reads_as_path(head: str) -> bool:
    """``/`` の直後の先頭語が、コマンド名ではなくパスの一部に見えるか。"""
    if "/" in head or not head.isascii():
        return True
    try:
        return Path("/" + head).exists()
    except OSError:
        return False


_URL_HOST = re.compile(
    r"^(\[[0-9A-Fa-f:]+\]|\d{1,3}(\.\d{1,3}){3}|localhost|([A-Za-z0-9-]+\.)+local)(:\d+)?(/.*)?$"
)


def _looks_like_url(arg: str) -> bool:
    """``://`` を含むか、IP・``localhost``・``*.local`` のホスト（ポートは任意）か。"""
    return "://" in arg or _URL_HOST.match(arg) is not None


def provider_args(args: list[str]) -> tuple[str, str | None, str | None] | None:
    """``/provider`` の引数を種類・URL・モデルに分ける（読めなければ ``None``）。"""
    head = args[0].lower()
    if head.isdigit() and 1 <= int(head) <= len(PROVIDER_CHOICES):
        head = PROVIDER_CHOICES[int(head) - 1]
    url: str | None = None
    model: str | None = None
    for arg in args[1:]:
        if _looks_like_url(arg) and url is None:
            url = arg
        elif not _looks_like_url(arg) and model is None:
            model = arg
        else:
            return None
    return head, url, model


def _status_text(status: ProviderStatus) -> str:
    """``/provider`` の一覧の 1 行の、接続先か鍵の有無。"""
    if status.key_env and status.key_set:
        text = ui("cli.commands.provider_key_set", name=status.key_env)
    elif status.key_env:
        text = ui("cli.commands.provider_no_key", name=status.key_env)
    elif status.endpoint is None:
        text = ui("cli.commands.provider_not_set")
    else:
        source = status.endpoint.source
        label = ui("cli.commands.provider_last_used") if source == "profile" else source
        text = ui("cli.commands.provider_endpoint", url=status.endpoint.url, source=label)
    if status.model and not status.current:
        text += f" · {status.model}"
    return text


class SlashDispatcher:
    """spec 照合 → ハンドラ実行。"""

    def __init__(
        self,
        rt: ConversationRuntime,
        console: Console,
        trace_log: TraceLog | None = None,
        *,
        trusted: bool = False,
        render: RichRenderer | None = None,
    ) -> None:
        self.rt = rt
        self.console = console
        self.trace_log = trace_log
        self.trusted = trusted
        self.render = render
        self._fast_prev: ReasoningEffort | None = None
        self._fast_on = False

    def _print(self, *objects: object, **kwargs: Any) -> None:
        """自動の色付けを切って出す。"""
        self.console.print(*objects, highlight=False, **kwargs)

    async def dispatch(self, cmd: ParsedCommand) -> CommandOutcome:
        command = _BY_NAME.get(cmd.name)
        if command is None:
            shown = escape(cmd.raw or cmd.name)
            unknown = ui("cli.commands.unknown_command_help_for_the", shown=shown)
            self._print(f"[yellow]{unknown}[/yellow]")
            return CommandOutcome.HANDLED
        handler = getattr(self, f"_cmd_{command.name.replace('-', '_')}")
        return await handler(cmd.args) or CommandOutcome.HANDLED

    async def _cmd_help(self, args: list[str]) -> None:
        table = Table(show_header=True, header_style="bold")
        table.add_column(ui("cli.commands.command"))
        table.add_column(ui("cli.commands.description"))
        for command in SLASH_COMMANDS:
            shown = f"/{command.name}"
            if command.usage:
                shown += f" {command.usage}"
            if command.also:
                shown += " (" + ", ".join(f"/{other}" for other in command.also) + ")"
            table.add_row(escape(shown), escape(command.about()))
        self._print(table)

    async def _cmd_exit(self, args: list[str]) -> CommandOutcome:
        return CommandOutcome.EXIT

    async def _cmd_compact(self, args: list[str]) -> None:
        outcome = await self.rt.compactor.acompact(
            self.rt.session, limits=self.rt.compaction_limits()
        )
        if outcome is None:
            nothing = ui("cli.commands.nothing_to_fold_too_few")
            self._print(f"[dim]{nothing}[/dim]")
            return
        self.rt.store.save(self.rt.session)
        n = outcome.dropped_messages
        folded = ui("cli.commands.compacted_folded_messages_into_a", n=n)
        self._print(f"[dim]{folded}[/dim]")

    async def _cmd_effort(self, args: list[str]) -> None:
        sampling = self.rt.config.provider.sampling
        if not args:
            current = sampling.reasoning_effort or "default"
            self._print(ui("cli.commands.reasoning_effort_change_effort_low", current=current))
            return
        value = args[0].lower()
        if value not in _EFFORT_VALUES:
            values = "|".join(_EFFORT_VALUES)
            bad = ui("cli.commands.effort_must_be_one_of", values=values)
            self._print(f"[yellow]{bad}[/yellow]")
            return
        sampling.reasoning_effort = None if value == "default" else cast(ReasoningEffort, value)
        self._fast_on = False
        self._print(f"[dim]reasoning effort → {value}[/dim]")

    async def _cmd_language(self, args: list[str]) -> None:
        """画面の言語の表示と切り替え（設定の ``language.ui``）。"""
        language = self.rt.language
        values = "|".join(_SCREEN_VALUES)
        if not args:
            how = {
                None: ui("cli.commands.how_not_set"),
                language_gate.AUTO: ui("cli.commands.how_auto"),
            }.get(language.ui_setting) or ui("cli.commands.how_chosen")
            name = language_gate.native_name(language.ui_language)
            self._print(Text(ui("cli.commands.screen_language_now", name=name, how=how)))
            for line in language_gate.choice_lines(language.ui_language):
                self._print(line)
            change = ui("cli.commands.change_screen_language", values=values)
            self._print(Text(change, style="dim"))
            return
        value = language_gate.parse_answer(" ".join(args), language.ui_language)
        if value is None:
            numbers = "/".join(str(n) for n in range(1, len(screen_languages()) + 1))
            bad = ui("cli.commands.screen_language_must_be", values=values, numbers=numbers)
            self._print(Text(bad, style="yellow"))
            return
        path: Path | None = None
        error = ""
        try:
            path = save_screen_language(value)
        except (ConfigError, OSError) as exc:
            error = str(exc)
        overridden = language.set_ui(value)
        use_screen_language(language.chosen_ui)
        name = language_gate.native_name(value)
        if path is None:
            not_saved = ui(
                "cli.commands.screen_language_switched_not_saved", name=name, error=error
            )
            self._print(Text(not_saved, style="yellow"))
        else:
            switched = ui("cli.commands.screen_language_switched", name=name, path=path)
            self._print(Text(switched, style="dim"))
        if overridden is not None:
            note = ui("cli.commands.screen_language_overridden", value=overridden)
            self._print(Text(note, style="yellow"))

    async def _cmd_fast(self, args: list[str]) -> None:
        sampling = self.rt.config.provider.sampling
        if self._fast_on:
            sampling.reasoning_effort = self._fast_prev
            self._fast_on = False
            prev = self._fast_prev or "default"
            off = ui("cli.commands.fast_mode_off_effort", prev=prev)
            self._print(f"[dim]{off}[/dim]")
        else:
            self._fast_prev = sampling.reasoning_effort
            sampling.reasoning_effort = "low"
            self._fast_on = True
            on = ui("cli.commands.fast_mode_on_effort_low")
            self._print(f"[dim]{on}[/dim]")

    async def _cmd_model(self, args: list[str]) -> None:
        from satsumaestro.cli.app import _print_window, missing_model_text
        from satsumaestro.engine.context_fit import (
            check_model,
            list_models,
            missing_model_notice,
        )
        from satsumaestro.providers.detect import same_model
        from satsumaestro.providers.registry import expand_model_name

        current = self.rt.session.model
        if not args:
            self._print(f"model: {escape(current)}")
            names = await list_models(self.rt.config, current, ENVIRON)
            for name in names or []:
                mark = "*" if same_model(name, current) else " "
                self._print(f"  {mark} {escape(name)}")
            return
        resolved = expand_model_name(args[0])
        check = await check_model(self.rt.config, resolved, ENVIRON)
        if check is not None and check.missing:
            self._print(missing_model_text(missing_model_notice(resolved, check, startup=False)))
            kept = escape(current)
            self._print(f"[dim]{ui('cli.commands.model_stays', kept=kept)}[/dim]")
            return
        self.rt.session.model = resolved
        self._print(self._save_model(resolved))
        fit = await self.rt.refit_for_model()
        await _print_window(self.rt, fit, console=self.console, breakdown_line=True)

    async def _cmd_provider(self, args: list[str]) -> None:
        from satsumaestro.cli.app import _print_window, _unloaded_model, missing_model_text
        from satsumaestro.config.loader import ConfigLoader, save_provider
        from satsumaestro.engine.context_fit import local_server_for
        from satsumaestro.engine.provider_switch import (
            MissingModelError,
            ModelNeededError,
            switch_provider,
            unloaded_text,
        )

        loader = ConfigLoader(self.rt.cwd, trusted=self.trusted)
        loader.load_env()
        if not args:
            self._show_providers()
            return
        parsed = provider_args(args)
        if parsed is None:
            self._print(Text(ui("cli.commands.provider_usage_bad"), style="yellow"))
            return
        choice, url, model = parsed
        stays = Text(ui("cli.commands.provider_stays", name=self._provider_name()), style="dim")
        try:
            result = await switch_provider(self.rt, choice, env=ENVIRON, url=url, model=model)
        except MissingModelError as exc:
            self._print(missing_model_text(exc.notice))
            self._print(stays)
            return
        except ModelNeededError as exc:
            self._print(Text(str(exc), style="yellow"))
            for name in exc.models:
                self._print(Text(f"  {name}"))
            self._print(stays)
            return
        except SatsumaError as exc:
            self._print(Text(str(exc), style="yellow"))
            self._print(stays)
            return
        url_shown = result.endpoint.url if result.endpoint else ""
        detail = " · ".join(part for part in (url_shown, result.model) if part)
        try:
            path = save_provider(
                result.choice, result.model, result.base_url, remember=result.remember
            )
        except (ConfigError, OSError) as exc:
            not_saved = ui(
                "cli.commands.provider_switched_not_saved",
                name=result.choice,
                detail=detail,
                error=str(exc),
            )
            self._print(Text(not_saved, style="yellow"))
        else:
            switched = ui(
                "cli.commands.provider_switched", name=result.choice, detail=detail, path=path
            )
            self._print(Text(switched, style="dim"))
        for note in result.notes:
            self._print(Text(note, style="yellow"))
        for unloaded in result.unloaded:
            self._print(Text(unloaded_text(unloaded), style="dim"))
        kept = self._role_models()
        if kept:
            name = CHOICE_LABELS[result.choice]
            note = ui("cli.commands.provider_roles_kept", items=kept, name=name)
            self._print(Text(note, style="yellow"))
        if self.render is not None:
            server = local_server_for(self.rt.config, result.model, ENVIRON)
            kind = server.kind if server is not None else "ollama"
            self.render.use_local_server(kind, _unloaded_model(result.fit))
        await _print_window(self.rt, result.fit, console=self.console, breakdown_line=True)

    def _save_model(self, model: str) -> Text:
        """選んだモデルを今の設定とユーザーの設定に書き、知らせる文を返す。"""
        provider = self.rt.config.provider
        choice = provider.kind if provider.kind != "auto" else None
        profiles = dict(provider.profiles)
        profile = profiles.get(choice) if choice is not None else None
        if choice is not None and profile is not None:
            profiles[choice] = profile.model_copy(update={"model": model})
        update = {"default_model": model, "profiles": profiles}
        self.rt.config.provider = provider.model_copy(update=update)
        try:
            path = save_model(model, choice=choice)
        except (ConfigError, OSError) as exc:
            text = ui("cli.commands.model_switched_not_saved", name=model, error=str(exc))
            return Text(text, style="yellow")
        return Text(ui("cli.commands.model_switched", name=model, path=path), style="dim")

    def _show_providers(self) -> None:
        """今のプロバイダと、選べるプロバイダごとの接続先か鍵の有無を出す。"""
        provider, model = self.rt.config.provider, self.rt.session.model
        actual = current_choice(provider, model, ENVIRON)
        endpoint = (
            resolve_endpoint(provider, ENVIRON) if actual not in ("anthropic", "xai") else None
        )
        detail = " · ".join(part for part in (endpoint.url if endpoint else "", model) if part)
        name = self._provider_name()
        self._print(Text(ui("cli.commands.provider_header", name=name, detail=detail)))
        statuses = provider_statuses(provider, model, ENVIRON)
        width = max(len(status.choice) for status in statuses) + 2
        for n, status in enumerate(statuses, 1):
            label = f"{status.choice} ✓" if status.current else status.choice
            self._print(Text(f"  {n}) {label:<{width}}  {_status_text(status)}"))
        path = default_config_home() / "settings.json"
        self._print(Text(ui("cli.commands.provider_usage", path=path), style="dim"))

    def _provider_name(self) -> str:
        """明示の種類か、``auto`` なら推測した種類。"""
        provider = self.rt.config.provider
        if provider.kind != "auto":
            return provider.kind
        actual = current_choice(provider, self.rt.session.model, ENVIRON)
        return f"auto → {actual}" if actual is not None and is_choice(actual) else "auto"

    def _role_models(self) -> str:
        """切り替えても替わらないモデルの設定の名前。"""
        cfg = self.rt.config
        items = ["provider.fallback_model"] if cfg.provider.fallback_model else []
        items += [f"provider.roles.{r}" for r in ROLE_NAMES if getattr(cfg.provider.roles, r, None)]
        if cfg.subagent.default_model:
            items.append("subagent.default_model")
        return ", ".join(items)

    async def _cmd_session(self, args: list[str]) -> None:
        from satsumaestro.cli.app import _session_list

        sid = self.rt.session.session_id
        self._print(ui("cli.commands.current_session", sid=sid))
        _session_list(self.rt.cwd)

    async def _cmd_doctor(self, args: list[str]) -> None:
        from satsumaestro.cli.app import _doctor

        _doctor(self.rt.cwd, probe=False, trust=self.trusted)

    async def _cmd_mode(self, args: list[str]) -> None:
        """``/mode`` = 現在値と一覧、``/mode plan``、``/mode code [<sub>]``。"""
        session = self.rt.session
        if not args:
            self._print(f"mode: {mode_markup(session.mode, full=True)}")
            subs = " | ".join(_SUB_VALUES)
            hint = ui("cli.commands.main_plan_code_shift_tab", subs=subs)
            self._print(f"[dim]{hint}[/dim]")
            return
        main = args[0].lower()
        if main == "plan":
            if len(args) > 1:
                no_sub = ui("cli.commands.plan_takes_no_sub_mode")
                self._print(f"[yellow]{no_sub}[/yellow]")
                return
            self.rt.set_mode(main="plan")
            mode, plan_file = mode_markup(session.mode), self.rt.plan_file_relpath()
            entered = ui("cli.commands.mode_writes_other_than_the", mode=mode, plan_file=plan_file)
            self._print(f"[dim]{entered}[/dim]")
        elif main == "code":
            sub: PermissionMode | None = None
            if len(args) > 1:
                if args[1].lower() not in _SUB_VALUES or len(args) > 2:
                    self._print(_mode_hint())
                    return
                sub = PermissionMode(args[1].lower())
            self.rt.set_mode(main="code", sub=sub)
            self._print(f"[dim]mode → {mode_markup(session.mode, full=True)}[/dim]")
        else:
            self._print(_mode_hint())
            return
        self.rt.store.save(session)

    async def _cmd_sub_mode(self, args: list[str]) -> None:
        """接頭辞で省略した権限サブモードの確認・変更。"""
        session = self.rt.session
        if not args:
            when = (
                ui("cli.commands.in_plan_the_permissions_when") if session.mode.plan_active else ""
            )
            self._print(f"sub-mode: {session.mode.sub.value}{when}")
            subs = " | ".join(_SUB_VALUES)
            hint = ui("cli.commands.change_sub_mode_sub_the", subs=subs)
            self._print(f"[dim]{hint}[/dim]")
            return
        if session.mode.plan_active:
            locked = ui("cli.commands.the_sub_mode_cannot_be")
            self._print(f"[yellow]{locked}[/yellow]")
            return
        await self._cmd_mode(["code", *args])

    async def _cmd_plan(self, args: list[str]) -> None:
        active = self.rt.session.mode.main == "plan"
        want = args[0].lower() if args else ("off" if active else "on")
        if want == "on":
            await self._cmd_mode(["plan"])
        elif want == "off":
            await self._cmd_mode(["code"])
        else:
            self._print("[yellow]" + ui("cli.commands.use_plan_on_off") + "[/yellow]")

    async def _cmd_clear(self, args: list[str]) -> None:
        session = self.rt.session
        session.messages.clear()
        session.prompt_history.clear()
        session.todos.clear()
        session.plan_steps = []
        session.plan_step_index = 0
        session.plan_step_estimates = []
        session.plan_budget_lines = 0
        self.rt.set_mode(main="code")
        self.rt.store.save(session)
        cleared = ui("cli.commands.cleared_the_conversation_todos_and")
        self._print(f"[dim]{cleared}[/dim]")

    async def _cmd_insights(self, args: list[str]) -> None:
        if self.trace_log is None:
            off = ui("cli.commands.telemetry_is_off_enable_it")
            self._print(f"[yellow]{off}[/yellow]")
            return
        ins = insights_from(self.trace_log.entries)
        avg = f"{ins.iterations_avg:.1f}"
        self._print(
            ui(
                "cli.commands.turns_failed_recovered_bash_auto",
                turns=ins.turns,
                turns_failed=ins.turns_failed,
                recoveries=ins.recoveries,
                auto_allowed=ins.auto_allowed,
                avg=avg,
            )
        )
        if ins.tools:
            table = Table(show_header=True, header_style="bold")
            table.add_column(ui("cli.commands.tool"))
            table.add_column(ui("cli.commands.runs"), justify="right")
            table.add_column(ui("cli.commands.errors"), justify="right")
            for name, stat in sorted(ins.tools.items(), key=lambda kv: -kv[1].calls):
                table.add_row(name, str(stat.calls), str(stat.errors))
            self._print(table)
