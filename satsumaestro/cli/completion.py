"""入力欄で、スラッシュコマンドとその引数の候補を出す。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import (
    AsyncGenerator,
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING, Protocol

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from satsumaestro.cli.commands import (
    SLASH_COMMANDS,
    argument_choices,
    command_named,
    local_provider_choice,
)
from satsumaestro.engine.context_fit import list_models
from satsumaestro.engine.provider_switch import models_for
from satsumaestro.providers.choice import current_choice
from satsumaestro.providers.registry import MODEL_SHORTHANDS
from satsumaestro.util.environ import ENVIRON
from satsumaestro.util.errors import SatsumaError

if TYPE_CHECKING:
    from satsumaestro.engine.runtime import ConversationRuntime

Choice = tuple[str, str]
_REJECTED = "fg:ansired"


class ModelSource(Protocol):
    """モデルの候補の出どころ（``choice`` は ``/provider`` の切り替え先・空なら今の接続先）。"""

    def prefetch(self, choice: str = "") -> None: ...

    async def names(self, choice: str = "") -> Sequence[Choice]: ...


_Target = tuple[str, str, str, str]


class ModelNames:
    """今の接続先と切り替え先のモデルの名前（裏で取り、接続先かモデルが変わるまで持つ）。"""

    def __init__(
        self, rt: ConversationRuntime, *, timeout_s: float = 2.0, retry_s: float = 30.0
    ) -> None:
        self._rt = rt
        self._timeout_s = timeout_s
        self._retry_s = retry_s
        self._found: dict[_Target, tuple[tuple[str, ...], float]] = {}
        self._tasks: dict[_Target, asyncio.Task[None]] = {}

    def _target(self, choice: str) -> _Target:
        provider = self._rt.config.provider
        return choice, provider.kind, provider.base_url or "", self._rt.session.model

    def prefetch(self, choice: str = "") -> None:
        """一覧が無いか古ければ、裏で取り始める。"""
        target = self._target(choice)
        task = self._tasks.get(target)
        if task is not None and not task.done():
            return
        found = self._found.get(target)
        if found is not None and (found[0] or time.monotonic() - found[1] < self._retry_s):
            return
        self._tasks[target] = asyncio.get_running_loop().create_task(self._fetch(target))

    async def names(self, choice: str = "") -> list[Choice]:
        """候補の名前と説明（取っている途中なら、終わるのを待つ）。"""
        self.prefetch(choice)
        target = self._target(choice)
        task = self._tasks.get(target)
        if task is not None and not task.done():
            await asyncio.shield(task)
        found = self._found.get(target, ((), 0.0))[0]
        choices = [(name, "") for name in found]
        provider, model = self._rt.config.provider, self._rt.session.model
        if not choice and current_choice(provider, model, ENVIRON) == "anthropic":
            choices += [(short, f"→ {full}") for short, full in sorted(MODEL_SHORTHANDS.items())]
        return choices

    async def _fetch(self, target: _Target) -> None:
        choice, model = target[0], target[3]
        try:
            if choice:
                names = await models_for(self._rt, choice, ENVIRON, timeout_s=self._timeout_s)
            else:
                names = await list_models(
                    self._rt.config, model, ENVIRON, timeout_s=self._timeout_s
                )
        except SatsumaError:
            names = None
        self._found[target] = (tuple(names or ()), time.monotonic())


def _split(text: str) -> tuple[list[str], str] | None:
    """``/`` の後を、打ち終えた語と打ちかけの語に分ける（候補を出さない行は ``None``）。"""
    if not text.startswith("/") or "\n" in text:
        return None
    body = text[1:]
    words = body.split()
    current = "" if not body or body[-1].isspace() else words.pop()
    head = words[0] if words else current
    if "/" in head or not head.isascii():
        return None
    return words, current


def _first_argument_of(text: str) -> str | None:
    """最初の引数を打っているコマンドの名前。"""
    split = _split(text)
    if split is None:
        return None
    words, _ = split
    command = command_named(words[0]) if len(words) == 1 else None
    return command.name if command is not None else None


def _switch_target(text: str) -> str | None:
    """``/provider <ローカルの種類> `` の後のモデルを打っているなら、その種類。"""
    split = _split(text)
    if split is None:
        return None
    words, _ = split
    named = command_named(words[0]) if words else None
    return local_provider_choice(named.name, words[1:]) if named is not None else None


def slash_completions(
    text: str, models: Sequence[Choice] = (), *, providers: Mapping[str, str] | None = None
) -> Iterator[Completion]:
    """カーソルの前の ``text`` に出す候補（``models`` は ``/model`` と ``/provider`` のモデル）。"""
    split = _split(text)
    if split is None:
        return
    words, current = split
    typed = current.lower()
    if not words:
        names = [
            (name, command)
            for command in SLASH_COMMANDS
            for name in (command.name, *command.also)
            if name.startswith(typed) and (typed or name == command.name)
        ]
        for name, command in _exact_first(names, typed):
            yield Completion(
                name + " ", -len(current), display="/" + name, display_meta=command.about()
            )
        return
    named = command_named(words[0])
    if named is None:
        return
    given = words[1:]
    if named.name == "model" and given:
        return
    if named.name == "model" or local_provider_choice(named.name, given):
        first = [c for c in models if c[0].lower().startswith(typed)]
        rest = [c for c in models if typed in c[0].lower() and c not in first]
        choices: list[Choice] = [*first, *rest]
    else:
        found = argument_choices(named.name, given, providers=providers)
        choices = [c for c in found if c[0].startswith(typed)]
    for value, meta in _exact_first(choices, typed):
        later = [*given, value]
        more = bool(argument_choices(named.name, later) or local_provider_choice(named.name, later))
        yield Completion(
            value + (" " if more else ""), -len(current), display=value, display_meta=meta
        )


def _exact_first[T](items: Iterable[tuple[str, T]], typed: str) -> list[tuple[str, T]]:
    """打った語と同じ候補を先頭に（一覧は先頭を選んだ状態で出る）。"""
    return sorted(items, key=lambda item: item[0].lower() != typed)


class SlashCompleter(Completer):
    """行頭の ``/`` で、コマンドとその引数の候補を出す。"""

    def __init__(
        self,
        models: ModelSource | None = None,
        *,
        providers: Callable[[], Mapping[str, str]] | None = None,
        rejected: Callable[[], Collection[str]] | None = None,
    ) -> None:
        self._models = models
        self._providers = providers
        self._rejected = rejected

    def _provider_notes(self, text: str) -> Mapping[str, str] | None:
        if self._providers is None or _first_argument_of(text) != "provider":
            return None
        return self._providers()

    def _marked(self, text: str, completions: Iterable[Completion]) -> Iterator[Completion]:
        """断られた鍵のプロバイダの候補を赤に。"""
        if self._rejected is None or _first_argument_of(text) != "provider":
            yield from completions
            return
        rejected = self._rejected()
        for c in completions:
            if c.text.strip() in rejected:
                c = Completion(c.text, c.start_position, c.display, c.display_meta, style=_REJECTED)
            yield c

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        return self._marked(text, slash_completions(text, providers=self._provider_notes(text)))

    async def get_completions_async(
        self, document: Document, complete_event: CompleteEvent
    ) -> AsyncGenerator[Completion, None]:
        text = document.text_before_cursor
        models: Sequence[Choice] = ()
        target = _switch_target(text)
        if self._models is not None and _first_argument_of(text) == "model":
            models = await self._models.names()
        elif self._models is not None and target is not None:
            models = await self._models.names(target)
        elif self._models is not None and _split(text) is not None:
            self._models.prefetch()
        found = slash_completions(text, models, providers=self._provider_notes(text))
        for completion in self._marked(text, found):
            yield completion
