"""入力欄で、スラッシュコマンドとその引数の候補を出す。"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Protocol

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from satsumaestro.cli.commands import SLASH_COMMANDS, argument_choices, command_named
from satsumaestro.engine.context_fit import list_models
from satsumaestro.providers.choice import current_choice
from satsumaestro.providers.registry import MODEL_SHORTHANDS
from satsumaestro.util.errors import SatsumaError

if TYPE_CHECKING:
    from satsumaestro.engine.runtime import ConversationRuntime

Choice = tuple[str, str]


class ModelSource(Protocol):
    """``/model`` の候補の出どころ。"""

    def prefetch(self) -> None: ...

    async def names(self) -> Sequence[Choice]: ...


class ModelNames:
    """今の接続先のモデルの名前（裏で取り、接続先かモデルが変わるまで持つ）。"""

    def __init__(
        self, rt: ConversationRuntime, *, timeout_s: float = 2.0, retry_s: float = 30.0
    ) -> None:
        self._rt = rt
        self._timeout_s = timeout_s
        self._retry_s = retry_s
        self._names: tuple[str, ...] = ()
        self._fetched_for: tuple[str, str, str] | None = None
        self._fetched_at = 0.0
        self._task: asyncio.Task[None] | None = None

    def _target(self) -> tuple[str, str, str]:
        provider = self._rt.config.provider
        return provider.kind, provider.base_url or "", self._rt.session.model

    def prefetch(self) -> None:
        """一覧が無いか古ければ、裏で取り始める。"""
        if self._task is not None and not self._task.done():
            return
        target = self._target()
        if target == self._fetched_for and (
            self._names or time.monotonic() - self._fetched_at < self._retry_s
        ):
            return
        self._task = asyncio.get_running_loop().create_task(self._fetch(target))

    async def names(self) -> list[Choice]:
        """候補の名前と説明（取っている途中なら、終わるのを待つ）。"""
        self.prefetch()
        if self._task is not None and not self._task.done():
            await asyncio.shield(self._task)
        found = self._names if self._fetched_for == self._target() else ()
        choices = [(name, "") for name in found]
        provider, model = self._rt.config.provider, self._rt.session.model
        if current_choice(provider, model, os.environ) == "anthropic":
            choices += [(short, f"→ {full}") for short, full in sorted(MODEL_SHORTHANDS.items())]
        return choices

    async def _fetch(self, target: tuple[str, str, str]) -> None:
        try:
            names = await list_models(
                self._rt.config, target[2], os.environ, timeout_s=self._timeout_s
            )
        except SatsumaError:
            names = None
        self._names, self._fetched_for = tuple(names or ()), target
        self._fetched_at = time.monotonic()


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


def _wants_models(text: str) -> bool:
    split = _split(text)
    if split is None:
        return False
    words, _ = split
    command = command_named(words[0]) if len(words) == 1 else None
    return command is not None and command.name == "model"


def slash_completions(text: str, models: Sequence[Choice] = ()) -> Iterator[Completion]:
    """カーソルの前の ``text`` に出す候補（``models`` は ``/model`` の候補）。"""
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
    if named.name == "model":
        if given:
            return
        first = [c for c in models if c[0].lower().startswith(typed)]
        rest = [c for c in models if typed in c[0].lower() and c not in first]
        choices: list[Choice] = [*first, *rest]
    else:
        choices = [c for c in argument_choices(named.name, given) if c[0].startswith(typed)]
    for value, meta in _exact_first(choices, typed):
        more = bool(argument_choices(named.name, [*given, value]))
        yield Completion(
            value + (" " if more else ""), -len(current), display=value, display_meta=meta
        )


def _exact_first[T](items: Iterable[tuple[str, T]], typed: str) -> list[tuple[str, T]]:
    """打った語と同じ候補を先頭に（一覧は先頭を選んだ状態で出る）。"""
    return sorted(items, key=lambda item: item[0].lower() != typed)


class SlashCompleter(Completer):
    """行頭の ``/`` で、コマンドとその引数の候補を出す。"""

    def __init__(self, models: ModelSource | None = None) -> None:
        self._models = models

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        return slash_completions(document.text_before_cursor)

    async def get_completions_async(
        self, document: Document, complete_event: CompleteEvent
    ) -> AsyncGenerator[Completion, None]:
        text = document.text_before_cursor
        models: Sequence[Choice] = ()
        if self._models is not None and _wants_models(text):
            models = await self._models.names()
        elif self._models is not None and _split(text) is not None:
            self._models.prefetch()
        for completion in slash_completions(text, models):
            yield completion
