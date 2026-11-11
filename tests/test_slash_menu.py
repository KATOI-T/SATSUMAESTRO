"""入力欄の下の候補の一覧と、そのキー。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples, fragment_list_to_text
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import FloatContainer
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.output import DummyOutput

from satsumaestro.cli import completion as completion_mod
from satsumaestro.cli.completion import slash_completions
from satsumaestro.cli.completion_menu import SuggestionMenu, install_menu
from satsumaestro.cli.repl import _prompt_session
from satsumaestro.config.models import ProviderConfig, RuntimeConfig


def _menu_for(text: str, *, rows: int = 8) -> SuggestionMenu:
    buffer = Buffer()
    buffer.document = Document(text)
    menu = SuggestionMenu(buffer, rows=rows)
    completions = list(slash_completions(text, [(f"model-{n}", "") for n in range(12)]))
    buffer.complete_state = CompletionState(Document(text), completions)
    return menu


def _row(line: StyleAndTextTuples) -> str:
    return fragment_list_to_text(line)


def _highlighted(lines: list[StyleAndTextTuples]) -> list[str]:
    return [_row(line).strip() for line in lines if any("cyan" in s for s, _ in line)]


def test_rows_line_up_and_the_first_is_chosen() -> None:
    lines = _menu_for("/mo").lines(100)
    assert [_row(line).split()[0] for line in lines] == ["/model", "/mode"]
    assert _row(lines[0]).index("現在") == _row(lines[1]).index("メイン")
    assert _row(lines[0]).startswith("  /model")
    assert _highlighted(lines) == [_row(lines[0]).strip()]


def test_descriptions_are_cut_to_the_width() -> None:
    lines = _menu_for("/la").lines(40)
    for line in lines:
        text = _row(line)
        assert text.endswith("…")
        assert sum(2 if ord(c) > 0x2E7F else 1 for c in text) <= 39


def test_arguments_start_with_nothing_chosen() -> None:
    menu = _menu_for("/model ")
    assert menu.chosen() is None and _highlighted(menu.lines(80)) == []
    menu.move(-1)
    chosen = menu.chosen()
    assert chosen is not None and chosen.text == "model-11"
    assert _highlighted(menu.lines(80)) == ["model-11"]


def test_the_list_scrolls_to_keep_the_choice_in_view() -> None:
    menu = _menu_for("/", rows=8)
    for _ in range(9):
        menu.move(1)
    lines = menu.lines(100)
    assert len(lines) == 8
    assert _highlighted(lines)[0].startswith("/doctor")
    assert _row(lines[-1]).split()[0] == "/doctor"
    menu.move(-9)
    assert _row(menu.lines(100)[0]).split()[0] == "/help"


def test_fill_puts_the_choice_in_the_line() -> None:
    menu = _menu_for("/mo")
    menu.move(1)
    menu.fill(then_suggest=False)
    assert (menu.buffer.text, menu.buffer.cursor_position) == ("/mode ", 6)
    assert menu.buffer.complete_state is None
    menu = _menu_for("/model gp")
    menu.fill(then_suggest=False)
    assert menu.buffer.text == "/model gp"


def test_install_swaps_the_default_box() -> None:
    with create_pipe_input() as pipe:
        session: PromptSession[str] = PromptSession(input=pipe, output=DummyOutput())
    assert install_menu(session) is not None
    floats = [f for c in session.layout.walk() if isinstance(c, FloatContainer) for f in c.floats]
    assert not any(isinstance(f.content, CompletionsMenu) for f in floats)
    assert install_menu(SimpleNamespace()) is None  # type: ignore[arg-type]


def _runtime() -> Any:
    config = RuntimeConfig(provider=ProviderConfig(kind="ollama"))
    return SimpleNamespace(config=config, session=SimpleNamespace(model="gpt-oss:120b"))


@pytest.fixture
def models(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(cfg: Any, model: str, env: Any, **kwargs: Any) -> list[str]:
        return ["gpt-oss:120b", "gpt-oss:20b"]

    monkeypatch.setattr(completion_mod, "list_models", fake)


async def _until(check: Callable[[], object]) -> None:
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


def _shown(buffer: Buffer, text: str) -> list[str] | None:
    state = buffer.complete_state
    if buffer.text != text or state is None:
        return None
    return [c.display_text for c in state.completions]


class _Prompt:
    """パイプの入力で動かす本物の入力欄。"""

    def __init__(self, bindings: KeyBindings | None = None, **options: Any) -> None:
        self._pipe_cm = create_pipe_input()
        self.pipe = self._pipe_cm.__enter__()
        self.session = _prompt_session(
            _runtime(), bindings or KeyBindings(), input=self.pipe, output=DummyOutput(), **options
        )
        self.buffer = self.session.default_buffer
        self.answer = asyncio.create_task(self.session.prompt_async(set_exception_handler=False))

    async def type(self, text: str, *, then: Callable[[], object] | None = None) -> None:
        self.pipe.send_text(text)
        if then is not None:
            await _until(then)

    async def result(self, keys: str) -> str:
        self.pipe.send_text(keys)
        try:
            return await asyncio.wait_for(self.answer, 5)
        finally:
            self._pipe_cm.__exit__(None, None, None)


async def test_enter_runs_the_chosen_command(models: None) -> None:
    prompt = _Prompt()
    await prompt.type("/mo", then=lambda: _shown(prompt.buffer, "/mo") == ["/model", "/mode"])
    assert await prompt.result("\r") == "/model "


async def test_down_tab_and_enter_choose_a_model(models: None) -> None:
    prompt = _Prompt()
    buffer = prompt.buffer
    await prompt.type("/mo", then=lambda: _shown(buffer, "/mo"))
    await prompt.type("\x1b[B\t", then=lambda: _shown(buffer, "/mode ") == ["plan", "code"])
    await prompt.type("\x15/mo", then=lambda: _shown(buffer, "/mo"))
    both = ["gpt-oss:120b", "gpt-oss:20b"]
    await prompt.type("\t", then=lambda: _shown(buffer, "/model ") == both)
    assert await prompt.result("\x1b[B\x1b[B\r") == "/model gpt-oss:20b"


async def test_enter_after_tab_runs_the_command_itself(models: None) -> None:
    prompt = _Prompt()
    await prompt.type("/pro", then=lambda: _shown(prompt.buffer, "/pro"))
    await prompt.type("\t", then=lambda: _shown(prompt.buffer, "/provider "))
    assert await prompt.result("\r") == "/provider "


async def test_escape_closes_and_backspace_reopens(models: None) -> None:
    prompt = _Prompt()
    buffer = prompt.buffer
    await prompt.type("/mod", then=lambda: _shown(buffer, "/mod"))
    await prompt.type("\x1b", then=lambda: buffer.complete_state is None)
    await prompt.type("\x7f", then=lambda: _shown(buffer, "/mo") == ["/model", "/mode"])
    await prompt.type("\x1b", then=lambda: buffer.complete_state is None)
    assert await prompt.result("\r") == "/mo"


async def test_history_is_not_taken_over_by_the_list(models: None) -> None:
    history = InMemoryHistory()
    for line in ("older", "/model gpt-oss:20b"):
        history.append_string(line)
    prompt = _Prompt(history=history)
    await prompt.type("\x1b[A", then=lambda: prompt.buffer.text == "/model gpt-oss:20b")
    await asyncio.sleep(0.1)
    assert prompt.buffer.complete_state is None
    await prompt.type("\x1b[A", then=lambda: prompt.buffer.text == "older")
    assert await prompt.result("\r") == "older"


async def test_shift_tab_keeps_its_binding_and_plain_lines_get_no_list(models: None) -> None:
    bindings = KeyBindings()
    pressed: list[str] = []

    @bindings.add(Keys.BackTab)
    def _cycle(event: KeyPressEvent) -> None:
        pressed.append(event.current_buffer.text)

    prompt = _Prompt(bindings)
    buffer = prompt.buffer
    await prompt.type("/mo", then=lambda: _shown(buffer, "/mo"))
    await prompt.type("\x1b[Z", then=lambda: pressed)
    assert pressed == ["/mo"] and buffer.text == "/mo"
    await prompt.type("\x15hello mo", then=lambda: buffer.text == "hello mo")
    await asyncio.sleep(0.1)
    assert buffer.complete_state is None
    assert await prompt.result("\r") == "hello mo"
