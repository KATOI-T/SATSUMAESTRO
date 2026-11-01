"""画面の言語の設定（``language.ui``）と、設定が無いときの起動時の問い・``/language``。"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.console import Console

from satsumaestro.cli import app
from satsumaestro.cli import language_gate as gate
from satsumaestro.cli.commands import ParsedCommand, SlashDispatcher, parse_slash
from satsumaestro.config.loader import ConfigLoader, save_screen_language
from satsumaestro.config.models import HookConfig, LanguageConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.state import LanguageState
from satsumaestro.i18n.text import (
    pinned_ui_language,
    ui,
    ui_language,
    use_language,
    use_screen_language,
)
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.security.trust import TrustStore, trust_root
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ConfigError
from satsumaestro.util.paths import default_config_home
from tests.fixtures.mock_provider import SequencedMockProvider

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def _unpinned() -> Iterator[None]:
    """設定と環境の言語で画面の言語が決まるところを見る（テストの日本語の固定を外す）。"""
    with pinned_ui_language(None):
        yield


def _user_settings() -> Path:
    return default_config_home() / "settings.json"


def _write_user_settings(data: object) -> None:
    path = _user_settings()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_the_setting_accepts_auto_and_screen_languages() -> None:
    assert LanguageConfig().ui is None and LanguageConfig().chosen_ui is None
    assert LanguageConfig(ui="auto").chosen_ui is None
    assert LanguageConfig(ui="ja").chosen_ui == "ja"
    assert LanguageConfig(ui="en").chosen_ui == "en"
    with pinned_ui_language("en"), pytest.raises(ValidationError) as caught:
        LanguageConfig(ui="zz")
    assert "Unknown screen language: 'zz' (available: auto, ja, en)" in str(caught.value)


def test_the_chosen_language_does_not_follow_the_conversation() -> None:
    state = LanguageState(LanguageConfig(ui="en"), startup="ja")
    assert state.ui_language == "en"
    state.begin_turn("with_tax はどこで呼ばれていますか")
    assert state.current == "ja"
    assert state.ui_language == "en"
    assert state.for_child().ui_language == "en"


def test_auto_follows_the_conversation() -> None:
    state = LanguageState(LanguageConfig(ui="auto"), startup="en")
    assert state.ui_language == "en"
    state.begin_turn("with_tax はどこで呼ばれていますか")
    assert state.ui_language == "ja"


def test_rewriting_the_setting_takes_effect_next_turn() -> None:
    source = [LanguageConfig(ui="ja")]
    state = LanguageState(LanguageConfig(), startup="en")
    state.watch(lambda: source[0])
    state.begin_turn("Where is with_tax called from in this project?")
    assert state.ui_language == "ja"
    source[0] = LanguageConfig(ui="auto")
    state.begin_turn("Where is with_tax called from in this project?")
    assert state.ui_language == "en"


@pytest.mark.usefixtures("_unpinned")
def test_without_an_owner_the_chosen_language_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    assert ui_language() == "en"
    use_screen_language("ja")
    assert ui("cli.render.done") == "✓ 完了"
    use_language(LanguageState(LanguageConfig(ui="en"), startup="ja"))
    assert ui("cli.render.done") == "✓ Done"


def test_saving_keeps_the_other_settings(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text(
        json.dumps({"max_iterations": 9, "language": {"mode": "fixed", "fixed": "en"}}),
        encoding="utf-8",
    )
    path = save_screen_language("ja", tmp_path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "max_iterations": 9,
        "language": {"mode": "fixed", "fixed": "en", "ui": "ja"},
    }


def test_saving_creates_the_file(tmp_path: Path) -> None:
    path = save_screen_language("en", tmp_path / "home")
    assert json.loads(path.read_text(encoding="utf-8")) == {"language": {"ui": "en"}}


@pytest.mark.parametrize("text", ["{oops", '{"language": "ja"}'])
def test_saving_does_not_overwrite_unreadable_settings(tmp_path: Path, text: str) -> None:
    (tmp_path / "settings.json").write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        save_screen_language("ja", tmp_path)
    assert (tmp_path / "settings.json").read_text(encoding="utf-8") == text


def test_the_question_lists_every_screen_language() -> None:
    lines = gate.question_lines()
    assert [line.plain for line in lines] == [
        "画面の言語を選んでください / Choose the screen language",
        "  1) 日本語",
        "  2) English",
    ]
    assert lines[0].style == gate.HEADING_STYLE
    for line, number in zip(lines[1:], ("1)", "2)"), strict=True):
        (span,) = line.spans
        assert line.plain[span.start : span.end] == number
        assert span.style == gate.NUMBER_STYLE
    assert gate.question("ja") == "(1/2) [1]: "
    assert gate.question("en") == "(1/2) [2]: "


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        ("", "ja"),
        ("1", "ja"),
        ("2", "en"),
        (" en ", "en"),
        ("English", "en"),
        ("日本語", "ja"),
        ("AUTO", "auto"),
        ("3", None),
        ("fr", None),
    ],
)
def test_answers(answer: str, code: str | None) -> None:
    assert gate.parse_answer(answer, "ja") == code


def _choose(
    monkeypatch: pytest.MonkeyPatch,
    cwd: Path,
    answers: Iterable[str],
    *,
    terminal: bool = True,
    trust: bool = False,
) -> str:
    """起動時の問いを ``answers`` で答えて流し、画面に出た文を返す（尽きたら EOF）。"""
    out = io.StringIO()
    console = Console(file=out, width=300)
    replies = iter(answers)

    def answer(prompt: str = "", **_: object) -> str:
        console.print(prompt, end="", markup=False)
        try:
            return next(replies)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(console, "input", answer)
    monkeypatch.setattr(app, "_console", console)
    monkeypatch.setattr(app, "_stdin_is_terminal", lambda: terminal)
    app._choose_screen_language(cwd, trust=trust)
    return out.getvalue()


@pytest.mark.usefixtures("_unpinned")
def test_the_first_start_asks_and_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LC_MESSAGES", "ja_JP.UTF-8")
    printed = _choose(monkeypatch, tmp_path, ["9", "2"])
    assert printed.startswith("画面の言語を選んでください / Choose the screen language\n")
    assert printed.count("(1/2) [1]: ") == 2
    assert "Screen language: English (saved as language.ui in " in printed
    assert json.loads(_user_settings().read_text(encoding="utf-8")) == {"language": {"ui": "en"}}
    assert ui_language() == "en"
    assert _choose(monkeypatch, tmp_path, []) == ""
    assert ui_language() == "en"


@pytest.mark.usefixtures("_unpinned")
def test_answering_auto_follows_the_conversation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LC_MESSAGES", "ja_JP.UTF-8")
    printed = _choose(monkeypatch, tmp_path, ["auto"])
    assert "画面の言語: auto（" in printed
    assert json.loads(_user_settings().read_text(encoding="utf-8")) == {"language": {"ui": "auto"}}
    assert ui_language() == "ja"


@pytest.mark.usefixtures("_unpinned")
def test_no_question_without_a_terminal_or_on_eof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _choose(monkeypatch, tmp_path, ["2"], terminal=False) == ""
    printed = _choose(monkeypatch, tmp_path, [])
    assert "Choose the screen language" in printed
    assert not _user_settings().exists()


@pytest.mark.usefixtures("_unpinned")
def test_a_broken_user_settings_file_is_left_to_the_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _user_settings().parent.mkdir(parents=True, exist_ok=True)
    _user_settings().write_text("{oops", encoding="utf-8")
    assert _choose(monkeypatch, tmp_path, ["2"]) == ""
    assert _user_settings().read_text(encoding="utf-8") == "{oops"


@pytest.mark.usefixtures("_unpinned")
def test_the_project_setting_counts_only_when_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "repo"
    (project / ".satsuma").mkdir(parents=True)
    (project / ".satsuma" / "settings.local.json").write_text(
        json.dumps({"language": {"ui": "en"}}), encoding="utf-8"
    )
    assert "Choose the screen language" in _choose(monkeypatch, project, [])
    assert _choose(monkeypatch, project, [], trust=True) == ""
    assert ui_language() == "en"
    TrustStore().trust(trust_root(project))
    assert _choose(monkeypatch, project, []) == ""


def test_doctor_shows_the_screen_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)

    def row(value: str | None) -> tuple[str, str]:
        return app._screen_language_row(RuntimeConfig(language=LanguageConfig(ui=value)))

    assert row("ja") == ("ok", "日本語（設定の language.ui）")
    assert row("auto") == ("ok", "auto: 会話の言語に合わせる（決まるまでは環境の言語の English）")
    assert row(None) == (
        "-",
        "未設定: 端末の REPL が起動時に一度だけ聞く（それまでは環境の言語の English）",
    )


def test_help_follows_the_user_setting() -> None:
    _write_user_settings({"language": {"ui": "ja"}})
    env = {k: v for k, v in os.environ.items() if k not in ("LC_ALL", "LC_MESSAGES")}
    env.update(LANG="C.UTF-8", COLUMNS="200", SATSUMA_LOG_FILE="off")
    done = subprocess.run(
        [sys.executable, "-m", "satsumaestro.cli.app", "--help"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "エージェント型コーディング CLI" in done.stdout


def _dispatcher(
    tmp_path: Path, *, trusted: bool = False, cwd: Path | None = None
) -> tuple[SlashDispatcher, Console, LanguageState]:
    """REPL と同じく、設定ファイルの language をターンの頭で読み直す持ち主で組む。"""
    workspace = cwd if cwd is not None else tmp_path
    loader = ConfigLoader(workspace, trusted=trusted)
    language = LanguageState.from_config(loader.load_language(), {"LANG": "C.UTF-8"})
    language.watch(loader.load_language)
    use_language(language)
    rt = ConversationRuntime(
        session=new_session(workspace, "mock-model"),
        provider=SequencedMockProvider([[MessageStop()]]),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), workspace),
        config=RuntimeConfig(),
        language=language,
    )
    console = Console(record=True, width=200)
    return SlashDispatcher(rt, console), console, language


async def _run(d: SlashDispatcher, console: Console, line: str) -> str:
    command = parse_slash(line)
    assert command is not None
    await d.dispatch(command)
    return console.export_text(clear=True)


@pytest.mark.usefixtures("_unpinned")
async def test_language_command_shows_the_current_language(tmp_path: Path) -> None:
    _write_user_settings({"language": {"ui": "ja"}})
    d, console, _ = _dispatcher(tmp_path)
    printed = await _run(d, console, "/language")
    assert printed.splitlines() == [
        "画面の言語: 日本語（設定の language.ui）",
        "  1) 日本語 ✓",
        "  2) English",
        "変更: /language ja|en|auto（上の番号でも選べます）",
    ]


@pytest.mark.usefixtures("_unpinned")
async def test_language_command_switches_and_records(tmp_path: Path) -> None:
    _write_user_settings({"language": {"ui": "ja"}, "max_iterations": 9})
    d, console, language = _dispatcher(tmp_path)
    printed = await _run(d, console, "/language en")
    assert printed.startswith("Screen language → English (saved as language.ui in ")
    assert ui_language() == "en"
    assert json.loads(_user_settings().read_text(encoding="utf-8")) == {
        "language": {"ui": "en"},
        "max_iterations": 9,
    }
    language.begin_turn("with_tax はどこで呼ばれていますか")
    assert (language.current, ui_language()) == ("ja", "en")
    await _run(d, console, "/language 1")
    assert ui_language() == "ja"
    await _run(d, console, "/language auto")
    assert ui_language() == "ja"
    language.begin_turn("Where is with_tax called from in this project?")
    assert ui_language() == "en"
    assert json.loads(_user_settings().read_text(encoding="utf-8"))["language"] == {"ui": "auto"}


@pytest.mark.usefixtures("_unpinned")
async def test_language_command_keeps_the_fixed_reply_language(tmp_path: Path) -> None:
    d, console, language = _dispatcher(tmp_path)
    await _run(d, console, "/lang en")
    await _run(d, console, "/language ja")
    language.begin_turn("with_tax はどこで呼ばれていますか")
    assert (language.mode, language.current, ui_language()) == ("fixed", "en", "ja")


@pytest.mark.usefixtures("_unpinned")
async def test_language_command_rejects_unknown_values(tmp_path: Path) -> None:
    _write_user_settings({"language": {"ui": "ja"}})
    d, console, _ = _dispatcher(tmp_path)
    printed = await _run(d, console, "/language fr")
    assert printed.strip() == "画面の言語は ja|en|auto か番号（1/2）で選びます"
    assert ui_language() == "ja"
    assert json.loads(_user_settings().read_text(encoding="utf-8")) == {"language": {"ui": "ja"}}


@pytest.mark.usefixtures("_unpinned")
async def test_language_command_under_a_project_setting(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    (project / ".satsuma").mkdir(parents=True)
    (project / ".satsuma" / "settings.local.json").write_text(
        json.dumps({"language": {"ui": "en"}}), encoding="utf-8"
    )
    d, console, language = _dispatcher(tmp_path, trusted=True, cwd=project)
    printed = await _run(d, console, "/language ja")
    assert "このプロジェクトの設定の language.ui（en）が優先されるので、次の起動では en" in printed
    language.begin_turn("with_tax はどこで呼ばれていますか")
    assert ui_language() == "ja"


@pytest.mark.usefixtures("_unpinned")
async def test_language_command_when_the_settings_cannot_be_written(tmp_path: Path) -> None:
    d, console, _ = _dispatcher(tmp_path)
    _user_settings().parent.mkdir(parents=True, exist_ok=True)
    _user_settings().write_text('{"language": "ja"}', encoding="utf-8")
    printed = await _run(d, console, "/language en")
    assert printed.startswith("Screen language → English (could not save it: ")
    assert ui_language() == "en"
    assert _user_settings().read_text(encoding="utf-8") == '{"language": "ja"}'


async def test_help_lists_the_language_command(tmp_path: Path) -> None:
    assert parse_slash("/language") == ParsedCommand(name="language", args=[], raw="language")
    d, console, _ = _dispatcher(tmp_path)
    printed = await _run(d, console, "/help")
    assert "/language [ja|en|auto]" in printed
    assert "画面の言語を表示 / 切り替える（会話の言語に合わせるなら auto。応答の言語は" in printed
