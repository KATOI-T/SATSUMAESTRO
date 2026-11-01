"""言語の定義と、会話の言語の持ち主（``LanguageState``）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from satsumaestro.cli import app as cli_app
from satsumaestro.config.loader import ConfigLoader
from satsumaestro.config.models import (
    HookConfig,
    LanguageConfig,
    RuntimeConfig,
    SubagentConfig,
    ToolsConfig,
)
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import RuntimeFactory, resolve_subagent_type
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.schema import DEFAULT_PATH, catalog, load_catalog
from satsumaestro.i18n.state import LanguageState, locale_language
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ConfigError
from satsumaestro.util.lang import detect_language, language_name, requests_other_language
from tests.fixtures.mock_provider import MockProvider, SequencedMockProvider

runner = CliRunner()

_JA = "with_tax はどこで呼ばれていますか"
_EN = "Where is with_tax called from in this project?"


def test_the_packaged_definitions_hold_japanese_and_english() -> None:
    langs = catalog()
    assert langs.codes() == ("ja", "en")
    assert (langs.name("ja"), langs.name("en")) == ("Japanese", "English")
    assert langs.get("ja").native_name == "日本語"  # type: ignore[union-attr]
    assert all(spec.ui for spec in langs.languages)
    assert language_name("ja") == "Japanese"
    assert language_name("xx") == "xx"
    assert requests_other_language("英語で説明して")
    assert requests_other_language("Answer in Japanese please")
    assert requests_other_language("translate this into French")
    assert not requests_other_language("この関数を説明して")


_KOREAN = """
[[language]]
code = "ko"
name = "Korean"
native_name = "한국어"
locales = ["ko"]
request_patterns = ['한국어로', '\\bin\\s+korean\\b']

[language.detect]
kind = "script"
ranges = [[0xAC00, 0xD7AF], [0x1100, 0x11FF], [0x3130, 0x318F]]
min_count = 2
min_ratio = 0.10
"""


_HANGUL_RANGES = ("[0xAC00, 0xD7AF]", "[0x1100, 0x11FF]", "[0x3130, 0x318F]")


def _packaged_text(*, without_hangul: bool = False) -> str:
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    if without_hangul:
        lines = text.splitlines(keepends=True)
        text = "".join(ln for ln in lines if not any(r in ln for r in _HANGUL_RANGES))
    return text


def test_a_language_is_added_by_a_definition(tmp_path: Path) -> None:
    path = tmp_path / "languages.toml"
    path.write_text(_packaged_text(without_hangul=True) + _KOREAN, encoding="utf-8")
    langs = load_catalog(path)
    assert langs.codes() == ("ja", "en", "ko")
    assert detect_language("이 함수는 어디에서 호출됩니까", languages=langs) == "ko"
    assert detect_language("この関数はどこで呼ばれますか", languages=langs) == "ja"
    assert detect_language("이 함수는 어디에서 호출됩니까") == "und"
    assert langs.request_pattern().search("한국어로 답해 주세요")
    assert not langs.get("ko").ui  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (_KOREAN, "ranges overlap"),
        ('[[language]]\ncode = "ja"\nname = "J"\nnative_name = "J"\n', "duplicate"),
        (
            '[[language]]\ncode = "fr"\nname = "French"\nnative_name = "Français"\n'
            '[language.detect]\nkind = "latin"\n',
            "only one language",
        ),
        (
            '[[language]]\ncode = "zh"\nname = "Chinese"\nnative_name = "中文"\n'
            '[language.detect]\nkind = "script"\n',
            "needs ranges",
        ),
        (
            '[[language]]\ncode = "de"\nname = "German"\nnative_name = "Deutsch"\n'
            "request_patterns = ['(auf deutsch']\n",
            "regular expression",
        ),
    ],
)
def test_broken_definitions_are_refused(tmp_path: Path, extra: str, message: str) -> None:
    path = tmp_path / "languages.toml"
    path.write_text(_packaged_text() + extra, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_catalog(path)


def test_definitions_without_english_are_refused(tmp_path: Path) -> None:
    text = _packaged_text()
    path = tmp_path / "languages.toml"
    path.write_text(text[: text.index('[[language]]\ncode = "en"')], encoding="utf-8")
    with pytest.raises(ConfigError, match="definition of en"):
        load_catalog(path)


def test_fixed_defaults_to_japanese_and_takes_defined_codes() -> None:
    assert LanguageConfig().fixed == "ja"
    assert LanguageConfig(fixed="en").fixed == "en"
    with pytest.raises(ValidationError, match="定義に無い言語"):
        LanguageConfig(fixed="zh")


def test_the_loader_reads_only_the_language_settings(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.json").write_text('{"language": {"mode": "fixed", "fixed": "ja"}}')
    project = tmp_path / "proj"
    (project / ".satsuma").mkdir(parents=True)
    (project / ".satsuma" / "settings.local.json").write_text('{"language": {"fixed": "en"}}')
    assert ConfigLoader(project, home, trusted=True).load_language() == LanguageConfig(
        mode="fixed", fixed="en"
    )
    assert ConfigLoader(project, home, trusted=False).load_language().fixed == "ja"


@pytest.mark.parametrize(
    ("environ", "code"),
    [
        ({"LANG": "ja_JP.UTF-8"}, "ja"),
        ({"LANG": "en_US.UTF-8"}, "en"),
        ({"LANG": "fr_FR.UTF-8"}, "en"),
        ({"LANG": "C"}, "en"),
        ({}, "en"),
        ({"LC_ALL": "en_US.UTF-8", "LANG": "ja_JP.UTF-8"}, "en"),
        ({"LC_MESSAGES": "ja_JP", "LANG": "en_US"}, "ja"),
        ({"LC_ALL": "", "LANG": "ja"}, "ja"),
    ],
)
def test_the_environment_language(environ: dict[str, str], code: str) -> None:
    assert locale_language(environ) == code


def _state(**settings: object) -> LanguageState:
    return LanguageState(LanguageConfig(**settings))


def test_the_owner_follows_the_user_and_keeps_the_last_language() -> None:
    state = _state()
    assert state.current == ""
    assert state.begin_turn(_JA) == "ja"
    assert state.begin_turn("```\nprint(1)\n```") == "ja"
    assert state.begin_turn(_EN) == "en"


def test_fixed_and_off() -> None:
    assert _state(mode="fixed", fixed="ja").begin_turn(_EN) == "ja"
    off = _state(mode="off")
    assert off.begin_turn(_JA) == ""
    assert off.current == ""


def test_the_owner_writes_the_session_and_resumes_from_it(tmp_path: Path) -> None:
    session = new_session(tmp_path, "m")
    state = _state()
    state.bind(session)
    state.begin_turn(_JA)
    assert session.language == "ja"
    resumed = _state()
    resumed.bind(session)
    assert resumed.current == "ja"
    assert resumed.begin_turn("```\nprint(1)\n```") == "ja"


def test_set_fixed_and_set_auto(tmp_path: Path) -> None:
    session = new_session(tmp_path, "m")
    state = _state()
    state.bind(session)
    state.set_fixed("en")
    assert (state.mode, state.fixed, state.current, session.language) == (
        "fixed",
        "en",
        "en",
        "en",
    )
    assert state.begin_turn(_JA) == "en"
    state.set_auto()
    assert state.begin_turn(_JA) == "ja"
    with pytest.raises(ValueError, match="unknown language"):
        state.set_fixed("zh")


def test_a_rewritten_settings_file_applies_from_the_next_turn() -> None:
    settings = [LanguageConfig()]
    state = _state()
    state.watch(lambda: settings[0])
    assert state.begin_turn(_JA) == "ja"
    settings[0] = LanguageConfig(mode="fixed", fixed="en")
    assert state.begin_turn(_JA) == "en"
    state.set_fixed("ja")
    assert state.begin_turn(_EN) == "ja"
    settings[0] = LanguageConfig()
    assert state.begin_turn(_EN) == "en"


def test_an_unreadable_settings_file_keeps_the_language() -> None:
    def broken() -> LanguageConfig:
        raise ConfigError("設定ファイルの JSON が不正")

    state = _state(mode="fixed", fixed="ja")
    state.watch(broken)
    assert state.begin_turn(_EN) == "ja"


def test_a_child_reads_the_parent_and_does_not_detect_again() -> None:
    parent = _state()
    child = parent.for_child()
    assert child.mode == "auto"
    assert child.begin_turn(_EN) == "en"
    assert parent.current == ""
    parent.begin_turn(_JA)
    assert (child.mode, child.pinned) == ("fixed", True)
    assert child.begin_turn(_EN + " Report format: list every path.") == "ja"
    parent.begin_turn(_EN)
    assert child.current == "en"
    grandchild = child.for_child()
    assert grandchild.current == "en"


def test_a_child_keeps_the_language_control_off() -> None:
    parent = _state(mode="off")
    parent.begin_turn(_JA)
    assert parent.for_child().begin_turn(_JA) == ""


def test_the_screen_language_before_the_first_utterance() -> None:
    assert LanguageState.from_config(LanguageConfig(), {"LANG": "ja_JP.UTF-8"}).ui_language == "ja"
    assert LanguageState.from_config(LanguageConfig(), {"LANG": "de_DE"}).ui_language == "en"
    fixed = LanguageState.from_config(LanguageConfig(mode="fixed", fixed="ja"), {"LANG": "C"})
    assert fixed.ui_language == "ja"
    state = LanguageState.from_config(LanguageConfig(), {"LANG": "ja_JP.UTF-8"})
    state.begin_turn(_EN)
    assert state.ui_language == "en"
    assert state.for_child().ui_language == "en"


def _factory(tmp_path: Path, provider: object, config: RuntimeConfig) -> RuntimeFactory:
    return RuntimeFactory(
        provider=provider,  # type: ignore[arg-type]
        model="mock-model",
        config=config,
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        hooks=HookRunner(HookConfig(), tmp_path),
        cwd=tmp_path,
        provider_builder=lambda _m: provider,  # type: ignore[arg-type,return-value]
    )


def _runtime(
    tmp_path: Path, provider: object, config: RuntimeConfig, factory: RuntimeFactory | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config,
        runtime_factory=factory,
    )


async def test_the_runtime_and_its_children_share_one_owner(tmp_path: Path) -> None:
    config = RuntimeConfig(
        subagent=SubagentConfig(default_model="mock-model"),
        tools=ToolsConfig(deferred_tool_loading=False),
    )
    provider = SequencedMockProvider([[TextDelta(text="cart.py の total です"), MessageStop()]])
    factory = _factory(tmp_path, provider, config)
    rt = _runtime(tmp_path, provider, config, factory)
    assert rt.language is factory.language
    await rt.run_turn(_JA)
    child = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    assert child.language.current == "ja"
    assert child.language.pinned
    assert child.session.language == "ja"
    assert config.language == LanguageConfig()


async def test_the_settings_file_changes_the_language_between_turns(tmp_path: Path) -> None:
    (tmp_path / ".satsuma").mkdir()
    local = tmp_path / ".satsuma" / "settings.local.json"
    local.write_text("{}", encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [TextDelta(text="cart.py の total です"), MessageStop()],
            [TextDelta(text="It is cart.py, in the total function."), MessageStop()],
        ]
    )
    rt = _runtime(tmp_path, provider, RuntimeConfig(language=LanguageConfig(repair=False)))
    rt.language.watch(ConfigLoader(tmp_path, tmp_path / "home", trusted=True).load_language)
    await rt.run_turn(_JA)
    local.write_text(json.dumps({"language": {"mode": "fixed", "fixed": "en"}}), encoding="utf-8")
    await rt.run_turn(_JA)
    last_user = [m for m in rt.session.messages if m.role == "user"][-1]
    sent = last_user.blocks[0].text  # type: ignore[union-attr]
    assert sent.endswith(
        "(Reply in English. Code, paths, identifiers and command output stay as-is.)"
    )
    assert rt.session.language == "en"


def test_languages_lists_the_definitions_as_json() -> None:
    result = runner.invoke(cli_app.app, ["languages", "--json"])
    assert result.exit_code == 0, result.output
    listed = json.loads(result.stdout)
    assert [entry["code"] for entry in listed] == ["ja", "en"]
    assert listed[0] == {"code": "ja", "name": "Japanese", "native_name": "日本語", "ui": True}


def test_languages_lists_the_definitions_as_text() -> None:
    result = runner.invoke(cli_app.app, ["languages"])
    assert result.exit_code == 0, result.output
    assert "ja" in result.stdout and "日本語" in result.stdout


def _prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str
) -> tuple[object, MockProvider]:
    provider = MockProvider([TextDelta(text="It is cart.py."), MessageStop()])
    monkeypatch.setattr(cli_app, "build_provider", lambda config, model, env: provider)
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    result = runner.invoke(
        cli_app.app, ["prompt", _JA, "--output-format", "json", "--cwd", str(tmp_path), *extra]
    )
    return result, provider


def test_cli_lang_fixes_the_language_of_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, provider = _prompt(monkeypatch, tmp_path, "--lang", "en")
    assert result.exit_code == 0, result.output  # type: ignore[attr-defined]
    sent = provider.requests[0].messages[0].blocks[0].text  # type: ignore[union-attr]
    assert sent.endswith(
        "(Reply in English. Code, paths, identifiers and command output stay as-is.)"
    )


def test_cli_lang_refuses_an_undefined_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, _provider = _prompt(monkeypatch, tmp_path, "--lang", "zh")
    assert result.exit_code != 0  # type: ignore[attr-defined]
    assert "ja, en" in result.output  # type: ignore[attr-defined]
