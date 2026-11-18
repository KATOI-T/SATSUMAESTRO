"""入力欄のスラッシュコマンドと引数の候補。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from satsumaestro.cli import completion as completion_mod
from satsumaestro.cli.commands import SLASH_COMMANDS, provider_notes
from satsumaestro.cli.completion import ModelNames, SlashCompleter, slash_completions
from satsumaestro.config.models import ProviderConfig, RuntimeConfig
from satsumaestro.i18n.text import pinned_ui_language, ui
from satsumaestro.providers.registry import RECOMMENDED_MODELS
from satsumaestro.util.errors import ConfigError

MODELS = [("gpt-oss:120b", ""), ("gpt-oss:20b", ""), ("qwen3-coder:30b", ""), ("qwen/gpt-x", "")]


def _texts(text: str, models: Sequence[tuple[str, str]] = ()) -> list[str]:
    return [c.text.strip() for c in slash_completions(text, models)]


def test_a_slash_lists_every_command_with_its_description() -> None:
    shown = list(slash_completions("/"))
    assert [c.text for c in shown] == [command.name + " " for command in SLASH_COMMANDS]
    model = next(c for c in shown if c.text == "model ")
    assert model.display_text == "/model"
    assert model.display_meta_text == ui("cli.commands.show_switch_the_current_model")
    assert model.start_position == 0


def test_the_list_narrows_as_you_type() -> None:
    shown = list(slash_completions("/mo"))
    assert [c.text for c in shown] == ["model ", "mode "]
    assert all(c.start_position == -2 for c in shown)
    assert _texts("/MO") == ["model", "mode"]
    assert _texts("/he") == ["help"]
    assert _texts("/lan") == ["language"]


def test_aliases_are_offered_once_typed() -> None:
    assert _texts("/q") == ["q", "quit"]
    assert _texts("/sub") == ["sub-mode", "submode"]
    assert "quit" not in _texts("/")


def test_the_typed_word_comes_first() -> None:
    assert _texts("/mode") == ["mode", "model"]
    assert _texts("/plan") == ["plan"]
    models = [("gpt-oss:120b-q8", ""), ("gpt-oss:120b", "")]
    assert _texts("/model gpt-oss:120b", models) == ["gpt-oss:120b", "gpt-oss:120b-q8"]


def test_a_space_follows_when_more_can_be_chosen() -> None:
    inserted = {c.display_text: c.text for c in slash_completions("/mode ")}
    assert inserted == {"plan": "plan", "code": "code "}
    assert [c.text for c in slash_completions("/he")] == ["help "]
    assert [c.text for c in slash_completions("/language j")] == ["ja"]


def test_the_descriptions_follow_the_screen_language() -> None:
    with pinned_ui_language("en"):
        meta = next(c for c in slash_completions("/mo")).display_meta_text
    assert meta == "Show the current model and the available ones / switch and save"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "hello /mo",
        " /mo",
        "/Users/alice/x",
        "/Users/alice/x を消して",
        "/tmp/",
        "/日本",
        "/mo\nx",
        "/nothing ",
        "/help ",
        "/exit ",
        "/lang ",
    ],
)
def test_no_suggestions(text: str) -> None:
    assert _texts(text, MODELS) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/provider ", ["ollama", "lmstudio", "llamacpp", "openai", "anthropic", "xai"]),
        ("/provider l", ["lmstudio", "llamacpp"]),
        ("/PROVIDER ll", ["llamacpp"]),
        ("/mode ", ["plan", "code"]),
        ("/mode c", ["code"]),
        ("/mode code ", ["read-only", "workspace-write", "danger-full-access", "prompt", "allow"]),
        ("/mode code d", ["danger-full-access"]),
        ("/mode plan ", []),
        ("/sub-mode ", ["read-only", "workspace-write", "danger-full-access", "prompt", "allow"]),
        ("/submode w", ["workspace-write"]),
        ("/sub-mode prompt ", []),
        ("/plan ", ["on", "off"]),
        ("/effort ", ["none", "minimal", "low", "medium", "high", "xhigh", "max", "default"]),
        ("/effort m", ["minimal", "medium", "max"]),
        ("/language ", ["ja", "en", "auto"]),
        ("/language e", ["en"]),
    ],
)
def test_arguments_are_suggested(text: str, expected: list[str]) -> None:
    assert _texts(text) == expected


def test_a_cloud_provider_is_followed_by_its_recommended_models() -> None:
    assert _texts("/provider anthropic ") == list(RECOMMENDED_MODELS["anthropic"])
    assert _texts("/provider 6 ") == list(RECOMMENDED_MODELS["xai"])
    assert _texts("/provider xai grok-4.2") == [
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
    ]
    assert _texts("/provider OpenAI gpt-5.4") == ["gpt-5.4", "gpt-5.4-mini"]
    assert _texts("/provider anthropic claude-opus-5 ") == []
    inserted = {c.display_text: c.text for c in slash_completions("/provider ")}
    assert set(inserted.values()) == {f"{name} " for name in inserted}
    assert [c.text for c in slash_completions("/provider xai grok-4.7")] == ["grok-4.7"]


def test_a_local_provider_is_followed_by_the_models_on_its_server() -> None:
    assert _texts("/provider ollama ") == []
    assert _texts("/provider ollama ", MODELS) == [name for name, _ in MODELS]
    assert _texts("/provider 1 gpt", MODELS) == ["gpt-oss:120b", "gpt-oss:20b", "qwen/gpt-x"]
    assert _texts("/provider lmstudio 120", MODELS) == ["gpt-oss:120b"]
    assert [c.text for c in slash_completions("/provider ollama gpt-oss:20", MODELS)] == [
        "gpt-oss:20b"
    ]
    assert _texts("/provider ollama gpt-oss:120b ", MODELS) == []
    assert _texts("/provider xai ", MODELS) == list(RECOMMENDED_MODELS["xai"])


def _notes(text: str, **kwargs: Any) -> dict[str, str]:
    return {c.display_text: c.display_meta_text for c in slash_completions(text, **kwargs)}


def test_arguments_carry_a_label() -> None:
    assert _notes("/language ") == {"ja": "日本語", "en": "English", "auto": "会話の言語に合わせる"}
    assert _notes("/provider ")["llamacpp"] == "llama.cpp"
    assert _notes("/effort ")["high"] == "深く考える。遅くなる"
    assert _notes("/mode code ")["workspace-write"] == "作業場所の中の書き込みまで確認なし（既定）"
    with pinned_ui_language("en"):
        assert _notes("/effort h") == {"high": "Think deeply; slower"}
        assert _notes("/plan ") == {"on": "Enter plan mode", "off": "Go back to code mode"}


@pytest.mark.parametrize("language", ["ja", "en"])
@pytest.mark.parametrize("text", ["/effort ", "/mode ", "/mode code ", "/sub-mode ", "/plan "])
def test_every_fixed_value_has_a_one_line_note(language: str, text: str) -> None:
    with pinned_ui_language(language):
        notes = _notes(text)
    assert notes and all(note and "\n" not in note for note in notes.values()), notes


def test_providers_show_where_they_connect() -> None:
    notes = {"ollama": "Ollama ✓ · http://192.168.188.110:11434（OLLAMA_HOST）"}
    shown = _notes("/provider ", providers=notes)
    assert shown["ollama"] == notes["ollama"]
    assert shown["xai"] == "xAI"
    env = {"OLLAMA_HOST": "192.168.188.110:11434", "ANTHROPIC_API_KEY": "x"}
    found = provider_notes(ProviderConfig(), "gpt-oss:120b", env)
    assert found["ollama"].startswith("Ollama ✓ · http://192.168.188.110:11434")
    assert found["anthropic"] == "Anthropic · 鍵あり（ANTHROPIC_API_KEY）"
    assert found["xai"] == "xAI · 鍵なし（XAI_API_KEY）"
    assert found["lmstudio"] == "LM Studio · 未設定"


async def test_provider_notes_are_read_only_for_provider_lines() -> None:
    asked: list[int] = []

    def notes() -> dict[str, str]:
        asked.append(1)
        return {"openai": "OpenAI · 鍵あり（OPENAI_API_KEY）"}

    completer = SlashCompleter(providers=notes)
    events = completer.get_completions_async(Document("/provider o"), CompleteEvent())
    shown = {c.display_text: c.display_meta_text async for c in events}
    assert shown == {"ollama": "Ollama", "openai": "OpenAI · 鍵あり（OPENAI_API_KEY）"}
    assert await _complete(completer, "/mode ") == ["plan", "code"]
    assert len(asked) == 1


def test_models_match_anywhere_with_prefix_matches_first() -> None:
    assert _texts("/model ", MODELS) == [name for name, _ in MODELS]
    assert _texts("/model gpt", MODELS) == ["gpt-oss:120b", "gpt-oss:20b", "qwen/gpt-x"]
    assert _texts("/model 120", MODELS) == ["gpt-oss:120b"]
    assert _texts("/model QWEN", MODELS) == ["qwen3-coder:30b", "qwen/gpt-x"]
    assert _texts("/model gpt-oss:120b ", MODELS) == []
    assert _texts("/model ", []) == []


class _Source:
    def __init__(self, names: Sequence[tuple[str, str]]) -> None:
        self._names = names
        self.prefetched = 0
        self.asked = 0
        self.targets: list[str] = []

    def prefetch(self, choice: str = "") -> None:
        self.prefetched += 1

    async def names(self, choice: str = "") -> Sequence[tuple[str, str]]:
        self.asked += 1
        self.targets.append(choice)
        return self._names


async def _complete(completer: SlashCompleter, text: str) -> list[str]:
    document = Document(text)
    events = completer.get_completions_async(document, CompleteEvent())
    return [c.text.strip() async for c in events]


async def test_the_completer_asks_for_models_only_after_model() -> None:
    source = _Source(MODELS)
    completer = SlashCompleter(source)
    assert await _complete(completer, "/model q") == ["qwen3-coder:30b", "qwen/gpt-x"]
    assert source.asked == 1
    assert await _complete(completer, "/mo") == ["model", "mode"]
    assert (source.asked, source.prefetched) == (1, 1)
    assert await _complete(completer, "hello") == []
    assert await _complete(completer, "/Users/a") == []
    assert (source.asked, source.prefetched) == (1, 1)
    assert list(completer.get_completions(Document("/pl"), CompleteEvent()))[0].text == "plan "


async def test_the_completer_asks_the_server_after_a_local_provider() -> None:
    source = _Source(MODELS)
    completer = SlashCompleter(source)
    assert await _complete(completer, "/provider 3 qwen") == ["qwen3-coder:30b", "qwen/gpt-x"]
    assert await _complete(completer, "/provider ollama ") == [name for name, _ in MODELS]
    assert source.targets == ["llamacpp", "ollama"]
    assert await _complete(completer, "/provider xai grok-4.7") == ["grok-4.7"]
    assert await _complete(completer, "/provider ollama gpt-oss:120b ") == []
    assert source.targets == ["llamacpp", "ollama"]


def _runtime(kind: str = "ollama", model: str = "gpt-oss:120b") -> Any:
    config = RuntimeConfig(provider=ProviderConfig(kind=kind))  # type: ignore[arg-type]
    return SimpleNamespace(config=config, session=SimpleNamespace(model=model))


def _fake_list(
    monkeypatch: pytest.MonkeyPatch, answers: list[list[str] | None | Exception]
) -> list[str]:
    asked: list[str] = []

    async def fake(cfg: Any, model: str, env: Any, **kwargs: Any) -> list[str] | None:
        asked.append(model)
        answer = answers[min(len(asked), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(completion_mod, "list_models", fake)
    return asked


async def test_the_model_list_is_kept_until_the_model_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _fake_list(monkeypatch, [["gpt-oss:120b", "gpt-oss:20b"], ["qwen3:8b"]])
    rt = _runtime()
    names = ModelNames(rt)
    assert await names.names() == [("gpt-oss:120b", ""), ("gpt-oss:20b", "")]
    assert await names.names() == [("gpt-oss:120b", ""), ("gpt-oss:20b", "")]
    assert asked == ["gpt-oss:120b"]
    rt.session.model = "gpt-oss:20b"
    assert await names.names() == [("qwen3:8b", "")]
    assert asked == ["gpt-oss:120b", "gpt-oss:20b"]


async def test_a_server_that_does_not_answer_gives_no_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _fake_list(monkeypatch, [None, ConfigError("no endpoint"), ["gpt-oss:120b"]])
    names = ModelNames(_runtime(), retry_s=60.0)
    assert await names.names() == []
    assert await names.names() == []
    assert len(asked) == 1
    names = ModelNames(_runtime(), retry_s=0.0)
    assert await names.names() == []
    assert await names.names() == [("gpt-oss:120b", "")]


async def test_prefetch_does_not_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = asyncio.Event()

    async def slow(cfg: Any, model: str, env: Any, **kwargs: Any) -> list[str]:
        await gate.wait()
        return ["gpt-oss:120b"]

    monkeypatch.setattr(completion_mod, "list_models", slow)
    names = ModelNames(_runtime())
    names.prefetch()
    names.prefetch()
    waiting = asyncio.create_task(names.names())
    await asyncio.sleep(0)
    assert not waiting.done()
    gate.set()
    assert await waiting == [("gpt-oss:120b", "")]


async def test_the_models_of_the_switch_target_are_kept_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _fake_list(monkeypatch, [["gpt-oss:120b"]])
    targets: list[str] = []

    async def fake_for(rt: Any, choice: str, env: Any, **kwargs: Any) -> list[str] | None:
        targets.append(choice)
        return {"lmstudio": ["openai/gpt-oss-20b"]}.get(choice)

    monkeypatch.setattr(completion_mod, "models_for", fake_for)
    names = ModelNames(_runtime("anthropic", "claude-sonnet-5"))
    assert await names.names("lmstudio") == [("openai/gpt-oss-20b", "")]
    assert await names.names("lmstudio") == [("openai/gpt-oss-20b", "")]
    assert await names.names("ollama") == []
    assert targets == ["lmstudio", "ollama"]
    current = await names.names()
    assert current[0] == ("gpt-oss:120b", "") and ("sonnet", "→ claude-sonnet-5") in current
    assert len(asked) == 1


async def test_shorthands_are_offered_for_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_list(monkeypatch, [None])
    shown = await ModelNames(_runtime("anthropic", "claude-sonnet-5")).names()
    assert ("sonnet", "→ claude-sonnet-5") in shown
    assert await ModelNames(_runtime("ollama")).names() == []
