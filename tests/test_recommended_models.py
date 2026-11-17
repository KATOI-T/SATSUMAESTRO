"""クラウドで勧めるモデル（SATSUMAESTRO で確かめたもの）の一覧と、それを出す所。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

import satsumaestro.cli.app as cli_app
from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.cli.completion import ModelNames
from satsumaestro.config.models import GuidanceConfig, ProviderConfig, RuntimeConfig
from satsumaestro.engine.context_fit import list_models
from satsumaestro.engine.provider_switch import ModelNeededError, switch_provider
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.registry import (
    MODEL_SHORTHANDS,
    RECOMMENDED_MODELS,
    recommended_models,
    vendor_for_model,
)
from tests.fixtures.mock_provider import MockProvider
from tests.test_p165_plan_steps import _runtime

_VENDORS = {
    "openai": ProviderKind.OPENAI,
    "anthropic": ProviderKind.ANTHROPIC,
    "xai": ProviderKind.XAI,
}
_NOTE = "SATSUMAESTRO で確かめたモデルです"


def test_every_model_belongs_to_its_provider() -> None:
    assert set(RECOMMENDED_MODELS) == set(_VENDORS)
    for choice, names in RECOMMENDED_MODELS.items():
        assert names, choice
        assert [n for n in names if vendor_for_model(n) is not _VENDORS[choice]] == []


def test_no_model_is_listed_twice() -> None:
    names = [name for group in RECOMMENDED_MODELS.values() for name in group]
    assert len(names) == len(set(names))


def test_the_shorthands_point_at_recommended_models() -> None:
    assert set(MODEL_SHORTHANDS.values()) <= set(recommended_models("anthropic"))


def test_other_providers_have_no_recommended_models() -> None:
    assert recommended_models("ollama") == () and recommended_models("compatible") == ()


@pytest.mark.parametrize(
    ("provider", "model", "env", "expected"),
    [
        (ProviderConfig(kind="anthropic"), "claude-sonnet-5-5", {}, "anthropic"),
        (ProviderConfig(kind="xai"), "grok-4.3", {}, "xai"),
        (ProviderConfig(kind="openai"), "gpt-5.5", {}, "openai"),
        (ProviderConfig(), "claude-opus-5-5", {"ANTHROPIC_API_KEY": "k"}, "anthropic"),
        (ProviderConfig(), "gpt-5.5", {"OPENAI_BASE_URL": "https://api.openai.com/v1"}, "openai"),
        (ProviderConfig(base_url="https://llm.example.com/v1"), "qwen3:8b", {}, None),
    ],
)
async def test_a_cloud_lists_its_recommended_models(
    provider: ProviderConfig, model: str, env: dict[str, str], expected: str | None
) -> None:
    names = await list_models(RuntimeConfig(provider=provider), model, env)
    assert names == (list(RECOMMENDED_MODELS[expected]) if expected else None)


def _local_runtime(tmp_path: Path, model: str = "gpt-oss:120b") -> ConversationRuntime:
    cfg = RuntimeConfig(
        provider=ProviderConfig(kind="ollama", base_url="http://10.0.0.9:11434/v1"),
        guidance=GuidanceConfig(step_budget=False),
    )
    rt = _runtime(MockProvider([]), tmp_path, cfg=cfg)
    rt.session.model = model
    return rt


async def _dispatch(rt: ConversationRuntime, line: str) -> str:
    console = Console(record=True, width=200)
    await SlashDispatcher(rt, console).dispatch(parse_slash(line))  # type: ignore[arg-type]
    return console.export_text()


async def test_switching_to_a_cloud_without_a_model_offers_the_recommended_models(
    tmp_path: Path,
) -> None:
    rt = _local_runtime(tmp_path)
    with pytest.raises(ModelNeededError) as needed:
        await switch_provider(rt, "xai", env={"XAI_API_KEY": "k"})
    assert needed.value.models == list(RECOMMENDED_MODELS["xai"])


async def test_model_lists_the_recommended_models_on_a_cloud(tmp_path: Path) -> None:
    cfg = RuntimeConfig(
        provider=ProviderConfig(kind="anthropic"), guidance=GuidanceConfig(step_budget=False)
    )
    rt = _runtime(MockProvider([]), tmp_path, cfg=cfg)
    rt.session.model = "claude-sonnet-5-5"
    out = await _dispatch(rt, "/model")
    assert "model: claude-sonnet-5-5" in out
    assert "  * claude-sonnet-5-5" in out and "    claude-haiku-5-5" in out
    assert _NOTE in out


async def test_provider_offers_the_recommended_models_without_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XAI_API_KEY", "k")
    rt = _local_runtime(tmp_path)
    out = await _dispatch(rt, "/provider xai")
    assert "xAI で使うモデルを付けてください" in out
    assert "  grok-4.7" in out and "  grok-build-0.1" in out
    assert _NOTE in out
    assert "プロバイダは ollama のままです" in out
    assert rt.config.provider.kind == "ollama"


async def test_the_model_completion_offers_the_recommended_models_on_a_cloud() -> None:
    config = RuntimeConfig(provider=ProviderConfig(kind="anthropic"))
    rt = SimpleNamespace(config=config, session=SimpleNamespace(model="claude-sonnet-5"))
    shown = await ModelNames(rt).names()  # type: ignore[arg-type]
    listed = [(name, "") for name in RECOMMENDED_MODELS["anthropic"]]
    assert shown[: len(listed)] == listed
    assert ("sonnet", "→ claude-sonnet-5") in shown


def test_the_models_command_shows_the_recommended_models() -> None:
    result = CliRunner().invoke(cli_app.app, ["models"])
    assert result.exit_code == 0, result.output
    assert "cloud models tested with SATSUMAESTRO" in result.output
    for name in ("gpt-6.1-terra", "claude-opus-5-5", "grok-build-0.1"):
        assert name in result.output
