"""モデルが無いとき、prompt は決める場所と候補を出して止まり、REPL は /model へ案内する。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

import satsumaestro.cli.app as cli_app
from satsumaestro.cli import repl as repl_mod
from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.config.models import ProviderConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.provider_switch import ModelNotSetError, connect
from satsumaestro.providers import factory
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.factory import build_provider
from satsumaestro.providers.registry import RECOMMENDED_MODELS
from satsumaestro.session.store import JSONLSessionStore
from satsumaestro.util.environ import ENVIRON
from tests.fixtures.mock_provider import MockProvider
from tests.test_p153_missing_model import _ollama, _route_http
from tests.test_p165_plan_steps import _runtime


def _patch_provider(
    monkeypatch: pytest.MonkeyPatch, providers: list[MockProvider] | None = None
) -> list[str]:
    built: list[str] = []

    def build(config: RuntimeConfig, model: str, env: object) -> MockProvider:
        built.append(model)
        provider = MockProvider([TextDelta(text="ok"), MessageStop()])
        if providers is not None:
            providers.append(provider)
        return provider

    async def fit(cfg: RuntimeConfig, model: str, env: object) -> None:
        return None

    monkeypatch.setattr(cli_app, "build_provider", build)
    monkeypatch.setattr(factory, "build_provider", build)
    monkeypatch.setattr(cli_app, "apply_context_fit", fit)
    return built


class _Lines:
    """``PromptSession`` の代役（行を順に返し、尽きたら終わる）。"""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def prompt_async(self, *_: Any, **__: Any) -> str:
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


def _settings(provider: dict[str, Any]) -> Path:
    path = Path(os.environ["SATSUMA_CONFIG_HOME"]) / "settings.json"
    path.write_text(json.dumps({"provider": provider}), encoding="utf-8")
    return path


def test_prompt_stops_and_lists_the_models_on_the_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = _patch_provider(monkeypatch)
    _route_http(monkeypatch, _ollama())
    result = CliRunner().invoke(cli_app.app, ["prompt", "1+1", "--cwd", str(tmp_path)])
    assert result.exit_code == 1
    out = result.output.replace("\n", " ")
    assert "モデルが決まっていません。--model か、" in out
    assert "settings.json の provider.default_model か、環境変数か .env の OLLAMA_MODEL" in out
    assert "利用可能: gpt-oss:120b, qwen3-coder-next:q4_K_M, llama3.1:8b" in out
    assert built == []


async def test_repl_starts_and_points_to_model_before_sending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    providers: list[MockProvider] = []
    built = _patch_provider(monkeypatch, providers)
    _route_http(monkeypatch, _ollama())
    rt, prompter, render = await cli_app._assemble_runtime(
        None, tmp_path, interactive=True, trusted=True
    )
    lines = ["1+1", "/model", "/model gpt-oss:120b", "1+1"]
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _Lines(lines))
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    console = Console(record=True, width=300)
    try:
        assert rt.session.model == "" and built == [""]
        await repl_mod.run_repl(rt, prompter, render, console)  # type: ignore[arg-type]
    finally:
        await cli_app._shutdown(rt.mcp, rt.rag, rt.orchestrator, rt.background, rt.lsp)
    out = console.export_text()
    assert "送っていません。/model <名前> で使うモデルを選んでください" in out
    assert "model: （未選択）" in out and "gpt-oss:120b" in out
    assert built == ["", "gpt-oss:120b"] and providers[0].requests == []
    assert [req.model for req in providers[1].requests] == ["gpt-oss:120b"]
    assert rt._runtime_factory is not None and rt._runtime_factory.model == "gpt-oss:120b"


def test_a_cloud_lists_the_recommended_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = _patch_provider(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    _settings({"kind": "anthropic"})
    result = CliRunner().invoke(cli_app.app, ["prompt", "1+1", "--cwd", str(tmp_path)])
    assert result.exit_code == 1
    out = result.output.replace("\n", " ")
    assert "モデルが決まっていません" in out and "OLLAMA_MODEL" not in out
    assert "SATSUMAESTRO で確かめたモデル: " + ", ".join(RECOMMENDED_MODELS["anthropic"]) in out
    assert built == []


async def test_connect_refuses_an_empty_name_unless_told_otherwise() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(kind="xai"))
    with pytest.raises(ModelNotSetError) as stopped:
        await connect(cfg, " ", {})
    assert stopped.value.models == list(RECOMMENDED_MODELS["xai"])
    assert "provider.default_model" in str(stopped.value)
    built: list[str] = []

    def build(config: RuntimeConfig, model: str, env: object) -> MockProvider:
        built.append(model)
        return MockProvider([])

    connection = await connect(cfg, "", {"XAI_API_KEY": "k"}, build=build, model_required=False)
    assert built == [""] and connection.notes == ()


async def test_resume_without_a_model_uses_the_saved_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = _patch_provider(monkeypatch)
    _route_http(monkeypatch, _ollama())
    store = JSONLSessionStore()
    rt, _, _ = await cli_app._assemble_runtime(
        "gpt-oss:120b", tmp_path, interactive=False, trusted=True
    )
    store.save(rt.session)
    path = store.path_for(tmp_path, rt.session.session_id)
    resumed, _, _ = await cli_app._assemble_runtime(
        None, tmp_path, interactive=False, resume_path=path, trusted=True
    )
    assert resumed.session.model == "gpt-oss:120b"
    assert built == ["gpt-oss:120b", "gpt-oss:120b"]


@pytest.mark.parametrize(
    ("keys", "chosen", "kind"),
    [
        ({"ANTHROPIC_API_KEY": "sk-ant-k", "XAI_API_KEY": "xai-k"}, "grok-4.7", ProviderKind.XAI),
        (
            {
                "OPENAI_API_KEY": "sk-proj-k",
                "OPENAI_BASE_URL": "https://h/v1",
                "ANTHROPIC_API_KEY": "k",
            },
            "claude-sonnet-5-5",
            ProviderKind.ANTHROPIC,
        ),
    ],
)
async def test_a_model_chosen_after_starting_without_one_decides_the_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    keys: dict[str, str],
    chosen: str,
    kind: ProviderKind,
) -> None:
    """auto では名前で行き先が決まるので、起動時の（名前なしで組んだ）先を使い続けない。"""
    for name in ("OLLAMA_HOST", "OPENAI_API_KEY", "OPENAI_BASE_URL", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    started = build_provider(RuntimeConfig(), "", ENVIRON)
    assert started.kind is not kind
    rt = _runtime(started, tmp_path)
    rt.session.model = ""
    await SlashDispatcher(rt, Console(record=True, width=200)).dispatch(
        parse_slash(f"/model {chosen}")
    )
    assert rt.session.model == chosen and rt.provider.kind is kind


async def test_a_model_without_its_key_leaves_everything_as_it_was(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("OLLAMA_HOST", "OPENAI_API_KEY", "OPENAI_BASE_URL", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-k")
    started = MockProvider([])
    rt = _runtime(started, tmp_path)
    rt.session.model = ""
    console = Console(record=True, width=200)
    await SlashDispatcher(rt, console).dispatch(parse_slash("/model grok-4.7"))
    out = console.export_text()
    assert "XAI_API_KEY" in out and "model は （未選択） のままです" in out
    assert "コマンド実行に失敗しました" not in out
    assert rt.session.model == "" and rt.provider is started
