"""接続できなくても REPL は起動し、理由と案内を出して指示を送らない（prompt は止まる）。"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

import satsumaestro.cli.app as cli_app
from satsumaestro.cli import repl as repl_mod
from satsumaestro.config.models import ProviderConfig, RuntimeConfig
from satsumaestro.engine.provider_switch import UnconnectedProvider, connect
from satsumaestro.providers.base import ApiRequest
from satsumaestro.util.errors import ConfigError, SatsumaError
from tests.fixtures.mock_provider import MockProvider
from tests.test_model_not_set import _Lines

_DOWN = "http://127.0.0.1:11434"
_UNREACHABLE = "Ollama に接続できません: " + _DOWN
_NO_OLLAMA_URL = "Ollama の接続先がありません"
_NO_AUTO_URL = "OpenAI 互換の接続先が決まりません"
_NO_KEY = "ANTHROPIC_API_KEY が設定されていません"

_CASES = [
    pytest.param(
        {"kind": "ollama", "base_url": f"{_DOWN}/v1", "default_model": "qwen3:8b"},
        _UNREACHABLE,
        id="unreachable",
    ),
    pytest.param({"kind": "ollama", "default_model": "qwen3:8b"}, _NO_OLLAMA_URL, id="no-url"),
    pytest.param({"default_model": "qwen3:8b"}, _NO_AUTO_URL, id="auto-no-url"),
    pytest.param({"kind": "anthropic", "default_model": "claude-sonnet-4-5"}, _NO_KEY, id="no-key"),
]


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("Connection refused")


@pytest.fixture(autouse=True)
def _nothing_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    real = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(_refused)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)


def _settings(provider: dict[str, Any]) -> None:
    path = Path(os.environ["SATSUMA_CONFIG_HOME"]) / "settings.json"
    path.write_text(json.dumps({"provider": provider}), encoding="utf-8")


async def _drain(stream: AsyncIterator[object]) -> None:
    async for _ in stream:
        pass


@pytest.mark.parametrize(("provider", "reason"), _CASES)
async def test_connect_gives_the_reason_instead_of_stopping(
    provider: dict[str, Any], reason: str
) -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(**provider))
    model = cfg.provider.default_model
    with pytest.raises(SatsumaError) as stopped:
        await connect(cfg, model, {})
    assert reason in str(stopped.value)
    connection = await connect(cfg, model, {}, allow_unconnected=True)
    assert connection.problem == str(stopped.value)
    assert isinstance(connection.provider, UnconnectedProvider)
    with pytest.raises(ConfigError) as refused:
        await _drain(connection.provider.stream(ApiRequest(model=model, system="", messages=[])))
    assert str(refused.value) == connection.problem


async def test_a_connection_that_works_has_no_problem() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(kind="anthropic"))
    built: list[str] = []

    def build(config: RuntimeConfig, model: str, env: object) -> MockProvider:
        built.append(model)
        return MockProvider([])

    connection = await connect(cfg, "m", {}, build=build, allow_unconnected=True)
    assert built == ["m"] and connection.problem is None
    assert isinstance(connection.provider, MockProvider)


@pytest.mark.parametrize(("provider", "reason"), _CASES)
async def test_the_repl_starts_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: dict[str, Any], reason: str
) -> None:
    _settings(provider)
    startup = Console(record=True, width=400)
    monkeypatch.setattr(cli_app, "_err_console", startup)
    rt, prompter, render = await cli_app._assemble_runtime(
        None, tmp_path, interactive=True, trusted=True
    )
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _Lines(["1+1"]))
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    console = Console(record=True, width=400)
    try:
        assert rt.unconnected is not None and reason in rt.unconnected
        assert isinstance(rt.provider, UnconnectedProvider)
        await repl_mod.run_repl(rt, prompter, render, console)  # type: ignore[arg-type]
    finally:
        await cli_app._shutdown(rt.mcp, rt.rag, rt.orchestrator, rt.background, rt.lsp)
    shown = startup.export_text()
    assert reason in shown and "接続できないまま起動しました。/provider で" in shown
    out = console.export_text()
    assert reason in out and "接続できていないので送っていません。/provider で" in out
    assert rt.session.messages == []


async def test_a_new_provider_clears_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _settings({"kind": "anthropic", "default_model": "claude-sonnet-4-5"})
    monkeypatch.setattr(cli_app, "_err_console", Console(record=True))
    rt, _, _ = await cli_app._assemble_runtime(None, tmp_path, interactive=True, trusted=True)
    try:
        assert rt.unconnected is not None
        rt.use_provider(MockProvider([]), "claude-sonnet-4-5", compactor=rt.compactor)
        assert rt.unconnected is None
    finally:
        await cli_app._shutdown(rt.mcp, rt.rag, rt.orchestrator, rt.background, rt.lsp)


def test_prompt_still_stops(tmp_path: Path) -> None:
    _settings({"kind": "anthropic", "default_model": "claude-sonnet-4-5"})
    result = CliRunner().invoke(cli_app.app, ["prompt", "1+1", "--cwd", str(tmp_path)])
    assert result.exit_code == 1
    assert _NO_KEY in result.output
