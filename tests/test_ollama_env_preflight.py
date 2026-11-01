"""接続先の決め方と起動前の到達確認。"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from satsumaestro.config.loader import ConfigLoader, prefer_dotenv_ollama
from satsumaestro.config.models import ProviderConfig, RuntimeConfig
from satsumaestro.engine.context_fit import check_ollama_reachable


def test_dotenv_ollama_wins_over_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "OLLAMA_HOST=http://192.0.2.10:11434/v1\nOLLAMA_MODEL=\nOTHER_KEY=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "shell-model")
    monkeypatch.setenv("OTHER_KEY", "from-shell")
    prefer_dotenv_ollama(dotenv)
    assert os.environ["OLLAMA_HOST"] == "http://192.0.2.10:11434/v1"
    assert os.environ["OLLAMA_MODEL"] == "shell-model"
    assert os.environ["OTHER_KEY"] == "from-shell"


def test_config_loader_uses_dotenv_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.10:11434")
    (tmp_path / ".env").write_text("OLLAMA_HOST=http://10.0.0.5:11434/v1\n", encoding="utf-8")
    cfg = ConfigLoader(tmp_path, trusted=True).load()
    assert cfg.provider.base_url == "http://10.0.0.5:11434/v1"


def _cfg(base_url: str) -> RuntimeConfig:
    return RuntimeConfig(provider=ProviderConfig(base_url=base_url))


async def test_reachable_ollama_passes() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"models": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        problem = await check_ollama_reachable(
            _cfg("http://10.0.0.5:11434/v1"), "gpt-oss:120b", {}, client=client
        )
    assert problem is None
    assert seen == ["/api/tags"]


async def test_unreachable_ollama_names_the_host() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        problem = await check_ollama_reachable(
            _cfg("http://10.0.0.5:11434/v1"), "gpt-oss:120b", {}, client=client
        )
    assert problem is not None
    assert "http://10.0.0.5:11434" in problem and "OLLAMA_HOST" in problem


async def test_non_ollama_provider_is_not_probed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("照会してはいけない")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert (
            await check_ollama_reachable(
                RuntimeConfig(),
                "claude-sonnet-5",
                {"ANTHROPIC_API_KEY": "x"},
                client=client,
            )
            is None
        )
        assert (
            await check_ollama_reachable(_cfg("http://localhost:8080/v1"), "m", {}, client=client)
            is None
        )
