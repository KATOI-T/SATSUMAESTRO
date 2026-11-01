"""``OLLAMA_HOST`` をスキームなしで書いたときの接続先（Ollama 本体と同じ読み方）。"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from satsumaestro.config.loader import ConfigLoader
from satsumaestro.config.models import ProviderConfig, RuntimeConfig
from satsumaestro.engine.context_fit import check_ollama_reachable
from satsumaestro.providers.detect import is_ollama_endpoint
from satsumaestro.providers.factory import openai_endpoint
from satsumaestro.util.urls import ollama_host_url


@pytest.mark.parametrize(
    ("raw", "url"),
    [
        ("192.0.2.10:11434", "http://192.0.2.10:11434"),
        ("192.0.2.10", "http://192.0.2.10:11434"),
        ("localhost", "http://localhost:11434"),
        ("0.0.0.0", "http://0.0.0.0:11434"),
        (":11434", "http://127.0.0.1:11434"),
        ("[::1]:11434", "http://[::1]:11434"),
        ("::1", "http://[::1]:11434"),
        ("ollama.example:8080/proxy", "http://ollama.example:8080/proxy"),
        ("  192.0.2.10:11434  ", "http://192.0.2.10:11434"),
        ("http://192.0.2.10:11434", "http://192.0.2.10:11434"),
        ("https://ollama.example", "https://ollama.example"),
    ],
)
def test_ollama_host_url(raw: str, url: str) -> None:
    assert ollama_host_url(raw) == url


def test_endpoint_from_a_schemeless_ollama_host() -> None:
    env = {"OLLAMA_HOST": "192.0.2.10:11434"}
    assert openai_endpoint(RuntimeConfig(), env) == "http://192.0.2.10:11434/v1"
    assert openai_endpoint(RuntimeConfig(), {"OLLAMA_HOST": "localhost"}) == (
        "http://localhost:11434/v1"
    )


def test_schemeless_base_url_gets_http() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(base_url="localhost:8080"))
    assert openai_endpoint(cfg, {}) == "http://localhost:8080/v1"


def test_config_loader_normalizes_ollama_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
    cfg = ConfigLoader(tmp_path, trusted=True).load()
    assert cfg.provider.base_url == "http://192.0.2.10:11434"
    assert openai_endpoint(cfg, {}) == "http://192.0.2.10:11434/v1"


def test_schemeless_ollama_host_on_another_port_is_detected() -> None:
    assert is_ollama_endpoint("http://192.0.2.10:8080/v1", {"OLLAMA_HOST": "192.0.2.10:8080"})


async def test_preflight_reaches_the_schemeless_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OLLAMA_HOST", "192.0.2.10:11434")
    cfg = ConfigLoader(tmp_path, trusted=True).load()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"models": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        problem = await check_ollama_reachable(
            cfg, "gpt-oss:120b", {"OLLAMA_HOST": "192.0.2.10:11434"}, client=client
        )
    assert problem is None
    assert seen == ["http://192.0.2.10:11434/api/tags"]
