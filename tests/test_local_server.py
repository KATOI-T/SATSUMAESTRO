"""ローカルのサーバの判定（Ollama の判定と同じ結果になること）。"""

from __future__ import annotations

import pytest

from satsumaestro.providers.detect import OllamaDetectMode, is_ollama_endpoint, ollama_root
from satsumaestro.providers.local import OllamaServer, server_for_endpoint

_CASES: list[tuple[str, dict[str, str], OllamaDetectMode]] = [
    ("http://localhost:11434/v1", {}, "auto"),
    ("http://192.168.0.5:11434/v1", {}, "auto"),
    ("http://gpu.lan:8000/v1", {"OLLAMA_HOST": "gpu.lan:8000"}, "auto"),
    ("http://gpu.lan:8000/v1", {"OLLAMA_HOST": "other.lan:8000"}, "auto"),
    ("http://localhost:1234/v1", {}, "auto"),
    ("http://localhost:8080/v1", {}, "auto"),
    ("http://localhost:8080/v1", {}, "on"),
    ("http://localhost:11434/v1", {}, "off"),
    ("https://api.example.com/v1", {}, "auto"),
]


@pytest.mark.parametrize(("endpoint", "env", "detect"), _CASES)
def test_the_server_matches_the_ollama_rule(
    endpoint: str, env: dict[str, str], detect: OllamaDetectMode
) -> None:
    server = server_for_endpoint(endpoint, env, detect)
    if is_ollama_endpoint(endpoint, env, detect):
        assert server == OllamaServer(ollama_root(endpoint))
        assert server.kind == "ollama"
    else:
        assert server is None
