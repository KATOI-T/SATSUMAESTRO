"""ローカルの LLM サーバ（Ollama）の判定とネイティブ API。"""

from __future__ import annotations

from collections.abc import Mapping

from satsumaestro.providers.detect import OllamaDetectMode, is_ollama_endpoint, ollama_root
from satsumaestro.providers.local.base import LocalKind, LocalServer
from satsumaestro.providers.local.ollama import OllamaServer

__all__ = ["LocalKind", "LocalServer", "OllamaServer", "server_for_endpoint"]


def server_for_endpoint(
    endpoint: str, env: Mapping[str, str], ollama_detect: OllamaDetectMode = "auto"
) -> LocalServer | None:
    """OpenAI 互換の接続先がローカルのサーバならそれを返す（通信はしない）。"""
    if is_ollama_endpoint(endpoint, env, ollama_detect):
        return OllamaServer(ollama_root(endpoint))
    return None
