"""Ollama 検出・ネイティブ API クライアントのテスト。"""

from __future__ import annotations

import json
from typing import Any

import httpx

from satsumaestro.providers.detect import (
    fetch_model_context,
    is_ollama_endpoint,
    ollama_root,
    touch_keep_alive,
)


def test_ollama_root_strips_openai_suffix() -> None:
    assert ollama_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert ollama_root("http://localhost:11434/v1/") == "http://localhost:11434"
    assert ollama_root("http://host:11434/api/v1") == "http://host:11434"
    assert ollama_root("http://host:11434") == "http://host:11434"
    assert ollama_root("https://gw.example/ollama/v1") == "https://gw.example/ollama"


def test_is_ollama_endpoint_auto() -> None:
    assert is_ollama_endpoint("http://localhost:11434/v1", {}) is True
    assert is_ollama_endpoint("http://10.0.0.5:11434/v1", {}) is True
    assert is_ollama_endpoint("http://localhost:8080/v1", {}) is False
    env = {"OLLAMA_HOST": "http://gpu-box:9999"}
    assert is_ollama_endpoint("http://gpu-box:9999/v1", env) is True
    assert is_ollama_endpoint("http://gpu-box:9998/v1", env) is False
    assert is_ollama_endpoint("http://gpu-box:9999/v1", {"OLLAMA_HOST": "gpu-box:9999"}) is True


def test_is_ollama_endpoint_forced() -> None:
    assert is_ollama_endpoint("http://localhost:8080/v1", {}, "on") is True
    assert is_ollama_endpoint("http://localhost:11434/v1", {}, "off") is False


def _client(
    calls: list[tuple[str, str, Any]],
    *,
    ps: list[dict[str, Any]] | None = None,
    show: dict[str, Any] | None = None,
    ps_fail: bool = False,
    show_fail: bool = False,
    generate_status: int = 200,
) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/api/ps":
            if ps_fail:
                raise httpx.ConnectError("down")
            return httpx.Response(200, json={"models": ps or []})
        if request.url.path == "/api/show":
            if show_fail:
                raise httpx.ConnectError("down")
            return httpx.Response(200, json=show or {})
        if request.url.path == "/api/generate":
            return httpx.Response(generate_status, json={"done": True})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


_SHOW_FULL: dict[str, Any] = {
    "parameters": "num_ctx 8192\nstop <|end|>",
    "model_info": {"gptoss.block_count": 36, "gptoss.context_length": 131072},
}
_SHOW_NATIVE_ONLY: dict[str, Any] = {
    "parameters": "stop <|end|>",
    "model_info": {"gptoss.context_length": 131072},
}


async def test_fetch_context_prefers_loaded_ps() -> None:
    calls: list[tuple[str, str, Any]] = []
    ps = [{"name": "gpt-oss:120b", "context_length": 16384, "expires_at": "2026-08-23T13:00:00Z"}]
    async with _client(calls, ps=ps, show=_SHOW_FULL) as client:
        info = await fetch_model_context("http://ollama", "gpt-oss:120b", client=client)
    assert info.effective == 16384 and info.source == "ps"
    assert info.native == 131072 and info.loaded is True
    assert info.expires_at == "2026-08-23T13:00:00Z"
    assert info.context_length == 16384
    assert [(m, p) for m, p, _ in calls] == [("GET", "/api/ps"), ("POST", "/api/show")]
    assert calls[1][2] == {"model": "gpt-oss:120b"}


async def test_fetch_context_falls_back_to_modelfile_num_ctx() -> None:
    calls: list[tuple[str, str, Any]] = []
    async with _client(calls, ps=[], show=_SHOW_FULL) as client:
        info = await fetch_model_context("http://ollama", "gpt-oss:120b", client=client)
    assert info.effective == 8192 and info.source == "show.parameters"
    assert info.loaded is False and info.native == 131072
    assert info.ps_ok is True


async def test_fetch_context_native_only_when_unloaded() -> None:
    calls: list[tuple[str, str, Any]] = []
    async with _client(calls, ps=[], show=_SHOW_NATIVE_ONLY) as client:
        info = await fetch_model_context("http://ollama", "gpt-oss:120b", client=client)
    assert info.effective is None and info.native == 131072
    assert info.source == "show.model_info" and info.context_length == 131072


async def test_fetch_context_matches_latest_tag() -> None:
    calls: list[tuple[str, str, Any]] = []
    ps = [{"model": "phi4:latest", "context_length": 4096}]
    async with _client(calls, ps=ps, show={}) as client:
        info = await fetch_model_context("http://ollama", "phi4", client=client)
    assert info.effective == 4096 and info.loaded is True


async def test_fetch_context_unreachable_is_harmless() -> None:
    calls: list[tuple[str, str, Any]] = []
    async with _client(calls, ps_fail=True, show_fail=True) as client:
        info = await fetch_model_context("http://ollama", "gpt-oss:120b", client=client)
    assert info.source == "none" and info.context_length is None and info.loaded is False
    assert info.ps_ok is False


async def test_touch_keep_alive_posts_generate() -> None:
    calls: list[tuple[str, str, Any]] = []
    async with _client(calls) as client:
        await touch_keep_alive("http://ollama", "gpt-oss:120b", "30m", client=client)
    assert calls == [("POST", "/api/generate", {"model": "gpt-oss:120b", "keep_alive": "30m"})]


async def test_touch_keep_alive_swallows_failures() -> None:
    calls: list[tuple[str, str, Any]] = []
    async with _client(calls, generate_status=500) as client:
        await touch_keep_alive("http://ollama", "m", "30m", client=client)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
        await touch_keep_alive("http://ollama", "m", "30m", client=client)
