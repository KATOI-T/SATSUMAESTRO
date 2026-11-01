"""コンテキスト長自動整合のテスト。"""

from __future__ import annotations

import json
from typing import Any

import httpx

from satsumaestro.config.models import CompactionConfig, ProviderConfig, RuntimeConfig
from satsumaestro.engine.context_fit import apply_context_fit, fit_compaction_threshold


def test_fit_compaction_threshold_math() -> None:
    assert fit_compaction_threshold(120_000, 16384) == 13107
    assert fit_compaction_threshold(10_000, 16384) is None
    assert fit_compaction_threshold(13107, 16384) is None
    assert fit_compaction_threshold(120_000, None) is None
    assert fit_compaction_threshold(120_000, 16384, enabled=False) is None
    assert fit_compaction_threshold(0, 16384) is None
    assert fit_compaction_threshold(120_000, 131072, ratio=0.5) == 65536


def _client(ps: list[dict[str, Any]], *, fail: bool = False) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if fail:
            raise httpx.ConnectError("down")
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": ps})
        if request.url.path == "/api/show":
            body = json.loads(request.content)
            assert body == {"model": "gpt-oss:120b"}
            return httpx.Response(200, json={"model_info": {"x.context_length": 131072}})
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _cfg(base_url: str = "http://localhost:11434/v1", **kw: Any) -> RuntimeConfig:
    return RuntimeConfig(provider=ProviderConfig(base_url=base_url), **kw)


_PS = [{"name": "gpt-oss:120b", "context_length": 16384}]


async def test_apply_context_fit_clamps_shared_config() -> None:
    cfg = _cfg()
    async with _client(_PS) as client:
        result = await apply_context_fit(cfg, "gpt-oss:120b", {}, client=client)
    assert result is not None and result.clamped
    assert (result.original_threshold, result.effective_threshold) == (120_000, 13107)
    assert cfg.compact_token_threshold == 13107
    assert cfg.model_copy(update={"max_iterations": 3}).compact_token_threshold == 13107
    assert "16384" in result.describe() and "13107" in result.describe()


async def test_apply_context_fit_disabled_keeps_threshold() -> None:
    cfg = _cfg(compaction=CompactionConfig(auto_context_fit=False))
    async with _client(_PS) as client:
        assert await apply_context_fit(cfg, "gpt-oss:120b", {}, client=client) is None
    assert cfg.compact_token_threshold == 120_000


async def test_apply_context_fit_skips_non_ollama() -> None:
    cfg = _cfg("http://localhost:8080/v1")
    async with _client(_PS) as client:
        assert await apply_context_fit(cfg, "gpt-oss:120b", {}, client=client) is None
    assert cfg.compact_token_threshold == 120_000


async def test_apply_context_fit_unreachable_keeps_threshold() -> None:
    cfg = _cfg()
    async with _client(_PS, fail=True) as client:
        result = await apply_context_fit(cfg, "gpt-oss:120b", {}, client=client)
    assert result is not None and not result.clamped
    assert result.context_length is None
    assert cfg.compact_token_threshold == 120_000


async def test_apply_context_fit_uses_native_window_when_unloaded() -> None:
    cfg = _cfg()
    async with _client([]) as client:
        result = await apply_context_fit(cfg, "gpt-oss:120b", {}, client=client)
    assert result is not None and result.info.source == "show.model_info"
    assert cfg.compact_token_threshold == 104857


async def test_apply_context_fit_skips_non_openai_provider() -> None:
    cfg = RuntimeConfig()
    async with _client(_PS, fail=True) as client:
        result = await apply_context_fit(cfg, "claude-3-opus", {}, client=client)
    assert result is None
    assert cfg.compact_token_threshold == 120_000
