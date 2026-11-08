"""提供者並列能力の推定（``providers/capacity.py``）のテスト。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta
from satsumaestro.providers import capacity as cap
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.capacity import estimate_parallelism, probe_parallelism
from satsumaestro.util.clock import now_ms


async def test_capacity_auto_ollama_env_then_probe_then_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "capacity.json"
    probe_calls = 0

    async def probe2() -> int:
        nonlocal probe_calls
        probe_calls += 1
        return 2

    def est(**kwargs: object):  # noqa: ANN202
        defaults: dict[str, object] = {
            "configured": "auto",
            "is_ollama": True,
            "env": {},
            "cache_path": cache,
            "cache_key": "http://localhost:11434+gpt-oss:120b",
            "probe": probe2,
        }
        return estimate_parallelism(**{**defaults, **kwargs})  # type: ignore[arg-type]

    assert await est(configured=3, env={"OLLAMA_NUM_PARALLEL": "8"}) == 3
    assert await est(env={"OLLAMA_NUM_PARALLEL": "4"}) == 4
    assert await est(env={"OLLAMA_NUM_PARALLEL": "bogus"}) == 2
    assert probe_calls == 1
    assert await est(env={"OLLAMA_NUM_PARALLEL": "0"}) == 2
    assert probe_calls == 1
    later = now_ms() + 24 * 60 * 60 * 1000 + 1
    monkeypatch.setattr(cap, "now_ms", lambda: later)
    assert await est() == 2
    assert probe_calls == 2
    monkeypatch.undo()
    assert await est(cache_key="other-model", probe=None) == 1

    async def boom() -> int:
        raise RuntimeError("probe crashed")

    assert await est(cache_key="other-model", probe=boom) == 1
    assert await est(is_ollama=False) is None


async def test_capacity_takes_the_slots_the_server_reports() -> None:
    def est(**kwargs: object):  # noqa: ANN202
        defaults: dict[str, object] = {
            "configured": "auto",
            "is_ollama": False,
            "env": {},
            "cache_path": None,
            "cache_key": "http://localhost:9931+m",
        }
        return estimate_parallelism(**{**defaults, **kwargs})  # type: ignore[arg-type]

    assert await est(slots=4) == 4
    assert await est(slots=4, configured=2) == 2
    assert await est(slots=None) is None


class _DelayProvider:
    """prefill（TTFT）と decode を持つプロバイダ。"""

    def __init__(self, *, width: int = 64, prefill_s: float = 0.03, decode_s: float = 0.15) -> None:
        self.kind = ProviderKind.OPENAI
        self._sem = asyncio.Semaphore(width)
        self._prefill_s = prefill_s
        self._decode_s = decode_s
        self.requests = 0

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.requests += 1
        async with self._sem:
            await asyncio.sleep(self._prefill_s)
            yield TextDelta(text="ok")
            await asyncio.sleep(self._decode_s)
        yield MessageStop()

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


async def test_probe_parallelism_ttft_step_rule() -> None:
    assert await probe_parallelism(_DelayProvider(), "m") == 4
    assert await probe_parallelism(_DelayProvider(width=1), "m") == 1
    assert await probe_parallelism(_DelayProvider(width=2), "m") == 4


async def test_probe_parallelism_respects_limit() -> None:
    provider = _DelayProvider()
    assert await probe_parallelism(provider, "m", limit=2) == 2
    assert provider.requests == 4
    provider = _DelayProvider()
    assert await probe_parallelism(provider, "m", limit=1) == 1
    assert provider.requests == 0
