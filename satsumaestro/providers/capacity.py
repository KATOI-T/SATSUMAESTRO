"""提供者の実効並列能力の推定。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Literal

from satsumaestro.engine.events import closing_stream
from satsumaestro.providers.base import ApiRequest, Provider
from satsumaestro.session.models import ConversationMessage, TextBlock
from satsumaestro.util.clock import now_ms
from satsumaestro.util.jsonl import write_text_atomic

_log = logging.getLogger(__name__)

_PROBE_STEP_RATIO = 0.7
_CACHE_TTL_MS = 24 * 60 * 60 * 1000
_PROBE_PROMPT = "Reply with the single word: ok"
_PROBE_MAX_TOKENS = 16
_OLLAMA_AUTO_PARALLEL = 4

OLLAMA_NUM_PARALLEL_ENV = "OLLAMA_NUM_PARALLEL"


async def estimate_parallelism(
    *,
    configured: int | Literal["auto"],
    is_ollama: bool,
    env: Mapping[str, str],
    cache_path: Path | None,
    cache_key: str,
    probe: Callable[[], Awaitable[int]] | None = None,
    slots: int | None = None,
) -> int | None:
    """提供者側の並列能力を推定する（``slots`` はサーバが答えた同時処理の数）。"""
    if configured != "auto":
        return max(1, configured)
    if slots is not None:
        return max(1, slots)
    if not is_ollama:
        return None
    from_env = _parse_env(env.get(OLLAMA_NUM_PARALLEL_ENV))
    if from_env is not None:
        return from_env
    cached = _cache_get(cache_path, cache_key)
    if cached is not None:
        return cached
    if probe is None:
        return 1
    try:
        measured = max(1, await probe())
    except Exception:
        _log.debug("capacity probe failed; assuming 1", exc_info=True)
        return 1
    _cache_put(cache_path, cache_key, measured)
    return measured


async def probe_parallelism(provider: Provider, model: str, *, limit: int = 4) -> int:
    """校正プローブ: 同時受付の有無を TTFT 階段で測り、1 か min(4, limit) を返す。"""
    width = min(_OLLAMA_AUTO_PARALLEL, limit)
    if width < 2:
        return 1
    await _consume(provider, model)
    baseline_start = time.monotonic()
    await _consume(provider, model)
    baseline = time.monotonic() - baseline_start
    if baseline <= 0.0:
        return 1
    ttfts = sorted(await asyncio.gather(*(_consume(provider, model) for _ in range(width))))
    step = (ttfts[-1] - ttfts[0]) / (width - 1)
    return width if step < baseline * _PROBE_STEP_RATIO else 1


async def _consume(provider: Provider, model: str) -> float:
    """プローブ 1 リクエストを消費し、最初のイベント到達までの秒数（TTFT）を返す。"""
    req = ApiRequest(
        model=model,
        system="",
        messages=[ConversationMessage(role="user", blocks=[TextBlock(text=_PROBE_PROMPT)])],
        tools=[],
        max_tokens=_PROBE_MAX_TOKENS,
        thinking=False,
    )
    start = time.monotonic()
    ttft: float | None = None
    async with closing_stream(provider.stream(req)) as events:
        async for _ in events:
            if ttft is None:
                ttft = time.monotonic() - start
    return ttft if ttft is not None else time.monotonic() - start


def _parse_env(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value >= 1 else None


def _cache_get(path: Path | None, key: str) -> int | None:
    entry = _cache_load(path).get(key)
    if not isinstance(entry, dict):
        return None
    observed_at = entry.get("observed_at")
    parallelism = entry.get("parallelism")
    if not isinstance(observed_at, int) or not isinstance(parallelism, int):
        return None
    if now_ms() - observed_at > _CACHE_TTL_MS:
        return None
    return max(1, parallelism)


def _cache_put(path: Path | None, key: str, parallelism: int) -> None:
    if path is None:
        return
    data = _cache_load(path)
    data[key] = {"parallelism": parallelism, "observed_at": now_ms()}
    try:
        write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))
    except OSError:
        _log.debug("capacity cache write failed: %s", path, exc_info=True)


def _cache_load(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
