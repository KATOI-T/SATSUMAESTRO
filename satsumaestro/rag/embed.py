"""OpenAI 互換 ``/embeddings`` クライアント。"""

from __future__ import annotations

import asyncio
import os
from typing import Protocol

import httpx

from satsumaestro.util.errors import RagError

DEFAULT_EMBED_ENDPOINT = "http://localhost:11434/v1"
DEFAULT_BATCH = 64
_EMBED_TRIES = 3
_BACKOFF_BASE_S = 0.5
_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class EmbedderLike(Protocol):
    """埋め込み器の最小契約。"""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


class Embedder:
    """OpenAI 互換 ``/embeddings`` のバッチクライアント。"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "bge-m3",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        batch_size: int = DEFAULT_BATCH,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = base_url or os.environ.get("SATSUMA_EMBED_BASE_URL") or DEFAULT_EMBED_ENDPOINT
        self.base_url = resolved.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("SATSUMA_EMBED_API_KEY")
        self.timeout_s = timeout_s
        self.batch_size = max(1, batch_size)
        self._client: httpx.AsyncClient | None = client
        self._batch_supported = True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """テキスト列を埋め込みベクトル列へ変換する（入力と同じ順序を保証）。"""
        if not texts:
            return []
        vectors: list[list[float]] = []
        pos = 0
        while pos < len(texts):
            size = self.batch_size if self._batch_supported else 1
            batch = texts[pos : pos + size]
            try:
                vectors.extend(await self._embed_group(batch))
            except RagError:
                if len(batch) == 1:
                    raise
                self._batch_supported = False
                continue
            pos += len(batch)
        return vectors

    async def _embed_group(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, object] = {"model": self.model, "input": batch}
        data = await self._post_with_retry(payload)
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(batch):
            raise RagError(
                f"embeddings response mismatch: expected {len(batch)} vectors, "
                f"got {len(items) if isinstance(items, list) else 'none'}"
            )
        ordered: list[list[float]] = [[] for _ in batch]
        for pos, item in enumerate(items):
            if not isinstance(item, dict):
                raise RagError("embeddings response contains a non-object entry")
            idx = item.get("index", pos)
            vector = item.get("embedding")
            if not isinstance(idx, int) or not 0 <= idx < len(batch):
                raise RagError(f"embeddings response has an out-of-range index: {idx!r}")
            if not isinstance(vector, list) or not vector:
                raise RagError("embeddings response contains an empty embedding")
            ordered[idx] = [float(x) for x in vector]
        if any(not v for v in ordered):
            raise RagError("embeddings response is missing one or more vectors")
        return ordered

    async def _post_with_retry(self, payload: dict[str, object]) -> dict[str, object]:
        client = self._ensure_client()
        last: Exception | None = None
        for attempt in range(_EMBED_TRIES):
            try:
                resp = await client.post("/embeddings", json=payload)
            except httpx.HTTPError as e:
                last = e
            else:
                if resp.status_code < 400:
                    body = resp.json()
                    if not isinstance(body, dict):
                        raise RagError("embeddings response is not a JSON object")
                    return body
                detail = resp.text[:200].strip()
                error = RagError(f"embeddings HTTP {resp.status_code}: {detail}")
                if resp.status_code not in _RETRY_STATUS:
                    raise error
                last = error
            if attempt < _EMBED_TRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE_S * (2**attempt))
        raise RagError(f"embeddings unreachable at {self.base_url}: {last}") from last

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout_s, headers=headers
            )
        return self._client

    async def aclose(self) -> None:
        """HTTP クライアントを閉じる（多重呼び出し安全）。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
