"""ネットワーク非依存の決定論埋め込み。"""

from __future__ import annotations

import hashlib
import re

_DIM = 64
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class MockEmbedder:
    """``Embedder`` と同一シグネチャの決定論スタブ。"""

    def __init__(self, dim: int = _DIM, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail
        self.calls: list[list[str]] = []
        self.closed = False
        self.base_url = "mock://embeddings"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from satsumaestro.util.errors import RagError

        self.calls.append(list(texts))
        if self.fail:
            raise RagError("mock embedder is offline")
        return [self._vector(text) for text in texts]

    async def aclose(self) -> None:
        self.closed = True

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dim] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector
