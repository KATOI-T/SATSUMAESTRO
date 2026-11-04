"""ローカルの LLM サーバの共通の形。"""

from __future__ import annotations

from typing import Literal, Protocol

import httpx

from satsumaestro.providers.detect import ModelContextInfo

LocalKind = Literal["ollama"]


class LocalServer(Protocol):
    """ローカルの LLM サーバのネイティブ API（届くか・モデルの有無・窓・ロードの状態）。"""

    @property
    def kind(self) -> LocalKind: ...

    @property
    def root(self) -> str: ...

    async def reachable(
        self, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> str | None:
        """届かなければその理由。"""
        ...

    async def has_model(
        self, model: str, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> bool | None:
        """モデルがあるか（決められなければ ``None``）。"""
        ...

    async def list_models(
        self, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> list[str] | None:
        """使えるモデルの名前。"""
        ...

    async def context_info(
        self, model: str, *, client: httpx.AsyncClient | None = None
    ) -> ModelContextInfo:
        """モデルの窓の大きさ。"""
        ...

    async def model_loaded(self, model: str) -> bool | None:
        """モデルが載っているか（分からなければ ``None``）。"""
        ...
