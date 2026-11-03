"""Ollama のネイティブ API（``/api/*``）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from satsumaestro.providers.detect import ModelContextInfo, fetch_model_context, model_loaded
from satsumaestro.providers.local.base import LocalKind

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OllamaServer:
    """Ollama のサーバ（``root`` は ``/v1`` を除いたルート）。"""

    root: str

    @property
    def kind(self) -> LocalKind:
        return "ollama"

    async def reachable(
        self, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> str | None:
        """``/api/tags`` に届くか。届かなければその理由。"""
        owned = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
        try:
            resp = await http.get(f"{self.root}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            if owned:
                await http.aclose()
        return None

    async def has_model(
        self, model: str, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> bool | None:
        """``/api/show`` でモデルの有無を確かめる（決められなければ ``None``）。"""
        owned = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
        try:
            resp = await http.post(f"{self.root}/api/show", json={"model": model})
        except httpx.HTTPError as exc:
            _log.debug("Ollama /api/show query failed: %s", exc)
            return None
        finally:
            if owned:
                await http.aclose()
        if resp.status_code == 404:
            return False
        return True if resp.is_success else None

    async def list_models(
        self, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> list[str] | None:
        """``/api/tags`` のモデル名。"""
        owned = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
        try:
            resp = await http.get(f"{self.root}/api/tags")
            resp.raise_for_status()
            models = resp.json().get("models") or []
            names = (m.get("name") or m.get("model") for m in models)
            return [str(n) for n in names if n]
        except Exception as exc:
            _log.debug("Ollama /api/tags query failed: %s", exc)
            return None
        finally:
            if owned:
                await http.aclose()

    async def context_info(
        self, model: str, *, client: httpx.AsyncClient | None = None
    ) -> ModelContextInfo:
        """``/api/ps`` と ``/api/show`` から窓の大きさを読む。"""
        return await fetch_model_context(self.root, model, client=client)

    async def model_loaded(self, model: str) -> bool | None:
        """``/api/ps`` にモデルが載っているか。"""
        return await model_loaded(self.root, model)
