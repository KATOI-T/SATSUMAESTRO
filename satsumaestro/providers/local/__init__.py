"""ローカルの LLM サーバ（Ollama・LM Studio）の判定とネイティブ API。"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Mapping
from urllib.parse import urlparse

import httpx

from satsumaestro.config.models import ProviderConfig
from satsumaestro.providers.detect import LocalKind, is_ollama_endpoint, ollama_root
from satsumaestro.providers.local.base import LocalServer
from satsumaestro.providers.local.lmstudio import LMStudioServer
from satsumaestro.providers.local.ollama import OllamaServer

__all__ = [
    "LMStudioServer",
    "LocalKind",
    "LocalServer",
    "OllamaServer",
    "looks_local",
    "reset_probed",
    "resolve_server",
    "server_for_endpoint",
]

_log = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 1.5

_PROBED: dict[str, LocalKind | None] = {}


def reset_probed() -> None:
    """問い合わせの記録を空にする（テスト用）。"""
    _PROBED.clear()


def _lmstudio(root: str, env: Mapping[str, str]) -> LMStudioServer:
    return LMStudioServer(root, token=env.get("LM_API_TOKEN") or None)


def server_for_endpoint(
    endpoint: str, env: Mapping[str, str], provider: ProviderConfig
) -> LocalServer | None:
    """設定・Ollama の規則・問い合わせの記録から、ローカルのサーバを決める（通信はしない）。"""
    chosen = provider.local.server
    root = ollama_root(endpoint)
    if chosen == "generic":
        return None
    if chosen == "lmstudio":
        return _lmstudio(root, env)
    if chosen == "ollama" or is_ollama_endpoint(endpoint, env, provider.ollama.detect):
        return OllamaServer(root)
    if provider.ollama.detect == "on":
        return None
    return _lmstudio(root, env) if _PROBED.get(root) == "lmstudio" else None


def looks_local(endpoint: str) -> bool:
    """ループバック・プライベートアドレス・``localhost``・``*.local`` のホストか。"""
    host = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}").hostname or ""
    if host == "localhost" or host.endswith((".localhost", ".local")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


async def resolve_server(
    endpoint: str,
    env: Mapping[str, str],
    provider: ProviderConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> LocalServer | None:
    """自動の判定で、ローカルに見えるホストだけに 1 回問い合わせてサーバの種類を記録する。"""
    known = server_for_endpoint(endpoint, env, provider)
    root = ollama_root(endpoint)
    if (
        known is not None
        or provider.local.server != "auto"
        or provider.ollama.detect != "auto"
        or root in _PROBED
        or not looks_local(endpoint)
    ):
        return known
    try:
        _PROBED[root] = await _probe(_lmstudio(root, env), client)
    except httpx.HTTPError as exc:
        _log.debug("local server probe failed for %s: %s", root, exc)
    return server_for_endpoint(endpoint, env, provider)


async def _probe(server: LMStudioServer, client: httpx.AsyncClient | None) -> LocalKind | None:
    """LM Studio なら ``"lmstudio"``、別のサーバなら ``None``（届かなければ例外）。"""
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=PROBE_TIMEOUT_S)
    try:
        resp = await http.get(f"{server.root}/api/v1/models", headers=server.headers())
        if resp.status_code == 200 and _lmstudio_v1(resp):
            return "lmstudio"
        resp = await http.get(f"{server.root}/api/v0/models", headers=server.headers())
        return "lmstudio" if resp.status_code == 200 and _lmstudio_v0(resp) else None
    finally:
        if owned:
            await http.aclose()


def _json(resp: httpx.Response) -> object:
    try:
        return resp.json()
    except ValueError:
        return None


def _lmstudio_v1(resp: httpx.Response) -> bool:
    data = _json(resp)
    models = data.get("models") if isinstance(data, dict) else None
    return isinstance(models, list) and all(
        isinstance(m, dict) and "key" in m and "loaded_instances" in m for m in models
    )


def _lmstudio_v0(resp: httpx.Response) -> bool:
    data = _json(resp)
    models = data.get("data") if isinstance(data, dict) else None
    return isinstance(models, list) and all(
        isinstance(m, dict) and "state" in m and "max_context_length" in m for m in models
    )
