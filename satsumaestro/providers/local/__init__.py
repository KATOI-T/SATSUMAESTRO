"""ローカルの LLM サーバ（Ollama・LM Studio・llama.cpp）の判定とネイティブ API。"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from satsumaestro.config.models import ProviderConfig
from satsumaestro.providers.detect import (
    LocalKind,
    forget_used_model,
    is_ollama_endpoint,
    ollama_root,
    unload_model,
    used_models,
)
from satsumaestro.providers.local import llamacpp, lmstudio
from satsumaestro.providers.local.base import LocalServer
from satsumaestro.providers.local.llamacpp import LlamaCppServer
from satsumaestro.providers.local.lmstudio import LMStudioServer
from satsumaestro.providers.local.ollama import OllamaServer

__all__ = [
    "LMStudioServer",
    "LlamaCppServer",
    "LocalKind",
    "LocalServer",
    "OllamaServer",
    "Unloaded",
    "looks_local",
    "reset_probed",
    "resolve_server",
    "server_for_endpoint",
    "unload_used",
    "unreached_cause",
]

_log = logging.getLogger(__name__)

PROBE_TIMEOUT_S = 1.5

_PROBED: dict[str, LocalKind | None] = {}
_UNREACHED: dict[str, str] = {}


def reset_probed() -> None:
    """問い合わせの記録を空にする（テスト用）。"""
    _PROBED.clear()
    _UNREACHED.clear()


def unreached_cause(endpoint: str) -> str | None:
    """起動時の問い合わせでつながらなかった接続先なら、その理由。"""
    return _UNREACHED.get(ollama_root(endpoint))


def _lmstudio(root: str, env: Mapping[str, str]) -> LMStudioServer:
    return LMStudioServer(root, token=env.get("LM_API_TOKEN") or None)


def _llamacpp(root: str, env: Mapping[str, str]) -> LlamaCppServer:
    return LlamaCppServer(root, token=env.get("LLAMA_API_KEY") or None)


def _server(kind: LocalKind | None, root: str, env: Mapping[str, str]) -> LocalServer | None:
    if kind == "lmstudio":
        return _lmstudio(root, env)
    if kind == "llamacpp":
        return _llamacpp(root, env)
    return OllamaServer(root) if kind == "ollama" else None


def server_for_endpoint(
    endpoint: str, env: Mapping[str, str], provider: ProviderConfig
) -> LocalServer | None:
    """設定・Ollama の規則・問い合わせの記録から、ローカルのサーバを決める（通信はしない）。"""
    chosen = provider.local.server
    root = ollama_root(endpoint)
    if provider.kind in ("ollama", "lmstudio", "llamacpp"):
        return _server(provider.kind, root, env)
    if provider.kind != "auto" or chosen == "generic":
        return None
    if chosen in ("lmstudio", "llamacpp"):
        return _server(chosen, root, env)
    if chosen == "ollama" or is_ollama_endpoint(endpoint, env, provider.ollama.detect):
        return OllamaServer(root)
    if provider.ollama.detect == "on":
        return None
    return _server(_PROBED.get(root), root, env)


@dataclass(frozen=True)
class Unloaded:
    """降ろしたモデルと、そのサーバの種類。"""

    server: LocalKind
    models: tuple[str, ...]


async def unload_used(provider: ProviderConfig, *, root: str | None = None) -> list[Unloaded]:
    """このプロセスがローカルのサーバで使ったモデルを降ろす（``root`` ならそのサーバだけ）。"""
    done: list[Unloaded] = []
    if provider.ollama.unload_on_exit:
        used = [(r, m) for r, m in used_models() if root is None or r == root]
        results = await asyncio.gather(*(unload_model(r, m) for r, m in used))
        names: list[str] = []
        for (used_root, model), ok in zip(used, results, strict=True):
            if ok:
                forget_used_model(used_root, model)
                names.append(model)
        if names:
            done.append(Unloaded("ollama", tuple(names)))
    if not provider.local.unload_on_exit:
        return done
    names = await lmstudio.unload_models_loaded_here(root=root)
    if names:
        done.append(Unloaded("lmstudio", tuple(names)))
    names = await llamacpp.unload_models_loaded_here(root=root)
    if names:
        done.append(Unloaded("llamacpp", tuple(names)))
    return done


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
        or provider.kind != "auto"
        or provider.local.server != "auto"
        or provider.ollama.detect != "auto"
        or root in _PROBED
        or not looks_local(endpoint)
    ):
        return known
    try:
        _PROBED[root] = await _probe(_lmstudio(root, env), _llamacpp(root, env), client)
        _UNREACHED.pop(root, None)
    except httpx.HTTPError as exc:
        _log.debug("local server probe failed for %s: %s", root, exc)
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            _UNREACHED[root] = f"{type(exc).__name__}: {exc}"
    return server_for_endpoint(endpoint, env, provider)


async def _probe(
    lmstudio: LMStudioServer, llamacpp: LlamaCppServer, client: httpx.AsyncClient | None
) -> LocalKind | None:
    """LM Studio か llama.cpp ならその種類、別のサーバなら ``None``（届かなければ例外）。"""
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=PROBE_TIMEOUT_S)
    try:
        resp = await http.get(f"{lmstudio.root}/api/v1/models", headers=lmstudio.headers())
        if resp.status_code == 200 and _lmstudio_v1(resp):
            return "lmstudio"
        resp = await http.get(f"{llamacpp.root}/props", headers=llamacpp.headers())
        if resp.status_code == 200 and _llamacpp_props(resp):
            return "llamacpp"
        resp = await http.get(f"{lmstudio.root}/api/v0/models", headers=lmstudio.headers())
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


def _llamacpp_props(resp: httpx.Response) -> bool:
    data = _json(resp)
    if not isinstance(data, dict) or not isinstance(data.get("default_generation_settings"), dict):
        return False
    return "total_slots" in data or data.get("role") == "router"


def _lmstudio_v0(resp: httpx.Response) -> bool:
    data = _json(resp)
    models = data.get("data") if isinstance(data, dict) else None
    return isinstance(models, list) and all(
        isinstance(m, dict) and "state" in m and "max_context_length" in m for m in models
    )
