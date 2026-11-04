"""設定 + 環境からプロバイダを生成する。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlparse

import httpx

from satsumaestro.config.models import RuntimeConfig, SamplingConfig
from satsumaestro.i18n.text import ui
from satsumaestro.providers.anthropic_provider import AnthropicProvider
from satsumaestro.providers.base import Provider, ProviderKind
from satsumaestro.providers.detect import choose_provider_kind
from satsumaestro.providers.fallback import FallbackProvider
from satsumaestro.providers.local import server_for_endpoint
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.providers.prompt_cache import PromptCacheMonitor
from satsumaestro.providers.registry import expand_model_name
from satsumaestro.providers.retry import RetryingProvider, RetryPolicy
from satsumaestro.util.errors import ProviderError
from satsumaestro.util.urls import ollama_host_url

_XAI_BASE_URL = "https://api.x.ai/v1"

_CONNECT_TIMEOUT_S = 10.0


def build_bare(
    config: RuntimeConfig,
    model: str,
    env: Mapping[str, str] | None = None,
) -> Provider:
    """モデル ID・設定・環境から**素の** :class:`Provider` を生成する。"""
    env = env if env is not None else os.environ
    resolved = expand_model_name(model)
    kind = choose_provider_kind(resolved, env, config.provider.base_url)
    timeout = _timeouts_for(config)

    if kind is ProviderKind.ANTHROPIC:
        api_key = env.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError(ui("providers.factory.key_is_not_set", name="ANTHROPIC_API_KEY"))
        return AnthropicProvider(api_key, config.provider.base_url, timeout=timeout)

    if kind is ProviderKind.XAI:
        api_key = env.get("XAI_API_KEY")
        if not api_key:
            raise ProviderError(ui("providers.factory.key_is_not_set", name="XAI_API_KEY"))
        base_url = config.provider.base_url or _XAI_BASE_URL
        return OpenAICompatProvider(
            api_key,
            base_url,
            kind=ProviderKind.XAI,
            timeout=timeout,
            openai=config.provider.openai,
        )

    endpoint = openai_endpoint(config, env)
    if not endpoint:
        raise ProviderError(ui("providers.factory.openai_endpoint_unresolved"))
    api_key = env.get("OPENAI_API_KEY") or "ollama"
    ollama = config.provider.ollama
    server = server_for_endpoint(endpoint, env, ollama.detect)
    root = server.root if server is not None and server.kind == "ollama" else None
    return OpenAICompatProvider(
        api_key,
        endpoint,
        kind=ProviderKind.OPENAI,
        timeout=timeout,
        ollama_root=root,
        keep_alive=ollama.keep_alive or None,
        openai=config.provider.openai,
    )


def build_provider(
    config: RuntimeConfig,
    model: str,
    env: Mapping[str, str] | None = None,
) -> Provider:
    """堅牢化済みの :class:`Provider` を生成する。"""
    env = env if env is not None else os.environ
    resolved = expand_model_name(model)
    policy = _policy(config)
    primary = _with_retry(build_bare(config, resolved, env), config, policy)

    configured = config.provider.fallback_model
    if not configured:
        return primary
    fallback_model = expand_model_name(configured)
    if fallback_model.casefold() == resolved.casefold():
        return primary

    return FallbackProvider(
        primary,
        build_fallback=lambda: _with_retry(build_bare(config, fallback_model, env), config, policy),
        fallback_model=fallback_model,
    )


def resolve_role_model(config: RuntimeConfig, role: str, *, default: str) -> str:
    """役割のモデルを ``provider.roles.<role>`` → ``default`` の順で解決する。"""
    configured = getattr(config.provider.roles, role, None)
    return expand_model_name(configured or default)


def resolve_role_sampling(config: RuntimeConfig, role: str) -> SamplingConfig:
    """役割のサンプリングを ``roles.sampling_overrides`` → ``provider.sampling`` で解決する。"""
    override = config.provider.roles.sampling_overrides.get(role)
    return override if override is not None else config.provider.sampling


def _policy(config: RuntimeConfig) -> RetryPolicy:
    retry = config.provider.retry
    return RetryPolicy(
        max_retries=retry.max_retries,
        initial_backoff_s=retry.initial_backoff_s,
        max_backoff_s=retry.max_backoff_s,
        timeout_s=config.api_timeout_s,
        tool_parse_max_retries=retry.tool_parse_max_retries,
        read_timeout_max_retries=retry.read_timeout_max_retries,
    )


def _timeouts_for(config: RuntimeConfig) -> httpx.Timeout:
    """1 回の API 呼び出しに与えるタイムアウト。"""
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT_S, read=config.read_timeout_s(), write=30.0, pool=10.0
    )


def _with_retry(bare: Provider, config: RuntimeConfig, policy: RetryPolicy) -> RetryingProvider:
    return RetryingProvider(
        bare,
        policy,
        cache_monitor=PromptCacheMonitor(config.provider.retry.cache_break_min_drop),
    )


def openai_endpoint(config: RuntimeConfig, env: Mapping[str, str]) -> str | None:
    """OpenAI 互換エンドポイントを設定 → OPENAI_BASE_URL → OLLAMA_HOST の順で解決する。"""
    host = env.get("OLLAMA_HOST")
    raw = (
        config.provider.base_url
        or env.get("OPENAI_BASE_URL")
        or (ollama_host_url(host) if host else None)
    )
    return _normalize_openai_base(raw) if raw else None


def _normalize_openai_base(url: str) -> str:
    """裸ホストには ``/v1`` を補い、明示パスはそのまま返す。"""
    trimmed = url.strip().rstrip("/")
    if "://" not in trimmed:
        trimmed = f"http://{trimmed}"
    if not urlparse(trimmed).path:
        return trimmed + "/v1"
    return trimmed
