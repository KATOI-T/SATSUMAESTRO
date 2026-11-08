"""設定 + 環境からプロバイダを生成する。"""

from __future__ import annotations

import os
from collections.abc import Mapping

import httpx

from satsumaestro.config.models import RuntimeConfig, SamplingConfig
from satsumaestro.engine.prefill import keep_alive_seconds
from satsumaestro.i18n.text import ui
from satsumaestro.providers.anthropic_provider import AnthropicProvider
from satsumaestro.providers.base import Provider, ProviderKind
from satsumaestro.providers.choice import (
    CHOICE_LABELS,
    HOST_ENV,
    OPENAI_API_BASE_URL,
    resolve_endpoint,
)
from satsumaestro.providers.detect import choose_provider_kind
from satsumaestro.providers.fallback import FallbackProvider
from satsumaestro.providers.local import LlamaCppServer, LMStudioServer, server_for_endpoint
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.providers.prompt_cache import PromptCacheMonitor
from satsumaestro.providers.registry import expand_model_name
from satsumaestro.providers.retry import RetryingProvider, RetryPolicy
from satsumaestro.util.errors import ProviderError

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
    choice = config.provider.kind
    kind = choose_provider_kind(resolved, env, config.provider.base_url, choice=choice)
    timeout = _timeouts_for(config)
    explicit = choice != "auto"

    if kind is ProviderKind.ANTHROPIC:
        api_key = env.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError(ui("providers.factory.key_is_not_set", name="ANTHROPIC_API_KEY"))
        base_url = None if explicit else config.provider.base_url
        return AnthropicProvider(api_key, base_url, timeout=timeout)

    if kind is ProviderKind.XAI:
        api_key = env.get("XAI_API_KEY")
        if not api_key:
            raise ProviderError(ui("providers.factory.key_is_not_set", name="XAI_API_KEY"))
        base_url = _XAI_BASE_URL if explicit else config.provider.base_url or _XAI_BASE_URL
        return OpenAICompatProvider(
            api_key,
            base_url,
            kind=ProviderKind.XAI,
            timeout=timeout,
            openai=config.provider.openai,
        )

    if choice == "openai":
        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError(ui("providers.factory.key_is_not_set", name="OPENAI_API_KEY"))
        return OpenAICompatProvider(
            api_key,
            OPENAI_API_BASE_URL,
            kind=ProviderKind.OPENAI,
            timeout=timeout,
            openai=config.provider.openai,
        )

    endpoint = openai_endpoint(config, env)
    if not endpoint:
        if choice in HOST_ENV:
            raise ProviderError(
                ui(
                    "providers.factory.no_endpoint_for",
                    name=CHOICE_LABELS[choice],
                    var=HOST_ENV[choice],
                )
            )
        raise ProviderError(ui("providers.factory.openai_endpoint_unresolved"))
    server = server_for_endpoint(endpoint, env, config.provider)
    if isinstance(server, LMStudioServer):
        ttl = keep_alive_seconds(config.provider.local.ttl)
        return OpenAICompatProvider(
            server.token or _local_key(env) or "lm-studio",
            endpoint,
            kind=ProviderKind.OPENAI,
            timeout=timeout,
            extra_body={"ttl": int(ttl)} if ttl else None,
            on_used=server.note_used,
            openai=config.provider.openai,
        )
    if isinstance(server, LlamaCppServer):
        return OpenAICompatProvider(
            server.token or _local_key(env) or "llama.cpp",
            endpoint,
            kind=ProviderKind.OPENAI,
            timeout=timeout,
            on_used=server.note_used,
            openai=config.provider.openai,
        )
    api_key = env.get("OPENAI_API_KEY") or "ollama"
    ollama = config.provider.ollama
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


_CLOUD_KEY_PREFIXES = ("sk-proj-", "sk-svcacct-", "sk-admin-", "sk-ant-", "xai-")


def _local_key(env: Mapping[str, str]) -> str | None:
    """ローカルのサーバに送ってよい ``OPENAI_API_KEY``（クラウドの形の鍵は送らない）。"""
    key = env.get("OPENAI_API_KEY")
    return key if key and not key.startswith(_CLOUD_KEY_PREFIXES) else None


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
    """今の設定で使う OpenAI 互換の接続先（種類ごとの順は :func:`resolve_endpoint`）。"""
    endpoint = resolve_endpoint(config.provider, env)
    return endpoint.url if endpoint is not None else None
