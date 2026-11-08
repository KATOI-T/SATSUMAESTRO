"""プロバイダの種類の選び方と、種類ごとの接続先（通信しない）。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args
from urllib.parse import urlparse

from satsumaestro.config.models import ChosenProvider, ProviderConfig
from satsumaestro.i18n.text import ui
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.detect import choose_provider_kind
from satsumaestro.providers.local import server_for_endpoint
from satsumaestro.util.urls import host_url, ollama_host_url, openai_base

PROVIDER_CHOICES: tuple[ChosenProvider, ...] = get_args(ChosenProvider)
LOCAL_CHOICES: tuple[ChosenProvider, ...] = ("ollama", "lmstudio", "llamacpp")
CHOICE_LABELS: dict[str, str] = {
    "ollama": "Ollama",
    "lmstudio": "LM Studio",
    "llamacpp": "llama.cpp",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "xai": "xAI",
}
HOST_ENV: dict[str, str] = {
    "ollama": "OLLAMA_HOST",
    "lmstudio": "LMSTUDIO_HOST",
    "llamacpp": "LLAMACPP_HOST",
}
DEFAULT_PORTS: dict[str, int] = {"ollama": 11434, "lmstudio": 1234, "llamacpp": 9931}
CLOUD_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
}
OPENAI_API_BASE_URL = "https://api.openai.com/v1"

EndpointSource = Literal[
    "argument",
    "settings",
    "profile",
    "OLLAMA_HOST",
    "LMSTUDIO_HOST",
    "LLAMACPP_HOST",
    "OPENAI_BASE_URL",
    "fixed",
]


@dataclass(frozen=True)
class Endpoint:
    """OpenAI 互換の接続先と、その出どころ。"""

    url: str
    source: EndpointSource


@dataclass(frozen=True)
class ProviderStatus:
    """``/provider`` の一覧の 1 行（ほかの画面の選択肢にも使う）。"""

    choice: ChosenProvider
    current: bool
    endpoint: Endpoint | None
    key_env: str | None
    key_set: bool
    model: str

    @property
    def ready(self) -> bool:
        """切り替えに要る接続先か鍵があるか。"""
        return self.key_set if self.key_env else self.endpoint is not None


def is_choice(value: str) -> bool:
    """``/provider`` で選べる種類の名前か。"""
    return value in PROVIDER_CHOICES


def host_endpoint(choice: str, env: Mapping[str, str]) -> Endpoint | None:
    """種類ごとの ``*_HOST`` から決まる接続先。"""
    var = HOST_ENV.get(choice)
    raw = env.get(var, "").strip() if var else ""
    if not var or not raw:
        return None
    return Endpoint(openai_base(host_url(raw, DEFAULT_PORTS[choice])), _host_source(var))


def resolve_endpoint(provider: ProviderConfig, env: Mapping[str, str]) -> Endpoint | None:
    """今の設定で使う OpenAI 互換の接続先（anthropic と xai は ``None``）。"""
    choice = provider.kind
    if choice == "auto":
        return _auto_endpoint(provider, env)
    if choice == "openai":
        return Endpoint(OPENAI_API_BASE_URL, "fixed")
    if choice in LOCAL_CHOICES:
        if provider.base_url:
            return Endpoint(openai_base(provider.base_url), "settings")
        return host_endpoint(choice, env)
    return None


def switch_endpoint(
    provider: ProviderConfig, choice: str, env: Mapping[str, str], *, current: str | None
) -> Endpoint | None:
    """切り替えに使う接続先（今の種類なら今の接続先、ほかは覚えた URL → ``*_HOST``）。"""
    if choice == "openai":
        return Endpoint(OPENAI_API_BASE_URL, "fixed")
    if choice not in LOCAL_CHOICES:
        return None
    if choice == current and (active := resolve_endpoint(provider, env)) is not None:
        return active
    profile = provider.profiles.get(choice)
    if profile is not None and profile.base_url:
        return Endpoint(openai_base(profile.base_url), "profile")
    return host_endpoint(choice, env)


def current_choice(provider: ProviderConfig, model: str, env: Mapping[str, str]) -> str | None:
    """今の設定で使う種類（``auto`` は推測する。ほかの OpenAI 互換は ``"compatible"``）。"""
    if provider.kind != "auto":
        return provider.kind
    vendor = choose_provider_kind(model, env, provider.base_url)
    if vendor is ProviderKind.ANTHROPIC:
        return "anthropic"
    if vendor is ProviderKind.XAI:
        return "xai"
    endpoint = _auto_endpoint(provider, env)
    if endpoint is None:
        return None
    server = server_for_endpoint(endpoint.url, env, provider)
    if server is not None:
        return server.kind
    host = (urlparse(endpoint.url).hostname or "").lower()
    if host in {h.lower() for h in provider.openai.hosts}:
        return "openai"
    return "compatible"


def provider_statuses(
    provider: ProviderConfig, model: str, env: Mapping[str, str]
) -> list[ProviderStatus]:
    """選べるプロバイダごとの、接続先か鍵の有無と覚えたモデル。"""
    current = current_choice(provider, model, env)
    statuses: list[ProviderStatus] = []
    for choice in PROVIDER_CHOICES:
        key_env = CLOUD_KEYS.get(choice)
        profile = provider.profiles.get(choice)
        remembered = model if choice == current else (profile.model if profile else "")
        statuses.append(
            ProviderStatus(
                choice=choice,
                current=choice == current,
                endpoint=(
                    switch_endpoint(provider, choice, env, current=current)
                    if choice in LOCAL_CHOICES
                    else None
                ),
                key_env=key_env,
                key_set=bool(key_env and env.get(key_env)),
                model=remembered,
            )
        )
    return statuses


def missing_text(choice: str) -> str:
    """切り替えに要る接続先か鍵が無いときの、種類ごとの案内。"""
    if choice == "ollama":
        return ui("providers.choice.no_ollama_endpoint")
    if choice == "lmstudio":
        return ui("providers.choice.no_lmstudio_endpoint")
    if choice == "llamacpp":
        return ui("providers.choice.no_llamacpp_endpoint")
    return ui("providers.choice.key_is_not_set", name=CLOUD_KEYS[choice])


def _auto_endpoint(provider: ProviderConfig, env: Mapping[str, str]) -> Endpoint | None:
    host = env.get("OLLAMA_HOST")
    if provider.base_url:
        from_host = bool(host) and ollama_host_url(host or "") == provider.base_url
        return Endpoint(openai_base(provider.base_url), "OLLAMA_HOST" if from_host else "settings")
    base = env.get("OPENAI_BASE_URL")
    if base:
        return Endpoint(openai_base(base), "OPENAI_BASE_URL")
    if host:
        return Endpoint(openai_base(ollama_host_url(host)), "OLLAMA_HOST")
    return None


def _host_source(var: str) -> EndpointSource:
    if var == "LMSTUDIO_HOST":
        return "LMSTUDIO_HOST"
    return "LLAMACPP_HOST" if var == "LLAMACPP_HOST" else "OLLAMA_HOST"


__all__ = [
    "CHOICE_LABELS",
    "CLOUD_KEYS",
    "DEFAULT_PORTS",
    "HOST_ENV",
    "LOCAL_CHOICES",
    "OPENAI_API_BASE_URL",
    "PROVIDER_CHOICES",
    "Endpoint",
    "EndpointSource",
    "ProviderStatus",
    "current_choice",
    "host_endpoint",
    "is_choice",
    "missing_text",
    "provider_statuses",
    "resolve_endpoint",
    "switch_endpoint",
]
