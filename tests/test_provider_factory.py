"""build_bare / build_provider の種別選択テスト。"""

from __future__ import annotations

import pytest

from satsumaestro.config.models import OllamaConfig, ProviderConfig, RuntimeConfig
from satsumaestro.providers.anthropic_provider import AnthropicProvider
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.factory import build_bare, build_provider
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.util.errors import ProviderError


def test_anthropic_from_model_and_key() -> None:
    p = build_bare(RuntimeConfig(), "claude-opus-4-8", {"ANTHROPIC_API_KEY": "sk-ant-x"})
    assert isinstance(p, AnthropicProvider)
    assert p.kind is ProviderKind.ANTHROPIC
    wrapped = build_provider(RuntimeConfig(), "claude-opus-4-8", {"ANTHROPIC_API_KEY": "sk-ant-x"})
    assert wrapped.kind is ProviderKind.ANTHROPIC


def test_ollama_host_selects_openai_compat() -> None:
    p = build_bare(RuntimeConfig(), "qwen2.5-coder", {"OLLAMA_HOST": "http://localhost:11434"})
    assert isinstance(p, OpenAICompatProvider)
    assert p.kind is ProviderKind.OPENAI


def test_openai_from_model_and_base_url() -> None:
    env = {"OPENAI_API_KEY": "sk-x", "OPENAI_BASE_URL": "https://api.openai.com/v1"}
    p = build_bare(RuntimeConfig(), "gpt-4o", env)
    assert isinstance(p, OpenAICompatProvider)


def test_xai_requires_key() -> None:
    with pytest.raises(ProviderError):
        build_provider(RuntimeConfig(), "grok-2", {})


def test_anthropic_missing_key_raises() -> None:
    with pytest.raises(ProviderError):
        build_provider(RuntimeConfig(), "claude-opus-4-8", {})


def test_openai_unresolved_endpoint_raises() -> None:
    with pytest.raises(ProviderError):
        build_provider(RuntimeConfig(), "gpt-4o", {"OPENAI_API_KEY": "sk-x"})


def test_configured_base_url_wins_over_cloud_keys() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(base_url="http://localhost:11434/v1"))
    p = build_bare(cfg, "qwen3:8b", {"ANTHROPIC_API_KEY": "sk-ant-x", "XAI_API_KEY": "x"})
    assert isinstance(p, OpenAICompatProvider)
    assert p.kind is ProviderKind.OPENAI


def test_base_url_from_config() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(base_url="http://localhost:8080/v1"))
    p = build_bare(cfg, "gpt-4o", {"OPENAI_API_KEY": "sk-x"})
    assert isinstance(p, OpenAICompatProvider)


def test_bare_host_base_url_gets_v1_suffix() -> None:
    from satsumaestro.providers.factory import openai_endpoint

    assert (
        openai_endpoint(RuntimeConfig(), {"OLLAMA_HOST": "http://127.0.0.1:11434"})
        == "http://127.0.0.1:11434/v1"
    )
    bare_cfg = RuntimeConfig(provider=ProviderConfig(base_url="http://127.0.0.1:11434"))
    assert openai_endpoint(bare_cfg, {}) == "http://127.0.0.1:11434/v1"


def test_explicit_version_path_preserved() -> None:
    from satsumaestro.providers.factory import openai_endpoint

    assert (
        openai_endpoint(RuntimeConfig(), {"OPENAI_BASE_URL": "https://api.openai.com/v1"})
        == "https://api.openai.com/v1"
    )
    router = RuntimeConfig(provider=ProviderConfig(base_url="https://openrouter.ai/api/v1"))
    assert openai_endpoint(router, {}) == "https://openrouter.ai/api/v1"


def test_official_openai_is_detected_by_host_but_never_for_ollama() -> None:
    env = {"OPENAI_API_KEY": "sk-x", "OPENAI_BASE_URL": "https://api.openai.com/v1"}
    official = build_bare(RuntimeConfig(), "gpt-5-nano", env)
    assert isinstance(official, OpenAICompatProvider) and official._official is True

    local = build_bare(RuntimeConfig(), "qwen3:8b", {"OLLAMA_HOST": "http://localhost:11434"})
    assert isinstance(local, OpenAICompatProvider) and local._official is False

    forced = ProviderConfig(base_url="https://api.openai.com/v1", ollama=OllamaConfig(detect="on"))
    proxied = build_bare(RuntimeConfig(provider=forced), "gpt-5-nano", {"OPENAI_API_KEY": "sk-x"})
    assert isinstance(proxied, OpenAICompatProvider) and proxied._official is False
