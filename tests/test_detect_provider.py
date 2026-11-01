"""リクエストを送るプロバイダの判定のテスト。"""

from __future__ import annotations

import pytest

from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.detect import choose_provider_kind

_ALL_KEYS = {"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "x", "XAI_API_KEY": "x"}


@pytest.mark.parametrize("model", ["gpt-oss:120b", "claude-opus-4-8", "opus", "grok-3", "o3-mini"])
def test_ollama_host_keeps_every_request_local(model: str) -> None:
    env = {"OLLAMA_HOST": "http://localhost:11434", **_ALL_KEYS}
    assert choose_provider_kind(model, env) is ProviderKind.OPENAI


@pytest.mark.parametrize("model", ["qwen3:8b", "claude-opus-4-8", "grok-3"])
def test_configured_base_url_keeps_every_request_local(model: str) -> None:
    base_url = "http://localhost:11434/v1"
    assert choose_provider_kind(model, _ALL_KEYS, base_url) is ProviderKind.OPENAI


@pytest.mark.parametrize(
    ("model", "kind"),
    [
        ("claude-opus-4-8", ProviderKind.ANTHROPIC),
        ("opus", ProviderKind.ANTHROPIC),
        ("grok-3", ProviderKind.XAI),
        ("gpt-4o", ProviderKind.OPENAI),
    ],
)
def test_model_id_shape_decides_before_base_url_and_keys(model: str, kind: ProviderKind) -> None:
    env = {"OPENAI_BASE_URL": "http://localhost:8080/v1", **_ALL_KEYS}
    assert choose_provider_kind(model, env) is kind


def test_openai_base_url_serves_names_without_a_vendor_shape() -> None:
    env = {"OPENAI_BASE_URL": "http://localhost:8080/v1", "ANTHROPIC_API_KEY": "x"}
    assert choose_provider_kind("local-model", env) is ProviderKind.OPENAI


@pytest.mark.parametrize(
    ("keys", "kind"),
    [
        ({"OPENAI_API_KEY": "x"}, ProviderKind.OPENAI),
        ({"ANTHROPIC_API_KEY": "x"}, ProviderKind.ANTHROPIC),
        ({"XAI_API_KEY": "x"}, ProviderKind.XAI),
        (_ALL_KEYS, ProviderKind.OPENAI),
        ({"ANTHROPIC_API_KEY": "x", "XAI_API_KEY": "x"}, ProviderKind.ANTHROPIC),
        ({"ANTHROPIC_API_KEY": "", "XAI_API_KEY": "x"}, ProviderKind.XAI),
    ],
)
def test_keys_decide_names_without_a_vendor_shape(keys: dict[str, str], kind: ProviderKind) -> None:
    assert choose_provider_kind("unknown", keys) is kind


def test_nothing_configured_falls_back_to_openai_compatible() -> None:
    assert choose_provider_kind("unknown", {}) is ProviderKind.OPENAI
