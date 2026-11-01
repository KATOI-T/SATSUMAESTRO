"""モデルの略称と、モデル ID の形から分かるプロバイダのテスト。"""

from __future__ import annotations

import pytest

from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.registry import MODEL_SHORTHANDS, expand_model_name, vendor_for_model


@pytest.mark.parametrize(
    ("shorthand", "expected"),
    [
        ("opus", "claude-opus-4-8"),
        ("sonnet", "claude-sonnet-5"),
        ("haiku", "claude-haiku-4-5-20251001"),
        ("Opus", "claude-opus-4-8"),
        ("HAIKU", "claude-haiku-4-5-20251001"),
    ],
)
def test_shorthand_expands_to_the_model_id(shorthand: str, expected: str) -> None:
    assert expand_model_name(shorthand) == expected


def test_shorthands_are_only_for_anthropic_families() -> None:
    assert sorted(MODEL_SHORTHANDS) == ["haiku", "opus", "sonnet"]
    assert all(vendor_for_model(t) is ProviderKind.ANTHROPIC for t in MODEL_SHORTHANDS.values())


@pytest.mark.parametrize("name", ["claude-opus-4-8", "gpt-oss:120b", "My-Custom-Model", "grok"])
def test_other_names_pass_through_unchanged(name: str) -> None:
    assert expand_model_name(name) == name


@pytest.mark.parametrize(
    ("model", "vendor"),
    [
        ("opus", ProviderKind.ANTHROPIC),
        ("claude-sonnet-5", ProviderKind.ANTHROPIC),
        ("Claude-3-5-Haiku-Latest", ProviderKind.ANTHROPIC),
        ("grok-3", ProviderKind.XAI),
        ("gpt-4o", ProviderKind.OPENAI),
        ("gpt-oss:120b", ProviderKind.OPENAI),
        ("chatgpt-4o-latest", ProviderKind.OPENAI),
        ("o1", ProviderKind.OPENAI),
        ("o3-mini", ProviderKind.OPENAI),
        ("o4-mini", ProviderKind.OPENAI),
        ("opencoder:8b", None),
        ("qwen3.8:27b", None),
        ("grok", None),
        ("", None),
    ],
)
def test_vendor_for_model(model: str, vendor: ProviderKind | None) -> None:
    assert vendor_for_model(model) is vendor
