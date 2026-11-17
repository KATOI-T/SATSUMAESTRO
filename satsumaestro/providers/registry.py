"""モデルの略称と、モデル ID の形から分かるプロバイダ。"""

from __future__ import annotations

import re

from satsumaestro.providers.base import ProviderKind

MODEL_SHORTHANDS: dict[str, str] = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5-20251001",
}


RECOMMENDED_MODELS: dict[str, tuple[str, ...]] = {
    "openai": (
        "gpt-6.1-terra",
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.2",
    ),
    "anthropic": (
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-5-5",
        "claude-haiku-4-5-20251001",
    ),
    "xai": (
        "grok-4.7",
        "grok-4.6",
        "grok-4.5",
        "grok-4.3",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "grok-build-0.1",
    ),
}


def expand_model_name(model: str) -> str:
    """略称なら正式なモデル ID に広げる。"""
    return MODEL_SHORTHANDS.get(model.lower(), model)


def recommended_models(choice: str) -> tuple[str, ...]:
    """クラウドのプロバイダで勧めるモデル（SATSUMAESTRO で確かめたもの。ほかの種類は空）。"""
    return RECOMMENDED_MODELS.get(choice, ())


_VENDOR_NAMES: tuple[tuple[re.Pattern[str], ProviderKind], ...] = (
    (re.compile(r"claude-"), ProviderKind.ANTHROPIC),
    (re.compile(r"grok-"), ProviderKind.XAI),
    (re.compile(r"(?:chat)?gpt-|o[0-9]+(?:-|$)|chat-latest$"), ProviderKind.OPENAI),
)


def vendor_for_model(model: str) -> ProviderKind | None:
    """モデル ID の形から、そのモデルを出しているプロバイダを見当づける。"""
    name = expand_model_name(model).lower()
    return next((kind for pattern, kind in _VENDOR_NAMES if pattern.match(name)), None)
