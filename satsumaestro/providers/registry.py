"""モデルの略称と、モデル ID の形から分かるプロバイダ。"""

from __future__ import annotations

import re

from satsumaestro.providers.base import ProviderKind

MODEL_SHORTHANDS: dict[str, str] = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5-20251001",
}


def expand_model_name(model: str) -> str:
    """略称なら正式なモデル ID に広げる。"""
    return MODEL_SHORTHANDS.get(model.lower(), model)


_VENDOR_NAMES: tuple[tuple[re.Pattern[str], ProviderKind], ...] = (
    (re.compile(r"claude-"), ProviderKind.ANTHROPIC),
    (re.compile(r"grok-"), ProviderKind.XAI),
    (re.compile(r"(?:chat)?gpt-|o[0-9]+(?:-|$)|chat-latest$"), ProviderKind.OPENAI),
)


def vendor_for_model(model: str) -> ProviderKind | None:
    """モデル ID の形から、そのモデルを出しているプロバイダを見当づける。"""
    name = expand_model_name(model).lower()
    return next((kind for pattern, kind in _VENDOR_NAMES if pattern.match(name)), None)
