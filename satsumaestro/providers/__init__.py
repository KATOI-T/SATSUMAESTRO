"""プロバイダ層: LLM プロバイダ抽象・アダプタ・堅牢化ラッパ。"""

from __future__ import annotations

from satsumaestro.providers.anthropic_provider import AnthropicProvider
from satsumaestro.providers.base import ApiRequest, Provider, ProviderKind
from satsumaestro.providers.detect import choose_provider_kind
from satsumaestro.providers.factory import build_bare, build_provider
from satsumaestro.providers.fallback import FallbackProvider
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.providers.prompt_cache import CacheLoss, PromptCacheMonitor
from satsumaestro.providers.registry import (
    MODEL_SHORTHANDS,
    expand_model_name,
    vendor_for_model,
)
from satsumaestro.providers.retry import (
    RetryingProvider,
    RetryPolicy,
    backoff_delay,
    is_retryable,
    is_tool_call_parse_error,
    retry_after_of,
)

__all__ = [
    "MODEL_SHORTHANDS",
    "AnthropicProvider",
    "ApiRequest",
    "CacheLoss",
    "FallbackProvider",
    "OpenAICompatProvider",
    "PromptCacheMonitor",
    "Provider",
    "ProviderKind",
    "RetryPolicy",
    "RetryingProvider",
    "backoff_delay",
    "build_bare",
    "build_provider",
    "choose_provider_kind",
    "expand_model_name",
    "is_retryable",
    "is_tool_call_parse_error",
    "retry_after_of",
    "vendor_for_model",
]
