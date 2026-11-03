"""サンプリング制御のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from satsumaestro.config.models import HookConfig, ProviderConfig, RuntimeConfig, SamplingConfig
from satsumaestro.engine.compaction import Compactor, _summarize_llm
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.openai_provider import _request_kwargs
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ConversationMessage, TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import MockProvider


def _req(sampling: SamplingConfig | None = None) -> ApiRequest:
    return ApiRequest(model="m", system="s", messages=[], tools=[], sampling=sampling)


def test_request_kwargs_omits_unset_sampling() -> None:
    kwargs = _request_kwargs(_req())
    assert kwargs["max_tokens"] == 32000
    assert kwargs["stream_options"] == {"include_usage": True}
    assert not {"temperature", "top_p", "reasoning_effort"} & kwargs.keys()
    assert "temperature" not in _request_kwargs(_req(SamplingConfig()))


def test_request_kwargs_sends_only_set_fields() -> None:
    kwargs = _request_kwargs(_req(SamplingConfig(temperature=0.3)))
    assert kwargs["temperature"] == 0.3 and "top_p" not in kwargs
    kwargs = _request_kwargs(_req(SamplingConfig(top_p=0.9, reasoning_effort="low")))
    assert kwargs["top_p"] == 0.9 and kwargs["reasoning_effort"] == "low"
    assert "temperature" not in kwargs


def test_sampling_config_validation() -> None:
    assert SamplingConfig(reasoning_effort="none").reasoning_effort == "none"
    assert SamplingConfig(reasoning_effort="xhigh").reasoning_effort == "xhigh"
    with pytest.raises(ValidationError):
        SamplingConfig(temperature=3.0)
    with pytest.raises(ValidationError):
        SamplingConfig(reasoning_effort="ultra")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        SamplingConfig.model_validate({"temprature": 0.3})


async def test_runtime_passes_sampling_to_provider(tmp_path: Path) -> None:
    sampling = SamplingConfig(temperature=0.2, reasoning_effort="medium")
    cfg = RuntimeConfig(provider=ProviderConfig(sampling=sampling))
    provider = MockProvider([TextDelta(text="ok"), MessageStop()])
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg,
    )
    await rt.run_turn("hi")
    assert provider.requests[0].sampling == sampling
    assert RuntimeConfig().provider.sampling == SamplingConfig()


async def test_compaction_summary_passes_sampling() -> None:
    sampling = SamplingConfig(temperature=0.1)
    provider = MockProvider([TextDelta(text="summary"), MessageStop()])
    messages = [ConversationMessage(role="user", blocks=[TextBlock(text="x")])]
    text = await _summarize_llm(messages, provider, "m", sampling=sampling)  # type: ignore[arg-type]
    assert text == "summary"
    assert provider.requests[0].sampling == sampling
    assert provider.requests[0].tools == []

    cfg = RuntimeConfig(provider=ProviderConfig(sampling=sampling))
    compactor = Compactor.from_config(cfg, provider=provider, model="m")  # type: ignore[arg-type]
    assert compactor._sampling == sampling
