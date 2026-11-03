"""xAI 向けの 400 交渉（本文で名指しされた引数の調整）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import httpx
import pytest
from openai import NOT_GIVEN, APIStatusError

from satsumaestro.config.models import OpenAIConfig, SamplingConfig
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.detect import model_quirks_for, reset_capabilities
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.util.errors import ProviderError

_XAI = "https://api.x.ai/v1"
_TOOL = {
    "name": "a",
    "description": "",
    "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}},
}


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_capabilities()


def _err(text: str, status: int = 400) -> APIStatusError:
    """xAI の 400 を SDK が包んだ形（本文の ``error`` が文字列なので ``param`` は付かない）。"""
    request = httpx.Request("POST", f"{_XAI}/chat/completions")
    response = httpx.Response(status, request=request)
    return APIStatusError(f"Error code: {status} - {text}", response=response, body=text)


def _chunks() -> AsyncIterator[NS]:
    async def gen() -> AsyncIterator[NS]:
        yield NS(
            choices=[NS(delta=NS(content="ok", tool_calls=None), finish_reason="stop")], usage=None
        )

    return gen()


def _provider(
    monkeypatch: pytest.MonkeyPatch,
    errors: list[APIStatusError | None],
    *,
    openai: OpenAIConfig | None = None,
) -> tuple[OpenAICompatProvider, list[dict[str, Any]]]:
    provider = OpenAICompatProvider("k", _XAI, ProviderKind.XAI, openai=openai)
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> AsyncIterator[NS]:
        calls.append(kwargs)
        err = errors[min(len(calls) - 1, len(errors) - 1)]
        if err is not None:
            raise err
        return _chunks()

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    return provider, calls


def _req(model: str = "grok-4.7", effort: str | None = None) -> ApiRequest:
    sampling = SamplingConfig(reasoning_effort=effort) if effort else None
    return ApiRequest(model=model, system="s", messages=[], tools=[_TOOL], sampling=sampling)


async def _run(provider: OpenAICompatProvider, req: ApiRequest) -> None:
    async for _ in provider.stream(req):
        pass


async def test_xai_keeps_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [None])
    await _run(provider, _req())
    assert calls[0]["max_tokens"] == 32000 and "max_completion_tokens" not in calls[0]


async def test_a_model_without_effort_gets_none_and_it_is_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected = _err(
        "{'code': 'invalid-argument', "
        "'error': 'Model grok-build-0.1 does not support parameter reasoningEffort.'}"
    )
    provider, calls = _provider(monkeypatch, [rejected, None])
    await _run(provider, _req("grok-build-0.1", "low"))
    assert [c.get("reasoning_effort") for c in calls] == ["low", None]
    assert calls[1]["tools"] is not NOT_GIVEN
    quirks = model_quirks_for(_XAI, "grok-build-0.1")
    assert quirks.rejected_values == {("reasoning_effort", "low")}

    await _run(provider, _req("grok-build-0.1", "low"))
    assert len(calls) == 3 and "reasoning_effort" not in calls[2]


@pytest.mark.parametrize(
    ("effort", "text"),
    [
        ("none", "This model does not support `reasoning_effort` value `none`."),
        ("max", "Invalid reasoning effort."),
    ],
)
async def test_a_rejected_effort_value_is_dropped_and_others_still_sent(
    monkeypatch: pytest.MonkeyPatch, effort: str, text: str
) -> None:
    rejected = _err(f"{{'code': 'invalid-argument', 'error': '{text}'}}")
    provider, calls = _provider(monkeypatch, [rejected, None])
    await _run(provider, _req(effort=effort))
    assert [c.get("reasoning_effort") for c in calls] == [effort, None]

    await _run(provider, _req(effort="high"))
    assert calls[2]["reasoning_effort"] == "high"


async def test_an_unknown_400_does_not_touch_the_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [_err("prompt is too long")])
    with pytest.raises(ProviderError):
        await _run(provider, _req())
    assert len(calls) == 1


async def test_negotiate_off_also_covers_xai(monkeypatch: pytest.MonkeyPatch) -> None:
    rejected = _err("Invalid reasoning effort.")
    provider, calls = _provider(monkeypatch, [rejected], openai=OpenAIConfig(negotiate=False))
    with pytest.raises(ProviderError):
        await _run(provider, _req(effort="max"))
    assert len(calls) == 1
