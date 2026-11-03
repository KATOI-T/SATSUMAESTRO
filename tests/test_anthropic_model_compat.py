"""Anthropic のモデルごとの機能に合わせた要求の形。"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace as NS
from typing import Any, NoReturn

import anthropic
import httpx
import pytest

from satsumaestro.config.models import SamplingConfig
from satsumaestro.providers import notices
from satsumaestro.providers.anthropic_provider import AnthropicProvider
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.detect import anthropic_model_info_for, reset_capabilities
from satsumaestro.session.models import ConversationMessage, TextBlock, ThinkingBlock
from satsumaestro.util.errors import ProviderError

ALL_EFFORT = ("low", "medium", "high", "xhigh", "max")


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    reset_capabilities()
    yield
    reset_capabilities()


def _model(
    *,
    max_tokens: int = 128000,
    thinking: tuple[str, ...] = ("adaptive",),
    effort: tuple[str, ...] = ALL_EFFORT,
) -> NS:
    """Models API の応答を真似たもの（機能の木は ``model_dump`` で返る）。"""
    types = {t: {"supported": t in thinking} for t in ("adaptive", "enabled", "disabled")}
    levels = {lv: {"supported": lv in effort} for lv in ALL_EFFORT}
    tree = {
        "thinking": {"supported": bool(thinking), "types": types},
        "effort": {"supported": bool(effort), **levels},
    }
    return NS(max_tokens=max_tokens, capabilities=NS(model_dump=lambda: tree))


def _req(effort: str | None = None, **kw: Any) -> ApiRequest:
    sampling = SamplingConfig(reasoning_effort=effort) if effort else None
    return ApiRequest(model="claude-x", system="s", messages=[], sampling=sampling, **kw)


class _Stream:
    """``messages.stream`` の代わり。呼ばれた引数を残し、決めた順に失敗か成功を返す。"""

    def __init__(self, failures: list[BaseException] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._failures = list(failures or [])

    def __call__(self, **params: Any) -> _Stream:
        self.calls.append(params)
        return self

    async def __aenter__(self) -> _Stream:
        if self._failures:
            raise self._failures.pop(0)
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> NoReturn:
        raise StopAsyncIteration


def _provider(monkeypatch: pytest.MonkeyPatch, found: Any) -> tuple[AnthropicProvider, list[str]]:
    provider = AnthropicProvider(api_key="x")
    lookups: list[str] = []

    async def retrieve(model: str, **_: Any) -> Any:
        lookups.append(model)
        if isinstance(found, BaseException):
            raise found
        return found

    monkeypatch.setattr(provider._client.models, "retrieve", retrieve)
    return provider, lookups


def _bad_request(message: str) -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(
        message, response=httpx.Response(400, request=request), body=None
    )


async def _drain(provider: AnthropicProvider, req: ApiRequest) -> None:
    async for _ in provider.stream(req):
        pass


async def _params(provider: AnthropicProvider, req: ApiRequest) -> dict[str, Any]:
    info = await provider._model_info(req.model)
    return provider._build_params(req, info)


async def test_current_models_keep_adaptive_thinking_and_the_chosen_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model())
    params = await _params(provider, _req("xhigh"))
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "xhigh"}
    assert params["max_tokens"] == 32000


async def test_models_without_adaptive_thinking_get_a_budget_and_no_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model(max_tokens=64000, thinking=("enabled",), effort=()))
    params = await _params(provider, _req())
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert "output_config" not in params


async def test_effort_falls_to_the_nearest_level_the_model_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model(effort=("low", "medium", "high", "max")))
    assert (await _params(provider, _req("xhigh")))["output_config"] == {"effort": "high"}
    assert (await _params(provider, _req("none")))["output_config"] == {"effort": "low"}
    assert (await _params(provider, _req("minimal")))["output_config"] == {"effort": "low"}


async def test_the_output_cap_and_the_budget_follow_the_model_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model(max_tokens=64000, thinking=("enabled",)))
    params = await _params(provider, _req(max_tokens=128000))
    assert params["max_tokens"] == 64000
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 32000}


async def test_no_thinking_is_sent_when_the_request_turns_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model(thinking=("enabled",)))
    params = await _params(provider, _req(thinking=False, max_tokens=1024))
    assert params["thinking"] is anthropic.NOT_GIVEN


async def test_an_unknown_model_is_sent_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _provider(monkeypatch, httpx.ConnectError("offline"))
    params = await _params(provider, _req())
    assert params["thinking"] == {"type": "adaptive"}
    assert params["output_config"] == {"effort": "high"}


async def test_the_lookup_runs_once_and_a_4xx_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, lookups = _provider(monkeypatch, _model())
    await _params(provider, _req())
    await _params(provider, _req())
    assert lookups == ["claude-x"]

    request = httpx.Request("GET", "https://api.anthropic.com/v1/models/claude-y")
    missing = anthropic.NotFoundError(
        "not found", response=httpx.Response(404, request=request), body=None
    )
    provider, lookups = _provider(monkeypatch, missing)
    await _params(provider, _req().model_copy(update={"model": "claude-y"}))
    await _params(provider, _req().model_copy(update={"model": "claude-y"}))
    assert lookups == ["claude-y"]


async def test_a_network_failure_is_looked_up_again(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, lookups = _provider(monkeypatch, httpx.ConnectError("offline"))
    await _params(provider, _req())
    await _params(provider, _req())
    assert lookups == ["claude-x", "claude-x"]


async def test_a_changed_effort_is_announced_once_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model(effort=("low", "medium", "high")))
    monkeypatch.setattr(provider._client.messages, "stream", _Stream())
    seen: list[notices.ProviderNotice] = []
    with notices.bind(seen.append):
        await _drain(provider, _req("max"))
        await _drain(provider, _req("max"))
    assert [(n.kind, n.requested, n.sent) for n in seen] == [("effort_adjusted", "max", "high")]
    assert "'high'" in notices.describe(seen[0])


async def test_an_effort_the_model_cannot_take_is_announced_as_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model(thinking=("enabled",), effort=()))
    monkeypatch.setattr(provider._client.messages, "stream", _Stream())
    seen: list[notices.ProviderNotice] = []
    with notices.bind(seen.append):
        await _drain(provider, _req("low"))
    assert [(n.kind, n.requested, n.sent) for n in seen] == [("effort_adjusted", "low", "")]


async def test_the_default_effort_is_not_announced(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _provider(monkeypatch, _model(thinking=("enabled",), effort=()))
    monkeypatch.setattr(provider._client.messages, "stream", _Stream())
    seen: list[notices.ProviderNotice] = []
    with notices.bind(seen.append):
        await _drain(provider, _req())
    assert seen == []


UNBOUND = (
    "messages.1.content.0: Invalid `signature` in `thinking` block. "
    "The block is bound to a different conversation."
)


def _history_req() -> ApiRequest:
    messages = [
        ConversationMessage(role="user", blocks=[TextBlock(text="q")]),
        ConversationMessage(
            role="assistant",
            blocks=[ThinkingBlock(thinking="", signature="sig"), TextBlock(text="a")],
        ),
        ConversationMessage(role="user", blocks=[TextBlock(text="next")]),
    ]
    return ApiRequest(model="claude-x", system="s", messages=messages)


async def test_unbound_thinking_is_dropped_by_the_api_and_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model())
    stream = _Stream([_bad_request(UNBOUND)])
    monkeypatch.setattr(provider._client.messages, "stream", stream)
    await _drain(provider, _history_req())
    await _drain(provider, _history_req())

    first, retried, later = stream.calls
    assert "block_binding" not in first["thinking"]
    for params in (retried, later):
        assert params["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
        assert params["extra_headers"] == {"anthropic-beta": "thinking-binding-controls-2026-08-01"}
    info = anthropic_model_info_for(str(provider._client.base_url), "claude-x")
    assert info.drop_unbound_thinking is True


async def test_thinking_is_stripped_when_the_api_cannot_drop_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, _ = _provider(monkeypatch, _model())
    beta_refused = _bad_request("block_binding: Extra inputs are not permitted")
    stream = _Stream([_bad_request(UNBOUND), beta_refused])
    monkeypatch.setattr(provider._client.messages, "stream", stream)
    await _drain(provider, _history_req())

    stripped = stream.calls[-1]
    assert "block_binding" not in stripped["thinking"]
    assert [b["type"] for b in stripped["messages"][1]["content"]] == ["text"]
    info = anthropic_model_info_for(str(provider._client.base_url), "claude-x")
    assert info.drop_unbound_thinking is False


async def test_other_bad_requests_are_not_resent(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _ = _provider(monkeypatch, _model())
    stream = _Stream([_bad_request("max_tokens: too large")])
    monkeypatch.setattr(provider._client.messages, "stream", stream)
    with pytest.raises(ProviderError, match="too large"):
        await _drain(provider, _history_req())
    assert len(stream.calls) == 1
