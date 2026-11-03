"""プロバイダ境界の例外変換テスト。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any, NoReturn

import anthropic
import httpx
import openai
import pytest

from satsumaestro.engine.events import MessageStop, ToolUse, ToolUseDelta
from satsumaestro.providers.anthropic_provider import AnthropicProvider
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.openai_provider import OpenAICompatProvider, _assemble_openai
from satsumaestro.providers.retry import is_read_timeout, transport_retryable
from satsumaestro.util.errors import ProviderError


def _req() -> ApiRequest:
    return ApiRequest(model="test-model", system="", messages=[], tools=[], max_tokens=64)


async def test_openai_stream_wraps_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatProvider(api_key="x", base_url="http://localhost:1")
    response = httpx.Response(500, request=httpx.Request("POST", "http://localhost:1"))

    async def boom(**_: Any) -> NoReturn:
        raise openai.InternalServerError("error parsing tool call", response=response, body=None)

    monkeypatch.setattr(provider._client.chat.completions, "create", boom)
    with pytest.raises(ProviderError, match="error parsing tool call"):
        async for _ in provider.stream(_req()):
            pass


async def test_openai_malformed_tool_args_degrades_to_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def chunks() -> AsyncIterator[NS]:
        delta = NS(
            content=None,
            tool_calls=[NS(index=0, id="call_1", function=NS(name="read", arguments="We need"))],
        )
        yield NS(choices=[NS(delta=delta, finish_reason=None)], usage=None)
        done = NS(delta=NS(content=None, tool_calls=None), finish_reason="tool_calls")
        yield NS(choices=[done], usage=None)

    with caplog.at_level(logging.WARNING):
        out = [ev async for ev in _assemble_openai(chunks()) if not isinstance(ev, ToolUseDelta)]

    assert out == [
        ToolUse(id="call_1", name="read", input={}, input_fallback=True),
        MessageStop(stop_reason="tool_use"),
    ]
    assert "not valid JSON" in caplog.text


async def test_openai_boundary_attaches_retry_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatProvider(api_key="x", base_url="http://localhost:1")
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "http://localhost:1"),
        headers={"retry-after": "7"},
    )

    async def boom(**_: Any) -> NoReturn:
        raise openai.RateLimitError("slow down", response=response, body=None)

    monkeypatch.setattr(provider._client.chat.completions, "create", boom)
    with pytest.raises(ProviderError) as excinfo:
        async for _ in provider.stream(_req()):
            pass
    assert excinfo.value.status_code == 429
    assert excinfo.value.retry_after == 7.0


async def test_anthropic_stream_wraps_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider(api_key="x")

    def boom(**_: Any) -> NoReturn:
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://localhost:1"))

    monkeypatch.setattr(provider._client.messages, "stream", boom)
    with pytest.raises(ProviderError, match="Anthropic API 呼び出しに失敗"):
        async for _ in provider.stream(_req()):
            pass


class _BoomStream:
    """1 チャンク返した後に ``httpx.ReadTimeout`` を投げる非同期イテレータ。"""

    def __aiter__(self) -> _BoomStream:
        return self

    async def __anext__(self) -> NS:
        if not getattr(self, "_sent", False):
            self._sent = True
            return NS(
                choices=[NS(delta=NS(content="部分", tool_calls=None), finish_reason=None)],
                usage=None,
            )
        raise httpx.ReadTimeout("timed out", request=httpx.Request("POST", "http://localhost:1"))


async def test_openai_stream_wraps_read_timeout_mid_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OpenAICompatProvider(api_key="x", base_url="http://localhost:1")

    async def create(**_: Any) -> _BoomStream:
        return _BoomStream()

    monkeypatch.setattr(provider._client.chat.completions, "create", create)

    received: list[Any] = []
    with pytest.raises(ProviderError) as excinfo:
        async for ev in provider.stream(_req()):
            received.append(ev)

    assert received
    assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)
    assert excinfo.value.retryable is True
    assert is_read_timeout(excinfo.value)


async def test_anthropic_stream_wraps_read_timeout_mid_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = AnthropicProvider(api_key="x")

    class _Ctx:
        async def __aenter__(self) -> _BoomAnthropic:
            return _BoomAnthropic()

        async def __aexit__(self, *_: Any) -> bool:
            return False

    class _BoomAnthropic:
        def __aiter__(self) -> _BoomAnthropic:
            return self

        async def __anext__(self) -> NoReturn:
            raise httpx.ReadTimeout(
                "timed out", request=httpx.Request("POST", "http://localhost:1")
            )

    async def no_lookup(*_: Any, **__: Any) -> NoReturn:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(provider._client.models, "retrieve", no_lookup)
    monkeypatch.setattr(provider._client.messages, "stream", lambda **_: _Ctx())
    with pytest.raises(ProviderError) as excinfo:
        async for _ in provider.stream(_req()):
            pass
    assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)
    assert is_read_timeout(excinfo.value)


def test_connect_timeout_is_not_classified_as_read_timeout() -> None:
    """接続不能と「応答が返らない」を混同しない（復帰の発火条件を分ける）。"""
    request = httpx.Request("POST", "http://localhost:1")
    connect = ProviderError("x", retryable=True)
    connect.__cause__ = httpx.ConnectTimeout("nope", request=request)
    assert not is_read_timeout(connect)
    assert transport_retryable(connect) is True
