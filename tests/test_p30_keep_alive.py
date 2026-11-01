"""モデル常駐制御（keep_alive touch）のテスト。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from satsumaestro.engine.events import closing_stream, fold_events
from satsumaestro.providers import openai_provider
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.util.errors import ProviderError

_ROOT = "http://localhost:11434"


def _text_chunks() -> list[NS]:
    return [
        NS(choices=[NS(delta=NS(content="hi", tool_calls=None), finish_reason=None)], usage=None),
        NS(choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason="stop")], usage=None),
        NS(choices=[], usage=NS(prompt_tokens=1, completion_tokens=1)),
    ]


async def _chunks(items: list[NS]) -> AsyncIterator[NS]:
    for it in items:
        yield it


def _provider(
    monkeypatch: pytest.MonkeyPatch,
    touched: list[tuple[str, str, str]],
    *,
    ollama_root: str | None = _ROOT,
    keep_alive: str | None = "30m",
    fail: bool = False,
) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(
        "k", f"{_ROOT}/v1", ollama_root=ollama_root, keep_alive=keep_alive
    )

    async def fake_create(**kwargs: Any) -> AsyncIterator[NS]:
        if fail:
            raise APIConnectionError(request=httpx.Request("POST", f"{_ROOT}/v1/chat/completions"))
        return _chunks(_text_chunks())

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)

    async def fake_touch(root: str, model: str, keep: str, **_: Any) -> None:
        touched.append((root, model, keep))

    monkeypatch.setattr(openai_provider, "touch_keep_alive", fake_touch)
    return provider


def _req() -> ApiRequest:
    return ApiRequest(model="gpt-oss:120b", system="s", messages=[], tools=[])


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_keep_alive_touched_after_full_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    touched: list[tuple[str, str, str]] = []
    provider = _provider(monkeypatch, touched)
    events = [ev async for ev in provider.stream(_req())]
    assert events[-1].type == "message_stop"
    await _settle()
    assert touched == [(_ROOT, "gpt-oss:120b", "30m")]


async def test_keep_alive_touched_when_fold_events_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[tuple[str, str, str]] = []
    provider = _provider(monkeypatch, touched)
    async with closing_stream(provider.stream(_req())) as events:
        folded = await fold_events(events)
    assert folded.role == "assistant"
    await _settle()
    assert len(touched) == 1


async def test_keep_alive_skipped_without_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    touched: list[tuple[str, str, str]] = []
    provider = _provider(monkeypatch, touched, ollama_root=None)
    _ = [ev async for ev in provider.stream(_req())]
    await _settle()
    assert touched == []
    provider = _provider(monkeypatch, touched, keep_alive=None)
    _ = [ev async for ev in provider.stream(_req())]
    await _settle()
    assert touched == []


async def test_keep_alive_touched_on_truncated_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    touched: list[tuple[str, str, str]] = []
    provider = OpenAICompatProvider("k", f"{_ROOT}/v1", ollama_root=_ROOT, keep_alive="30m")
    truncated = [
        NS(choices=[NS(delta=NS(content="hi", tool_calls=None), finish_reason=None)], usage=None),
    ]

    async def fake_create(**kwargs: Any) -> AsyncIterator[NS]:
        return _chunks(truncated)

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)

    async def fake_touch(root: str, model: str, keep: str, **_: Any) -> None:
        touched.append((root, model, keep))

    monkeypatch.setattr(openai_provider, "touch_keep_alive", fake_touch)

    events = [ev async for ev in provider.stream(_req())]
    assert events[-1].type == "message_stop"
    await _settle()
    assert touched == [(_ROOT, "gpt-oss:120b", "30m")]


async def test_keep_alive_skipped_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    touched: list[tuple[str, str, str]] = []
    provider = _provider(monkeypatch, touched, fail=True)
    with pytest.raises(ProviderError):
        _ = [ev async for ev in provider.stream(_req())]
    await _settle()
    assert touched == []
