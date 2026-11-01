"""構造化出力の生成側強制（strict tools）のテスト。"""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import httpx
import pytest
from openai import APIStatusError

from satsumaestro.config.models import HookConfig, RuntimeConfig, ToolSchemaConfig
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.detect import capabilities_for, reset_capabilities
from satsumaestro.providers.openai_provider import OpenAICompatProvider, _strict_tools
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.schema import lint_tool_schema, strict_eligible
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import MockProvider

_BASE = "http://strict.test/v1"


@pytest.fixture(autouse=True)
def _fresh_capabilities() -> None:
    reset_capabilities()


_ALL_REQUIRED: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "p"},
        "items": {
            "type": "array",
            "description": "i",
            "items": {"$ref": "#/$defs/Item"},
        },
    },
    "required": ["path", "items"],
    "$defs": {
        "Item": {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "d"}},
            "required": ["id"],
        }
    },
}
_OPTIONAL: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "p"},
        "limit": {"type": "integer", "description": "l"},
    },
    "required": ["path"],
}


def test_strict_eligible_requires_all_properties() -> None:
    assert strict_eligible(_ALL_REQUIRED) is True
    assert strict_eligible(_OPTIONAL) is False
    nested = copy.deepcopy(_ALL_REQUIRED)
    nested["$defs"]["Item"]["required"] = []
    assert strict_eligible(nested) is False


def test_strict_eligible_rejects_freeform_objects() -> None:
    freeform = {
        "type": "object",
        "properties": {"payload": {"type": "object"}},
        "required": ["payload"],
    }
    assert strict_eligible(freeform) is False
    opened = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": True,
    }
    assert strict_eligible(opened) is False
    closed = copy.deepcopy(_ALL_REQUIRED)
    closed["additionalProperties"] = False
    assert strict_eligible(closed) is True


def test_strict_tools_shape() -> None:
    tools = [
        {
            "type": "function",
            "function": {"name": "a", "description": "", "parameters": _ALL_REQUIRED},
        },
        {"type": "function", "function": {"name": "b", "description": "", "parameters": _OPTIONAL}},
    ]
    before = copy.deepcopy(tools)
    out = _strict_tools(tools)
    assert tools == before
    a, b = out
    assert a["function"]["strict"] is True
    assert a["function"]["parameters"]["additionalProperties"] is False
    assert a["function"]["parameters"]["$defs"]["Item"]["additionalProperties"] is False
    assert "strict" not in b["function"]
    assert "additionalProperties" not in b["function"]["parameters"]


def test_lint_logs_strict_ineligible_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="satsumaestro.tools.schema"):
        lint_tool_schema("b", _OPTIONAL)
    assert "strict" in caplog.text and "b" in caplog.text


def _http_error(status: int) -> APIStatusError:
    request = httpx.Request("POST", f"{_BASE}/chat/completions")
    response = httpx.Response(status, request=request)
    return APIStatusError(f"http {status}", response=response, body=None)


def _chunks() -> AsyncIterator[NS]:
    async def gen() -> AsyncIterator[NS]:
        yield NS(
            choices=[NS(delta=NS(content="ok", tool_calls=None), finish_reason=None)], usage=None
        )
        yield NS(
            choices=[NS(delta=NS(content=None, tool_calls=None), finish_reason="stop")], usage=None
        )

    return gen()


def _provider(monkeypatch: pytest.MonkeyPatch, statuses: list[int | None]) -> tuple[Any, list[Any]]:
    """``create`` を差し替える。"""
    provider = OpenAICompatProvider("k", _BASE)
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> AsyncIterator[NS]:
        calls.append(kwargs)
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        if status is not None:
            raise _http_error(status)
        return _chunks()

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    return provider, calls


def _req(strict: bool = True) -> ApiRequest:
    return ApiRequest(
        model="m",
        system="s",
        messages=[],
        tools=[{"name": "a", "description": "", "input_schema": _ALL_REQUIRED}],
        strict_tools=strict,
    )


def _has_strict(call: dict[str, Any]) -> bool:
    return any(t["function"].get("strict") for t in call["tools"])


async def test_strict_400_falls_back_and_caches(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    provider, calls = _provider(monkeypatch, [400, None])
    with caplog.at_level(logging.WARNING):
        events = [ev async for ev in provider.stream(_req())]
    assert isinstance(events[0], TextDelta) and isinstance(events[-1], MessageStop)
    assert len(calls) == 2
    assert _has_strict(calls[0]) and not _has_strict(calls[1])
    assert capabilities_for(_BASE).strict_tools_unsupported is True
    assert "strict" in caplog.text

    _ = [ev async for ev in provider.stream(_req())]
    assert len(calls) == 3 and not _has_strict(calls[2])


async def test_strict_400_twice_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [400, 400])
    with pytest.raises(ProviderError) as exc:
        _ = [ev async for ev in provider.stream(_req())]
    assert exc.value.status_code == 400
    assert len(calls) == 2
    assert capabilities_for(_BASE).strict_tools_unsupported is False


async def test_non_400_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [500])
    with pytest.raises(ProviderError) as exc:
        _ = [ev async for ev in provider.stream(_req())]
    assert exc.value.status_code == 500
    assert len(calls) == 1 and _has_strict(calls[0])


async def test_strict_off_sends_plain_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [None])
    _ = [ev async for ev in provider.stream(_req(strict=False))]
    assert len(calls) == 1 and not _has_strict(calls[0])
    assert "additionalProperties" not in calls[0]["tools"][0]["function"]["parameters"]


async def test_runtime_passes_strict_flag(tmp_path: Path) -> None:
    def make(cfg: RuntimeConfig) -> tuple[ConversationRuntime, MockProvider]:
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
        return rt, provider

    rt, provider = make(RuntimeConfig())
    await rt.run_turn("hi")
    assert provider.requests[0].strict_tools is False
    rt, provider = make(RuntimeConfig(tool_schema=ToolSchemaConfig(strict_tools=True)))
    await rt.run_turn("hi")
    assert provider.requests[0].strict_tools is True
