"""OpenAI 公式 API 向けの 400 交渉（引数の調整・effort の強制・ツール外し）のテスト。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace as NS
from typing import Any

import httpx
import pytest
from openai import NOT_GIVEN, APIStatusError

from satsumaestro.config.models import OpenAIConfig, SamplingConfig
from satsumaestro.providers import notices
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.detect import capabilities_for, model_quirks_for, reset_capabilities
from satsumaestro.providers.openai_provider import _MAX_NEGOTIATIONS, OpenAICompatProvider
from satsumaestro.util.errors import ProviderError

_OFFICIAL = "https://api.openai.com/v1"
_CAP_MSG = (
    "Invalid 'max_completion_tokens': integer above maximum value. Expected a value <= 16384."
)
_TOOL = {
    "name": "a",
    "description": "",
    "input_schema": {
        "type": "object",
        "properties": {"p": {"type": "string", "description": "d"}},
        "required": ["p"],
    },
}


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_capabilities()


def _err(
    param: str | None = None, code: str | None = None, message: str = "bad", status: int = 400
) -> APIStatusError:
    request = httpx.Request("POST", f"{_OFFICIAL}/chat/completions")
    response = httpx.Response(status, request=request)
    body = {"param": param, "code": code, "message": message}
    return APIStatusError(message, response=response, body=body)


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
    base: str = _OFFICIAL,
    openai: OpenAIConfig | None = None,
    ollama_root: str | None = None,
) -> tuple[OpenAICompatProvider, list[dict[str, Any]]]:
    provider = OpenAICompatProvider("k", base, openai=openai, ollama_root=ollama_root)
    calls: list[dict[str, Any]] = []

    async def fake_create(**kwargs: Any) -> AsyncIterator[NS]:
        calls.append(kwargs)
        err = errors[min(len(calls) - 1, len(errors) - 1)]
        if err is not None:
            raise err
        return _chunks()

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    return provider, calls


def _req(
    model: str = "gpt-6-luna",
    *,
    tools: bool = True,
    sampling: SamplingConfig | None = None,
    strict: bool = False,
) -> ApiRequest:
    return ApiRequest(
        model=model,
        system="s",
        messages=[],
        tools=[_TOOL] if tools else [],
        sampling=sampling,
        strict_tools=strict,
    )


async def _run(provider: OpenAICompatProvider, req: ApiRequest) -> None:
    async for _ in provider.stream(req):
        pass


async def test_official_host_sends_max_completion_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [None])
    await _run(provider, _req())
    assert len(calls) == 1
    assert calls[0]["max_completion_tokens"] == 32000 and "max_tokens" not in calls[0]


async def test_cap_above_max_is_taken_from_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    cap_error = _err("max_completion_tokens", "integer_above_max_value", _CAP_MSG)
    provider, calls = _provider(monkeypatch, [cap_error, None])
    await _run(provider, _req("gpt-4o"))
    assert len(calls) == 2 and calls[1]["max_completion_tokens"] == 16384
    assert model_quirks_for(_OFFICIAL, "gpt-4o").token_cap == 16384

    await _run(provider, _req("gpt-4o"))
    assert len(calls) == 3 and calls[2]["max_completion_tokens"] == 16384


async def test_cap_error_naming_max_tokens_still_adjusts_the_sent_param(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = (
        "max_tokens is too large: 32000. This model supports at most 16384 completion tokens, "
        "whereas you provided 32000."
    )
    provider, calls = _provider(monkeypatch, [_err("max_tokens", "invalid_value", message), None])
    await _run(provider, _req("gpt-4o-mini"))
    assert len(calls) == 2
    assert calls[1]["max_completion_tokens"] == 16384 and "max_tokens" not in calls[1]


async def test_cap_error_without_a_number_drops_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    cap_error = _err("max_completion_tokens", "integer_above_max_value", "too large")
    provider, calls = _provider(monkeypatch, [cap_error, None])
    await _run(provider, _req("gpt-4o"))
    assert len(calls) == 2 and "max_completion_tokens" not in calls[1]
    assert model_quirks_for(_OFFICIAL, "gpt-4o").no_token_cap is True


async def test_unsupported_temperature_is_dropped_and_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, calls = _provider(monkeypatch, [_err("temperature", "unsupported_value"), None])
    req = _req("gpt-5.5", sampling=SamplingConfig(temperature=0.3))
    await _run(provider, req)
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.3 and "temperature" not in calls[1]
    assert calls[1]["tools"] is not NOT_GIVEN

    await _run(provider, req)
    assert len(calls) == 3 and "temperature" not in calls[2]


async def test_rejected_effort_value_does_not_block_other_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejected = _err("reasoning_effort", "unsupported_value", "max is not supported")
    provider, calls = _provider(monkeypatch, [rejected, None])
    await _run(provider, _req("gpt-5.5", sampling=SamplingConfig(reasoning_effort="max")))
    assert len(calls) == 2 and "reasoning_effort" not in calls[1]
    assert model_quirks_for(_OFFICIAL, "gpt-5.5").rejected_values == {("reasoning_effort", "max")}

    await _run(provider, _req("gpt-5.5", sampling=SamplingConfig(reasoning_effort="high")))
    assert len(calls) == 3 and calls[2]["reasoning_effort"] == "high"

    await _run(provider, _req("gpt-5.5", sampling=SamplingConfig(reasoning_effort="max")))
    assert len(calls) == 4 and "reasoning_effort" not in calls[3]


async def test_tools_force_effort_none_then_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[notices.ProviderNotice] = []
    provider, calls = _provider(monkeypatch, [_err(message="tools need effort none"), None])
    with notices.bind(seen.append):
        await _run(provider, _req())
        await _run(provider, _req())
    assert len(calls) == 3
    assert "reasoning_effort" not in calls[0]
    assert calls[1]["reasoning_effort"] == "none" and calls[1]["tools"] is not NOT_GIVEN
    assert calls[2]["reasoning_effort"] == "none"
    assert model_quirks_for(_OFFICIAL, "gpt-6-luna").tools_need_effort_none is True
    assert [n.kind for n in seen] == ["effort_forced"] and seen[0].model == "gpt-6-luna"


async def test_rejected_effort_none_drops_tools_without_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[notices.ProviderNotice] = []
    errors = [
        _err(message="tools need responses"),
        _err("reasoning_effort", "unsupported_value"),
        None,
    ]
    provider, calls = _provider(monkeypatch, errors)
    req = _req("gpt-6-astra", sampling=SamplingConfig(reasoning_effort="high"))
    with notices.bind(seen.append):
        await _run(provider, req)
        await _run(provider, req)
    assert len(calls) == 4
    assert calls[1]["reasoning_effort"] == "none"
    assert calls[2]["tools"] is NOT_GIVEN and calls[2]["reasoning_effort"] == "high"
    assert calls[3]["tools"] is NOT_GIVEN
    assert model_quirks_for(_OFFICIAL, "gpt-6-astra").tools_unsupported is True
    assert [n.kind for n in seen] == ["tools_dropped"]


async def test_rejected_user_effort_is_not_restored_when_tools_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = [
        _err("reasoning_effort", "unsupported_value", "xhigh is not supported"),
        _err(message="tools need effort none"),
        _err("reasoning_effort", "unsupported_value", "none is not supported"),
        None,
    ]
    provider, calls = _provider(monkeypatch, errors)
    req = _req("gpt-6-astra", sampling=SamplingConfig(reasoning_effort="xhigh"))
    await _run(provider, req)
    assert [c.get("reasoning_effort") for c in calls] == ["xhigh", None, "none", None]
    assert calls[3]["tools"] is NOT_GIVEN
    quirks = model_quirks_for(_OFFICIAL, "gpt-6-astra")
    assert quirks.tools_unsupported is True
    assert quirks.rejected_values == {("reasoning_effort", "xhigh")}


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (
            404,
            "This is not a chat model and thus not supported in the v1/chat/completions endpoint. "
            "Did you mean to use v1/completions?",
        ),
        (400, "This model is only supported in v1/responses and not in v1/chat/completions."),
    ],
)
async def test_responses_only_model_raises_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, status: int, message: str
) -> None:
    provider, calls = _provider(monkeypatch, [_err("model", None, message, status)])
    with pytest.raises(ProviderError) as exc:
        await _run(provider, _req("gpt-5.2-pro"))
    assert len(calls) == 1
    assert exc.value.status_code == status and "gpt-5.2-pro" in str(exc.value)
    assert exc.value.retryable is False


async def test_other_model_errors_are_not_negotiated(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [_err("model", "model_not_found", "no such model")])
    with pytest.raises(ProviderError, match="no such model"):
        await _run(provider, _req("gpt-7"))
    assert len(calls) == 1


def _is_strict(call: dict[str, Any]) -> bool:
    return bool(call["tools"][0]["function"].get("strict"))


async def test_cap_error_on_a_strict_request_keeps_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    cap_error = _err("max_completion_tokens", "integer_above_max_value", _CAP_MSG)
    provider, calls = _provider(monkeypatch, [cap_error, None])
    await _run(provider, _req("gpt-4o", strict=True))
    assert len(calls) == 2
    assert _is_strict(calls[0]) and _is_strict(calls[1])
    assert calls[1]["max_completion_tokens"] == 16384
    assert capabilities_for(_OFFICIAL).strict_tools_unsupported is False


async def test_strict_alone_fixing_the_400_is_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [_err(message="schema"), None])
    await _run(provider, _req(strict=True))
    assert len(calls) == 2
    assert _is_strict(calls[0]) and not _is_strict(calls[1])
    assert "reasoning_effort" not in calls[1]
    assert capabilities_for(_OFFICIAL).strict_tools_unsupported is True


async def test_effort_fix_after_dropping_strict_is_not_blamed_on_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider, calls = _provider(monkeypatch, [_err(message="x"), _err(message="x"), None])
    await _run(provider, _req(strict=True))
    assert len(calls) == 3
    assert _is_strict(calls[0]) and not _is_strict(calls[1]) and not _is_strict(calls[2])
    assert calls[2]["reasoning_effort"] == "none"
    assert capabilities_for(_OFFICIAL).strict_tools_unsupported is False
    assert model_quirks_for(_OFFICIAL, "gpt-6-luna").tools_need_effort_none is True

    await _run(provider, _req(strict=True))
    assert len(calls) == 4
    assert _is_strict(calls[3]) and calls[3]["reasoning_effort"] == "none"


async def test_cap_fix_after_dropping_strict_is_not_blamed_on_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cap_error = _err("max_completion_tokens", "integer_above_max_value", _CAP_MSG)
    provider, calls = _provider(monkeypatch, [_err(message="x"), cap_error, None])
    await _run(provider, _req("gpt-4o", strict=True))
    assert len(calls) == 3
    assert _is_strict(calls[0]) and not _is_strict(calls[1]) and not _is_strict(calls[2])
    assert calls[2]["max_completion_tokens"] == 16384
    assert capabilities_for(_OFFICIAL).strict_tools_unsupported is False


async def test_unknown_400_gives_up_after_effort_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [_err(message="first"), _err(message="later")])
    with pytest.raises(ProviderError, match="first"):
        await _run(provider, _req())
    assert len(calls) == 3
    assert calls[2]["tools"] is NOT_GIVEN
    assert model_quirks_for(_OFFICIAL, "gpt-6-luna").tools_unsupported is False


async def test_repeating_cap_error_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    cap_error = _err("max_completion_tokens", "integer_above_max_value", "<= 100")
    provider, calls = _provider(monkeypatch, [cap_error])
    with pytest.raises(ProviderError):
        await _run(provider, _req())
    assert len(calls) == _MAX_NEGOTIATIONS + 1


async def test_compatible_server_keeps_todays_request(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(monkeypatch, [_err(message="x")], base="http://strict.test/v1")
    with pytest.raises(ProviderError):
        await _run(provider, _req("m"))
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 32000 and "reasoning_effort" not in calls[0]


async def test_ollama_endpoint_is_never_official(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(
        monkeypatch,
        [_err("max_tokens", "unsupported_parameter")],
        base="http://localhost:11434/v1",
        ollama_root="http://localhost:11434",
    )
    with pytest.raises(ProviderError):
        await _run(provider, _req("qwen3:8b"))
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 32000 and calls[0]["tools"] is not NOT_GIVEN


async def test_negotiate_off_only_renames_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, calls = _provider(
        monkeypatch, [_err(message="x")], openai=OpenAIConfig(negotiate=False)
    )
    with pytest.raises(ProviderError):
        await _run(provider, _req())
    assert len(calls) == 1 and calls[0]["max_completion_tokens"] == 32000


async def test_configured_host_counts_as_official(monkeypatch: pytest.MonkeyPatch) -> None:
    tuning = OpenAIConfig(hosts=["my-proxy.example"])
    provider, calls = _provider(
        monkeypatch, [None], base="https://my-proxy.example/v1", openai=tuning
    )
    await _run(provider, _req())
    assert calls[0]["max_completion_tokens"] == 32000
