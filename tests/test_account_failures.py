"""429 は再送せず、残高切れ・利用の上限・鍵の拒否を状況の文で知らせる。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import anthropic
import httpx
import openai
import pytest
from typer.testing import CliRunner

import satsumaestro.cli.app as cli_app
from satsumaestro.cli.errors_ui import turn_failure
from satsumaestro.engine.events import TextDelta
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.failures import account_failure
from satsumaestro.providers.retry import RetryingProvider, RetryPolicy
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import ScriptedFailureProvider

_NO_CREDITS = {
    "error": {
        "message": "You have no credits remaining. Add credits to continue using the API at "
        "https://platform.openai.com/settings/organization/billing/.",
        "type": "insufficient_quota",
        "param": None,
        "code": "credit_balance_exhausted",
    }
}


def _sdk_error(
    sdk: str, url: str, status: int, body: dict[str, Any], headers: dict[str, str] | None = None
) -> ProviderError:
    """SDK が応答から作るのと同じ例外を、プロバイダの境界と同じく ``ProviderError`` で包む。"""
    response = httpx.Response(
        status, request=httpx.Request("POST", url), json=body, headers=headers or {}
    )
    if sdk == "anthropic":
        cause = anthropic.AsyncAnthropic(api_key="k")._make_status_error_from_response(response)
    else:
        cause = openai.AsyncOpenAI(api_key="k")._make_status_error_from_response(response)
    error = ProviderError(f"API call failed: {cause}", status_code=status)
    error.__cause__ = cause
    return error


def _openai(status: int, body: dict[str, Any], **kw: Any) -> ProviderError:
    return _sdk_error("openai", "https://api.openai.com/v1/chat/completions", status, body, **kw)


def _anthropic(status: int, message: str, kind: str) -> ProviderError:
    body = {"type": "error", "error": {"type": kind, "message": message}}
    return _sdk_error("anthropic", "https://api.anthropic.com/v1/messages", status, body)


def _req() -> ApiRequest:
    return ApiRequest(model="gpt-5.4-mini", system="sys", messages=[], tools=[])


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


async def test_a_429_is_not_sent_again(slept: list[float]) -> None:
    inner = ScriptedFailureProvider([_openai(429, _NO_CREDITS), [TextDelta(text="ok")]])
    with pytest.raises(ProviderError):
        async for _ in RetryingProvider(inner, RetryPolicy(max_retries=8)).stream(_req()):
            pass
    assert inner.attempts == 1 and slept == []


def test_no_credits_on_openai_says_so_with_the_providers_message() -> None:
    failure = turn_failure(_openai(429, _NO_CREDITS))
    assert failure.reason.startswith("OpenAI のアカウントの残高がありません（429）: You have no")
    assert failure.next_step == ""


def test_no_credits_on_anthropic_and_xai() -> None:
    low = "Your credit balance is too low to access the Anthropic API."
    refused = account_failure(_anthropic(400, low, "invalid_request_error"))
    assert refused is not None and (refused.kind, refused.vendor) == ("no_credits", "anthropic")
    team = "Your team abc has either used all available credits or reached its monthly limit."
    body = {"code": "Some resource has been exhausted", "error": team}
    xai = _sdk_error("openai", "https://api.x.ai/v1/chat/completions", 429, body)
    refused = account_failure(xai)
    assert refused is not None and (refused.kind, refused.vendor) == ("no_credits", "xai")
    assert refused.message == team


def test_a_rate_limit_says_so() -> None:
    body = {"error": {"message": "Rate limit reached for gpt-5.4-mini on tokens per min (TPM)."}}
    failure = turn_failure(_openai(429, body, headers={"retry-after": "12"}))
    assert failure.reason == (
        "OpenAI の利用の上限に達しました（429）: "
        "Rate limit reached for gpt-5.4-mini on tokens per min (TPM)."
    )
    assert failure.next_step == ""


def test_a_rejected_key_says_so() -> None:
    failure = turn_failure(_anthropic(401, "invalid x-api-key", "authentication_error"))
    assert failure.reason == (
        "Anthropic が API の鍵を受け付けませんでした（401）: invalid x-api-key"
    )
    assert failure.next_step == ""


def test_a_rejected_xai_key_comes_as_400() -> None:
    body = {
        "code": "invalid-argument",
        "error": "Incorrect API key provided. You can obtain an API key from https://console.x.ai.",
    }
    xai = _sdk_error("openai", "https://api.x.ai/v1/chat/completions", 400, body)
    assert turn_failure(xai).reason == (
        "xAI が API の鍵を受け付けませんでした（400）: Incorrect API key provided. "
        "You can obtain an API key from https://console.x.ai."
    )


def test_other_errors_keep_the_plain_text() -> None:
    failure = turn_failure(_openai(500, {"error": {"message": "boom"}}))
    assert failure.reason.startswith("モデルとの通信に失敗しました")
    assert account_failure(ProviderError("no response")) is None


def test_prompt_prints_the_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def build(config: object, model: str, env: object) -> ScriptedFailureProvider:
        return ScriptedFailureProvider([_openai(429, _NO_CREDITS)])

    async def fit(cfg: object, model: str, env: object) -> None:
        return None

    monkeypatch.setattr(cli_app, "build_provider", build)
    monkeypatch.setattr(cli_app, "apply_context_fit", fit)
    monkeypatch.setenv("OLLAMA_MODEL", "mock-model")
    result = CliRunner().invoke(cli_app.app, ["prompt", "hi", "--cwd", str(tmp_path)])
    assert result.exit_code == 1
    assert "error: OpenAI のアカウントの残高がありません（429）: You have no" in result.output
