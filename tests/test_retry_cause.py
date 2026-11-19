"""再送を待つ行に、失敗の理由（サーバのエラー・接続・応答なし・混雑）を添える。"""

from __future__ import annotations

import asyncio

import httpx
import openai
import pytest

from satsumaestro.engine.events import TextDelta
from satsumaestro.providers import notices
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.notices import ProviderNotice, describe
from satsumaestro.providers.retry import RetryingProvider, RetryPolicy, retry_cause
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import ScriptedFailureProvider

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _wrapped(cause: BaseException, **kw: object) -> ProviderError:
    error = ProviderError(f"API call failed: {cause}", **kw)  # type: ignore[arg-type]
    error.__cause__ = cause
    return error


def _status(status: int) -> ProviderError:
    response = httpx.Response(status, request=_REQUEST, json={"error": {"message": "boom"}})
    cause = openai.AsyncOpenAI(api_key="k")._make_status_error_from_response(response)
    return _wrapped(cause, status_code=status)


def _connection(inner: Exception) -> ProviderError:
    cause = openai.APIConnectionError(request=_REQUEST)
    cause.__cause__ = inner
    return _wrapped(cause, retryable=True)


def test_each_failure_gets_its_cause() -> None:
    assert retry_cause(_status(503)) == ("status", 503)
    assert retry_cause(_connection(httpx.ConnectError("refused"))) == ("connect", 0)
    assert retry_cause(_connection(httpx.RemoteProtocolError("peer closed"))) == ("dropped", 0)
    assert retry_cause(_wrapped(httpx.ReadTimeout("slow"), retryable=True)) == ("timeout", 0)
    busy = openai.APIError("busy", _REQUEST, body={"code": "resource-exhausted"})
    assert retry_cause(_wrapped(busy, retryable=True)) == ("busy", 0)
    assert retry_cause(ProviderError("the stream ended", retryable=True)) == ("", 0)


@pytest.mark.parametrize(
    ("cause", "status", "shown"),
    [
        ("status", 503, "（サーバのエラー 503）"),
        ("connect", 0, "（接続できない）"),
        ("dropped", 0, "（接続が切れた）"),
        ("timeout", 0, "（応答が来ない）"),
        ("busy", 0, "（サーバが混んでいる）"),
        ("", 0, ""),
    ],
)
def test_the_retry_line_says_why(cause: notices.RetryCause, status: int, shown: str) -> None:
    notice = ProviderNotice(
        kind="retry", attempt=1, limit=9, delay_s=2.4, cause=cause, status=status
    )
    assert describe(notice) == f"モデルの応答に失敗したので{shown}、2 秒後にもう一度試します（1/9）"


async def test_the_retrying_provider_sends_the_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_wait(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    seen: list[ProviderNotice] = []
    inner = ScriptedFailureProvider([_status(502), [TextDelta(text="ok")]])
    request = ApiRequest(model="gpt-5.4-mini", system="sys", messages=[], tools=[])
    with notices.bind(seen.append):
        events = [e async for e in RetryingProvider(inner, RetryPolicy()).stream(request)]
    assert events == [TextDelta(text="ok")]
    assert [(n.cause, n.status) for n in seen] == [("status", 502)]
    assert "（サーバのエラー 502）" in describe(seen[0])
