"""リトライラッパと例外分類。"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING

from satsumaestro.engine.events import AssistantEvent, Usage, aclose_quietly
from satsumaestro.providers import notices
from satsumaestro.providers.base import ApiRequest, Provider, ProviderKind
from satsumaestro.providers.prompt_cache import PromptCacheMonitor
from satsumaestro.util.errors import ProviderTimeoutError, RetriesExhaustedError

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetryPolicy:
    """リトライの回数・バックオフ・総時間予算。"""

    max_retries: int = 8
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 16.0
    timeout_s: float = 600.0
    tool_parse_max_retries: int = 1
    read_timeout_max_retries: int = 0


def status_is_transient(status: int) -> bool:
    """HTTP ステータスが、時間を置けば通る失敗を表すか。"""
    return status in (408, 409, 429) or status >= 500


_TOOL_PARSE_ERROR_SIGNATURES = (
    "error parsing tool call",
    "failed to parse tool call",
    "invalid tool call",
)

_TRANSPORT_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "TimeoutException",
    }
)

_READ_TIMEOUT_NAMES = frozenset({"ReadTimeout", "APITimeoutError", "TimeoutException"})


def _causes(exc: BaseException, limit: int = 5) -> Iterator[BaseException]:
    """例外と ``__cause__`` 連鎖を浅い順に辿る（循環・過深は打ち切り）。"""
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(limit):
        if current is None or id(current) in seen:
            return
        seen.add(id(current))
        yield current
        current = current.__cause__


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def status_code_of(exc: BaseException) -> int | None:
    """SDK 例外（``APIStatusError`` 系）の HTTP ステータスを取り出す。"""
    for e in _causes(exc):
        direct = _as_int(getattr(e, "status_code", None))
        if direct is not None:
            return direct
        via_response = _as_int(getattr(getattr(e, "response", None), "status_code", None))
        if via_response is not None:
            return via_response
    return None


def _header(exc: BaseException, name: str) -> str | None:
    """``exc.response.headers`` から 1 ヘッダを安全に読む（response 欠落に耐える）。"""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    get = getattr(headers, "get", None)
    if get is None:
        return None
    try:
        raw = get(name) or get(name.title())
    except Exception:
        return None
    return raw if isinstance(raw, str) else None


def _retry_after_seconds(raw: str) -> float | None:
    """``Retry-After`` を秒へ。"""
    value = raw.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        pass
    else:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def retry_after_of(exc: BaseException) -> float | None:
    """``Retry-After`` 相当の待機秒。"""
    for e in _causes(exc):
        explicit = getattr(e, "retry_after", None)
        if isinstance(explicit, int | float) and not isinstance(explicit, bool):
            return max(0.0, float(explicit))
        raw = _header(e, "retry-after")
        if raw is not None:
            parsed = _retry_after_seconds(raw)
            if parsed is not None:
                return parsed
    return None


def is_retryable(exc: BaseException) -> bool:
    """一過性失敗か。"""
    if isinstance(exc, asyncio.CancelledError | RetriesExhaustedError):
        return False
    for e in _causes(exc):
        flag = getattr(e, "retryable", None)
        if isinstance(flag, bool):
            return flag
        status = _as_int(getattr(e, "status_code", None)) or _as_int(
            getattr(getattr(e, "response", None), "status_code", None)
        )
        if status is not None:
            return status_is_transient(status)
        if type(e).__name__ in _TRANSPORT_ERROR_NAMES:
            return True
    return False


def transport_retryable(exc: BaseException) -> bool | None:
    """プロバイダ境界で ``ProviderError.retryable`` に載せる値を決める。"""
    if any(type(e).__name__ in _TRANSPORT_ERROR_NAMES for e in _causes(exc)):
        return True
    return None


def is_read_timeout(exc: BaseException) -> bool:
    """「送ったのに応答が来ない」タイムアウトか（接続不能・転送断とは区別する）。"""
    return any(type(e).__name__ in _READ_TIMEOUT_NAMES for e in _causes(exc))


def is_connection_failure(exc: BaseException) -> bool:
    """サーバに届かない・接続が途中で切れたことで失敗したか。"""
    return any(
        type(e).__name__ in _TRANSPORT_ERROR_NAMES and type(e).__name__ not in _READ_TIMEOUT_NAMES
        for e in _causes(exc)
    )


def describe(exc: BaseException) -> str:
    """プロバイダ例外の説明文（本文が空でもクラス名を返す）。"""
    text = str(exc).strip()
    return text or type(exc).__name__


def is_tool_call_parse_error(exc: BaseException) -> bool:
    """サーバ側の tool_call パース失敗か（モデル出力の欠陥＝投げ直しても同じ壊れ方をする）。"""
    for e in _causes(exc):
        text = str(e).casefold()
        if any(sig in text for sig in _TOOL_PARSE_ERROR_SIGNATURES):
            return True
    return False


def backoff_delay(attempt: int, policy: RetryPolicy) -> float:
    """``attempt`` 回目の失敗の後に待つ秒。"""
    doublings = min(max(attempt - 1, 0), 32)
    floor = min(policy.initial_backoff_s * 2.0**doublings, policy.max_backoff_s)
    return random.uniform(floor, 2.0 * floor)


class RetryingProvider:
    """一過性失敗を再試行する ``Provider`` デコレータ（責務の重ね順では**内側**）。"""

    kind: ProviderKind

    def __init__(
        self,
        inner: Provider,
        policy: RetryPolicy | None = None,
        *,
        cache_monitor: PromptCacheMonitor | None = None,
    ) -> None:
        self._inner = inner
        self._policy = policy if policy is not None else RetryPolicy()
        self._cache = cache_monitor
        self.kind = inner.kind

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        """``inner.stream`` を中継し、未 yield の一過性失敗のみ再試行する。"""
        policy = self._policy
        began_at = time.monotonic()
        deadline = began_at + policy.timeout_s
        attempt = 0
        last_error: BaseException | None = None

        while True:
            attempt += 1
            if time.monotonic() >= deadline:
                raise self._out_of_time(attempt - 1, began_at, last_error) from last_error

            failure: BaseException | None = None
            started = False
            usage_seen = False
            inner_it = self._inner.stream(req)
            try:
                while True:
                    try:
                        event = await inner_it.__anext__()
                    except StopAsyncIteration:
                        return
                    except Exception as exc:
                        if started or not is_retryable(exc):
                            raise
                        failure = exc
                        break
                    if isinstance(event, Usage) and not usage_seen:
                        usage_seen = True
                        self._observe_usage(req, event)
                    started = True
                    yield event
            finally:
                await aclose_quietly(inner_it)

            if failure is None:
                return
            last_error = failure

            read_timeout = is_read_timeout(failure)
            tool_parse = is_tool_call_parse_error(failure)
            if read_timeout:
                cap = policy.read_timeout_max_retries
            elif tool_parse:
                cap = policy.tool_parse_max_retries
            else:
                cap = policy.max_retries
            if attempt > cap:
                if read_timeout:
                    raise failure
                raise self._exhausted(attempt, began_at, failure) from failure

            delay = retry_after_of(failure)
            if delay is None:
                delay = backoff_delay(attempt, policy)
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or delay >= remaining:
                raise self._out_of_time(attempt, began_at, failure) from failure

            notices.notify(
                notices.ProviderNotice(
                    kind="retry",
                    attempt=attempt,
                    limit=cap + 1,
                    delay_s=delay,
                    reason="tool_parse" if tool_parse else "transient",
                )
            )
            _log.warning(
                "the provider stream failed (%d/%d%s); retrying in %.1fs: %s",
                attempt,
                cap + 1,
                ", limit lowered after a tool_call parse failure"
                if tool_parse
                else (", limit lowered after a silent timeout" if read_timeout else ""),
                delay,
                failure,
            )
            await asyncio.sleep(delay)

    async def count_tokens(self, req: ApiRequest) -> int:
        return await self._inner.count_tokens(req)

    def _exhausted(
        self, attempts: int, began_at: float, last: BaseException | None
    ) -> RetriesExhaustedError:
        return RetriesExhaustedError(
            attempts, time.monotonic() - began_at, last_error=str(last) if last else ""
        )

    def _out_of_time(
        self, attempts: int, began_at: float, last: BaseException | None
    ) -> ProviderTimeoutError:
        return ProviderTimeoutError(
            attempts, time.monotonic() - began_at, last_error=str(last) if last else ""
        )

    def _observe_usage(self, req: ApiRequest, usage: Usage) -> None:
        if self._cache is None:
            return
        loss = self._cache.observe(req, usage)
        if loss is None:
            return
        if loss.unexpected:
            _log.warning(
                "lost the prompt cache unexpectedly: the head of the request is unchanged but "
                "cache_read went %d -> %d (-%d tok)",
                loss.before,
                loss.after,
                loss.lost,
            )
        else:
            _log.debug(
                "lost the prompt cache (%s changed): cache_read %d -> %d",
                loss.changed,
                loss.before,
                loss.after,
            )


if TYPE_CHECKING:  # pragma: no cover

    def _conforms(p: RetryingProvider) -> Provider:
        return p
