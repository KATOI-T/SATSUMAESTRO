"""恒久失敗時のモデルフォールバック。"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from satsumaestro.engine.events import AssistantEvent, aclose_quietly
from satsumaestro.providers import notices
from satsumaestro.providers.base import ApiRequest, Provider, ProviderKind
from satsumaestro.providers.registry import expand_model_name
from satsumaestro.providers.retry import is_retryable, status_code_of
from satsumaestro.util.errors import RetriesExhaustedError, SatsumaError

_log = logging.getLogger(__name__)

_HOPELESS_STATUS = frozenset({400, 413, 422})


class FallbackProvider:
    """primary 恒久失敗 → ``fallback_model`` へ sticky 切替する ``Provider`` デコレータ。"""

    kind: ProviderKind

    def __init__(
        self,
        primary: Provider,
        build_fallback: Callable[[], Provider],
        fallback_model: str,
    ) -> None:
        self._primary = primary
        self._build_fallback = build_fallback
        self._backup_model = fallback_model
        self._fallback: Provider | None = None
        self.kind = primary.kind

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        """primary を中継し、恒久失敗かつ未 yield なら fallback へ切替えて再ストリームする。"""
        if self._fallback is not None:
            async for event in self._relay(self._fallback, self._switched(req)):
                yield event
            return

        started = False
        permanent: BaseException | None = None
        primary_it = self._primary.stream(req)
        try:
            while True:
                try:
                    event = await primary_it.__anext__()
                except StopAsyncIteration:
                    return
                except Exception as exc:
                    if started or not self._should_switch(exc, req):
                        raise
                    permanent = exc
                    break
                started = True
                yield event
        finally:
            await aclose_quietly(primary_it)

        if permanent is None:
            return

        fallback = self._acquire(permanent)
        notices.notify(
            notices.ProviderNotice(
                kind="fallback", from_model=req.model, to_model=self._backup_model
            )
        )
        _log.warning(
            "the primary %r failed for good; falling back to %r: %s",
            req.model,
            self._backup_model,
            permanent,
        )
        async for event in self._relay(fallback, self._switched(req)):
            yield event

    async def stream_without_switch(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        """今の接続先へそのまま流し、失敗しても切り替えない（本体以外の短い問い合わせ用）。"""
        if self._fallback is not None:
            async for event in self._relay(self._fallback, self._switched(req)):
                yield event
            return
        async for event in self._relay(self._primary, req):
            yield event

    async def count_tokens(self, req: ApiRequest) -> int:
        if self._fallback is not None:
            return await self._fallback.count_tokens(self._switched(req))
        return await self._primary.count_tokens(req)

    def _switched(self, req: ApiRequest) -> ApiRequest:
        """``req.model`` だけを差し替える（``session.model`` は不変）。"""
        return req.model_copy(update={"model": self._backup_model})

    def _should_switch(self, exc: BaseException, req: ApiRequest) -> bool:
        if not self._can_switch(req):
            return False
        if isinstance(exc, RetriesExhaustedError):
            return True
        if is_retryable(exc):
            return False
        return status_code_of(exc) not in _HOPELESS_STATUS

    def _can_switch(self, req: ApiRequest) -> bool:
        """fallback が primary と同一モデルなら切替は無意味（同じ失敗を 2 度払うだけ）。"""
        return (
            expand_model_name(self._backup_model).casefold()
            != expand_model_name(req.model).casefold()
        )

    def _acquire(self, primary_exc: BaseException) -> Provider:
        """fallback を遅延生成し sticky 化する。"""
        provider: Provider | None = None
        try:
            provider = self._build_fallback()
        except SatsumaError as exc:
            _log.error("cannot build the fallback %r: %s", self._backup_model, exc)
        if provider is None:
            raise primary_exc
        self._fallback = provider
        self.kind = provider.kind
        return provider

    async def _relay(self, provider: Provider, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        """fallback のストリームを素通し中継する（再試行しないので async for で足りる）。"""
        it = provider.stream(req)
        try:
            async for event in it:
                yield event
        finally:
            await aclose_quietly(it)


if TYPE_CHECKING:  # pragma: no cover

    def _conforms(p: FallbackProvider) -> Provider:
        return p
