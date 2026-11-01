"""LLM 非依存テスト用の ``MockProvider``。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from satsumaestro.engine.events import AssistantEvent
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.util.tokens import estimate_tokens


class MockProvider:
    """スクリプト応答を順に返すプロバイダ実装（``Provider`` Protocol 準拠）。"""

    def __init__(
        self,
        script: list[AssistantEvent],
        kind: ProviderKind = ProviderKind.ANTHROPIC,
    ) -> None:
        self.kind = kind
        self._script = script
        self.requests: list[ApiRequest] = []

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.requests.append(req)
        for event in self._script:
            yield event

    async def count_tokens(self, req: ApiRequest) -> int:
        body = "".join(m.model_dump_json() for m in req.messages)
        return estimate_tokens(req.system_text() + body)


class ScriptedFailureProvider:
    """呼び出しごとに「例外を投げる」か「イベント列を流す」かを切替えるプロバイダ。"""

    def __init__(
        self,
        steps: list[BaseException | list[AssistantEvent | BaseException]],
        kind: ProviderKind = ProviderKind.OPENAI,
    ) -> None:
        self.kind = kind
        self._steps = steps
        self.attempts = 0
        self.requests: list[ApiRequest] = []
        self.aclose_calls = 0

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.requests.append(req)
        idx = min(self.attempts, len(self._steps) - 1)
        self.attempts += 1
        step = self._steps[idx]
        try:
            if isinstance(step, BaseException):
                raise step
            for item in step:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.aclose_calls += 1

    async def count_tokens(self, req: ApiRequest) -> int:
        return estimate_tokens(req.system_text())


class SequencedMockProvider:
    """呼び出しごとに別スクリプトを返すプロバイダ（多ターンのループテスト用）。"""

    def __init__(
        self,
        scripts: list[list[AssistantEvent]],
        kind: ProviderKind = ProviderKind.ANTHROPIC,
    ) -> None:
        if not scripts:
            raise ValueError("scripts must be non-empty")
        self.kind = kind
        self._scripts = scripts
        self._turn = 0
        self.requests: list[ApiRequest] = []

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.requests.append(req)
        idx = min(self._turn, len(self._scripts) - 1)
        self._turn += 1
        for event in self._scripts[idx]:
            yield event

    async def count_tokens(self, req: ApiRequest) -> int:
        body = "".join(m.model_dump_json() for m in req.messages)
        return estimate_tokens(req.system_text() + body)
