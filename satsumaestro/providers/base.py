"""プロバイダ抽象の中核型。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from satsumaestro.config.models import SamplingConfig
from satsumaestro.engine.events import AssistantEvent
from satsumaestro.session.models import ConversationMessage


class ProviderKind(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    XAI = "xai"


class ApiRequest(BaseModel):
    model: str
    system: str
    system_dynamic: str = ""
    messages: list[ConversationMessage]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int = 32000
    thinking: bool = True
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    sampling: SamplingConfig | None = None
    strict_tools: bool = False
    first_byte_timeout_s: float | None = None

    def system_text(self) -> str:
        """静的前半 + 動的後半を 1 本へ連結する（system を分割できない経路用）。"""
        if not self.system_dynamic:
            return self.system
        return self.system + "\n\n" + self.system_dynamic


@runtime_checkable
class Provider(Protocol):
    """LLM プロバイダ抽象。"""

    kind: ProviderKind

    def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]: ...

    async def count_tokens(self, req: ApiRequest) -> int: ...
