"""バックグラウンドの実行 1 本（Task の子の会話か worker）で起きたことの記録。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from satsumaestro.util.clock import now_ms


class RunEventKind(StrEnum):
    """何が起きたか。"""

    STARTED = "started"
    READY = "ready"
    PROMPT_MISDELIVERED = "prompt_misdelivered"
    BLOCKED = "blocked"
    FINISHED = "finished"
    FAILED = "failed"
    CLOSED = "closed"

    @property
    def final(self) -> bool:
        """これ以降、この実行に出来事は無い。"""
        return self in (RunEventKind.FINISHED, RunEventKind.FAILED, RunEventKind.CLOSED)


class FailureCause(StrEnum):
    """blocked / failed に添える、何が原因かの分類。"""

    PROMPT_NOT_DELIVERED = "prompt_not_delivered"
    TRUST_PENDING = "trust_pending"
    BRANCH_BEHIND = "branch_behind"
    MCP_INIT = "mcp_init"
    TOOL_FAILURE = "tool_failure"
    OUTSIDE_WORKSPACE = "outside_workspace"
    INFRASTRUCTURE = "infrastructure"


class RunEvent(BaseModel):
    """実行の出来事 1 つ。"""

    kind: RunEventKind
    seq: int
    at: int = Field(default_factory=now_ms)
    cause: FailureCause | None = None
    detail: str | None = None
    data: dict[str, Any] | None = None

    @property
    def final(self) -> bool:
        return self.kind.final

    @classmethod
    def started(cls, seq: int, *, detail: str | None = None) -> RunEvent:
        return cls(kind=RunEventKind.STARTED, seq=seq, detail=detail)

    @classmethod
    def ready(cls, seq: int, *, detail: str | None = None) -> RunEvent:
        return cls(kind=RunEventKind.READY, seq=seq, detail=detail)

    @classmethod
    def prompt_misdelivered(cls, seq: int, *, detail: str | None = None) -> RunEvent:
        return cls(
            kind=RunEventKind.PROMPT_MISDELIVERED,
            seq=seq,
            cause=FailureCause.PROMPT_NOT_DELIVERED,
            detail=detail,
        )

    @classmethod
    def blocked(cls, seq: int, *, cause: FailureCause, detail: str | None = None) -> RunEvent:
        return cls(kind=RunEventKind.BLOCKED, seq=seq, cause=cause, detail=detail)

    @classmethod
    def finished(
        cls, seq: int, *, detail: str | None = None, data: dict[str, Any] | None = None
    ) -> RunEvent:
        return cls(kind=RunEventKind.FINISHED, seq=seq, detail=detail, data=data)

    @classmethod
    def failed(
        cls,
        seq: int,
        *,
        cause: FailureCause,
        detail: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> RunEvent:
        return cls(kind=RunEventKind.FAILED, seq=seq, cause=cause, detail=detail, data=data)

    @classmethod
    def closed(cls, seq: int, *, detail: str | None = None) -> RunEvent:
        return cls(kind=RunEventKind.CLOSED, seq=seq, detail=detail)
