"""ターンの出来事を外へ知らせる口。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from satsumaestro.engine.recovery import PlaybookStep
from satsumaestro.engine.turn import TurnSummary

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnBegan:
    """依頼を受けてターンを始めた。"""

    prompt_chars: int


@dataclass(frozen=True)
class ModelReplied:
    """モデルの応答を 1 つ受け取った。"""

    iteration: int
    blocks: int
    tool_calls: int


@dataclass(frozen=True)
class ToolDispatched:
    """ツールの呼び出しを処理し始めた（権限の判定より前。拒否された呼び出しも含む）。"""

    iteration: int
    tool: str


@dataclass(frozen=True)
class ToolReturned:
    """ツールの呼び出しの結果が決まった。"""

    iteration: int
    tool: str
    failed: bool


@dataclass(frozen=True)
class RecoveryAttempt:
    """既知の失敗に手当てを 1 手打った（台帳の 1 行と 1 対 1）。"""

    step: PlaybookStep


@dataclass(frozen=True)
class AutoAllowed:
    """読むだけと確かめられた bash を、確認なしで通した。"""

    iteration: int
    tool: str
    command: str


@dataclass(frozen=True)
class TurnEnded:
    """ターンが終わり、集計が出た。"""

    summary: TurnSummary


@dataclass(frozen=True)
class TurnAborted:
    """ターンが例外で終わった（集計は作られない）。"""

    iteration: int
    error: Exception


TurnEvent = (
    TurnBegan
    | ModelReplied
    | ToolDispatched
    | ToolReturned
    | RecoveryAttempt
    | AutoAllowed
    | TurnEnded
    | TurnAborted
)

Notify = Callable[[TurnEvent], None]


class TurnObserver(Protocol):
    """ターンの出来事の受け手。"""

    def observe(self, event: TurnEvent) -> None: ...


def guarded(observer: TurnObserver | None) -> Notify:
    """受け手に渡す関数を作る。"""
    if observer is None:
        return _ignore

    def notify(event: TurnEvent) -> None:
        try:
            observer.observe(event)
        except Exception:
            _log.debug("observer failed on %s; ignored", type(event).__name__, exc_info=True)

    return notify


def _ignore(event: TurnEvent) -> None:
    return None
