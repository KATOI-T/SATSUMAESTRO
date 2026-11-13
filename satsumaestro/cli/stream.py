"""stream-json 出力（Worker 子 CLI プロトコル）。"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from satsumaestro.engine.trace import (
    AutoAllowed,
    ModelReplied,
    RecoveryAttempt,
    ToolDispatched,
    ToolReturned,
    TurnEvent,
)
from satsumaestro.engine.turn import TurnSummary
from satsumaestro.util.clock import now_ms
from satsumaestro.util.environ import ENVIRON

WIRE_REVISION = 1
RECEIPT_ENV = "SATSUMA_WORKER_RECEIPT"


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def load_receipt(env: Mapping[str, str]) -> dict[str, Any] | None:
    """``SATSUMA_WORKER_RECEIPT`` を防御的に読む（未設定・壊れた JSON は None）。"""
    raw = env.get(RECEIPT_ENV, "")
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def emit_ready(session_id: str, cwd: Path) -> None:
    """必須の先頭行。"""
    _emit(
        {
            "event": "worker.ready",
            "protocol": WIRE_REVISION,
            "session_id": session_id,
            "cwd": str(cwd.resolve()),
            "pid": os.getpid(),
            "receipt": load_receipt(ENVIRON),
            "emitted_at_ms": now_ms(),
        }
    )


def emit_result(summary: TurnSummary, session_id: str, background: list[dict[str, str]]) -> None:
    """終端行（成功）。"""
    _emit(
        {
            "event": "worker.result",
            "final_text": summary.final_text,
            "stop_reason": summary.stop_reason,
            "iterations": summary.iterations,
            "usage": summary.usage.model_dump(),
            "delegated_usage": summary.delegated_usage.model_dump(),
            "compacted": summary.compacted,
            "health_warning": summary.health_warning,
            "background_tasks": background,
            "session_id": session_id,
            "emitted_at_ms": now_ms(),
        }
    )


def emit_error(error: BaseException, session_id: str | None) -> None:
    """終端行（失敗）。"""
    _emit(
        {
            "event": "worker.error",
            "error_type": type(error).__name__,
            "message": str(error),
            "session_id": session_id,
            "emitted_at_ms": now_ms(),
        }
    )


class ProgressLines:
    """ターンの出来事を ``worker.progress`` の行にする受け手。"""

    def observe(self, event: TurnEvent) -> None:
        match event:
            case ModelReplied():
                self._line("iteration", iteration=event.iteration, tool_calls=event.tool_calls)
            case ToolDispatched():
                self._line("tool_start", iteration=event.iteration, tool_name=event.tool)
            case ToolReturned():
                self._line(
                    "tool_end",
                    iteration=event.iteration,
                    tool_name=event.tool,
                    detail="error" if event.failed else "",
                )
            case RecoveryAttempt():
                step = event.step
                self._line("recovery", detail=f"{step.fault.value}:{step.outcome.value}")
            case AutoAllowed():
                self._line("auto_allow", iteration=event.iteration, tool_name=event.tool)

    def _line(self, kind: str, **fields: Any) -> None:
        _emit(
            {
                "event": "worker.progress",
                "kind": kind,
                **{k: v for k, v in fields.items() if v not in (None, "")},
                "emitted_at_ms": now_ms(),
            }
        )
