"""ターンの出来事を記録帳へ書く受け手と、``/insights`` の集計。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from satsumaestro.engine.trace import (
    AutoAllowed,
    ModelReplied,
    RecoveryAttempt,
    ToolDispatched,
    ToolReturned,
    TurnAborted,
    TurnBegan,
    TurnEnded,
    TurnEvent,
)
from satsumaestro.engine.turn import TurnSummary
from satsumaestro.session.models import TokenUsage
from satsumaestro.telemetry.log import TraceEntry, TraceLog
from satsumaestro.util.clock import now_ms

_COMMAND_CHARS = 200

_LEFT_OUT = frozenset(
    {"final_text", "last_text", "last_thinking", "health_warning", "decomposition"}
)


class TraceRecorder:
    """出来事を記録帳の 1 件にする受け手（``TurnObserver``）。"""

    def __init__(self, session_id: str, log: TraceLog) -> None:
        self._session_id = session_id
        self._log = log
        self._seq = 0

    def observe(self, event: TurnEvent) -> None:
        described = describe(event)
        if described is None:
            return
        name, data = described
        self._seq += 1
        self._log.add(
            TraceEntry(
                session=self._session_id, seq=self._seq, event=name, at_ms=now_ms(), data=data
            )
        )


def describe(event: TurnEvent) -> tuple[str, dict[str, object]] | None:
    """出来事を記録の名前と中身にする。"""
    match event:
        case TurnBegan():
            return "turn.begin", {"prompt_chars": event.prompt_chars}
        case ModelReplied():
            return "model.reply", {
                "iteration": event.iteration,
                "blocks": event.blocks,
                "tool_calls": event.tool_calls,
            }
        case ToolDispatched():
            return "tool.begin", {"iteration": event.iteration, "tool": event.tool}
        case ToolReturned():
            return "tool.end", {
                "iteration": event.iteration,
                "tool": event.tool,
                "failed": event.failed,
            }
        case RecoveryAttempt():
            step = event.step
            return "recovery", {
                "fault": step.fault.value,
                "remedy": step.remedy.value if step.remedy is not None else None,
                "outcome": step.outcome.value,
                "attempt": step.attempt,
            }
        case AutoAllowed():
            return "permission.auto_allow", {
                "iteration": event.iteration,
                "tool": event.tool,
                "command": event.command[:_COMMAND_CHARS],
            }
        case TurnEnded():
            return "turn.end", summary_fields(event.summary)
        case TurnAborted():
            step_at = getattr(event.error, "step", None)
            return "turn.abort", {
                "iteration": event.iteration,
                "error": type(event.error).__name__,
                "step": list(step_at) if step_at is not None else None,
                "timeout_recoveries": int(getattr(event.error, "timeout_recoveries", 0) or 0),
            }
    return None


def summary_fields(summary: TurnSummary) -> dict[str, object]:
    """ターンの集計を記録の中身にする。"""
    data: dict[str, object] = {}
    for name, info in TurnSummary.model_fields.items():
        if name in _LEFT_OUT:
            continue
        value = getattr(summary, name)
        if isinstance(value, TokenUsage):
            data[name] = value.model_dump()
        elif isinstance(value, list):
            data[name] = ",".join(value) if info.annotation == list[str] else len(value)
        elif value is None or isinstance(value, bool | int | float | str):
            data[name] = value
    return data


@dataclass
class ToolStat:
    calls: int = 0
    errors: int = 0


@dataclass
class Insights:
    """``/insights`` に出す集計。"""

    turns: int = 0
    turns_failed: int = 0
    iterations: list[int] = field(default_factory=list)
    tools: dict[str, ToolStat] = field(default_factory=dict)
    recoveries: int = 0
    auto_allowed: int = 0

    @property
    def iterations_avg(self) -> float:
        return sum(self.iterations) / len(self.iterations) if self.iterations else 0.0


def insights_from(entries: Iterable[TraceEntry]) -> Insights:
    """記録から、ターン数・ツールごとの実行とエラー・往復の数を数える。"""
    ins = Insights()
    for entry in entries:
        data = entry.data
        if entry.event == "turn.end":
            ins.turns += 1
            iterations = data.get("iterations")
            if isinstance(iterations, int):
                ins.iterations.append(iterations)
        elif entry.event == "turn.abort":
            ins.turns_failed += 1
        elif entry.event == "tool.end":
            stat = ins.tools.setdefault(str(data.get("tool", "?")), ToolStat())
            stat.calls += 1
            stat.errors += 1 if data.get("failed") else 0
        elif entry.event == "recovery":
            ins.recoveries += 1
        elif entry.event == "permission.auto_allow":
            ins.auto_allowed += 1
    return ins
