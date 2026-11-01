"""エンジン層: 会話ループ・イベント畳み込み・コンパクション。"""

from __future__ import annotations

from satsumaestro.engine.compaction import CompactionOutcome, Compactor
from satsumaestro.engine.condense import CondenseReport, condense
from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    Usage,
    fold_events,
)
from satsumaestro.engine.recovery import (
    Fallback,
    Fault,
    Playbook,
    PlaybookOutcome,
    PlaybookStep,
    RecoveryLedger,
    Remedy,
    RemedyRunner,
    SessionAbortError,
    fault_of,
    playbook_for,
    run_playbook,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import (
    NullSessionStore,
    RuntimeFactory,
    SubagentPreset,
    resolve_subagent_type,
)
from satsumaestro.engine.tool_check import check_tool_executor
from satsumaestro.engine.turn import (
    DenyPrompter,
    NullRenderer,
    PermissionPrompter,
    Renderer,
    TurnSummary,
)

__all__ = [
    "AssistantEvent",
    "CompactionOutcome",
    "Compactor",
    "CondenseReport",
    "ConversationRuntime",
    "DenyPrompter",
    "Fallback",
    "Fault",
    "MessageStop",
    "NullRenderer",
    "NullSessionStore",
    "PermissionPrompter",
    "Playbook",
    "PlaybookOutcome",
    "PlaybookStep",
    "RecoveryLedger",
    "Remedy",
    "RemedyRunner",
    "Renderer",
    "RuntimeFactory",
    "SessionAbortError",
    "SubagentPreset",
    "TextDelta",
    "Thinking",
    "ToolUse",
    "TurnSummary",
    "Usage",
    "check_tool_executor",
    "condense",
    "fault_of",
    "fold_events",
    "playbook_for",
    "resolve_subagent_type",
    "run_playbook",
]
