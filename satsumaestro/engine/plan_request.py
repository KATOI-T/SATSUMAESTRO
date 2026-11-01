"""plan モードのターンで、依頼の末尾に plan モードの通知を付ける。"""

from __future__ import annotations

import re

from satsumaestro.engine.plan_steps import plan_steps_request
from satsumaestro.session.models import ConversationMessage, TextBlock, ToolResultBlock

_WANTS_PLAN = re.compile(r"(計画|プラン|段取り|\bplan(s|ned|ning)?\b)", re.IGNORECASE)

PLAN_FILE_NEW = "not created yet — create it with write_file; its directory already exists"
PLAN_FILE_EXISTS = "it already exists — read it before updating"
PLAN_FILE_READ_MISSING = (
    "{plan_file} does not exist yet — you have not written the plan. Nothing to read: "
    "write the plan there with write_file (its directory already exists), then call "
    "exit_plan_mode to ask the user for approval. Do not read other files under "
    ".satsuma/plans/ — they belong to other sessions."
)

PLAN_REQUEST_SUFFIX = (
    "\n\n(Plan mode: treat this as a request to plan first, not to implement. Investigate "
    "with read-only tools, write the plan to {plan_file} with write_file ({state}), "
    "then call exit_plan_mode to ask the user for approval. Do not write other files "
    "or run commands until the plan is approved. Other files under .satsuma/plans/ belong "
    "to other sessions; do not read or reuse them."
    "{steps})"
)
PLAN_MODE_SUFFIX = (
    "\n\n(Plan mode is active: investigate with read-only tools, write the plan to "
    "{plan_file} with write_file ({state}), then call exit_plan_mode to ask the user for "
    "approval. Do not write other files or run commands until the plan is approved. Other "
    "files under .satsuma/plans/ belong to other sessions; do not read or reuse them."
    "{steps})"
)
PLAN_NOTE_MARKER = "\n\n(Plan mode"
PLAN_MODE_OFF_NOTE = (
    "Plan mode is no longer active: the user switched to the {sub} permission mode. The "
    "plan-mode limits in earlier messages no longer apply; file writes and commands now "
    "follow that permission mode."
)
_ENTERED = "Entered plan mode."
_EXITED = "Exited plan mode."


def wants_plan(text: str) -> bool:
    """ユーザー文が既に計画（プラン）を求めているか（語彙のみの判定）。"""
    return bool(_WANTS_PLAN.search(text))


def ensure_plan_request(
    text: str, plan_file: str, *, steps: bool = False, budget: bool = False, exists: bool = False
) -> str:
    """plan モード用: 依頼の末尾に plan モードの通知を付ける。"""
    suffix = PLAN_MODE_SUFFIX if wants_plan(text) else PLAN_REQUEST_SUFFIX
    return text + suffix.format(
        plan_file=plan_file,
        state=PLAN_FILE_EXISTS if exists else PLAN_FILE_NEW,
        steps=plan_steps_request(steps, budget),
    )


def plan_left_open(messages: list[ConversationMessage]) -> bool:
    """会話でモデルに最後に伝えた plan モードの状態が「入っている」のままか。"""
    off_head = PLAN_MODE_OFF_NOTE.split(":", 1)[0]
    for message in reversed(messages):
        for block in reversed(message.blocks):
            if isinstance(block, ToolResultBlock):
                if block.tool_name == "exit_plan_mode" and block.output.startswith(_EXITED):
                    return False
                if block.tool_name == "enter_plan_mode" and block.output.startswith(_ENTERED):
                    return True
            elif isinstance(block, TextBlock) and message.role == "user":
                if off_head in block.text:
                    return False
                if PLAN_NOTE_MARKER in block.text:
                    return True
    return False
