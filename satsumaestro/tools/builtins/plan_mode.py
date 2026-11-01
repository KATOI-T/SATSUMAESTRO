"""enter_plan_mode / exit_plan_mode 組み込みツール。"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import PLAN_DIR, Session, plan_file_relpath
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema

PLAN_EXIT_NOT_APPROVED = (
    "Plan presented; plan mode stays active. The user has not approved execution yet. "
    "Do not call exit_plan_mode again — stop and wait; the user will approve or switch "
    "to code mode (/mode code, Shift+Tab) when ready."
)
PLAN_EXIT_REVISE = (
    "The user did not approve the plan and asked for changes. Stay in plan mode, revise "
    "{plan_file} according to the feedback below, then call exit_plan_mode again."
)
PLAN_EXIT_NO_PLAN = (
    "No plan file yet, so there is nothing to show the user. Write the plan to {plan_file} "
    "with write_file (its directory already exists — do not try to create it), then call "
    "exit_plan_mode again."
)
ENTER_PLAN_DESC_ON_REQUEST = (
    "Enter plan mode: only read-only tools and writing the plan file are allowed until "
    "exit_plan_mode. Use it ONLY when the user explicitly asks for a plan; otherwise carry "
    "out the request directly without a planning step."
)
ENTER_PLAN_DESC_GUIDED = (
    "Enter plan mode: only read-only tools and writing the plan file are allowed "
    "until exit_plan_mode. Use it to research and write a plan for user approval "
    "before large changes."
)
PLAN_EXIT_APPROVED_WITH_NOTE = (
    "The user approved the plan and attached a note. Implement the plan, applying the "
    "following note as part of the work:\n\nUser note for implementation:\n{note}"
)
PLAN_EXIT_NOT_ACTIVE = (
    "Plan mode is not active, so there is nothing to exit. Continue the task with tools, or, "
    "if it is done, state your final answer as plain message text (no tool call)."
)
PLAN_EXIT_STEP_DONE_PREFIX = "Plan step recorded:"
PLAN_EXIT_STEP_DONE = (
    PLAN_EXIT_STEP_DONE_PREFIX + " step {index} of {total} is done. exit_plan_mode is not "
    "needed while running steps: ending your reply without a tool call does the same."
)
PLAN_EXIT_LAST_STEP_DONE = (
    PLAN_EXIT_STEP_DONE_PREFIX + " all {total} steps are done. Reply with a short final "
    "summary for the user (no tool call)."
)
FLAG_EXIT_NOT_ACTIVE = "plan-exit-not-active"
FLAG_EXIT_STEP_DONE = "plan-exit-step-done"
PLAN_EXIT_APPROVED_IMPLEMENT = (
    "The user approved the plan. Carry it out now: make the changes with tools (create and "
    "edit files with write_file / append_file / edit_file) and do not paste file contents "
    "into your reply. The plan is in {plan_file}."
)
PLAN_EXIT_APPROVED_NOTE_WHILE_STEPPING = (
    "The user also attached a note to the approval. Apply it throughout the work, "
    "starting with this step:\n\nUser note for implementation:\n{note}"
)


class EnterPlanMode(Tool):
    """plan mode へ入る（計画フェーズ: 計画ファイル以外の書込・bash は自動 deny）。"""

    class Input(BaseModel):
        pass

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="enter_plan_mode",
        description=ENTER_PLAN_DESC_ON_REQUEST,
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=False,
        deferred=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        from satsumaestro.engine.plan_request import PLAN_FILE_EXISTS, PLAN_FILE_NEW
        from satsumaestro.engine.plan_steps import plan_steps_request

        session = ctx.session
        if session is None:
            return ToolResult(output="no session attached; plan mode unavailable", is_error=True)
        plan_file = plan_file_relpath(session.session_id)
        state = PLAN_FILE_EXISTS if (ctx.workspace_root / plan_file).is_file() else PLAN_FILE_NEW
        if session.mode.main == "plan":
            return ToolResult(
                output=f"plan mode is already active (plan file: {plan_file}; {state})"
            )
        (ctx.workspace_root / PLAN_DIR).mkdir(parents=True, exist_ok=True)
        session.mode = session.mode.model_copy(update={"main": "plan"})
        session.plan_steps = []
        session.plan_step_index = 0
        session.plan_step_estimates = []
        session.plan_budget_lines = 0
        steps = plan_steps_request(ctx.config.guidance.plan_steps, ctx.config.guidance.step_budget)
        return ToolResult(
            output=(
                "Entered plan mode. File writes and bash are denied until exit_plan_mode, "
                f"except the plan file `{plan_file}` ({state}) which you may write with "
                "write_file / edit_file / append_file. Investigate with read-only tools, write "
                f"the plan there, then call exit_plan_mode to ask the user for approval.{steps}"
            )
        )


class ExitPlanMode(Tool):
    """plan mode を抜けて直前の code モードへ戻る（runtime がユーザー承認を確認する）。"""

    class Input(BaseModel):
        pass

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="exit_plan_mode",
        description=(
            "Present the plan file and ask the user to approve leaving plan mode. If "
            "approved, the previous permissions are restored and you may implement the "
            "plan (the user may attach a note to apply while implementing); otherwise "
            "plan mode stays active and the user's feedback is returned."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=False,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        session = ctx.session
        if session is None:
            return ToolResult(output="no session attached; plan mode unavailable", is_error=True)
        if session.mode.main != "plan":
            step = current_step(session)
            if step is not None:
                index, total = step
                template = PLAN_EXIT_LAST_STEP_DONE if index >= total else PLAN_EXIT_STEP_DONE
                return ToolResult(
                    output=template.format(index=index, total=total),
                    flags=[FLAG_EXIT_STEP_DONE],
                )
            return ToolResult(
                output=PLAN_EXIT_NOT_ACTIVE, is_error=True, flags=[FLAG_EXIT_NOT_ACTIVE]
            )
        session.mode = session.mode.model_copy(update={"main": "code"})
        return ToolResult(
            output=f"Exited plan mode. Permissions restored to {session.mode.sub.value}."
        )


def current_step(session: Session) -> tuple[int, int] | None:
    """承認した計画の段を実行中なら ``(いまの段, 段の数)``。"""
    total = len(session.plan_steps)
    index = session.plan_step_index
    return (index, total) if 0 < index <= total else None
