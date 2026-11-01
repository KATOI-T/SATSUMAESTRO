"""バックグラウンド Task・Team・Cron をモデルから操作する 11 のツール。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from satsumaestro.orchestration.ids import UnknownIdError
from satsumaestro.orchestration.task_ledger import TaskStatus
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.errors import OrchestrationError

if TYPE_CHECKING:
    from satsumaestro.orchestration.runner import Orchestrator

_UNAVAILABLE = "orchestration is not available in this context"
_KNOWN_IDS_LIMIT = 10


def _orchestrator(ctx: ToolContext) -> Orchestrator | None:
    """``ctx.orchestrator`` を :class:`Orchestrator` に絞る（循環を避けるため遅延 import）。"""
    from satsumaestro.orchestration.runner import Orchestrator

    orch = ctx.orchestrator
    return orch if isinstance(orch, Orchestrator) else None


def _json(payload: Any) -> ToolResult:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    return ToolResult(output=json.dumps(payload, ensure_ascii=False))


def _reject(message: str) -> ToolResult:
    return ToolResult(output=message, is_error=True)


def known_ids_hint(kind: str, orch: Orchestrator) -> str:
    """その種類の既知の ID を新しい方から並べた ``(known task_ids: a, b, …)``。"""
    if kind == "task":
        ids = [t.task_id for t in orch.tasks.list()]
    elif kind == "team":
        ids = [t.team_id for t in orch.teams.list()]
    elif kind == "cron":
        ids = [c.cron_id for c in orch.crons.list()]
    elif kind == "worker" and orch.workers is not None:
        ids = [w.worker_id for w in orch.workers.ledger.list()]
    else:
        ids = []
    label = f"{kind}_ids"
    if not ids:
        return f"(no known {label})"
    shown = ids[-_KNOWN_IDS_LIMIT:]
    rest = len(ids) - len(shown)
    tail = f", … and {rest} more" if rest else ""
    return f"(known {label}: {', '.join(shown)}{tail})"


def _failure(e: OrchestrationError, orch: Orchestrator) -> ToolResult:
    text = str(e)
    if isinstance(e, UnknownIdError):
        text = f"{text} {known_ids_hint(e.kind, orch)}"
    return ToolResult(output=text, is_error=True)


class _OrchestrationTool(Tool):
    """orchestrator を解いて :meth:`apply` に渡す共通の枠。"""

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        orch = _orchestrator(ctx)
        if orch is None:
            return _reject(_UNAVAILABLE)
        try:
            return await self.apply(inp, orch)
        except OrchestrationError as e:
            return _failure(e, orch)

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        raise NotImplementedError


class CreateTask(_OrchestrationTool):
    """バックグラウンド Task を起動し、すぐに task_id を返す。"""

    class Input(BaseModel):
        prompt: str = Field(
            description=(
                "The full task for the background sub-agent to perform autonomously. "
                "It runs in an isolated context; collect its result later with task_output."
            )
        )
        description: str = Field(
            default="", description="Short 3-5 word summary of the task (shown in task_list)."
        )
        model: str = Field(
            default="",
            description="Model or alias for the background sub-agent. Empty uses the default.",
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="task_create",
        description=(
            "Start a background task handled by an isolated sub-agent and return its task_id "
            "immediately. Use task_list / task_get to track progress and task_output to read "
            "the result when status is completed."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, CreateTask.Input)
        if not inp.prompt.strip():
            return _reject("prompt is required")
        task = await orch.create_task(inp.prompt, inp.description, model=inp.model)
        return _json(
            {
                "task_id": task.task_id,
                "status": task.status,
                "description": task.description,
                "created_at": task.created_at,
            }
        )


class ListTasks(_OrchestrationTool):
    """Task の一覧と board（active / blocked / finished と鮮度）。"""

    class Input(BaseModel):
        status: str = Field(
            default="",
            description=(
                "Optional status filter: created, running, blocked, completed, failed, stopped. "
                "Empty lists all tasks."
            ),
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="task_list",
        description=(
            "List background tasks on a status board (active / blocked / finished and "
            "heartbeat freshness). Optionally filter by status."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, ListTasks.Input)
        status: TaskStatus | None = None
        if inp.status.strip():
            try:
                status = TaskStatus(inp.status.strip().lower())
            except ValueError:
                return _reject(f"unknown status: {inp.status}")
        tasks = orch.list_tasks(status)
        return _json(
            {
                "count": len(tasks),
                "tasks": [
                    {
                        "task_id": t.task_id,
                        "status": t.status,
                        "description": t.description,
                        "prompt": t.prompt,
                        "team_id": t.team_id,
                        "created_at": t.created_at,
                        "updated_at": t.updated_at,
                        "output_ready": bool(t.output),
                    }
                    for t in tasks
                ],
                "board": orch.board().model_dump(mode="json"),
            }
        )


class GetTask(_OrchestrationTool):
    """Task のすべて（注記・出力・生存報告を含む）を返す。"""

    class Input(BaseModel):
        task_id: str = Field(description="ID returned by task_create.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="task_get",
        description=(
            "Get the full state of a background task (status, messages, output, heartbeat)."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, GetTask.Input)
        return _json(orch.get_task(inp.task_id))


class ReadTaskOutput(_OrchestrationTool):
    """Task の出力（子エージェントの最終要約）を返す。"""

    class Input(BaseModel):
        task_id: str = Field(description="ID of the task whose output to read.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="task_output",
        description=(
            "Read the output of a background task (the sub-agent's final summary). Empty until "
            "the task completes; check task_list for status first."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, ReadTaskOutput.Input)
        output = orch.task_output(inp.task_id)
        return _json({"task_id": inp.task_id, "output": output, "output_ready": bool(output)})


class AnnotateTask(_OrchestrationTool):
    """Task に注記を残す（走っている子には届けない）。"""

    class Input(BaseModel):
        task_id: str = Field(description="ID of the task to annotate.")
        message: str = Field(description="Note to record on the task (not delivered to the agent).")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="task_update",
        description=(
            "Record a note on a background task. The note is stored in the task's message log "
            "only; it is not delivered to the running sub-agent."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, AnnotateTask.Input)
        if not inp.message.strip():
            return _reject("message is required")
        task = orch.update_task(inp.task_id, inp.message)
        return _json(
            {
                "task_id": task.task_id,
                "status": task.status,
                "notes": len(task.notes),
                "latest_note": task.notes[-1].text if task.notes else None,
            }
        )


class StopTask(_OrchestrationTool):
    """走っている・待っている Task を止める（終わったものは拒否）。"""

    class Input(BaseModel):
        task_id: str = Field(description="ID of the task to stop.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="task_stop",
        description=(
            "Stop a background task. Rejected if the task already completed, failed or stopped."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, StopTask.Input)
        task = orch.stop_task(inp.task_id)
        return _json(
            {"task_id": task.task_id, "status": task.status, "finished_at": task.finished_at}
        )


class CreateTeam(_OrchestrationTool):
    """既にある Task を 1 つの Team に束ねる。"""

    class Input(BaseModel):
        name: str = Field(description="Human-readable team name.")
        task_ids: list[str] = Field(
            description="IDs of existing tasks (from task_create) to group under this team."
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="team_create",
        description=(
            "Group existing background tasks under a named team so they can be tracked and "
            "stopped together with team_delete."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, CreateTeam.Input)
        if not inp.name.strip():
            return _reject("name is required")
        team = orch.create_team(inp.name, inp.task_ids)
        return _json(
            {
                "team_id": team.team_id,
                "name": team.name,
                "status": team.status,
                "members": team.members,
                "created_at": team.created_at,
            }
        )


class DeleteTeam(_OrchestrationTool):
    """Team を論理削除し、終わっていないメンバーを止める。"""

    class Input(BaseModel):
        team_id: str = Field(description="ID returned by team_create.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="team_delete",
        description=(
            "Delete a team (soft delete) and stop every member task that has not finished yet."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, DeleteTeam.Input)
        team = orch.delete_team(inp.team_id)
        return _json(
            {
                "team_id": team.team_id,
                "name": team.name,
                "status": team.status,
                "members": team.members,
            }
        )


class CreateCron(_OrchestrationTool):
    """Cron を登録する（スケジューラは無い。登録簿だけ）。"""

    class Input(BaseModel):
        schedule: str = Field(description='Cron expression, e.g. "0 * * * *" (stored as-is).')
        prompt: str = Field(description="Prompt to run on each scheduled execution.")
        description: str = Field(default="", description="Optional human-readable note.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="cron_create",
        description=(
            "Register a recurring prompt with a cron schedule. The entry is recorded only; no "
            "scheduler runs it automatically in this version."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, CreateCron.Input)
        if not inp.schedule.strip() or not inp.prompt.strip():
            return _reject("schedule and prompt are required")
        return _json(orch.create_cron(inp.schedule, inp.prompt, inp.description or None))


class ListCrons(_OrchestrationTool):
    """Cron の一覧。"""

    class Input(BaseModel):
        enabled_only: bool = Field(default=False, description="If true, list only enabled entries.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="cron_list",
        description="List registered cron entries with their run counts.",
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, ListCrons.Input)
        jobs = orch.list_crons(inp.enabled_only)
        return _json({"count": len(jobs), "crons": [j.model_dump(mode="json") for j in jobs]})


class DeleteCron(_OrchestrationTool):
    """Cron を削除する（物理削除）。"""

    class Input(BaseModel):
        cron_id: str = Field(description="ID returned by cron_create.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="cron_delete",
        description="Remove a cron entry permanently.",
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, orch: Orchestrator) -> ToolResult:
        assert isinstance(inp, DeleteCron.Input)
        job = orch.delete_cron(inp.cron_id)
        return _json({"cron_id": job.cron_id, "schedule": job.schedule, "removed": True})


ORCHESTRATION_TOOLS: tuple[type[Tool], ...] = (
    CreateTask,
    ListTasks,
    GetTask,
    ReadTaskOutput,
    AnnotateTask,
    StopTask,
    CreateTeam,
    DeleteTeam,
    CreateCron,
    ListCrons,
    DeleteCron,
)

ORCHESTRATION_TOOL_NAMES: frozenset[str] = frozenset(t.spec.name for t in ORCHESTRATION_TOOLS)

MUTATING_TOOL_NAMES: frozenset[str] = frozenset(
    t.spec.name
    for t in ORCHESTRATION_TOOLS
    if t.spec.required_mode is PermissionMode.DANGER_FULL_ACCESS
)
