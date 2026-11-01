"""bash_output 組み込みツール。"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema


class BashOutput(Tool):
    """バックグラウンド bash タスクの増分出力・状態の読み取り。"""

    class Input(BaseModel):
        task_id: str = Field(description="Background task id returned by bash (e.g. 'bg-1').")
        wait_s: float = Field(
            default=0.0,
            ge=0.0,
            le=600.0,
            description=(
                "Seconds to wait for the task to exit before reading "
                "(0 = return immediately with whatever output has accumulated)."
            ),
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="bash_output",
        description="Read incremental output and status of a background bash task.",
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, BashOutput.Input)
        registry = ctx.background
        if registry is None:
            return ToolResult(output="no background tasks in this context", is_error=True)
        task = registry.get(inp.task_id)
        if task is None:
            known = ", ".join(registry.known_ids()) or "(none)"
            return ToolResult(
                output=f"background task not found: {inp.task_id} (known: {known})",
                is_error=True,
            )
        if inp.wait_s > 0 and task.running:
            try:
                await asyncio.wait_for(asyncio.shield(task.proc.wait()), timeout=inp.wait_s)
            except TimeoutError:
                pass
        if task.running:
            status = f"running (pid {task.proc.pid})"
        else:
            status = f"exited with code {task.proc.returncode}"
            task.notified = True
        chunk = registry.read_new_output(task)
        header = f"{task.task_id} [{status}] command: {task.command_preview}"
        return ToolResult(output=header + "\n" + (chunk or "(no new output)"))
