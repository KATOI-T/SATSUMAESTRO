"""agent 組み込みツール。"""

from __future__ import annotations

import logging
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.errors import SatsumaError

_log = logging.getLogger(__name__)


class Agent(Tool):
    """サブタスクを子エージェントへ委譲し、最終要約だけを返す。"""

    class Input(BaseModel):
        description: str = Field(description="Short 3-5 word summary of the delegated task.")
        prompt: str = Field(
            description=(
                "The full task for the sub-agent to perform autonomously. It runs in an "
                "isolated context and returns only its final summary text."
            )
        )
        subagent_type: str = Field(
            default="general-purpose",
            description=(
                "Sub-agent preset: 'Explore' (read-only search/reading), 'Plan' "
                "(read-only + todo_write), or 'general-purpose' (all tools)."
            ),
        )
        model: str = Field(
            default="",
            description=(
                "Model or alias for the sub-agent. Empty uses the configured default "
                "(the parent's model unless overridden)."
            ),
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="agent",
        description=(
            "Delegate an exploration or decomposition sub-task to an isolated sub-agent "
            "and return only its final summary text."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.WORKSPACE_WRITE,
        parallel_safe=True,
        deferred=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, Agent.Input)
        from satsumaestro.engine.subagent import RuntimeFactory, resolve_subagent_type
        from satsumaestro.engine.turn import resolve_output

        factory = ctx.runtime_factory
        if not isinstance(factory, RuntimeFactory):
            return ToolResult(
                output="subagent execution is not available in this context", is_error=True
            )
        missing = [name for name in ("description", "prompt") if not getattr(inp, name).strip()]
        if missing:
            return ToolResult(
                output=(
                    f"agent was not started: {' and '.join(missing)} "
                    f"{'is' if len(missing) == 1 else 'are'} empty. Put a few words "
                    "naming the task in description and the whole task in prompt."
                ),
                is_error=True,
            )
        max_depth = ctx.config.subagent.max_depth
        if ctx.depth >= max_depth:
            return ToolResult(
                output=f"subagent depth limit reached (max_depth={max_depth})", is_error=True
            )

        preset = resolve_subagent_type(inp.subagent_type)
        child = factory.build_child(preset=preset, depth=ctx.depth + 1, model=inp.model)
        try:
            summary = await child.run_turn(inp.prompt)
        except SatsumaError as e:
            return ToolResult(output=f"subagent failed ({type(e).__name__}): {e}", is_error=True)
        _log.info(
            "subagent %s finished: iterations=%d input_tokens=%d output_tokens=%d",
            preset.key,
            summary.iterations,
            summary.usage.input_tokens,
            summary.usage.output_tokens,
        )
        output, empty = resolve_output(summary)
        return ToolResult(output=output, is_error=empty, usage=summary.usage)
