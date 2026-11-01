"""tool_search 組み込みツール。"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.registry import ToolRegistry
from satsumaestro.tools.schema import model_tool_schema


class ToolSearch(Tool):
    """遅延ツールを探して読み込む（読み込んだツールは次の往復から呼べる）。"""

    class Input(BaseModel):
        query: str = Field(
            description=(
                "'select:' followed by tool names separated by commas, or keywords "
                "('+word' makes a word required)."
            )
        )
        max_results: int = Field(
            default=5, ge=1, le=20, description="How many tools to load at most."
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="tool_search",
        description=(
            "Load deferred tools so you can call them. Name them with 'select:name_a,name_b', "
            "or give keywords to search their names and descriptions."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=False,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, ToolSearch.Input)
        registry = ctx.registry
        if not isinstance(registry, ToolRegistry):
            return ToolResult(output="tool search is not available in this context", is_error=True)
        found = registry.search(inp.query, inp.max_results)
        if found:
            return ToolResult(
                output=f"Loaded: {', '.join(found)}. Call them by name from your next step."
            )
        names = ", ".join(registry.deferred_names()) or "(none)"
        return ToolResult(
            output=(
                f"Nothing matched `{inp.query}`. Deferred tools: {names}. "
                "Load one with select:<name>."
            )
        )
