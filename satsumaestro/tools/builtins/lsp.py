"""lsp 組み込みツール。"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field, field_validator

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.lsp.models import LspRequest, action_for
from satsumaestro.tools.lsp.registry import LanguageServers
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.errors import ToolError


class Lsp(Tool):
    """言語サーバによるコード知能の問い合わせ。"""

    class Input(BaseModel):
        action: str = Field(
            description=(
                "One of: diagnostics, hover, definition, references, completion, "
                "symbols, format (LSP method names like textDocument/definition also work)."
            )
        )
        path: str | None = Field(
            default=None,
            description=(
                "Target file inside the workspace. Optional only for diagnostics "
                "(aggregate cached) and symbols with query."
            ),
        )
        line: int | None = Field(
            default=None,
            ge=0,
            description="0-based line for hover/definition/references/completion.",
        )
        character: int | None = Field(
            default=None,
            ge=0,
            description="0-based character for hover/definition/references/completion.",
        )
        query: str | None = Field(
            default=None, description="Workspace symbol search text (action=symbols)."
        )

        @field_validator("action")
        @classmethod
        def _known_action(cls, value: str) -> str:
            action = action_for(value)
            if action is None:
                valid = ", ".join(a.value for a in LspRequest)
                raise ValueError(f"unknown lsp action {value!r}; use one of {valid}")
            return action.value

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="lsp",
        description=(
            "Query a language server for code intelligence: diagnostics, hover, "
            "definition, references, completion, symbols, format. Positions are 0-based."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, Lsp.Input)
        registry = ctx.lsp
        if not isinstance(registry, LanguageServers):
            return ToolResult(
                output="LSP integration is disabled (see the lsp.enabled setting).",
                is_error=True,
            )
        try:
            action = LspRequest(inp.action)
            path = None
            if inp.path is not None:
                path = ctx.jail.resolve(inp.path)
                if not path.is_file():
                    return ToolResult(output=f"file not found: {inp.path}", is_error=True)
            output = await registry.dispatch(
                action, path=path, line=inp.line, character=inp.character, query=inp.query
            )
        except ToolError as e:
            return ToolResult(output=str(e), is_error=True)
        return ToolResult(output=output)
