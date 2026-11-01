"""append_file 組み込みツール。"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools import fs_errors, related_code, syntax_check
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util import chunking
from satsumaestro.util.jsonl import write_text_atomic


class AppendFile(Tool):
    """jail 内ファイルの末尾への追記（無ければ作成）。"""

    class Input(BaseModel):
        path: str = Field(description="Workspace-relative path of the file to append to.")
        content: str = Field(
            description=(
                "UTF-8 text to add at the end of the file, exactly as given (no newline is "
                "inserted before it)."
            )
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="append_file",
        description=(
            "Append text to the end of a file within the workspace (created if missing). Use it "
            f"to build a long file in pieces: write_file for the first {chunking.LINES}, then "
            f"append_file for each following piece. Keep each piece under {chunking.SIZE}."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.WORKSPACE_WRITE,
        parallel_safe=False,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, AppendFile.Input)
        real = ctx.jail.resolve(inp.path)
        current = await asyncio.to_thread(syntax_check.before_text, real)
        updated = (current or "") + inp.content
        note = ""
        flags: list[str] = []
        if ctx.config.tools.syntax_check:
            check = await asyncio.to_thread(syntax_check.check_edit, real, current, updated)
            if check.after is not None and not check.after.ok:
                if check.before_ok:
                    note = syntax_check.break_warning(inp.path, check.after)
                    flags.append(syntax_check.FLAG_WARNING)
                else:
                    note = syntax_check.append_note(inp.path, check.after)
        try:
            await asyncio.to_thread(write_text_atomic, real, updated)
        except OSError as e:
            return fs_errors.error_result(e, inp.path)
        if ctx.reads is not None:
            ctx.reads.note_write(real, updated, tool="append_file")
        related, related_flags = await asyncio.to_thread(
            related_code.after_write, ctx, real, current, updated
        )
        note += related
        flags += related_flags
        total = len(updated)
        return ToolResult(
            output=f"appended {len(inp.content)} bytes to {inp.path} (now {total} bytes){note}",
            flags=flags,
        )
