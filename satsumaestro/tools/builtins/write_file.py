"""write_file 組み込みツール。"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools import fs_errors, overwrite_guard, related_code, syntax_check
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util import chunking
from satsumaestro.util.jsonl import write_text_atomic


class WriteFile(Tool):
    """jail 内ファイルへの原子的書込。"""

    class Input(BaseModel):
        path: str = Field(description="Workspace-relative path of the file to write.")
        content: str = Field(description="Full UTF-8 text to write. Replaces the whole file.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="write_file",
        description=(
            f"Write a UTF-8 text file within the workspace (atomic). Keep `content` under "
            f"{chunking.LINES_AND_SIZE} per call. For a longer file, write the first part here "
            "and continue with append_file."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.WORKSPACE_WRITE,
        parallel_safe=False,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, WriteFile.Input)
        real = ctx.jail.resolve(inp.path)
        note = ""
        flags: list[str] = []
        before = await asyncio.to_thread(syntax_check.before_text, real)
        if (
            before is not None
            and ctx.reads is not None
            and ctx.config.tools.overwrite_guard
            and overwrite_guard.is_guarded(real, ctx.jail.root, ctx.reads.written)
            and overwrite_guard.is_fragment(before, inp.content)
        ):
            first = overwrite_guard.first_line(inp.content)
            if ctx.reads.overwrite_override(real, first):
                flags.append(overwrite_guard.FLAG_OVERRIDE)
            else:
                ctx.reads.note_overwrite_rejected(real, first)
                return ToolResult(
                    output=overwrite_guard.reject_message(
                        inp.path, before, ctx.reads.written_by.get(real, [])
                    ),
                    is_error=True,
                    flags=[overwrite_guard.FLAG_REJECT],
                )
        if ctx.config.tools.syntax_check:
            check = await asyncio.to_thread(syntax_check.check_edit, real, before, inp.content)
            if check.after is not None and not check.after.ok:
                note = syntax_check.write_warning(inp.path, check.after)
                if check.before_ok:
                    flags.append(syntax_check.FLAG_WARNING)
        try:
            await asyncio.to_thread(write_text_atomic, real, inp.content)
        except OSError as e:
            return fs_errors.error_result(e, inp.path)
        if ctx.reads is not None:
            ctx.reads.note_write(real, inp.content, tool="write_file")
        related, related_flags = await asyncio.to_thread(
            related_code.after_write, ctx, real, before, inp.content
        )
        note += related
        flags += related_flags
        replaced = overwrite_guard.replaced_note(before or "")
        return ToolResult(
            output=f"wrote {len(inp.content)} bytes to {inp.path}{replaced}{note}", flags=flags
        )
