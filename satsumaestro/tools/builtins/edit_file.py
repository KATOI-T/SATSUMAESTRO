"""edit_file 組み込みツール。"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools import edit_match, fs_errors, read_ledger, related_code, syntax_check
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.jsonl import write_text_atomic


class EditFile(Tool):
    """jail 内ファイルの文字列置換。"""

    class Input(BaseModel):
        path: str = Field(description="Workspace-relative path of the file to edit.")
        old: str = Field(
            description="Exact text to replace. Must match the file byte for byte, "
            "including indentation."
        )
        new: str = Field(description="Replacement text.")
        replace_all: bool = Field(
            default=False,
            description="Replace every occurrence. Omit to require that old appears exactly once.",
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="edit_file",
        description="Replace an exact string in a file within the workspace.",
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.WORKSPACE_WRITE,
        parallel_safe=False,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, EditFile.Input)
        if inp.old == "":
            return ToolResult(
                output=edit_match.empty_old_message(inp.path),
                is_error=True,
                flags=[edit_match.FLAG_MISMATCH],
            )
        if inp.old == inp.new:
            return ToolResult(
                output=edit_match.same_text_message(inp.path),
                is_error=True,
                flags=[edit_match.FLAG_MISMATCH],
            )
        real = ctx.jail.resolve(inp.path)
        try:
            text = await asyncio.to_thread(real.read_text, encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return fs_errors.error_result(e, inp.path)
        old, new = inp.old, inp.new
        count = text.count(old)
        note = ""
        flags: list[str] = []
        if count == 0 and ctx.config.tools.edit_reindent:
            shifted = edit_match.reindent_edit(text, old, new)
            if shifted is not None:
                old, new, count = shifted.old, shifted.new, 1
                note = edit_match.reindent_note(shifted)
                flags.append(edit_match.FLAG_REINDENT)
        if count == 0:
            output = edit_match.not_found_message(inp.path, text, inp.old, inp.new)
            miss_flags = [edit_match.FLAG_MISMATCH]
            stale = ctx.reads.stale_note(real, inp.path, text) if ctx.reads is not None else None
            if stale is not None:
                output += "\n" + stale
                miss_flags.append(read_ledger.FLAG_STALE)
            return ToolResult(output=output, is_error=True, flags=miss_flags)
        if count > 1 and not inp.replace_all:
            return ToolResult(
                output=edit_match.not_unique_message(inp.path, text, inp.old, count),
                is_error=True,
                flags=[edit_match.FLAG_MISMATCH],
            )
        updated = text.replace(old, new)
        if ctx.config.tools.syntax_check:
            reason = syntax_check.patch_rejection(real, new, old=old)
            if reason is not None:
                return ToolResult(
                    output=syntax_check.reject_patch_message(inp.path, reason),
                    is_error=True,
                    flags=[syntax_check.FLAG_REJECT],
                )
            check = await asyncio.to_thread(syntax_check.check_edit, real, text, updated)
            if check.after is not None and not check.after.ok:
                if not check.before_known_broken:
                    if ctx.config.tools.syntax_reject_edits:
                        return ToolResult(
                            output=syntax_check.reject_syntax_message(inp.path, check.after),
                            is_error=True,
                            flags=[syntax_check.FLAG_REJECT],
                        )
                    note += syntax_check.break_warning(inp.path, check.after)
                    flags.append(syntax_check.FLAG_WARNING)
                else:
                    note += syntax_check.still_broken_note(inp.path, check.after)
        try:
            await asyncio.to_thread(write_text_atomic, real, updated)
        except OSError as e:
            return fs_errors.error_result(e, inp.path)
        if ctx.reads is not None:
            ctx.reads.note_write(real, updated, tool="edit_file")
        related, related_flags = await asyncio.to_thread(
            related_code.after_write, ctx, real, text, updated
        )
        note += related
        flags += related_flags
        return ToolResult(output=f"replaced {count} occurrence(s) in {inp.path}{note}", flags=flags)
