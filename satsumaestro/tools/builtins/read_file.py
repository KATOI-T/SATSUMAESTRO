"""read_file 組み込みツール。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools import fs_errors
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema

_BYTE_CAP = 2_000_000
FULL_READ_CAP_NOTE = f"\n... (truncated at {_BYTE_CAP} bytes)"


def _slice(text: str, offset: int, limit: int) -> str:
    """1 始まりの ``offset`` 行から ``limit`` 行を取り出す。"""
    if offset <= 1 and limit <= 0:
        return text
    lines = text.splitlines(keepends=True)
    start = max(offset - 1, 0)
    end = (start + limit) if limit > 0 else None
    return "".join(lines[start:end])


def _read_range(path: Path, offset: int, limit: int) -> tuple[str, int]:
    """``offset``/``limit`` 指定時のストリーム読取。"""
    start = max(offset - 1, 0)
    out: list[str] = []
    seen = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            seen = i + 1
            if i < start:
                continue
            if limit > 0 and len(out) >= limit:
                break
            out.append(line)
    return "".join(out), seen


def past_end_message(path: str, offset: int, total: int) -> str:
    """範囲に 1 行も入らなかったときの文。"""
    if total == 0:
        return f"read_file: {path} is empty (0 lines)."
    return (
        f"read_file: {path} has {total} lines; nothing to read from line {offset}. "
        f"Use offset 1-{total}."
    )


def _whole_text(path: Path) -> str:
    """全文読取。"""
    with path.open("rb") as f:
        data = f.read(_BYTE_CAP + 1)
    if len(data) <= _BYTE_CAP:
        return data.decode("utf-8")
    text = data[:_BYTE_CAP].decode("utf-8", errors="ignore")
    return text + FULL_READ_CAP_NOTE


class ReadFile(Tool):
    """jail 内ファイルの読み取り。"""

    class Input(BaseModel):
        path: str = Field(description="Workspace-relative path of the UTF-8 text file to read.")
        offset: int = Field(
            default=1,
            ge=0,
            description="1-based line number to start reading from. Omit to start at the top.",
        )
        limit: int = Field(
            default=0,
            ge=0,
            description="Maximum number of lines to read. Omit or 0 to read to the end.",
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="read_file",
        description=(
            "Read a UTF-8 text file within the workspace. Use offset (1-based line) and "
            "limit (line count) to read just a range of a large file; only that range is "
            "loaded. Full reads are truncated after 2000000 bytes with a note."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, ReadFile.Input)
        real = ctx.jail.resolve(inp.path)
        try:
            if inp.offset > 1 or inp.limit > 0:
                text, seen = await asyncio.to_thread(_read_range, real, inp.offset, inp.limit)
                if not text and inp.offset > 1:
                    text = past_end_message(inp.path, inp.offset, seen)
            else:
                text = await asyncio.to_thread(_whole_text, real)
        except (OSError, UnicodeDecodeError) as e:
            return fs_errors.error_result(e, inp.path)
        if ctx.reads is not None:
            await asyncio.to_thread(ctx.reads.note_read, real)
        return ToolResult(output=text)
