"""list_directory 組み込みツール。"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.rag.walk import is_ignored, load_ignore_spec
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools import fs_errors
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema

FILE_HINT = "use read_file to read it"


def format_size(size: int) -> str:
    """``1023 B`` / ``1.5 KB`` / ``3.0 MB`` / ``2.0 GB``（1,024 刻み・小数 1 桁）。"""
    if size < 1024:
        return f"{size} B"
    value = size / 1024
    for unit in ("KB", "MB"):
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _scan(base: Path, root: Path, limit: int, inside: Callable[[str], bool]) -> list[str]:
    """``base`` の直下を一覧した行を返す（同期。``asyncio.to_thread`` で呼ぶ）。"""
    base_rel = base.relative_to(root)
    spec = load_ignore_spec(root)
    dirs: list[str] = []
    files: list[tuple[str, int]] = []
    skipped = 0
    unreadable = 0
    with os.scandir(base) as it:
        for entry in it:
            try:
                is_dir = entry.is_dir()
            except OSError:
                unreadable += 1
                continue
            rel = (base_rel / entry.name).as_posix()
            if is_ignored(spec, rel, is_dir=is_dir):
                skipped += 1
                continue
            if not inside(entry.path):
                continue
            if is_dir:
                dirs.append(entry.name)
                continue
            try:
                files.append((entry.name, entry.stat().st_size))
            except OSError:
                unreadable += 1
                continue
    lines = [f"{d}/" for d in sorted(dirs)]
    lines.extend(f"{name}  {format_size(size)}" for name, size in sorted(files))
    truncated = len(lines) > limit
    lines = lines[:limit]
    if not lines:
        lines.append("(empty directory)")
    if skipped:
        lines.append(f"({skipped} entries skipped by ignore rules)")
    if unreadable:
        lines.append(f"({unreadable} entries could not be read)")
    if truncated:
        lines.append(f"... (truncated at {limit} entries)")
    return lines


class ListDirectory(Tool):
    """ディレクトリ直下の一覧（読み取り専用）。"""

    class Input(BaseModel):
        path: str = Field(
            default=".",
            description="Workspace-relative directory to list. Omit or '.' for the workspace root.",
        )
        limit: int = Field(
            default=200,
            ge=1,
            description=(
                "Maximum number of entries to return (default 200); the output ends with a "
                "note when truncated."
            ),
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="list_directory",
        description=(
            "List the direct children of one directory within the workspace (not recursive). "
            "Directories end with '/', files show their size. Entries ignored by .gitignore and "
            "VCS/dependency/build directories (.git, node_modules, .venv, dist, build) are "
            "skipped and counted. Use glob_search to find files by pattern across subdirectories."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, ListDirectory.Input)
        real = ctx.jail.resolve(inp.path)
        try:
            lines = await asyncio.to_thread(
                _scan, real, ctx.jail.root, inp.limit, ctx.jail.is_inside
            )
        except NotADirectoryError:
            return ToolResult(output=f"path is a file: {inp.path} — {FILE_HINT}", is_error=True)
        except FileNotFoundError:
            return ToolResult(output=f"directory not found: {inp.path}", is_error=True)
        except OSError as e:
            return fs_errors.error_result(e, inp.path)
        return ToolResult(output="\n".join(lines))
