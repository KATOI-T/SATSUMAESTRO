"""delete_path 組み込みツール。"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import STATE_DIR_UNDELETABLE
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.errors import PathEscapeError, StateDirError


class DeletePath(Tool):
    """jail 内のファイル・ディレクトリの削除。"""

    class Input(BaseModel):
        path: str = Field(description="Workspace-relative path of the file or directory to delete.")
        recursive: bool = Field(
            default=False,
            description=(
                "Delete a non-empty directory together with everything inside it. Required for "
                "non-empty directories; not needed for files or empty directories."
            ),
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="delete_path",
        description=(
            "Delete a file or directory within the workspace. Use this instead of rm/rmdir in "
            "bash. The user is always asked to confirm."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, DeletePath.Input)
        target = link_aware_target(ctx.jail.root, inp.path)
        ctx.jail.resolve(target.parent, allow_hidden=True)
        if ctx.jail.hides(target):
            raise StateDirError(f"{STATE_DIR_UNDELETABLE} {inp.path}")
        if target == ctx.jail.root:
            raise PathEscapeError(f"refusing to delete the workspace root: {inp.path}")
        if not target.exists() and not target.is_symlink():
            return ToolResult(output=f"not found: {inp.path}", is_error=True)
        if target.is_dir() and not target.is_symlink():
            count = count_entries(target)
            if count and not inp.recursive:
                return ToolResult(
                    output=(
                        f"directory not empty: {inp.path} ({count} entries). "
                        "Set recursive=true to delete it with its contents."
                    ),
                    is_error=True,
                )
            await asyncio.to_thread(shutil.rmtree, target)
            detail = f" and {count} entries inside" if count else ""
            return ToolResult(output=f"deleted directory {inp.path}{detail}")
        await asyncio.to_thread(target.unlink)
        return ToolResult(output=f"deleted {inp.path}")


def link_aware_target(root: Path, path: str) -> Path:
    """削除対象の絶対パス。"""
    joined = Path(os.path.normpath(Path(path) if os.path.isabs(path) else root / path))
    if joined.name in ("", ".", ".."):
        return joined.resolve()
    return joined.parent.resolve() / joined.name


def count_entries(directory: Path, limit: int = 100_000) -> int:
    """ディレクトリ配下のエントリ数（symlink は辿らない。上限で打ち切る）。"""
    n = 0
    for _root, dirs, files in os.walk(directory):
        n += len(dirs) + len(files)
        if n >= limit:
            break
    return n
