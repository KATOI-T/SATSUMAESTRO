"""notebook_edit 組み込みツール。"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools import fs_errors
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.errors import ToolError
from satsumaestro.util.jsonl import write_text_atomic


def _find_cell(cells: list[Any], cell_id: str) -> int:
    """``cells[].id`` が ``cell_id`` に一致するインデックスを返す。"""
    if not cell_id:
        raise ToolError("cell_id is required for this edit_mode")
    ids: list[str] = []
    for i, cell in enumerate(cells):
        cid = cell.get("id") if isinstance(cell, dict) else None
        if cid == cell_id:
            return i
        ids.append(str(cid))
    raise ToolError(f"cell {cell_id!r} not found; existing cell ids: {ids}")


def _blank_cell(cell_type: str, source: str) -> dict[str, Any]:
    """nbformat 4.5 相当の新セル（``id`` 採番済み）を組む。"""
    cell: dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


class NotebookEdit(Tool):
    """jail 内の Jupyter ノートブック（.ipynb）のセル編集。"""

    class Input(BaseModel):
        notebook_path: str = Field(
            description="Workspace-relative path of the .ipynb notebook to edit."
        )
        cell_id: str = Field(
            default="",
            description="Id of the target cell (cells[].id). Required for replace/delete. "
            "For insert the new cell goes after this cell; empty inserts at the top.",
        )
        new_source: str = Field(
            default="",
            description="New cell source. Required for replace and insert (an explicit empty "
            "string clears the cell); ignored for delete.",
        )
        cell_type: Literal["code", "markdown"] = Field(
            default="code",
            description="Type of the cell to insert. Ignored for replace and delete "
            "(the existing cell keeps its type).",
        )
        edit_mode: Literal["replace", "insert", "delete"] = Field(
            default="replace",
            description='"replace" rewrites the source of cell_id, "insert" adds a new '
            'cell, "delete" removes cell_id.',
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="notebook_edit",
        description=(
            "Edit one cell of a Jupyter notebook (.ipynb) in the workspace. "
            "edit_mode=replace rewrites the source of the cell with cell_id; insert adds "
            "a new cell of cell_type after cell_id (or at the top when cell_id is empty); "
            "delete removes the cell. Cell sources only: the notebook is never executed "
            "and outputs are not recomputed."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.WORKSPACE_WRITE,
        parallel_safe=False,
        deferred=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, NotebookEdit.Input)
        real = ctx.jail.resolve(inp.notebook_path)
        if real.suffix != ".ipynb":
            raise ToolError(f"notebook_edit only edits .ipynb files: {inp.notebook_path}")
        try:
            raw = await asyncio.to_thread(real.read_text, encoding="utf-8")
        except FileNotFoundError as e:
            raise ToolError(f"notebook not found: {inp.notebook_path}") from e
        except (OSError, UnicodeDecodeError) as e:
            raise ToolError(fs_errors.describe(e, inp.notebook_path)) from e
        try:
            nb = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ToolError(f"invalid notebook JSON: {e}") from e
        cells = nb.get("cells") if isinstance(nb, dict) else None
        if not isinstance(cells, list):
            raise ToolError("invalid notebook: no cells array")
        if inp.edit_mode in ("replace", "insert") and "new_source" not in inp.model_fields_set:
            raise ToolError(f"new_source is required for edit_mode={inp.edit_mode}")

        if inp.edit_mode == "replace":
            cells[_find_cell(cells, inp.cell_id)]["source"] = inp.new_source
            summary = f"replaced source of cell {inp.cell_id}"
        elif inp.edit_mode == "insert":
            new_cell = _blank_cell(inp.cell_type, inp.new_source)
            index = _find_cell(cells, inp.cell_id) + 1 if inp.cell_id else 0
            cells.insert(index, new_cell)
            summary = f"inserted {inp.cell_type} cell at index {index} (id={new_cell['id']})"
        else:
            cells.pop(_find_cell(cells, inp.cell_id))
            summary = f"deleted cell {inp.cell_id}"

        text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        await asyncio.to_thread(write_text_atomic, real, text)
        return ToolResult(output=f"{summary}; notebook now has {len(cells)} cells")
