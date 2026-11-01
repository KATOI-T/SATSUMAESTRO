"""todo_write 組み込みツール。"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import TodoItem
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema

STATUS_MARKS: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
}

_CHECK_WORDS = re.compile(
    r"\b(?:verif|test|check|lint|pytest)|確認|検証|テスト|試験", re.IGNORECASE
)
_CHECK_NOTE_MIN_ITEMS = 3
_CHECK_NOTE = (
    "Every item is completed, but none of them confirms the result. Before you report, confirm "
    "it with the tools you have (run the tests or the build with bash, or read what you wrote) "
    "and say what you confirmed."
)


def format_todos(todos: list[TodoItem]) -> str:
    """TODO リストを ``[ ]/[~]/[x]`` マーク付きで整形する（prompt 注入と共用）。"""
    return "\n".join(f"{STATUS_MARKS[t.status]} {t.content}" for t in todos)


def _validate(todos: list[TodoItem]) -> str | None:
    """入力検証。"""
    if not todos:
        return "todo_write got no items; send the whole list, with every item"
    for i, t in enumerate(todos):
        if not t.content.strip():
            return f"todos[{i}] has no content; write the task as a short instruction"
    return None


class TodoWrite(Tool):
    """セッション永続のステートフル TODO 書き込み。"""

    class Input(BaseModel):
        todos: list[TodoItem] = Field(
            description="The complete TODO list. Always send every item, not just the changed ones."
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="todo_write",
        description=(
            "Record the current TODO list for the session. Replaces the whole list; "
            "keep statuses updated as work progresses."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=False,
        deferred=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, TodoWrite.Input)
        error = _validate(inp.todos)
        if error is not None:
            return ToolResult(output=error, is_error=True)

        all_done = all(t.status == "completed" for t in inp.todos)
        if ctx.session is not None:
            ctx.session.todos = [] if all_done else [t.model_copy(deep=True) for t in inp.todos]

        output = format_todos(inp.todos)
        if all_done and len(inp.todos) >= _CHECK_NOTE_MIN_ITEMS and not _has_check(inp.todos):
            output += f"\n\n{_CHECK_NOTE}"
        return ToolResult(output=output)


def _has_check(todos: list[TodoItem]) -> bool:
    """結果を確かめる項目が計画にあるか。"""
    return any(_CHECK_WORDS.search(t.content) for t in todos)
