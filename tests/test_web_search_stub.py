"""web_search スタブ・todo_write テスト。"""

from __future__ import annotations

from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.todo_write import TodoItem, TodoWrite
from satsumaestro.tools.builtins.web_search import WebSearch


async def test_web_search_stub_is_error(tool_ctx: ToolContext) -> None:
    tool_ctx.config.web_search.provider = None
    result = await WebSearch().run(WebSearch.Input(query="anything"), tool_ctx)
    assert result.is_error is True
    assert "no search backend configured" in result.output


def test_web_search_schema_valid() -> None:
    schema = WebSearch.Input.model_json_schema()
    assert "query" in schema["properties"]


async def test_todo_write_formats(tool_ctx: ToolContext) -> None:
    result = await TodoWrite().run(
        TodoWrite.Input(
            todos=[
                TodoItem(content="A", active_form="Doing A", status="completed"),
                TodoItem(content="B", active_form="Doing B", status="in_progress"),
                TodoItem(content="C", active_form="Doing C"),
            ]
        ),
        tool_ctx,
    )
    assert result.output == "[x] A\n[~] B\n[ ] C"
    assert result.is_error is False
