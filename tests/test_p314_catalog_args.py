"""遅延ツールのカタログに必須の引数名を添え、``todo_write`` を既定でコアにする。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from satsumaestro.config.models import RuntimeConfig, ToolsConfig
from satsumaestro.prompt.builder import SystemPromptBuilder, format_tool_catalog
from satsumaestro.session.models import TodoItem
from satsumaestro.tools import default_registry
from satsumaestro.tools.builtins.todo_write import TodoWrite


def _catalog(reg: object) -> dict[str, dict[str, str]]:
    assert hasattr(reg, "deferred_catalog")
    return {e["name"]: e for e in reg.deferred_catalog()}


def test_deferred_catalog_carries_required_argument_names() -> None:
    entries = _catalog(default_registry())
    assert entries["web_search"]["required"] == "query"
    assert entries["web_fetch"]["required"] == "url"
    assert entries["todo_write"]["required"] == "todos"
    assert entries["enter_plan_mode"]["required"] == ""


def test_catalog_lines_show_required_arguments() -> None:
    text = format_tool_catalog(default_registry().deferred_catalog())
    assert "- web_search(query): Search the web" in text
    assert "- web_fetch(url): Fetch a URL" in text
    assert "- todo_write(todos): Record the current TODO list" in text
    assert "- enter_plan_mode: Enter plan mode" in text
    assert "its schema is loaded on the first call" in text


def test_entries_without_required_keep_the_old_form() -> None:
    text = format_tool_catalog([{"name": "x_tool", "description": "Do x. Second sentence."}])
    assert "- x_tool: Do x" in text and "Second sentence" not in text
    text = format_tool_catalog([{"name": "y", "description": "Do y.", "required": "a, b"}])
    assert "- y(a, b): Do y" in text


def test_todo_write_is_core_with_the_default_settings() -> None:
    cfg = RuntimeConfig()
    assert cfg.tools.core == ["todo_write"]
    reg = default_registry(core_override=cfg.tools.core)
    assert not reg.is_deferred("todo_write")
    assert "todo_write" in {s["name"] for s in reg.schemas_for(set())}
    assert "todo_write" not in _catalog(reg)
    assert reg.is_deferred("web_fetch") and "web_fetch" in _catalog(reg)


def test_settings_can_put_todo_write_back_to_deferred() -> None:
    cfg = RuntimeConfig(tools=ToolsConfig(core=[]))
    reg = default_registry(core_override=cfg.tools.core)
    assert reg.is_deferred("todo_write") and "todo_write" in _catalog(reg)
    assert default_registry().is_deferred("todo_write")


def test_todo_item_reads_known_variants_as_content() -> None:
    """gpt-oss:120b は schema が見えていても項目を task / description と書いた（3/3）。"""
    assert TodoItem.model_validate({"task": "a.txt に 1 行書く"}).content == "a.txt に 1 行書く"
    item = TodoItem.model_validate({"description": "b", "status": "in_progress", "id": "2"})
    assert item.content == "b" and item.status == "in_progress"
    assert TodoItem.model_validate({"content": "c", "task": "x"}).content == "c"
    assert TodoItem.model_validate({"content": "", "title": "t"}).content == "t"
    with pytest.raises(ValidationError):
        TodoItem.model_validate({"id": "1", "status": "pending"})
    for shape in ({"task": "a", "status": "pending", "id": "1"}, {"description": "a"}):
        inp = TodoWrite.Input.model_validate({"todos": [shape]})
        assert inp.todos[0].content == "a"


def test_static_half_shows_required_arguments() -> None:
    cfg = RuntimeConfig()
    reg = default_registry(core_override=cfg.tools.core)
    builder = SystemPromptBuilder(tool_catalog=format_tool_catalog(reg.deferred_catalog()))
    static = builder._static_sections("m", cfg)
    assert "- web_search(query):" in static and "- web_fetch(url):" in static
    assert "todo_write" not in static.split("# Deferred tools", 1)[1].split("\n#", 1)[0]
