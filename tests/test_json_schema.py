"""JSON schema 生成テスト。"""

from __future__ import annotations

from satsumaestro.tools import default_registry
from satsumaestro.tools.builtins.read_file import ReadFile
from satsumaestro.tools.schema import model_tool_schema


def test_json_schemas_shape() -> None:
    reg = default_registry()
    schemas = reg.json_schemas()
    assert schemas
    for entry in schemas:
        assert set(entry) == {"name", "description", "input_schema"}
        assert isinstance(entry["input_schema"], dict)


def test_json_schemas_matches_input_model() -> None:
    reg = default_registry()
    by_name = {e["name"]: e for e in reg.json_schemas()}
    assert by_name["read_file"]["input_schema"] == model_tool_schema(ReadFile.Input)


def test_json_schemas_covers_all_builtins() -> None:
    reg = default_registry()
    names = {e["name"] for e in reg.json_schemas()}
    assert names == {
        "read_file",
        "glob_search",
        "grep_search",
        "write_file",
        "edit_file",
        "bash",
        "web_fetch",
        "web_search",
        "todo_write",
        "enter_plan_mode",
        "exit_plan_mode",
        "notebook_edit",
        "agent",
        "bash_output",
        "tool_search",
        "lsp",
        "delete_path",
        "append_file",
        "list_directory",
    }
