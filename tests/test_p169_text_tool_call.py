"""見出しの後ろに置いた、テキストのツール呼び出しも検出する。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.engine.runtime import _detect_text_tool_call
from satsumaestro.util.errors import ToolCallingUnsupportedError
from tests.test_p339_text_only_reply import _runtime, _text

_TOOLS = {"read_file", "write_file", "glob_search", "bash"}
_WRITE = '{"name": "write_file", "parameters": {"path": "plan.md", "content": "x"}}'
_GLOB = '{"name": "glob_search", "parameters": {"path": ".", "pattern": "**/*"}}'


@pytest.mark.parametrize(
    ("text", "name"),
    [
        (f"回答\n{_WRITE}\nplan.md は plan mode にて読み込まれます。", "write_file"),
        (f"検索対象\n {_GLOB}\n\n結果\nファイルの一覧が取得されます。", "glob_search"),
        (f"回答\n```json\n{_WRITE}\n```", "write_file"),
        (f"確認します。\n対象\n{_GLOB}", "glob_search"),
    ],
)
def test_a_call_after_a_heading_is_detected(text: str, name: str) -> None:
    assert _detect_text_tool_call(text, _TOOLS) == name


@pytest.mark.parametrize(
    "text",
    [
        f"手順を説明します。\n1 行目\n2 行目\n3 行目\n{_WRITE}",
        f"呼び出しは {_WRITE} の形で送ります。",
        '回答\n{"name": "list_workspace", "parameters": {}}',
        '回答\n{"name": "write_file", "path": "a"}',
        '結果\n{"answer": 42}',
        '回答\n{"name": "write_file", "parameters": {',
    ],
)
def test_other_json_is_not_detected(text: str) -> None:
    assert _detect_text_tool_call(text, _TOOLS) is None


async def test_the_turn_stops_with_a_clear_error(tmp_path: Path) -> None:
    rt, script, _ = _runtime(tmp_path)
    script[:] = [_text(f"検索対象\n {_GLOB}\n\n結果\nファイルの一覧が取得されます。")]
    with pytest.raises(ToolCallingUnsupportedError):
        await rt.run_turn("消えたファイルを探して")
