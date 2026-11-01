"""レジストリの型付き dispatch テスト。"""

from __future__ import annotations

import pytest

from satsumaestro.engine.events import ToolUse
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext, ToolResult
from satsumaestro.util.errors import ToolError


@pytest.mark.asyncio
async def test_typed_dispatch_read_file(tool_ctx: ToolContext) -> None:
    (tool_ctx.jail.root / "hello.txt").write_text("hi", encoding="utf-8")
    reg = default_registry()
    call = ToolUse(id="1", name="read_file", input={"path": "hello.txt"})
    inp = reg.validate_input(call)
    result = await reg.run("read_file", inp, tool_ctx)
    assert isinstance(result, ToolResult)
    assert result.output == "hi"
    assert result.is_error is False


def test_unknown_tool_validate() -> None:
    reg = default_registry()
    call = ToolUse(id="1", name="does_not_exist", input={})
    with pytest.raises(ToolError):
        reg.validate_input(call)


@pytest.mark.asyncio
async def test_unknown_tool_run(tool_ctx: ToolContext) -> None:
    reg = default_registry()
    from satsumaestro.tools.builtins.read_file import ReadFile

    with pytest.raises(ToolError):
        await reg.run("does_not_exist", ReadFile.Input(path="x"), tool_ctx)


def test_invalid_input_raises_toolerror() -> None:
    reg = default_registry()
    call = ToolUse(id="1", name="read_file", input={"offset": 1})
    with pytest.raises(ToolError):
        reg.validate_input(call)


def test_get_returns_class_or_none() -> None:
    reg = default_registry()
    assert reg.get("bash") is not None
    assert reg.get("nope") is None
