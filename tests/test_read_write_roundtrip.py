"""write→read 往復テスト。"""

from __future__ import annotations

import pytest

from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.edit_file import EditFile
from satsumaestro.tools.builtins.read_file import ReadFile
from satsumaestro.tools.builtins.write_file import WriteFile
from satsumaestro.util.errors import PathEscapeError


async def test_write_then_read(tool_ctx: ToolContext) -> None:
    await WriteFile().run(WriteFile.Input(path="dir/a.txt", content="日本語\nsecond"), tool_ctx)
    result = await ReadFile().run(ReadFile.Input(path="dir/a.txt"), tool_ctx)
    assert result.output == "日本語\nsecond"
    assert (tool_ctx.jail.root / "dir" / "a.txt").read_text(encoding="utf-8") == "日本語\nsecond"


async def test_read_offset_limit(tool_ctx: ToolContext) -> None:
    await WriteFile().run(WriteFile.Input(path="lines.txt", content="l1\nl2\nl3\nl4\n"), tool_ctx)
    result = await ReadFile().run(ReadFile.Input(path="lines.txt", offset=2, limit=2), tool_ctx)
    assert result.output == "l2\nl3\n"


async def test_write_escapes_jail(tool_ctx: ToolContext) -> None:
    with pytest.raises(PathEscapeError):
        await WriteFile().run(WriteFile.Input(path="../evil.txt", content="x"), tool_ctx)


async def test_edit_replace(tool_ctx: ToolContext) -> None:
    await WriteFile().run(WriteFile.Input(path="e.txt", content="foo bar foo"), tool_ctx)
    res = await EditFile().run(EditFile.Input(path="e.txt", old="foo", new="baz"), tool_ctx)
    assert res.is_error and res.output.startswith("old string is not unique in e.txt")
    await EditFile().run(
        EditFile.Input(path="e.txt", old="foo", new="baz", replace_all=True), tool_ctx
    )
    result = await ReadFile().run(ReadFile.Input(path="e.txt"), tool_ctx)
    assert result.output == "baz bar baz"


async def test_edit_missing_old_is_error(tool_ctx: ToolContext) -> None:
    await WriteFile().run(WriteFile.Input(path="m.txt", content="abc"), tool_ctx)
    res = await EditFile().run(EditFile.Input(path="m.txt", old="zzz", new="x"), tool_ctx)
    assert res.is_error and res.output.startswith("old string not found in m.txt")
    assert (await ReadFile().run(ReadFile.Input(path="m.txt"), tool_ctx)).output == "abc"
