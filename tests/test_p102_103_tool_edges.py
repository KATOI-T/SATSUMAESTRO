"""edit_file の old == new と、MCP のツール名の正規化。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent

from satsumaestro.config.models import McpServerConfig, RuntimeConfig
from satsumaestro.mcp import manager as manager_mod
from satsumaestro.mcp.manager import MCPManager
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.tools import default_registry, edit_match
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.edit_file import EditFile
from satsumaestro.tools.mcp_bridge import mcp_tool_name, normalize_tool, register_mcp_tools


def _ctx(root: Path) -> ToolContext:
    jail = WorkspaceJail(root)
    return ToolContext(workspace_root=jail.root, jail=jail, config=RuntimeConfig())


@pytest.mark.parametrize("replace_all", [False, True])
async def test_same_old_and_new_changes_nothing(tmp_path: Path, replace_all: bool) -> None:
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    inp = EditFile.Input(path="a.py", old="x = 1", new="x = 1", replace_all=replace_all)

    res = await EditFile().run(inp, _ctx(tmp_path))

    assert res.is_error
    assert res.output.startswith("old and new are the same, so a.py was not changed")
    assert res.flags == [edit_match.FLAG_MISMATCH]
    assert path.read_text(encoding="utf-8") == "x = 1\n"


async def test_a_real_change_still_applies(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text("x = 1\n", encoding="utf-8")
    res = await EditFile().run(
        EditFile.Input(path="a.py", old="x = 1", new="x = 2"), _ctx(tmp_path)
    )
    assert not res.is_error
    assert path.read_text(encoding="utf-8") == "x = 2\n"


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("get.weather", "get_weather"),
        ("read file", "read_file"),
        ("my-tool", "my-tool"),
        ("search_v2", "search_v2"),
    ],
)
def test_tool_names_keep_only_what_the_api_accepts(raw: str, normalized: str) -> None:
    assert normalize_tool(raw) == normalized
    assert mcp_tool_name("github.com", raw) == f"mcp__github_com__{normalized}"


class _FakeSession(ClientSession):
    """ツール一覧と呼び出しだけを持つ擬似のセッション（接続はしない）。"""

    listed: list[str]
    called: list[str]

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        tools = [
            SimpleNamespace(name=n, description=f"tool {n}", inputSchema={"type": "object"})
            for n in self.listed
        ]
        return SimpleNamespace(tools=tools)

    async def call_tool(self, name: str, *args: Any, **kwargs: Any) -> CallToolResult:
        self.called.append(name)
        return CallToolResult(content=[TextContent(type="text", text=f"ran {name}")])


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    session = object.__new__(_FakeSession)
    session.listed = ["get.weather", "get_weather", "my-tool"]
    session.called = []

    async def _open(stack: object, cfg: object) -> _FakeSession:
        return session

    monkeypatch.setattr(manager_mod, "open_session", _open)
    return session


async def test_mcp_tools_are_registered_by_normalized_name_and_called_by_raw_name(
    fake_session: _FakeSession,
) -> None:
    mgr = await MCPManager.start({"srv": McpServerConfig(transport="stdio", command="x")})
    try:
        (handle,) = mgr.handles()
        assert [t.name for t in handle.tools] == ["mcp__srv__get_weather", "mcp__srv__my-tool"]
        assert handle.raw_names["mcp__srv__get_weather"] == "get.weather"

        registry = default_registry()
        assert register_mcp_tools(registry, mgr) == 2
        tool = registry.get("mcp__srv__get_weather")
        assert tool is not None
        res = await tool().run(tool.Input(), _ctx(Path.cwd()))
        assert res.output == "ran get.weather"
        assert fake_session.called == ["get.weather"]
    finally:
        await mgr.aclose()
