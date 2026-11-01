"""MCP ツール → registry ブリッジ。"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from satsumaestro.mcp.manager import MCPManager
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.registry import ToolRegistry


def normalize_server(server: str) -> str:
    """サーバ名をツール名に使える形へ正規化する（例 ``github.com`` → ``github_com``）。"""
    return re.sub(r"[^A-Za-z0-9_]", "_", server)


def normalize_tool(tool: str) -> str:
    """ツール名を API が受け付ける形へ正規化する（例 ``get.weather`` → ``get_weather``）。"""
    return re.sub(r"[^A-Za-z0-9_-]", "_", tool)


def mcp_tool_name(server: str, tool: str) -> str:
    """``mcp__<正規化サーバ名>__<正規化ツール名>`` を組む。"""
    return f"mcp__{normalize_server(server)}__{normalize_tool(tool)}"


def raw_tool_name(server: str, spec_name: str) -> str:
    """``mcp__<server>__`` プレフィックスを外す。"""
    prefix = f"mcp__{normalize_server(server)}__"
    return spec_name[len(prefix) :] if spec_name.startswith(prefix) else spec_name


def to_toolspec(server: str, t: Any) -> ToolSpec:
    """MCP ``Tool`` を :class:`ToolSpec` へ写す。"""
    return ToolSpec(
        name=mcp_tool_name(server, t.name),
        description=t.description or "",
        input_schema=t.inputSchema or {"type": "object"},
        required_mode=PermissionMode.PROMPT,
        parallel_safe=False,
    )


class _McpInput(BaseModel):
    """任意の MCP 入力スキーマを通す passthrough（検証は MCP サーバ側に委ねる）。"""

    model_config = ConfigDict(extra="allow")


def make_bridge_tool(mgr: MCPManager, server: str, tool: str, spec: ToolSpec) -> type[Tool]:
    """1 MCP ツール = 1 :class:`Tool` クラスを生成する。"""

    class _BridgeTool:
        spec: ClassVar[ToolSpec]
        Input: ClassVar[type[BaseModel]] = _McpInput

        async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
            return await mgr.call(server, tool, inp.model_dump())

    _BridgeTool.spec = spec
    return _BridgeTool


def register_mcp_tools(registry: ToolRegistry, mgr: MCPManager) -> int:
    """接続済みサーバの全ツールを registry へ登録し、登録数を返す。"""
    count = 0
    for handle in mgr.handles():
        if not handle.connected:
            continue
        for spec in handle.tools:
            raw = handle.raw_names.get(spec.name) or raw_tool_name(handle.name, spec.name)
            registry.register(make_bridge_tool(mgr, handle.name, raw, spec), strict=False)
            count += 1
    return count
