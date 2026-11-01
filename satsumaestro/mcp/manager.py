"""複数 MCP サーバのライフサイクル管理。"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from satsumaestro.config.models import McpServerConfig
from satsumaestro.mcp.client import open_session
from satsumaestro.tools.base import ToolResult, ToolSpec

_log = logging.getLogger(__name__)

CALL_TIMEOUT_S = 60.0
_MAX_OUTPUT_CHARS = 60_000


class MCPServerHandle(BaseModel):
    """1 MCP サーバの接続状態・ツール一覧・失敗理由。"""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    connected: bool
    session: ClientSession | None = None
    tools: list[ToolSpec] = []
    raw_names: dict[str, str] = {}
    error: str | None = None


class MCPManager:
    """設定された MCP サーバ群への接続・ツール集約・確実な後始末。"""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._servers: dict[str, MCPServerHandle] = {}

    @classmethod
    async def start(cls, servers: dict[str, McpServerConfig]) -> MCPManager:
        """全サーバへ接続する。"""
        mgr = cls()
        for name, cfg in servers.items():
            try:
                await mgr._connect(name, cfg)
            except Exception as e:  # noqa: BLE001
                mgr._degrade(name, e)
        return mgr

    async def _connect(self, name: str, cfg: McpServerConfig) -> None:
        from satsumaestro.tools.mcp_bridge import to_toolspec

        session = await open_session(self._stack, cfg)
        listed = await session.list_tools()
        tools: list[ToolSpec] = []
        raw_names: dict[str, str] = {}
        for t in listed.tools:
            spec = to_toolspec(name, t)
            if spec.name in raw_names:
                _log.warning(
                    "MCP server %s: tool %r has the same name as %r after normalizing (%s); "
                    "skipped",
                    name,
                    t.name,
                    raw_names[spec.name],
                    spec.name,
                )
                continue
            raw_names[spec.name] = t.name
            tools.append(spec)
        self._servers[name] = MCPServerHandle(
            name=name, connected=True, session=session, tools=tools, raw_names=raw_names
        )
        _log.info("MCP server %s connected: %d tool(s)", name, len(tools))

    def _degrade(self, name: str, err: Exception) -> None:
        _log.warning("MCP server %s degraded: %s", name, err)
        self._servers[name] = MCPServerHandle(
            name=name, connected=False, error=f"{type(err).__name__}: {err}"
        )

    def handles(self) -> list[MCPServerHandle]:
        """doctor / ``mcp`` 表示用にサーバハンドルを返す（設定順を保持）。"""
        return list(self._servers.values())

    async def call(self, server: str, tool: str, args: dict[str, object]) -> ToolResult:
        """MCP ツールを 1 回呼び出し、:class:`ToolResult` へ写す。"""
        handle = self._servers.get(server)
        if handle is None or handle.session is None or not handle.connected:
            return ToolResult(output=f"MCP server '{server}' is not connected", is_error=True)
        try:
            res = await asyncio.wait_for(
                handle.session.call_tool(tool, args), timeout=CALL_TIMEOUT_S
            )
        except TimeoutError:
            return ToolResult(output=f"MCP tool '{tool}' timed out", is_error=True)
        except Exception as e:  # noqa: BLE001
            _log.warning("MCP call %s/%s failed: %s", server, tool, e)
            return ToolResult(output=f"{type(e).__name__}: {e}", is_error=True)
        return _to_tool_result(res)

    async def aclose(self) -> None:
        """全 session/subprocess を確実に閉じる（逆順 aclose・子プロセス回収）。"""
        await self._stack.aclose()
        self._servers.clear()


def _to_tool_result(res: CallToolResult) -> ToolResult:
    """``CallToolResult`` の content を平文へ連結し、上限で truncate する。"""
    parts = [c.text for c in res.content if isinstance(c, TextContent)]
    text = "\n".join(parts) if parts else ""
    if len(text) > _MAX_OUTPUT_CHARS:
        text = text[:_MAX_OUTPUT_CHARS] + "\n… (output truncated because it was too long)"
    return ToolResult(output=text, is_error=bool(res.isError))
