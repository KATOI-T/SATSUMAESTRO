"""単一 MCP サーバ接続の薄いラッパ。"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from satsumaestro.config.models import McpServerConfig
from satsumaestro.i18n.text import ui
from satsumaestro.util.errors import ConfigError

HANDSHAKE_TIMEOUT_S = 30.0


async def open_session(stack: AsyncExitStack, cfg: McpServerConfig) -> ClientSession:
    """``cfg`` のトランスポートで接続し、``initialize`` 済みセッションを返す。"""
    if cfg.transport == "stdio":
        if not cfg.command:
            raise ConfigError(ui("mcp.client.a_stdio_mcp_server_needs"))
        params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None)
        read, write = await stack.enter_async_context(stdio_client(params))
    else:
        if not cfg.url:
            raise ConfigError(ui("mcp.client.a_streamable_http_mcp_server"))
        read, write, _ = await stack.enter_async_context(streamablehttp_client(cfg.url))
    session = await stack.enter_async_context(ClientSession(read, write))
    await asyncio.wait_for(session.initialize(), timeout=HANDSHAKE_TIMEOUT_S)
    return session
