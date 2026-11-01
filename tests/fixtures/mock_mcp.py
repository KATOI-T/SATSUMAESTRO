"""テスト用の最小 stdio MCP サーバ。"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("mock")


@server.tool()
def echo(text: str) -> str:
    """入力 ``text`` をそのまま返す。"""
    return text


@server.tool()
def big() -> str:
    """truncate 検証用に長い文字列を返す。"""
    return "x" * 200_000


if __name__ == "__main__":
    server.run(transport="stdio")
