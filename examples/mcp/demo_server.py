"""動作確認用の自作 MCP サーバ（stdio トランスポート）。

`.satsuma/settings.local.json` の ``mcp_servers`` へ登録し、``satsumaestro mcp list`` /
``satsumaestro prompt`` で MCP 経路（接続 → ツール列挙 → 権限 → call_tool）を
手動確認するためのサンプル。鍵・ネットワーク不要。

    uv run python examples/mcp/demo_server.py   # 単体起動（stdio なので通常は不要）

pytest 用のモックは [tests/fixtures/mock_mcp.py] にあり、用途が異なる（自動テスト専用）。
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("demo")


@server.tool()
def upper_echo(text: str) -> str:
    """Return the text in upper case."""
    return text.upper()


@server.tool()
def server_info() -> str:
    """Return the name and the tools of this server."""
    return "demo MCP server v1 / tools=upper_echo,server_info"


if __name__ == "__main__":
    server.run(transport="stdio")
