"""外の内容（web・MCP）を読んだ回数（親と子の runtime で共有する）。"""

from __future__ import annotations

from dataclasses import dataclass

WEB_TOOLS = frozenset({"web_fetch", "web_search"})
MCP_PREFIX = "mcp__"


def is_external_tool(name: str) -> bool:
    """結果が外の内容になるツールか。"""
    return name in WEB_TOOLS or name.startswith(MCP_PREFIX)


@dataclass
class ExternalReads:
    """外の内容を読んだ回数と、最後に読んだ物の名前と、最上位のターンの始めの回数。"""

    count: int = 0
    last: str = ""
    turn_mark: int = 0

    def begin_turn(self, top: bool) -> int:
        """ターンの始めの回数（子は最上位のターンの始めの回数を引き継ぐ）。"""
        if top:
            self.turn_mark = self.count
        return self.turn_mark

    def note(self, tool: str, is_error: bool, label: str) -> None:
        """ツールの結果が外の内容なら数える（web の失敗は外を読んでいないので数えない）。"""
        if not is_external_tool(tool) or (is_error and tool in WEB_TOOLS):
            return
        self.count += 1
        self.last = label
