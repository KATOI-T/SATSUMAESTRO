"""enforcer.classify テスト。"""

from __future__ import annotations

from satsumaestro.security import PermissionMode, classify

RO = PermissionMode.READ_ONLY
WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS


def test_bash_always_top() -> None:
    assert classify("bash", {"command": "ls"}, RO) is DFA
    assert classify("bash", {"command": "rm -rf /"}, WW) is DFA


def test_write_tools() -> None:
    assert classify("write_file", {"path": "a"}, RO) is WW
    assert classify("edit_file", {"path": "a"}, RO) is WW


def test_read_tools() -> None:
    for name in (
        "read_file",
        "glob_search",
        "grep_search",
        "web_fetch",
        "rag_search",
        "list_directory",
    ):
        assert classify(name, {}, WW) is RO


def test_unknown_tool_uses_spec_default() -> None:
    assert classify("mcp__fs__read", {}, RO) is RO
    assert classify("mcp__fs__write", {}, WW) is WW
