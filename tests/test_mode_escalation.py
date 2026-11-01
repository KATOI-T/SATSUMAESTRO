"""モード昇格テスト。"""

from __future__ import annotations

from satsumaestro.security import (
    PermissionMode,
    PermissionPolicy,
    default_policy,
)

RO = PermissionMode.READ_ONLY
WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS
PROMPT = PermissionMode.PROMPT
ALLOW = PermissionMode.ALLOW


def test_read_only_blocks_write() -> None:
    p = default_policy(RO)
    assert p.authorize("write_file", {"path": "a"}, WW).decision == "ask"
    assert p.authorize("read_file", {"path": "a"}, RO).decision == "allow"


def test_workspace_write_allows_write_blocks_bash() -> None:
    p = default_policy(WW)
    assert p.authorize("write_file", {"path": "a"}, WW).decision == "allow"
    assert p.authorize("bash", {"command": "ls"}, DFA).decision == "ask"


def test_danger_allows_everything_by_mode() -> None:
    p = default_policy(DFA)
    assert p.authorize("bash", {"command": "rm x"}, DFA).decision == "allow"
    assert p.authorize("write_file", {"path": "a"}, WW).decision == "allow"


def test_prompt_mode_always_asks() -> None:
    p = PermissionPolicy(mode=PROMPT)
    assert p.authorize("read_file", {"path": "a"}, RO).decision == "ask"
    assert p.authorize("bash", {"command": "ls"}, DFA).decision == "ask"


def test_allow_mode_always_allows() -> None:
    p = PermissionPolicy(mode=ALLOW)
    assert p.authorize("bash", {"command": "rm -rf /"}, DFA).decision == "allow"


def test_allow_mode_still_respects_denied_tools() -> None:
    p = PermissionPolicy(mode=ALLOW, blocked_tools=["bash"])
    assert p.authorize("bash", {"command": "ls"}, DFA).decision == "deny"
