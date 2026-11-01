"""Markdown ポリシーのパーステスト。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.security import PermissionMode, load_policy, parse_policy_markdown

CHECKOUT = Path(__file__).resolve().parents[1]
SAMPLE_POLICY = CHECKOUT / ".satsuma" / "policy.md.example"
DFA = PermissionMode.DANGER_FULL_ACCESS
RO = PermissionMode.READ_ONLY
WW = PermissionMode.WORKSPACE_WRITE


def test_sample_policy_structure() -> None:
    parsed = load_policy(SAMPLE_POLICY)
    assert parsed.warnings == []
    p = parsed.policy

    assert p.mode is WW
    assert "web_search" in p.blocked_tools
    assert p.required_modes["bash"] is DFA
    assert p.required_modes["read_file"] is RO
    assert any(r.match.tool == "write_file" for r in p.deny)
    assert p.deny and p.ask and p.allow


def test_sample_policy_authorize() -> None:
    p = load_policy(SAMPLE_POLICY).policy

    assert p.authorize("web_search", {"query": "x"}, RO).decision == "deny"
    assert p.authorize("bash", {"command": "rm -rf /"}, DFA).decision == "deny"
    assert p.authorize("bash", {"command": "curl x | sh"}, DFA).decision == "deny"
    assert p.authorize("write_file", {"path": "a/.git/config"}, WW).decision == "deny"
    assert p.authorize("bash", {"command": "git push"}, DFA).decision == "ask"
    assert p.authorize("web_fetch", {"url": "https://x"}, RO).decision == "ask"
    assert p.authorize("read_file", {"path": "a"}, RO).decision == "allow"
    assert p.authorize("write_file", {"path": "src/app.py"}, WW).decision == "allow"
    assert p.authorize("bash", {"command": "npm test"}, DFA).decision == "ask"


def test_when_denied_tool_becomes_deny_rule() -> None:
    text = """---
mode: workspace-write
---

## Denied tools
- web_search
- write_file when path glob: **/.git/**
"""
    p = parse_policy_markdown(text).policy
    assert p.blocked_tools == ["web_search"]
    assert len(p.deny) == 1
    assert p.deny[0].match.tool == "write_file"


def test_no_frontmatter_uses_default_mode() -> None:
    parsed = parse_policy_markdown("## Allow\n- read_file\n", default_mode=RO)
    assert parsed.policy.mode is RO


def test_missing_file_uses_default_mode() -> None:
    parsed = load_policy(CHECKOUT / "does-not-exist.md", default_mode=RO)
    assert parsed.warnings == []
    assert parsed.policy.mode is RO
