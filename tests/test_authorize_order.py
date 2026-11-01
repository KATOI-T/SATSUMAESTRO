"""authorize 評価順テスト。"""

from __future__ import annotations

from satsumaestro.security import (
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
    PermissionRule,
    RuleMatch,
)
from satsumaestro.security.allowlist import PermissionResolver

RO = PermissionMode.READ_ONLY
WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS


def _policy() -> PermissionPolicy:
    return PermissionPolicy(
        mode=WW,
        blocked_tools=["forbidden"],
        deny=[
            PermissionRule(
                match=RuleMatch(tool="bash", field="command", regex=r"rm"),
                label="deny rm",
            )
        ],
        ask=[
            PermissionRule(
                match=RuleMatch(tool="bash", field="command", regex=r"push"),
                label="ask push",
            )
        ],
        allow=[PermissionRule(match=RuleMatch(tool="read_file"), label="allow read")],
    )


def test_denied_tools_win_first() -> None:
    p = _policy()
    out = p.authorize("forbidden", {}, RO)
    assert out.decision == "deny"
    assert out.reason == "forbidden is listed under Denied tools in the policy"


def test_deny_rule_beats_override() -> None:
    p = _policy()
    allow_override = PermissionOutcome(decision="allow", reason="hook")
    out = p.authorize("bash", {"command": "rm -rf /"}, DFA, override=allow_override)
    assert out.decision == "deny"


def test_ask_rule_beats_an_allowing_hook() -> None:
    p = _policy()
    override = PermissionOutcome(decision="allow", reason="hook")
    out = p.authorize("bash", {"command": "git push"}, DFA, override=override)
    assert out.decision == "ask"
    assert "ask rule" in out.reason


def test_an_allowing_hook_skips_the_mode_escalation() -> None:
    p = _policy()
    override = PermissionOutcome(decision="allow", reason="hook")
    assert p.authorize("bash", {"command": "git status"}, DFA).decision == "ask"
    out = p.authorize("bash", {"command": "git status"}, DFA, override=override)
    assert out.decision == "allow"
    assert out.reason == "hook"


def test_the_resolver_keeps_the_same_order() -> None:
    resolver = PermissionResolver(_policy())
    override = PermissionOutcome(decision="allow", reason="hook")
    out = resolver.authorize("bash", {"command": "git push"}, DFA, override=override)
    assert out.decision == "ask"


def test_a_denying_or_asking_hook_beats_ask_and_allow_rules() -> None:
    p = _policy()
    for decision in ("deny", "ask"):
        override = PermissionOutcome(decision=decision, reason="hook")  # type: ignore[arg-type]
        out = p.authorize("read_file", {"path": "a"}, RO, override=override)
        assert (out.decision, out.reason) == (decision, "hook")


def test_ask_rule_matches() -> None:
    p = _policy()
    out = p.authorize("bash", {"command": "git push"}, DFA)
    assert out.decision == "ask"
    assert "ask rule" in out.reason


def test_allow_rule_within_mode() -> None:
    p = _policy()
    out = p.authorize("read_file", {"path": "a"}, RO)
    assert out.decision == "allow"
    assert "allow rule" in out.reason


def test_mode_fallthrough_allow_and_ask() -> None:
    p = _policy()
    assert p.authorize("other", {}, RO).decision == "allow"
    out = p.authorize("other", {}, DFA)
    assert out.decision == "ask"
    assert "requires" in out.reason


def test_allow_rule_permits_prompt_required_tool() -> None:
    p = PermissionPolicy(
        mode=WW,
        allow=[PermissionRule(match=RuleMatch(tool="mcp__x__t"), label="allow mcp")],
    )
    out = p.authorize("mcp__x__t", {}, PermissionMode.PROMPT)
    assert out.decision == "allow"
    assert "allow rule" in out.reason


def test_prompt_required_tool_without_allow_rule_asks() -> None:
    p = PermissionPolicy(mode=WW)
    assert p.authorize("mcp__x__t", {}, PermissionMode.PROMPT).decision == "ask"


def test_deny_rule_beats_allow_rule_for_prompt_required_tool() -> None:
    p = PermissionPolicy(
        mode=WW,
        deny=[PermissionRule(match=RuleMatch(tool="mcp__x__t"), label="deny mcp")],
        allow=[PermissionRule(match=RuleMatch(tool="mcp__x__t"), label="allow mcp")],
    )
    assert p.authorize("mcp__x__t", {}, PermissionMode.PROMPT).decision == "deny"


def test_allow_rule_blocked_when_required_exceeds_mode() -> None:
    p = PermissionPolicy(
        mode=WW,
        allow=[
            PermissionRule(
                match=RuleMatch(tool="bash", field="command", glob="npm test*"),
                label="allow npm",
            )
        ],
    )
    out = p.authorize("bash", {"command": "npm test"}, DFA)
    assert out.decision == "ask"
