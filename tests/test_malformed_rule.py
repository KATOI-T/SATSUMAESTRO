"""不正ルールの警告収集テスト。"""

from __future__ import annotations

from satsumaestro.security import PermissionMode, parse_policy_markdown

WW = PermissionMode.WORKSPACE_WRITE
RO = PermissionMode.READ_ONLY


def test_malformed_rules_collected_others_survive() -> None:
    text = r"""---
mode: workspace-write
---

## Deny
- bash when command re: [unclosed
- bash when command re: ^\s*rm
- bash when badfield glob: *

## Allow
- read_file
- bogus when nonsense
"""
    parsed = parse_policy_markdown(text)
    p = parsed.policy

    assert any(r.match.regex == r"^\s*rm" for r in p.deny)
    assert any(r.match.tool == "read_file" for r in p.allow)
    assert len(parsed.warnings) == 3
    joined = "\n".join(parsed.warnings)
    assert "[unclosed" in joined
    assert "badfield" in joined or "nonsense" in joined


def test_invalid_mode_falls_back_with_warning() -> None:
    text = "---\nmode: bogus-mode\n---\n## Allow\n- read_file\n"
    parsed = parse_policy_markdown(text, default_mode=WW)
    assert parsed.policy.mode is WW
    assert any("invalid mode" in w for w in parsed.warnings)


def test_broken_frontmatter_fails_safe_to_read_only() -> None:
    text = "---\nmode: allow\n## Allow\n- bash\n"
    parsed = parse_policy_markdown(text)
    assert parsed.policy.mode is RO
    assert any("front-matter" in w for w in parsed.warnings)


def test_missing_tool_rule_is_malformed_not_noop() -> None:
    text = "---\nmode: workspace-write\n---\n## Deny\n- when path glob: **/x/**\n"
    parsed = parse_policy_markdown(text)
    assert parsed.policy.deny == []
    assert any("malformed" in w for w in parsed.warnings)


def test_empty_tool_in_when_split_is_rejected() -> None:
    from satsumaestro.security.policy_markdown import _parse_rule

    warnings: list[str] = []
    assert _parse_rule(" when path glob: x", warnings, "deny") is None
    assert any("missing tool" in w for w in warnings)


def test_requirement_rejects_active_only_modes() -> None:
    text = (
        "---\nmode: workspace-write\n---\n"
        "## Tool requirements\n- bash: allow\n- web_fetch: prompt\n- read_file: read-only\n"
    )
    parsed = parse_policy_markdown(text)
    assert "bash" not in parsed.policy.required_modes
    assert "web_fetch" not in parsed.policy.required_modes
    assert parsed.policy.required_modes["read_file"] is RO
    assert sum("active-mode only" in w for w in parsed.warnings) == 2


def test_malformed_requirement_line() -> None:
    text = "---\nmode: workspace-write\n---\n## Tool requirements\n- bash danger\n"
    parsed = parse_policy_markdown(text)
    assert parsed.policy.required_modes == {}
    assert any("malformed" in w for w in parsed.warnings)
