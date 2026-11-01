"""Markdown 権限ポリシーのパーサ。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import regex
from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode, is_required_mode
from satsumaestro.security.policy import (
    PermissionPolicy,
    PermissionRule,
    RuleMatch,
    default_policy,
)

_VALID_FIELDS = {"command", "path", "url"}
_SECTIONS = {
    "denied tools": "denied",
    "deny": "deny",
    "ask": "ask",
    "allow": "allow",
    "tool requirements": "requirements",
}


class ParsedPolicy(BaseModel):
    """パース結果。"""

    policy: PermissionPolicy
    warnings: list[str] = Field(default_factory=list)


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    """先頭 ``---`` フェンスの front-matter 本文と残り本文を返す。"""

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :])
            return fm, body
    return None, ""


def _parse_mode(frontmatter: str, warnings: list[str], default: PermissionMode) -> PermissionMode:
    for raw in frontmatter.splitlines():
        line = raw.strip()
        if not line.startswith("mode:"):
            continue
        value = line[len("mode:") :].split("#", 1)[0].strip()
        try:
            return PermissionMode(value)
        except ValueError:
            warnings.append(f"invalid mode: {value!r} (fallback to {default.value})")
            return default
    return default


def _parse_matcher(rest: str) -> tuple[str | None, str | None, str | None]:
    """``<field> <glob:|re:><value>`` を (field, glob, regex) に分解。"""

    parts = rest.split(None, 1)
    if len(parts) != 2:
        return None, None, None
    field, matcher = parts[0].strip(), parts[1].strip()
    if field not in _VALID_FIELDS:
        return None, None, None
    if matcher.startswith("glob:"):
        return field, matcher[len("glob:") :].strip(), None
    if matcher.startswith("re:"):
        return field, None, matcher[len("re:") :].strip()
    return None, None, None


def _parse_rule(item: str, warnings: list[str], where: str) -> RuleMatch | None:
    """ルール文法 1 行を :class:`RuleMatch` へ。"""

    if " when " not in item:
        tool = item.strip()
        if not tool or any(c.isspace() for c in tool):
            warnings.append(f"{where}: malformed rule: {item!r}")
            return None
        return RuleMatch(tool=tool)
    tool_part, cond = item.split(" when ", 1)
    tool = tool_part.strip()
    if not tool:
        warnings.append(f"{where}: malformed rule (missing tool): {item!r}")
        return None
    field, glob, regexp = _parse_matcher(cond.strip())
    if field is None:
        warnings.append(f"{where}: malformed rule: {item!r}")
        return None
    if regexp is not None:
        try:
            regex.compile(regexp)
        except regex.error as e:
            warnings.append(f"{where}: invalid regex {regexp!r}: {e}")
            return None
    return RuleMatch(tool=tool, field=field, glob=glob, regex=regexp)


def _iter_sections(body: str) -> list[tuple[str, list[str]]]:
    """本文を ``##`` 見出しで分割し (section_key, items) を返す。"""

    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.startswith("##"):
            title = line.lstrip("#").strip().lower()
            key = _SECTIONS.get(title)
            current = (key, []) if key else None
            if current:
                sections.append(current)
            continue
        stripped = line.strip()
        if current is None or not stripped.startswith("- "):
            continue
        current[1].append(stripped[2:].strip())
    return sections


def parse_policy_markdown(
    text: str, default_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE
) -> ParsedPolicy:
    """Markdown 文字列を :class:`ParsedPolicy` にパースする（フェイルセーフ込み）。"""

    warnings: list[str] = []
    fm, body = _split_frontmatter(text)
    if fm is None and not body:
        warnings.append("broken front-matter; falling back to read-only")
        return ParsedPolicy(policy=default_policy(PermissionMode.READ_ONLY), warnings=warnings)

    active = _parse_mode(fm, warnings, default_mode) if fm is not None else default_mode
    policy = PermissionPolicy(mode=active)

    for key, items in _iter_sections(body):
        for item in items:
            if key == "requirements":
                _apply_requirement(policy, item, warnings)
            elif key == "denied":
                _apply_denied(policy, item, warnings)
            elif key in ("deny", "ask", "allow"):
                rule = _parse_rule(item, warnings, key)
                if rule is not None:
                    _bucket(policy, key).append(PermissionRule(match=rule, label=item))
    return ParsedPolicy(policy=policy, warnings=warnings)


def _bucket(policy: PermissionPolicy, key: str) -> list[PermissionRule]:
    return {
        "deny": policy.deny,
        "ask": policy.ask,
        "allow": policy.allow,
    }[key]


def _apply_requirement(policy: PermissionPolicy, item: str, warnings: list[str]) -> None:
    if ":" not in item:
        warnings.append(f"requirements: malformed line: {item!r}")
        return
    tool, value = (p.strip() for p in item.split(":", 1))
    try:
        mode = PermissionMode(value)
    except ValueError:
        warnings.append(f"requirements: invalid mode {value!r} for {tool!r}")
        return
    if not is_required_mode(mode):
        warnings.append(f"requirements: {mode.value!r} is active-mode only for {tool!r}")
        return
    policy.required_modes[tool] = mode


def _apply_denied(policy: PermissionPolicy, item: str, warnings: list[str]) -> None:
    if " when " in item:
        rule = _parse_rule(item, warnings, "denied tools")
        if rule is not None:
            policy.deny.append(PermissionRule(match=rule, label=item))
    else:
        policy.blocked_tools.append(item.strip())


def load_policy(
    policy_file: str | Path,
    default_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE,
) -> ParsedPolicy:
    """ポリシーファイルを読み込む。"""

    path = Path(policy_file)
    if not path.is_file():
        return ParsedPolicy(policy=default_policy(default_mode))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return ParsedPolicy(
            policy=default_policy(PermissionMode.READ_ONLY),
            warnings=[f"cannot read {path}: {e}; falling back to read-only"],
        )
    return parse_policy_markdown(text, default_mode)


def load_policies(
    policy_files: Sequence[str | Path],
    default_mode: PermissionMode = PermissionMode.WORKSPACE_WRITE,
) -> ParsedPolicy:
    """根に近い順に並べた policy.md を重ねる（無いファイルは飛ばす）。"""
    paths = [Path(p) for p in policy_files if Path(p).is_file()]
    if not paths:
        return ParsedPolicy(policy=default_policy(default_mode))
    if len(paths) == 1:
        return load_policy(paths[0], default_mode)
    merged = load_policy(paths[0], default_mode)
    warnings = [f"{paths[0]}: {w}" for w in merged.warnings]
    policy = merged.policy
    for path in paths[1:]:
        parsed = load_policy(path, policy.mode)
        warnings.extend(f"{path}: {w}" for w in parsed.warnings)
        near = parsed.policy
        policy = PermissionPolicy(
            mode=near.mode,
            required_modes={**policy.required_modes, **near.required_modes},
            blocked_tools=[*policy.blocked_tools, *near.blocked_tools],
            deny=[*policy.deny, *near.deny],
            ask=[*policy.ask, *near.ask],
            allow=[*policy.allow, *near.allow],
        )
    return ParsedPolicy(policy=policy, warnings=warnings)
