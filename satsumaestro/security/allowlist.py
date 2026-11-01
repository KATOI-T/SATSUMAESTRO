"""3層許可の解決レイヤ: session > memory > policy。"""

from __future__ import annotations

import logging
from pathlib import Path

import regex

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionOutcome, PermissionPolicy, RuleMatch, field_value
from satsumaestro.security.policy_markdown import parse_policy_markdown
from satsumaestro.security.remember_guard import remember_blocker
from satsumaestro.util.jsonl import write_text_atomic

_log = logging.getLogger(__name__)

_PRIMARY_FIELD: dict[str, str] = {
    "bash": "command",
    "read_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "append_file": "path",
    "notebook_edit": "path",
    "web_fetch": "url",
}

_MEMORY_TEMPLATE = (
    "# SATSUMAESTRO permission memory (per-user settings / the memory layer)\n"
    '# Records the actions answered with "always allow (a)" in the REPL. You can edit it.\n'
    "# Precedence: session > this file (memory) > policy.md (a deny in policy always wins)\n"
)
_ALLOW_HEADING = "## Allow"


def signature_for(tool_name: str, tool_input: dict[str, object]) -> RuleMatch:
    """呼び出しから「厳密一致」の :class:`RuleMatch` を作る。"""
    field = _PRIMARY_FIELD.get(tool_name)
    value = field_value(dict(tool_input), field) if field is not None else None
    if field is None or not isinstance(value, str):
        return RuleMatch(tool=tool_name)
    pattern = r"\A" + regex.escape(value) + r"\Z"
    return RuleMatch(tool=tool_name, field=field, regex=pattern)


def rule_to_line(rule: RuleMatch) -> str:
    """:class:`RuleMatch` を policy 文法の 1 行（``- `` 抜き本文）へ直列化する。"""
    if rule.field is not None and rule.regex is not None:
        return f"{rule.tool} when {rule.field} re: {rule.regex}"
    return str(rule.tool)


class SessionAllowlist:
    """当該プロセス（REPL セッション）限りの in-memory allowlist。"""

    def __init__(self) -> None:
        self._rules: list[RuleMatch] = []
        self.first_use_cleared: set[str] = set()

    def find(self, tool_name: str, tool_input: dict[str, object]) -> RuleMatch | None:
        """当たった許可（無ければ ``None``）。"""
        return next((r for r in self._rules if r.matches(tool_name, tool_input)), None)

    def matches(self, tool_name: str, tool_input: dict[str, object]) -> bool:
        return self.find(tool_name, tool_input) is not None

    def add(self, rule: RuleMatch) -> None:
        line = rule_to_line(rule)
        if all(rule_to_line(r) != line for r in self._rules):
            self._rules.append(rule)


class MemoryLayer:
    """Markdown で書かれたユーザ個別の権限レイヤ（``.satsuma/memory.md`` 等）。"""

    def __init__(self, path: Path, *, writable: bool) -> None:
        self.path = path
        self.writable = writable
        self._policy = self._load()

    def _load(self) -> PermissionPolicy:
        if not self.path.is_file():
            return PermissionPolicy(mode=PermissionMode.READ_ONLY)
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as e:
            _log.warning("cannot read memory layer %s: %s", self.path, e)
            return PermissionPolicy(mode=PermissionMode.READ_ONLY)
        return parse_policy_markdown(text).policy

    def match(self, tool_name: str, tool_input: dict[str, object]) -> PermissionOutcome | None:
        return self._policy.explicit(tool_name, tool_input)

    def remember(self, rule: RuleMatch) -> None:
        """書き込み可能層なら ``## Allow`` へ追記し永続化・再読込する。"""
        if not self.writable:
            return
        line = rule_to_line(rule)
        try:
            base = (
                self.path.read_text(encoding="utf-8") if self.path.is_file() else _MEMORY_TEMPLATE
            )
            if f"- {line}" in base:
                return
            updated = _append_allow_line(base, f"- {line}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(self.path, updated)
        except OSError as e:
            _log.warning("cannot persist memory layer %s: %s", self.path, e)
            return
        self._policy = self._load()


class PermissionResolver:
    """session > memory > policy を解決する権限判定器。"""

    def __init__(
        self,
        base: PermissionPolicy,
        memory_layers: list[MemoryLayer] | None = None,
        session: SessionAllowlist | None = None,
    ) -> None:
        self.base = base
        self.layers = memory_layers or []
        self.session = session or SessionAllowlist()
        self._writable = next((m for m in self.layers if m.writable), None)

    @property
    def required_modes(self) -> dict[str, PermissionMode]:
        return self.base.required_modes

    def authorize(
        self,
        tool_name: str,
        tool_input: dict[str, object],
        required: PermissionMode,
        override: PermissionOutcome | None = None,
    ) -> PermissionOutcome:
        hard = self.base.hard_deny(tool_name, tool_input)
        if hard is not None:
            return hard
        blocker = remember_blocker(tool_name, tool_input)
        rule = self.session.find(tool_name, tool_input) if blocker is None else None
        if rule is not None:
            return PermissionOutcome(
                decision="allow", reason="session allowlist", scoped=rule.field is not None
            )
        for layer in self.layers:
            outcome = layer.match(tool_name, tool_input)
            if outcome is None:
                continue
            if outcome.decision == "allow" and blocker is not None:
                _log.info(
                    "ignored a remembered allow in %s for %s (%s): %s",
                    layer.path,
                    tool_name,
                    blocker,
                    outcome.reason,
                )
                continue
            return outcome
        return self.base.authorize(tool_name, tool_input, required, override)

    def record(self, tool_name: str, tool_input: dict[str, object]) -> None:
        """`a`（永続許可）を session と書き込み可能 memory 層へ記録する。"""
        blocker = remember_blocker(tool_name, tool_input)
        if blocker is not None:
            _log.info("not remembering %s (%s)", tool_name, blocker)
            return
        rule = signature_for(tool_name, tool_input)
        self.session.add(rule)
        if self._writable is not None:
            self._writable.remember(rule)


def _append_allow_line(text: str, item_line: str) -> str:
    """``## Allow`` セクション末尾へ ``item_line`` を挿入した本文を返す。"""
    lines = text.splitlines()
    heading_idx = next((i for i, ln in enumerate(lines) if ln.strip().lower() == "## allow"), None)
    if heading_idx is None:
        prefix = text if text.endswith("\n") or text == "" else text + "\n"
        return f"{prefix}\n{_ALLOW_HEADING}\n{item_line}\n"
    insert_at = len(lines)
    for i in range(heading_idx + 1, len(lines)):
        if lines[i].lstrip().startswith("##"):
            insert_at = i
            break
    while insert_at > heading_idx + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines.insert(insert_at, item_line)
    return "\n".join(lines) + "\n"
