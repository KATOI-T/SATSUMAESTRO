"""権限ポリシーのモデルと ``authorize`` 評価。"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

import regex
from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode, mode_le

_log = logging.getLogger(__name__)

_MAX_MATCH_LEN = 8192
_MATCH_TIMEOUT_S = 1.0


def _glob_to_regex(pat: str) -> str:
    """glob パターンを正規表現へ変換する（``**`` は ``/`` を跨ぐ再帰マッチ）。"""

    i, n, out = 0, len(pat), []
    while i < n:
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("/**", i) and i + 3 == n:
            out.append("(?:/.*)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    return "".join(out)


def _regex_search(pattern: regex.Pattern[str], text: str) -> bool:
    """長さ制限＋実タイムアウト付きで ``search`` を実行する。"""

    clipped = text[:_MAX_MATCH_LEN]
    try:
        return pattern.search(clipped, timeout=_MATCH_TIMEOUT_S) is not None
    except TimeoutError:
        _log.warning("regex match timed out (possible ReDoS): %s", pattern.pattern)
        return False


_FIELD_KEYS: dict[str, tuple[str, ...]] = {
    "command": ("command",),
    "path": ("path", "notebook_path"),
    "url": ("url",),
}


def field_value(tool_input: dict[str, Any], field: str) -> object:
    """規則の欄 ``field`` に当たる入力の値（無ければ ``None``）。"""
    for key in _FIELD_KEYS.get(field, (field,)):
        if key in tool_input:
            return tool_input[key]
    return None


class RuleMatch(BaseModel):
    """1 ルールのマッチ条件。"""

    tool: str | None = None
    field: Literal["command", "path", "url"] | None = None
    glob: str | None = None
    regex: str | None = None

    def _tool_matches(self, tool_name: str) -> bool:
        return self.tool in (None, "*") or self.tool == tool_name

    def matches(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """``tool_name`` / ``tool_input`` がこのルールに合致するか。"""

        if not self._tool_matches(tool_name):
            return False
        has_matcher = self.glob is not None or self.regex is not None
        if self.field is None and not has_matcher:
            return True
        if self.field is None or not has_matcher:
            return False
        value = field_value(tool_input, self.field)
        if not isinstance(value, str):
            return False
        if self.glob is not None:
            return re.fullmatch(_glob_to_regex(self.glob), value) is not None
        if self.regex is not None:
            try:
                compiled = regex.compile(self.regex)
            except regex.error:
                return False
            return _regex_search(compiled, value)
        return False


class PermissionRule(BaseModel):
    """マッチ条件 + 由来ラベル（診断用）。"""

    match: RuleMatch
    label: str = ""


class PermissionOutcome(BaseModel):
    """authorize の判定結果。"""

    decision: Literal["allow", "deny", "ask"]
    reason: str = ""
    remember: bool = False
    origin: str | None = None
    scoped: bool = False


Verdict = Literal["deny", "ask", "allow"]

BLOCKED_TOOL_SUFFIX = " is listed under Denied tools in the policy"


class PermissionPolicy(BaseModel):
    """policy.md を読んだ権限の下地。"""

    mode: PermissionMode
    required_modes: dict[str, PermissionMode] = Field(default_factory=dict)
    blocked_tools: list[str] = Field(default_factory=list)
    deny: list[PermissionRule] = Field(default_factory=list)
    ask: list[PermissionRule] = Field(default_factory=list)
    allow: list[PermissionRule] = Field(default_factory=list)

    def authorize(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        required: PermissionMode,
        override: PermissionOutcome | None = None,
    ) -> PermissionOutcome:
        """1 回のツール呼び出しを allow / deny / ask に決める。"""
        stop = self.hard_deny(tool_name, tool_input)
        if stop is not None:
            return stop
        if override is not None and override.decision != "allow":
            return override
        asked = self._first_rule("ask", tool_name, tool_input)
        if asked is not None:
            return asked
        if override is not None:
            return override
        allowed = self._first_rule("allow", tool_name, tool_input)
        if allowed is not None and self._allow_reaches(required):
            return allowed
        return self._by_mode(required)

    def hard_deny(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionOutcome | None:
        """policy の拒否（名前で止めるツールと ``## Deny`` の規則）だけを見る。"""
        if tool_name in self.blocked_tools:
            return PermissionOutcome(decision="deny", reason=tool_name + BLOCKED_TOOL_SUFFIX)
        return self._first_rule("deny", tool_name, tool_input)

    def explicit(self, tool_name: str, tool_input: dict[str, Any]) -> PermissionOutcome | None:
        """書いてある規則（拒否 → 確認 → 許可）だけで決める。"""
        return (
            self.hard_deny(tool_name, tool_input)
            or self._first_rule("ask", tool_name, tool_input)
            or self._first_rule("allow", tool_name, tool_input)
        )

    def _first_rule(
        self, verdict: Verdict, tool_name: str, tool_input: dict[str, Any]
    ) -> PermissionOutcome | None:
        """``verdict`` の規則のうち、書いた順で最初に当たったもの。"""
        rules = {"deny": self.deny, "ask": self.ask, "allow": self.allow}[verdict]
        matching = (r for r in rules if r.match.matches(tool_name, tool_input))
        if verdict == "allow":
            hits = list(matching)
            hit = next((r for r in hits if r.match.field is not None), hits[0] if hits else None)
        else:
            hit = next(matching, None)
        if hit is None:
            return None
        return PermissionOutcome(
            decision=verdict,
            reason=f"{verdict} rule: {hit.label}",
            scoped=hit.match.field is not None,
        )

    def _allow_reaches(self, required: PermissionMode) -> bool:
        """``## Allow`` に当たった呼び出しを、ツールが求めるモードの点で通してよいか。"""
        return required is PermissionMode.PROMPT or mode_le(required, self.mode)

    def _by_mode(self, required: PermissionMode) -> PermissionOutcome:
        """どの規則にも当たらなかった呼び出しを、基準のモードで決める。"""
        if self.mode is PermissionMode.ALLOW:
            return PermissionOutcome(decision="allow", reason="mode=allow")
        if self.mode is PermissionMode.PROMPT:
            return PermissionOutcome(decision="ask", reason="mode=prompt")
        if mode_le(required, self.mode):
            return PermissionOutcome(decision="allow", reason="within active mode")
        return PermissionOutcome(
            decision="ask",
            reason=f"requires {required.value} > {self.mode.value}",
            origin="mode-escalation",
        )


_DEFAULT_READ_TOOLS = (
    "read_file",
    "glob_search",
    "list_directory",
    "grep_search",
    "rag_search",
)


def default_policy(mode: PermissionMode) -> PermissionPolicy:
    """ポリシー未配置/パース失敗時のデフォルト（Allow に読み取り系のみ）。"""

    allow = [
        PermissionRule(match=RuleMatch(tool=t), label=f"default allow: {t}")
        for t in _DEFAULT_READ_TOOLS
    ]
    return PermissionPolicy(mode=mode, allow=allow)
