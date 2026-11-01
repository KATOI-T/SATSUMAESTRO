"""非対話時の ask→deny テスト。"""

from __future__ import annotations

from satsumaestro.security import (
    PermissionMode,
    PermissionOutcome,
    PermissionPolicy,
)

DFA = PermissionMode.DANGER_FULL_ACCESS


def _resolve_noninteractive(outcome: PermissionOutcome, interactive: bool) -> str:
    """ask の最終処理（会話ループ側の想定挙動）: 非対話なら deny、対話ならプロンプトへ。"""

    if outcome.decision == "ask" and not interactive:
        return "deny"
    return outcome.decision


def test_ask_becomes_deny_when_noninteractive() -> None:
    p = PermissionPolicy(mode=PermissionMode.PROMPT)
    out = p.authorize("bash", {"command": "ls"}, DFA)
    assert out.decision == "ask"
    assert _resolve_noninteractive(out, interactive=False) == "deny"


def test_ask_stays_ask_when_interactive() -> None:
    p = PermissionPolicy(mode=PermissionMode.PROMPT)
    out = p.authorize("bash", {"command": "ls"}, DFA)
    assert _resolve_noninteractive(out, interactive=True) == "ask"
