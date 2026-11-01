"""ユーザー発話による plan 承認の判定。"""

from __future__ import annotations

import re

APPROVAL_TEXT_LIMIT = 80

_APPROVE_JA = re.compile(
    r"(実装(して|に移|に進|を進め|を始め|開始|に入|お願い|で)|"
    r"(進め|すすめ)(て|ろ|よう|ましょう|ください)|"
    r"(始め|はじめ|開始し|着手し|作業し|実行し|やっ|やり|作っ|書い)(て|ろ|よう|ましょう|ください)|"
    r"承認|オッケー|おっけー|それで(いい|お願い|進め|よろしく)|(いい|良い)(よ|です|と思います|ね)$|"
    r"(お願い|よろしく)(します|お願いします)?$|"
    r"(この|その)(計画|プラン|内容)で)"
)
_APPROVE_EN = re.compile(
    r"\b(approved?|go ahead|proceed|implement(\s+it|\s+this|\s+the plan)?|do it|lgtm|"
    r"looks good|ship it|let'?s go|start( implementing| the implementation)?|"
    r"sounds good|make it so|execute( the plan)?|run it)\b",
    re.IGNORECASE,
)
_BARE_YES = re.compile(
    r"^(ok|okay|yes|y|yep|sure|go|はい|うん|ok(です|で)?|了解|おk|ええ)[.!。！]?$",
    re.IGNORECASE,
)

_NEGATE = re.compile(
    r"(ない|ません|まだ|だめ|ダメ|待っ|やめ|中止|保留|直し|修正|変更|見直|やり直|ではなく|"
    r"\bnot\b|\bdon'?t\b|\bwait\b|\bno\b|\bhold\b|\bstop\b|\bbut\b|\bhowever\b|\binstead\b|"
    r"\bchange\b|\brevise\b|\bfix\b)",
    re.IGNORECASE,
)


def has_negation(text: str) -> bool:
    """否定・保留・条件の語を含むか（「続けないで」「まだ待って」）。"""
    return _NEGATE.search(text) is not None


def is_plan_approval_text(text: str) -> bool:
    """ユーザー発話が「計画を承認して実装へ進め」の意図か（保守的判定）。"""
    t = text.strip()
    if not t or len(t) > APPROVAL_TEXT_LIMIT:
        return False
    if t.endswith(("?", "？")):
        return False
    if _BARE_YES.match(t):
        return True
    if has_negation(t):
        return False
    return bool(_APPROVE_JA.search(t) or _APPROVE_EN.search(t))
