"""要求に載せる量の見積もりと、1 回の応答のツールの結果の上限。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from satsumaestro.i18n.text import ui
from satsumaestro.session.models import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.tools.builtins.read_file import FULL_READ_CAP_NOTE
from satsumaestro.util.tokens import estimate_upper

_MIN_BUDGET_TOKENS = 1_000
_MIN_KEEP_TOKENS = 200
WINDOW_INPUT_SHARE = 0.9
_MESSAGE_OVERHEAD = 4
_LOWER_CHARS_PER_TOKEN = 8
_ARG_ESCAPED = "<>&"
_ARG_ESCAPE_TOKENS = 5


def estimate_lower(text: str) -> int:
    """``text`` のトークン数を、少なめに出る側で見積もる（切り詰めの検知に使う）。"""
    return len(text) // _LOWER_CHARS_PER_TOKEN


def _block_texts(message: ConversationMessage) -> list[str]:
    texts: list[str] = []
    for block in message.blocks:
        if isinstance(block, TextBlock):
            texts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            texts.append(block.name + json.dumps(block.input, ensure_ascii=False))
        elif isinstance(block, ToolResultBlock):
            texts.append(block.output)
    return texts


def message_floor(message: ConversationMessage) -> int:
    """発言 1 つの見積もり（少なめに出る側）。"""
    return sum(estimate_lower(t) for t in _block_texts(message))


def request_floor(system: str, tools_json: str, messages: Sequence[ConversationMessage]) -> int:
    """要求全体の見積もり（少なめに出る側）。"""
    return (
        estimate_lower(system)
        + estimate_lower(tools_json)
        + sum(message_floor(m) for m in messages)
    )


def tool_input_extra(message: ConversationMessage) -> int:
    """ツールの呼び出しの引数を、サーバが ``<`` ``>`` ``&`` を描き直した形で数えるときの足し分。"""
    escaped = 0
    for block in message.blocks:
        if isinstance(block, ToolUseBlock):
            text = json.dumps(block.input, ensure_ascii=False)
            escaped += sum(text.count(c) for c in _ARG_ESCAPED)
    return _ARG_ESCAPE_TOKENS * escaped


def message_tokens(message: ConversationMessage) -> int:
    """発言 1 つの見積もり（多めに出る側）。"""
    return (
        _MESSAGE_OVERHEAD
        + sum(estimate_upper(t) for t in _block_texts(message))
        + tool_input_extra(message)
    )


def request_tokens(system: str, tools_json: str, messages: Sequence[ConversationMessage]) -> int:
    """要求全体の見積もり（多めに出る側）。"""
    return (
        estimate_upper(system)
        + estimate_upper(tools_json)
        + sum(message_tokens(m) for m in messages)
    )


def input_limit(window: int) -> int | None:
    """1 回の要求に載せてよい量。"""
    return int(window * WINDOW_INPUT_SHARE) if window > 0 else None


def result_budget(threshold: int, share: float) -> int | None:
    """1 回の応答のツールの結果に使える量。"""
    if threshold <= 0 or share <= 0:
        return None
    return max(int(threshold * share), _MIN_BUDGET_TOKENS)


@dataclass(frozen=True)
class ResultCut:
    """切ったツールの結果 1 つ（画面の 1 行と記録に使う）。"""

    tool: str
    kept_tokens: int
    total_tokens: int
    budget: int
    lines: tuple[int, int] | None = None
    total_lines: int | None = None

    def describe(self) -> str:
        """利用者向けの 1 行。"""
        tool, kept, budget = self.tool, f"{self.kept_tokens:,}", f"{self.budget:,}"
        limit = ui("engine.context_budget.the_limit_for_the_results", budget=budget)
        if self.lines is not None:
            first, last = f"{self.lines[0]:,}", f"{self.lines[1]:,}"
            if self.total_lines:
                total = f"{self.total_lines:,}"
                return ui(
                    "engine.context_budget.cut_lines_of_total",
                    tool=tool,
                    first=first,
                    last=last,
                    total=total,
                    kept=kept,
                    limit=limit,
                )
            return ui(
                "engine.context_budget.cut_lines",
                tool=tool,
                first=first,
                last=last,
                kept=kept,
                limit=limit,
            )
        total_tokens = f"{self.total_tokens:,}"
        return ui(
            "engine.context_budget.cut_head",
            tool=tool,
            kept=kept,
            total_tokens=total_tokens,
            limit=limit,
        )


def fit_tool_results(
    results: Sequence[ToolResultBlock], calls: Sequence[ToolUseBlock], budget: int
) -> tuple[list[ToolResultBlock], list[ResultCut]]:
    """結果の合計が ``budget`` を超えたら、大きい結果から同じ量まで切る。"""
    sizes = [estimate_upper(r.output) for r in results]
    if sum(sizes) <= budget:
        return list(results), []
    level = max(_water_level(sizes, budget), _MIN_KEEP_TOKENS)
    inputs = {c.id: c.input for c in calls}
    fitted: list[ToolResultBlock] = []
    cuts: list[ResultCut] = []
    for block, size in zip(results, sizes, strict=True):
        if size <= level:
            fitted.append(block)
            continue
        cut_block, cut = _cut(block, size, level, budget, inputs.get(block.tool_use_id, {}))
        fitted.append(cut_block)
        cuts.append(cut)
    return fitted, cuts


def fit_history_results(
    messages: Sequence[ConversationMessage], budget: int
) -> tuple[list[ConversationMessage], list[ResultCut]]:
    """会話にあるツールの結果の合計を ``budget`` に収める。"""
    results = [
        b for m in messages if m.role == "tool" for b in m.blocks if isinstance(b, ToolResultBlock)
    ]
    calls = [b for m in messages for b in m.blocks if isinstance(b, ToolUseBlock)]
    fitted, cuts = fit_tool_results(results, calls, budget)
    if not cuts:
        return list(messages), []
    replaced = {id(old): new for old, new in zip(results, fitted, strict=True) if old is not new}
    out = [
        m.model_copy(update={"blocks": [replaced.get(id(b), b) for b in m.blocks]})
        if any(id(b) in replaced for b in m.blocks)
        else m
        for m in messages
    ]
    return out, cuts


def tool_result_tokens(messages: Sequence[ConversationMessage]) -> int:
    """会話にあるツールの結果の本文の見積もりの合計。"""
    return sum(
        estimate_upper(b.output)
        for m in messages
        if m.role == "tool"
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    )


def _water_level(sizes: Sequence[int], budget: int) -> int:
    """大きい結果だけを同じ量まで削って、合計を ``budget`` に収めるときの量。"""
    remaining = budget
    ordered = sorted(sizes)
    for i, size in enumerate(ordered):
        left = len(ordered) - i
        if size * left > remaining:
            return remaining // left
        remaining -= size
    return ordered[-1] if ordered else budget


def _prefix_chars(text: str, size: int, level: int) -> int:
    """見積もりが ``level`` 以下に収まる先頭の文字数。"""
    end = int(len(text) * level / size)
    while end > 0 and estimate_upper(text[:end]) > level:
        end = int(end * 0.9)
    return end


def _cut(
    block: ToolResultBlock, size: int, level: int, budget: int, args: Mapping[str, Any]
) -> tuple[ToolResultBlock, ResultCut]:
    text = block.output
    head = text[: _prefix_chars(text, size, level)]
    newline = head.rfind("\n")
    if newline >= len(head) // 2:
        head = head[: newline + 1]
    kept = estimate_upper(head)
    if block.tool_name == "read_file" and not block.is_error and head.endswith("\n"):
        first = max(_int_arg(args, "offset"), 1)
        last = first + head.count("\n") - 1
        end = _line_count(text, first)
        if _int_arg(args, "limit") > 0:
            total = None
            whole = f"of the requested {first:,}-{end:,}"
        elif text.endswith(FULL_READ_CAP_NOTE):
            total = None
            whole = f"of more than {last:,}"
        else:
            total = end
            whole = f"of {total:,}"
        cut = ResultCut(block.tool_name, kept, size, budget, (first, last), total)
        note = (
            f"\n[read_file: this result was cut to fit the context window. Shown: lines "
            f"{first:,}-{last:,} {whole} (about {kept:,} of {size:,} tokens; the tool results "
            f"of one reply may use about {budget:,} tokens). Read the rest in parts, e.g. "
            f"read_file offset={last + 1} limit={last - first + 1}.]"
        )
    else:
        cut = ResultCut(block.tool_name, kept, size, budget)
        note = (
            f"\n\n[{block.tool_name}: this result was cut to fit the context window. Shown: the "
            f"first {len(head):,} of {len(text):,} characters (about {kept:,} of {size:,} tokens; "
            f"the tool results of one reply may use about {budget:,} tokens). Ask for less at a "
            "time (a narrower range, path or pattern) to see the rest.]"
        )
    return block.model_copy(update={"output": head + note}), cut


def _int_arg(args: Mapping[str, Any], name: str) -> int:
    value = args.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _line_count(text: str, first: int) -> int:
    """結果の本文が ``first`` 行目から始まるときの、最後の行の番号。"""
    lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    return first + lines - 1


__all__ = [
    "WINDOW_INPUT_SHARE",
    "ResultCut",
    "estimate_lower",
    "estimate_upper",
    "fit_history_results",
    "fit_tool_results",
    "input_limit",
    "message_floor",
    "message_tokens",
    "request_floor",
    "request_tokens",
    "result_budget",
    "tool_input_extra",
    "tool_result_tokens",
]
