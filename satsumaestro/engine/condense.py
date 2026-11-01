"""要約の前に、畳む部分から読む価値の薄いメッセージを間引く。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import PurePosixPath

from satsumaestro.config.models import CondenseConfig
from satsumaestro.session.models import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

_READING = frozenset({"read_file"})
_CHANGING = frozenset({"write_file", "edit_file", "append_file"})
_CHANGING_ANY = _CHANGING | {"delete_path", "notebook_edit"}

_BRIEF_CHARS = 160
_QUOTE_CHARS = 100
_GROUP_PATHS = 8


@dataclass
class CondenseReport:
    """間引いた内訳。"""

    messages_before: int = 0
    messages_after: int = 0
    chars_before: int = 0
    chars_after: int = 0
    stale_dropped: int = 0
    brief_runs: int = 0
    brief_folded: int = 0
    step_groups: int = 0
    steps_grouped: int = 0

    @property
    def tokens_saved(self) -> int:
        """減った文字数の 1/4（トークン数の目安）。"""
        return max(0, self.chars_before - self.chars_after) // 4

    def describe(self) -> str:
        return (
            f"condensed {self.messages_before} -> {self.messages_after} messages "
            f"(about {self.tokens_saved} tokens fewer): "
            f"{self.stale_dropped} stale file operation(s) dropped, "
            f"{self.brief_runs} run(s) of brief messages folded ({self.brief_folded} messages), "
            f"{self.step_groups} group(s) of similar tool steps ({self.steps_grouped} messages)"
        )


def condense(
    messages: list[ConversationMessage], cfg: CondenseConfig
) -> tuple[list[ConversationMessage], CondenseReport]:
    """3 つの規則を順に当てる（渡したリストは変えない）。"""
    report = CondenseReport(messages_before=len(messages), chars_before=_size_in_chars(messages))
    out = list(messages)
    if cfg.supersede_enabled:
        out = _drop_stale_file_ops(out, report)
    if cfg.collapse_enabled:
        out = _fold_brief_runs(out, cfg.collapse_threshold, report)
    if cfg.cluster_enabled:
        out = _group_similar_steps(out, cfg, report)
    report.messages_after = len(out)
    report.chars_after = _size_in_chars(out)
    return out, report


def _drop_stale_file_ops(
    messages: list[ConversationMessage], report: CondenseReport
) -> list[ConversationMessage]:
    """あとで書き換えたファイルへの、それより前の読み書きを落とす。"""
    changed_later: set[str] = set()
    stale: set[str] = set()
    for msg in reversed(messages):
        calls = _calls(msg)
        for call in calls:
            if call.name in _READING | _CHANGING and _file_path(call) in changed_later:
                stale.add(call.id)
        changed_later.update(
            p for c in calls if c.name in _CHANGING and (p := _file_path(c)) is not None
        )

    kept: list[ConversationMessage] = []
    dropped_calls: set[str] = set()
    for msg in messages:
        calls = _calls(msg)
        answers = [b.tool_use_id for b in msg.blocks if isinstance(b, ToolResultBlock)]
        if calls and _text_len(msg) < _BRIEF_CHARS and all(c.id in stale for c in calls):
            dropped_calls.update(c.id for c in calls)
            report.stale_dropped += 1
        elif answers and all(a in dropped_calls for a in answers):
            report.stale_dropped += 1
        else:
            kept.append(msg)
    return kept


def _is_brief(msg: ConversationMessage) -> bool:
    if msg.role not in ("user", "assistant") or _has_tool_block(msg):
        return False
    return _text_len(msg) < _BRIEF_CHARS


def _fold_brief_runs(
    messages: list[ConversationMessage], min_run: int, report: CondenseReport
) -> list[ConversationMessage]:
    """短い発言が ``min_run`` 件以上続く所を、発言を引用した 1 件の覚え書きにまとめる。"""
    out: list[ConversationMessage] = []
    for brief, run_iter in groupby(messages, key=_is_brief):
        run = list(run_iter)
        if not brief or len(run) < min_run:
            out.extend(run)
            continue
        lines = [f"({len(run)} brief messages in a row)"]
        lines.extend(f"{m.role}: {_quote(m)}" for m in run)
        out.append(ConversationMessage(role="user", blocks=[TextBlock(text="\n".join(lines))]))
        report.brief_runs += 1
        report.brief_folded += len(run)
    return out


def _is_tool_step(msg: ConversationMessage) -> bool:
    """まとめてよいツール手順か（ファイルを変える手順は含めない）。"""
    if not _has_tool_block(msg) or _text_len(msg) >= _BRIEF_CHARS:
        return False
    changing = any(isinstance(b, ToolUseBlock) and b.name in _CHANGING_ANY for b in msg.blocks)
    answering = any(
        isinstance(b, ToolResultBlock) and b.tool_name in _CHANGING_ANY for b in msg.blocks
    )
    return not (changing or answering)


def _features(msg: ConversationMessage) -> frozenset[str]:
    """似ているかを比べる特徴: 役割・ツール名・触ったディレクトリ・大きさの桁（4 倍ごと）。"""
    feats = {f"role={msg.role}"}
    size = 0
    for b in msg.blocks:
        if isinstance(b, ToolUseBlock):
            feats.add(f"tool={b.name}")
            if (path := _file_path(b)) is not None:
                feats.add(f"dir={PurePosixPath(path).parent}")
            size += len(str(b.input))
        elif isinstance(b, ToolResultBlock):
            feats.add(f"tool={b.tool_name}")
            size += len(b.output)
        elif isinstance(b, TextBlock):
            size += len(b.text)
    feats.add(f"size={size.bit_length() // 2}")
    return frozenset(feats)


def _similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """共通の特徴の数 ÷ どちらかにある特徴の数（Jaccard 係数）。"""
    return len(a & b) / len(a | b)


def _group_similar_steps(
    messages: list[ConversationMessage], cfg: CondenseConfig, report: CondenseReport
) -> list[ConversationMessage]:
    """似たツール手順を、最初の 1 件の位置に置いた 1 行へまとめる。"""
    steps = [i for i, m in enumerate(messages) if _is_tool_step(m)]
    if len(steps) > cfg.cluster_max_messages:
        return messages
    feats = {i: _features(messages[i]) for i in steps}
    taken: set[int] = set()
    groups: dict[int, list[int]] = {}
    for pos, anchor in enumerate(steps):
        if anchor in taken:
            continue
        members = [anchor] + [
            j
            for j in steps[pos + 1 :]
            if j not in taken
            and _similarity(feats[anchor], feats[j]) >= cfg.cluster_similarity_threshold
        ]
        if len(members) >= cfg.cluster_min_size:
            taken.update(members)
            groups[anchor] = members
            report.step_groups += 1
            report.steps_grouped += len(members)

    out: list[ConversationMessage] = []
    for i, msg in enumerate(messages):
        if i in groups:
            out.append(_group_note([messages[k] for k in groups[i]]))
        elif i not in taken:
            out.append(msg)
    return out


def _group_note(group: list[ConversationMessage]) -> ConversationMessage:
    names = sorted(
        {b.name for m in group for b in m.blocks if isinstance(b, ToolUseBlock)}
        | {b.tool_name for m in group for b in m.blocks if isinstance(b, ToolResultBlock)}
    )
    paths = sorted({p for m in group for c in _calls(m) if (p := _file_path(c)) is not None})
    results = [b for m in group for b in m.blocks if isinstance(b, ToolResultBlock)]
    note = f"({len(group)} similar tool steps) tools: {', '.join(names)}"
    if paths:
        shown = ", ".join(paths[:_GROUP_PATHS])
        rest = len(paths) - _GROUP_PATHS
        note += f"; files: {shown}" + (f" and {rest} more" if rest > 0 else "")
    if results:
        failed = sum(1 for r in results if r.is_error)
        note += f"; results: {len(results) - failed} ok, {failed} failed"
    return ConversationMessage(role="assistant", blocks=[TextBlock(text=note)])


def _calls(msg: ConversationMessage) -> list[ToolUseBlock]:
    return [b for b in msg.blocks if isinstance(b, ToolUseBlock)]


def _has_tool_block(msg: ConversationMessage) -> bool:
    return any(isinstance(b, ToolUseBlock | ToolResultBlock) for b in msg.blocks)


def _file_path(call: ToolUseBlock) -> str | None:
    value = call.input.get("path")
    return value if isinstance(value, str) and value else None


def _text_len(msg: ConversationMessage) -> int:
    return sum(len(b.text) for b in msg.blocks if isinstance(b, TextBlock))


def _quote(msg: ConversationMessage) -> str:
    text = " ".join(" ".join(b.text for b in msg.blocks if isinstance(b, TextBlock)).split())
    if not text:
        return "(empty)"
    return text if len(text) <= _QUOTE_CHARS else text[: _QUOTE_CHARS - 1] + "…"


def _size_in_chars(messages: list[ConversationMessage]) -> int:
    """本文・ツールの入力・結果の文字数の合計。"""
    total = 0
    for msg in messages:
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                total += len(b.text)
            elif isinstance(b, ToolUseBlock):
                total += len(str(b.input))
            elif isinstance(b, ToolResultBlock):
                total += len(b.output)
    return total


__all__ = ["CondenseReport", "condense"]
