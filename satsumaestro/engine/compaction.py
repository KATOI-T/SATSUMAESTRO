"""自動コンパクション。"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from satsumaestro.config.models import CondenseConfig, RuntimeConfig, SamplingConfig
from satsumaestro.engine.condense import CondenseReport, condense
from satsumaestro.engine.context_budget import message_tokens
from satsumaestro.engine.plan_request import PLAN_NOTE_MARKER
from satsumaestro.session.models import (
    ConversationMessage,
    Session,
    SessionCompaction,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.util.clock import now_ms
from satsumaestro.util.lang import LANGUAGE_DIRECTIVE

if TYPE_CHECKING:
    from satsumaestro.providers.base import Provider

_log = logging.getLogger(__name__)

_SUMMARY_PREFIX = "[Earlier conversation summarized to save context]"

_CONTINUATION_NOTICE = (
    "The conversation reached its context limit, so the older messages were replaced by "
    "the summary below. Everything after this notice is the recent exchange, kept as it "
    "was. Pick up the work from this point: treat the summary as settled, do not redo or "
    "revisit what it records, and do not ask the user again for anything it already "
    "contains."
)

MAX_TRANSCRIPT_CHARS = 24_000
_TRANSCRIPT_WINDOW_SHARE = 0.6
_TAIL_WINDOW_DIVISOR = 4
_SUMMARY_SYSTEM_PROMPT = (
    "You compress earlier parts of a coding-assistant conversation to save context. "
    "Summarize the transcript below into a concise but faithful summary that preserves "
    "decisions made, files changed, tool results, open questions, and any state the "
    "assistant must remember to continue. Use terse bullet points. Do not invent details."
)


@dataclass(frozen=True)
class CompactionLimits:
    """圧縮 1 回の量の上限（窓から決める）。"""

    tail_tokens: int | None = None
    transcript_chars: int = MAX_TRANSCRIPT_CHARS


def limits_for(
    window: int, threshold: int, fixed_tokens: int, summary_max_tokens: int
) -> CompactionLimits:
    """窓・圧縮の閾値・要求の固定部（system とツール定義）の見積もりから、圧縮の量を決める。"""
    if window <= 0:
        return CompactionLimits()
    tail = window // _TAIL_WINDOW_DIVISOR
    if fixed_tokens > 0 and threshold > 0:
        tail = min(tail, max((threshold - fixed_tokens) // 2, 0))
    transcript = int((window - summary_max_tokens) * _TRANSCRIPT_WINDOW_SHARE)
    return CompactionLimits(tail, max(min(MAX_TRANSCRIPT_CHARS, transcript), 0))


@dataclass(frozen=True)
class CompactionOutcome:
    """1 回の圧縮結果。"""

    summary: str
    dropped_messages: int
    condensed: CondenseReport
    used_llm: bool


class Compactor:
    """コンパクション実行器。"""

    def __init__(
        self,
        keep_last: int = 6,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        use_llm_summary: bool = True,
        llm_summary_max_tokens: int = 1024,
        condense: CondenseConfig | None = None,
        sampling: SamplingConfig | None = None,
    ) -> None:
        self._keep_last = keep_last
        self._provider = provider
        self._model = model
        self._use_llm_summary = use_llm_summary
        self._llm_summary_max_tokens = llm_summary_max_tokens
        self._condense = condense or CondenseConfig()
        self._sampling = sampling

    @classmethod
    def from_config(
        cls,
        cfg: RuntimeConfig,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        sampling: SamplingConfig | None = None,
    ) -> Compactor:
        """``RuntimeConfig.compaction`` から構築する（結線ヘルパ）。"""
        c = cfg.compaction
        return cls(
            c.keep_last,
            provider=provider,
            model=model,
            use_llm_summary=c.use_llm_summary,
            llm_summary_max_tokens=c.llm_summary_max_tokens,
            condense=c.trident,
            sampling=sampling if sampling is not None else cfg.provider.sampling,
        )

    def needs_compaction(self, usage: TokenUsage, cfg: RuntimeConfig) -> bool:
        """渡されたトークン量が閾値超過なら True（閾値 0 以下は無効）。"""
        if cfg.compact_token_threshold <= 0:
            return False
        used = usage.input_tokens + usage.output_tokens
        return used >= cfg.compact_token_threshold

    async def acompact(
        self,
        session: Session,
        *,
        allow_llm: bool = True,
        limits: CompactionLimits | None = None,
    ) -> CompactionOutcome | None:
        """要約の前の間引き → 要約（LLM 既定/機械 fallback）→ 継続メッセージ差し込み。"""
        limits = limits or CompactionLimits()
        cut = _cut_index(session.messages, self._keep_last, limits.tail_tokens)
        if cut is None:
            return None
        dropped = session.messages[:cut]
        kept = session.messages[cut:]

        condensed, report = condense([_flatten_prior_summary(m) for m in dropped], self._condense)
        asks = _asks_section(_asks_of(dropped))

        summary_text: str | None = None
        used_llm = False
        if (
            allow_llm
            and self._use_llm_summary
            and self._provider is not None
            and self._model
            and limits.transcript_chars > 0
        ):
            try:
                summary_text = await _summarize_llm(
                    condensed,
                    self._provider,
                    self._model,
                    max_tokens=self._llm_summary_max_tokens,
                    sampling=self._sampling,
                    transcript_chars=limits.transcript_chars,
                )
            except Exception as e:
                _log.warning("LLM summary failed; falling back to mechanical: %s", e)
                summary_text = None
            if summary_text is not None and summary_text.strip():
                summary_text = summary_text.strip()
                if asks:
                    summary_text = "\n".join(asks) + "\n\n" + summary_text
                used_llm = True
            else:
                summary_text = None
        if summary_text is None:
            summary_text = _rule_summary(dropped)

        session.messages = [_continuation_message(summary_text), *kept]
        session.compaction = SessionCompaction(
            compacted_at_ms=now_ms(),
            summary=summary_text,
            dropped_messages=len(dropped),
        )
        return CompactionOutcome(
            summary=summary_text,
            dropped_messages=len(dropped),
            condensed=report,
            used_llm=used_llm,
        )

    def compact(self, session: Session) -> bool:
        """既存同期 API（機械要約のみ）。"""
        cut = _cut_index(session.messages, self._keep_last)
        if cut is None:
            return False
        dropped = session.messages[:cut]
        kept = session.messages[cut:]
        summary_text = _rule_summary(dropped)
        summary_msg = ConversationMessage(
            role="user",
            blocks=[TextBlock(text=f"{_SUMMARY_PREFIX}\n{summary_text}")],
        )
        session.messages = [summary_msg, *kept]
        session.compaction = SessionCompaction(
            compacted_at_ms=now_ms(),
            summary=summary_text,
            dropped_messages=len(dropped),
        )
        return True


def _cut_index(
    messages: list[ConversationMessage], keep_last: int, tail_tokens: int | None = None
) -> int | None:
    """畳む/残すの境界 index を返す。"""
    total = len(messages)
    kept = min(keep_last, total)
    if tail_tokens is not None and kept > 1:
        used = message_tokens(messages[-1])
        count = 1
        while count < kept:
            size = message_tokens(messages[-1 - count])
            if used + size > tail_tokens:
                break
            used += size
            count += 1
        kept = count
    cut = total - kept
    if cut <= 1:
        return None
    if cut == total:
        return cut
    while cut > 0 and messages[cut].role == "tool":
        cut -= 1
    return cut if cut > 0 else None


def _continuation_message(summary_text: str) -> ConversationMessage:
    """継続定型＋要約本文の合成 user メッセージ。"""
    return ConversationMessage(
        role="user",
        blocks=[TextBlock(text=f"{_SUMMARY_PREFIX}\n{_CONTINUATION_NOTICE}\n\n{summary_text}")],
    )


def _flatten_prior_summary(msg: ConversationMessage) -> ConversationMessage:
    """以前の要約メッセージから定型（接頭辞・継続の文）を剥がし、印を付けて本文だけを渡す。"""
    body = _summary_body(msg)
    if body is None:
        return msg
    return ConversationMessage(
        role="user",
        blocks=[TextBlock(text=f"{_FOLDED_MARK}\n{body}")],
        usage=msg.usage,
    )


_ASK_COUNT = 3
_ASK_CHARS = 160
_FIRST_ASK_CHARS = 300
_NOTE_CHARS = 240
_CARRY_CHARS = 300
_PATH_COUNT = 10
_PATH_LINE_CHARS = 300
_USAGE_LINE_CHARS = 200
_STEP_COUNT = 5
_STEP_CHARS = 100

_CHANGING_TOOLS = frozenset(
    {"write_file", "edit_file", "append_file", "delete_path", "notebook_edit"}
)
_PATH_KEYS = ("path", "notebook_path")
_TARGET_KEYS = ("path", "notebook_path", "command", "pattern", "query", "url")

_FOLDED_MARK = "[summary from an earlier compaction]"

_FOLDED_HEAD = "Earlier messages folded:"
_ASKS_HEAD = "What the user asked (oldest first):"
_CARRY_HEAD = "Carried over from the previous summary: "
_CARRY_FIRST = ("Files changed:", "Last assistant note:")
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_NOTE_HEADS = (PLAN_NOTE_MARKER, LANGUAGE_DIRECTIVE.split("{name}", 1)[0])


def _one_line(text: str, limit: int) -> str:
    """空白を詰めて 1 行にし、``limit`` 字を超えたら末尾を「…」で切る。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _words_of(message: ConversationMessage) -> str:
    return " ".join(b.text for b in message.blocks if isinstance(b, TextBlock)).strip()


def _summary_body(message: ConversationMessage) -> str | None:
    """以前の圧縮で作った要約メッセージなら、定型を除いた本文を返す。"""
    if message.role != "user" or len(message.blocks) != 1:
        return None
    block = message.blocks[0]
    if not isinstance(block, TextBlock) or not block.text.startswith(_SUMMARY_PREFIX):
        return None
    body = block.text.removeprefix(_SUMMARY_PREFIX).lstrip("\n")
    return body.removeprefix(_CONTINUATION_NOTICE).lstrip("\n")


def _ask_text(message: ConversationMessage) -> str:
    """利用者の発言から、ハーネスが足した文を除いた本文。"""
    text = _REMINDER_RE.sub("", _words_of(message))
    for head in _NOTE_HEADS:
        text = text.split(head, 1)[0]
    return text.strip()


def _prior_asks(body: str) -> list[str]:
    """以前の要約の依頼の欄にある依頼を、古い順に返す。"""
    lines = body.splitlines()
    if _ASKS_HEAD not in lines:
        return []
    asks: list[str] = []
    for line in lines[lines.index(_ASKS_HEAD) + 1 :]:
        if not line.startswith("  - "):
            break
        asks.append(line[4:])
    return asks


def _asks_of(messages: list[ConversationMessage]) -> list[str]:
    """畳む発言にある依頼を古い順に返す。"""
    asks: list[str] = []
    for m in messages:
        if (body := _summary_body(m)) is not None:
            asks.extend(_prior_asks(body))
        elif m.role == "user" and (text := _ask_text(m)):
            asks.append(text)
    return asks


def _asks_section(asks: list[str]) -> list[str]:
    """依頼の欄の行。"""
    if not asks:
        return []
    first = asks[0]
    rest = [a for a in asks[1:] if a != first][-_ASK_COUNT:]
    return [
        _ASKS_HEAD,
        f"  - {_one_line(first, _FIRST_ASK_CHARS)}",
        *(f"  - {_one_line(a, _ASK_CHARS)}" for a in rest),
    ]


def _carry_text(body: str) -> str:
    """以前の要約から持ち越す文。"""
    first: list[str] = []
    rest: list[str] = []
    in_asks = False
    for line in body.splitlines():
        if line.startswith(_FOLDED_HEAD):
            continue
        if line == _ASKS_HEAD:
            in_asks = True
            continue
        if in_asks and line.startswith("  - "):
            continue
        in_asks = False
        line = line.removeprefix(_CARRY_HEAD).strip()
        if line:
            (first if line.startswith(_CARRY_FIRST) else rest).append(line)
    return " ".join([*first, *rest])


def _input_text(call: ToolUseBlock, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = call.input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _latest_unique(items: list[str], count: int) -> list[str]:
    """重複を除き、最後に現れた順で新しいほうから ``count`` 件（古い順に並べ直して返す）。"""
    seen: dict[str, None] = {}
    for item in reversed(items):
        if item not in seen:
            seen[item] = None
        if len(seen) == count:
            break
    return list(reversed(seen))


def _rule_summary(messages: list[ConversationMessage]) -> str:
    """畳むメッセージ群から、LLM を使わずに作業の要点を組み立てる（決定的）。"""
    by_role = Counter(m.role for m in messages)
    lines = [
        f"{_FOLDED_HEAD} {len(messages)} (user {by_role['user']}, "
        f"assistant {by_role['assistant']}, tool results {by_role['tool']})"
    ]

    carried = ""
    note = ""
    calls: list[ToolUseBlock] = []
    failed: dict[str, bool] = {}
    for m in messages:
        if (body := _summary_body(m)) is not None:
            carried = _carry_text(body)
            continue
        words = _words_of(m)
        if m.role == "assistant" and words:
            note = words
        for b in m.blocks:
            if isinstance(b, ToolUseBlock):
                calls.append(b)
            elif isinstance(b, ToolResultBlock):
                failed[b.tool_use_id] = b.is_error

    lines.extend(_asks_section(_asks_of(messages)))
    if carried:
        lines.append(f"{_CARRY_HEAD}{_one_line(carried, _CARRY_CHARS)}")
    if note:
        lines.append(f"Last assistant note: {_one_line(note, _NOTE_CHARS)}")

    changed = [p for c in calls if c.name in _CHANGING_TOOLS and (p := _input_text(c, _PATH_KEYS))]
    read = [p for c in calls if c.name == "read_file" and (p := _input_text(c, _PATH_KEYS))]
    if changed:
        files = ", ".join(_latest_unique(changed, _PATH_COUNT))
        lines.append(_one_line(f"Files changed: {files}", _PATH_LINE_CHARS))
    if read:
        files = ", ".join(_latest_unique(read, _PATH_COUNT))
        lines.append(_one_line(f"Files read: {files}", _PATH_LINE_CHARS))

    if calls:
        uses = Counter(c.name for c in calls)
        errors = Counter(c.name for c in calls if failed.get(c.id))
        parts = [
            f"{name} {n}" + (f" ({errors[name]} failed)" if errors[name] else "")
            for name, n in sorted(uses.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        lines.append(_one_line("Tool use: " + ", ".join(parts), _USAGE_LINE_CHARS))
        lines.append("Last steps:")
        for c in calls[-_STEP_COUNT:]:
            outcome = "no result" if c.id not in failed else ("failed" if failed[c.id] else "ok")
            target = _input_text(c, _TARGET_KEYS)
            step = f"{c.name} {target}: {outcome}" if target else f"{c.name}: {outcome}"
            lines.append(f"  - {_one_line(step, _STEP_CHARS)}")
    return "\n".join(lines)


RULE_SUMMARY_MAX_CHARS = (
    80
    + (len(_CARRY_HEAD) + _CARRY_CHARS + 1)
    + (len(_ASKS_HEAD) + 1)
    + (4 + _FIRST_ASK_CHARS + 1)
    + _ASK_COUNT * (4 + _ASK_CHARS + 1)
    + (len("Last assistant note: ") + _NOTE_CHARS + 1)
    + 2 * (_PATH_LINE_CHARS + 1)
    + (_USAGE_LINE_CHARS + 1)
    + (len("Last steps:") + 1)
    + _STEP_COUNT * (4 + _STEP_CHARS + 1)
)


async def _summarize_llm(
    messages: list[ConversationMessage],
    provider: Provider,
    model: str,
    *,
    max_tokens: int = 1024,
    sampling: SamplingConfig | None = None,
    transcript_chars: int = MAX_TRANSCRIPT_CHARS,
) -> str:
    """LLM を用いた意味的要約（既定経路）。"""
    from satsumaestro.engine.events import closing_stream, fold_events
    from satsumaestro.providers.base import ApiRequest

    transcript = _render_transcript(messages)[-transcript_chars:]
    req = ApiRequest(
        model=model,
        system=_SUMMARY_SYSTEM_PROMPT,
        messages=[ConversationMessage(role="user", blocks=[TextBlock(text=transcript)])],
        tools=[],
        max_tokens=max_tokens,
        thinking=False,
        sampling=sampling,
    )
    async with closing_stream(provider.stream(req)) as events:
        folded = await fold_events(events)
    return "".join(b.text for b in folded.blocks if isinstance(b, TextBlock)).strip()


_MAX_BLOCK_CHARS = 1_000


def _clip_block(text: str) -> str:
    if len(text) <= _MAX_BLOCK_CHARS:
        return text
    return text[:_MAX_BLOCK_CHARS] + f"…({len(text) - _MAX_BLOCK_CHARS} chars trimmed)"


def _render_transcript(messages: list[ConversationMessage]) -> str:
    """メッセージ群を LLM 要約入力用の可読テキストへ平坦化する（ブロック単位で上限）。"""
    lines: list[str] = []
    for m in messages:
        parts: list[str] = []
        for b in m.blocks:
            if isinstance(b, TextBlock):
                parts.append(_clip_block(b.text))
            elif isinstance(b, ThinkingBlock):
                parts.append(f"(thinking) {_clip_block(b.thinking)}")
            elif isinstance(b, ToolUseBlock):
                parts.append(f"(tool call: {b.name} {_clip_block(str(b.input))})")
            elif isinstance(b, ToolResultBlock):
                status = "error" if b.is_error else "ok"
                parts.append(f"(tool result [{status}] {b.tool_name}: {_clip_block(b.output)})")
        lines.append(f"{m.role}: " + " ".join(parts))
    return "\n".join(lines)
