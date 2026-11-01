"""承認済み計画を段へ割り、1 段ずつモデルへ渡すための純関数群。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from satsumaestro.engine.plan_approval import (
    APPROVAL_TEXT_LIMIT,
    has_negation,
    is_plan_approval_text,
)
from satsumaestro.util import chunking

MAX_STEP_CHARS = 300

_STEP_TAIL = (
    "When this step is complete, reply with a one-line summary and stop — do not start the "
    "next step. If you are blocked or need a decision from the user, start your reply with "
    "BLOCKED: and explain what you need. Creating directories never needs a decision: "
    "write_file creates parent directories itself."
)
_ALREADY_DONE = (
    "If an earlier step already did this, do not redo or polish it — just say so in one line."
)
_LAST_STEP_TAIL = (
    "When it is complete, write the final answer to the original request in your reply: "
    "what was done, and any answer the request asked for."
)

STEP_WRITE_REJECT_LIMIT = 2


def step_write_reject_message(tool: str, path: str, lines: int, limit: int) -> str:
    """段の上限を超えた 1 回の書き込みの差し戻し（書いていない）。"""
    if tool == "append_file":
        split = f"- Append the content in parts of at most {limit} lines, one append_file each."
    else:
        split = (
            f"- To write a long file, write_file the first part (at most {limit} lines), then "
            "append_file each following part of the same size."
        )
    return (
        f"{tool} rejected: {lines} lines is over this step's limit of {limit} lines per write\n"
        f"{path} was not changed. One response has to finish within 120 seconds; the limit keeps "
        "one write well inside that for this model.\n"
        "- Write only what this step of the plan needs; leave later steps for later.\n"
        f"{split}"
    )


PLAN_STEPS_REQUEST = (
    'End the plan with a "## Steps" section: a numbered list of the work you will do with tools '
    "after the user approves. End each step with the amount of new content it creates, written "
    "as (~N lines); use (~0 lines) for steps that only read, run or verify. No step may create "
    f"more than {chunking.LINES} of new content — the exact limit is measured from this model's "
    "speed and checked when you call exit_plan_mode — so a longer file is several steps: "
    "write_file for the first part, append_file for each following part (write_file replaces "
    "the whole file; once a file exists, add to it with append_file). Use the fewest steps "
    "that respect the limit. Planning, approval and anything the user does are not steps."
)
PLAN_STEPS_REQUEST_LEGACY = (
    'End the plan with a "## Steps" section: a numbered list of the work you will carry out '
    "after the user approves. Each step is one self-contained change you make in the "
    "workspace with tools, small enough to finish in a few tool calls (no step should create "
    f"more than {chunking.LINES} of new content). Do not list steps for planning, for "
    "approval, or for anything the user does — those are finished before the steps run. "
    "Split the work that produces the output: one large file is not one step. If a step needs "
    "sub-items to say what it contains, each of those sub-items is a step — write the first "
    "part with write_file and append the rest, one step each. Once the user approves, the "
    "plan is handed back to you one step at a time."
)


def plan_steps_request(plan_steps: bool, step_budget: bool) -> str:
    """設定に応じた「段の節を求める 1 文」（先頭に区切りの空白つき。無効なら空文字）。"""
    if not plan_steps:
        return ""
    return f" {PLAN_STEPS_REQUEST if step_budget else PLAN_STEPS_REQUEST_LEGACY}"


@dataclass(frozen=True)
class PlanStep:
    """計画の 1 段（本文は :data:`MAX_STEP_CHARS` で切り詰め済み）と、その見積もり行数。"""

    text: str
    estimated_lines: int | None = None


_ESTIMATE_RE = re.compile(
    r"[（(]\s*(?:~|約|about|approx\.?)?\s*(\d{1,5})\s*(?:lines?|行)\s*[)）]"
    r"|(?:~|約)\s*(\d{1,5})\s*(?:lines?|行)",
    re.IGNORECASE,
)


def parse_step_estimate(raw: str) -> int | None:
    """段の本文から見積もり行数を取る（複数あれば**最大**。無ければ None）。"""
    values = [int(m.group(1) or m.group(2)) for m in _ESTIMATE_RE.finditer(raw)]
    return max(values) if values else None


_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_RULE_RE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")
_ITEM_RE = re.compile(r"^(?:[-*+][ \t]+\[[ xX~-]\][ \t]+|[-*+][ \t]+|\d+[.)][ \t]+)(.*)$")
_TITLE_TRIM_RE = re.compile(r"^[\d.)\s]*|[*_`#\s]+$")
_TITLE_MARKS_RE = re.compile(r"[*_`]")
_EXACT_TITLES = frozenset(
    {
        "steps",
        "step",
        "手順",
        "ステップ",
        "implementation steps",
        "実装手順",
        "作業手順",
        "実行手順",
    }
)
_REQUESTED_TITLES = frozenset({"steps", "step"})
_STEP_WORD_RE = re.compile(r"(?:^|[^A-Za-z])(steps?|手順|ステップ)(?:[^A-Za-z]|$)", re.IGNORECASE)


def _normalized_title(raw: str) -> str:
    return _TITLE_TRIM_RE.sub("", _TITLE_MARKS_RE.sub("", raw)).strip().lower()


def _steps_headings(lines: list[str]) -> list[tuple[int, int]]:
    """段の節の**候補**を優先順に返す（各要素は ``(行番号, #の数)``）。"""
    requested: list[tuple[int, int]] = []
    exact: list[tuple[int, int]] = []
    loose: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m is None:
            continue
        title = _normalized_title(m.group(2))
        if title in _REQUESTED_TITLES:
            requested.append((i, len(m.group(1))))
        elif title in _EXACT_TITLES:
            exact.append((i, len(m.group(1))))
        elif _STEP_WORD_RE.search(title):
            loose.append((i, len(m.group(1))))
    return requested + exact + loose


def _truncate(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_STEP_CHARS else text[: MAX_STEP_CHARS - 1].rstrip() + "…"


def _heading_steps(body: list[str], depth: int) -> list[str]:
    """``### Step 1: …`` 形式の段。"""
    headings = [
        (i, len(m.group(1)))
        for i, line in enumerate(body)
        if (m := _HEADING_RE.match(line)) is not None and len(m.group(1)) > depth
    ]
    if not headings:
        return []
    level = min(d for _, d in headings)
    at_level = [i for i, d in headings if d == level]
    if len(at_level) < 2:
        return []
    if any(line[:1] not in (" ", "\t") and _ITEM_RE.match(line) for line in body[: at_level[0]]):
        return []

    steps: list[str] = []
    for i, line in enumerate(body[at_level[0] :], start=at_level[0]):
        stripped = line.strip()
        if not stripped:
            continue
        if _RULE_RE.match(line) is not None:
            if i > at_level[-1]:
                break
            continue
        m = _HEADING_RE.match(line)
        if m is not None and len(m.group(1)) == level:
            steps.append(m.group(2).strip())
        else:
            steps[-1] = f"{steps[-1]} {stripped}"
    return steps


def parse_steps(plan_text: str | None) -> list[str]:
    """計画本文から段の一覧を取り出す（2 段未満・見出し無しは ``[]``）。"""
    steps = parse_plan_steps(plan_text)
    return [s.text for s in steps] if len(steps) >= 2 else []


def parse_plan_steps(plan_text: str | None) -> list[PlanStep]:
    """計画本文から段と見積もりを取り出す。"""
    if not plan_text:
        return []
    lines = plan_text.splitlines()
    single: list[str] = []
    for start, depth in _steps_headings(lines):
        raw = _section_steps(lines, start, depth)
        if len(raw) >= 2:
            return [PlanStep(_truncate(s), parse_step_estimate(s)) for s in raw]
        if len(raw) == 1 and not single:
            single = raw
    return [PlanStep(_truncate(s), parse_step_estimate(s)) for s in single]


def _section_steps(lines: list[str], start: int, depth: int) -> list[str]:
    """``start`` 行の見出しが開く 1 つの節から段を取り出す。"""
    end = len(lines)
    for i in range(start + 1, len(lines)):
        m = _HEADING_RE.match(lines[i])
        if m is not None and len(m.group(1)) <= depth:
            end = i
            break
    body = lines[start + 1 : end]

    steps = _heading_steps(body, depth)
    if not steps:
        for line in body:
            stripped = line.strip()
            if _HEADING_RE.match(line) is not None:
                if steps:
                    break
                continue
            if not stripped:
                continue
            if line[:1] not in (" ", "\t"):
                item = _ITEM_RE.match(line)
                if item is not None:
                    steps.append(item.group(1).strip())
                continue
            if steps and _ITEM_RE.match(stripped) is None:
                steps[-1] = f"{steps[-1]} {stripped}"

    return [s for s in steps if s.strip()]


_RESUME_RE = re.compile(
    r"(続き|続け|再開|次の段|つづき|(?<!不)許可|\bcontinue\b|\bresume\b|\bgo on\b|"
    r"\bkeep going\b|\bnext step\b|\bcarry on\b)",
    re.IGNORECASE,
)
_BLOCKED_RE = re.compile(r"^\W{0,3}BLOCKED\b", re.IGNORECASE)


def wants_resume(text: str) -> bool:
    """ユーザー発話が「止まっている段を続けて」の意図か（保守的判定）。"""
    t = text.strip()
    if not t or len(t) > APPROVAL_TEXT_LIMIT:
        return False
    if t.endswith(("?", "？")):
        return False
    if has_negation(t):
        return False
    return bool(_RESUME_RE.search(t)) or is_plan_approval_text(t)


def is_blocked_reply(text: str) -> bool:
    """モデルの最終本文が ``BLOCKED:`` で始まるか（段を進めずユーザーへ返す合図）。"""
    return bool(_BLOCKED_RE.match(text.strip()))


def _budget_line(budget_lines: int | None, estimated_lines: int | None = None) -> str:
    """段の注入文に添える 1 回の書き込みの上限（予算が無ければ空）。"""
    if not budget_lines:
        return ""
    note = (
        f"Keep each write_file/append_file call under ~{budget_lines} lines (the limit "
        "measured for this model). write_file replaces the whole file: use it only for a file "
        "that does not exist yet, and add to an existing file with append_file. "
    )
    if estimated_lines:
        note += (
            f"This step adds about ~{estimated_lines} lines in total across all its calls; "
            "later steps add the rest. "
        )
    return note


def first_step_note(
    total: int,
    text: str,
    plan_file: str,
    *,
    budget_lines: int | None = None,
    estimated_lines: int | None = None,
) -> str:
    """承認直後に段 1 を渡す文（``exit_plan_mode`` の結果末尾 / 承認発話ターンの冒頭）。"""
    tail = _LAST_STEP_TAIL if total == 1 else _STEP_TAIL
    return (
        f"Approved. The plan has {total} steps and the full plan is in {plan_file}. "
        f"Do ONLY step 1 of {total} now: {text}\n"
        f"{_budget_line(budget_lines, estimated_lines)}{tail}"
    )


def next_step_note(
    index: int,
    total: int,
    text: str,
    *,
    budget_lines: int | None = None,
    estimated_lines: int | None = None,
) -> str:
    """段 ``index``（2..total）を渡す文。"""
    budget = _budget_line(budget_lines, estimated_lines)
    if index >= total:
        return (
            f"Step {index - 1} is recorded. Now do the last step ({total} of {total}): "
            f"{text}\n{budget}{_ALREADY_DONE} {_LAST_STEP_TAIL}"
        )
    return (
        f"Step {index - 1} is recorded. Now do ONLY step {index} of {total}: {text}\n"
        f"{budget}{_ALREADY_DONE} {_STEP_TAIL}"
    )


def resume_step_note(
    index: int,
    total: int,
    text: str,
    *,
    budget_lines: int | None = None,
    estimated_lines: int | None = None,
) -> str:
    """中断していた段 ``index`` を再開する文（``BLOCKED:`` の回答・「続けて」）。"""
    tail = _LAST_STEP_TAIL if index >= total else _STEP_TAIL
    return (
        f"Resuming step {index} of {total} of the approved plan: {text}\n"
        "The user's messages since you stopped contain the answer to your question or the "
        "go-ahead to continue. Continue from where you stopped; do not redo work that already "
        f"succeeded.\n{_budget_line(budget_lines, estimated_lines)}{tail}"
    )


def compacted_step_note(
    index: int,
    total: int,
    text: str,
    *,
    budget_lines: int | None = None,
    estimated_lines: int | None = None,
) -> str:
    """段 ``index`` の途中で会話を畳み、段を渡した文が要約に入ったときに、段を渡し直す文。"""
    tail = _LAST_STEP_TAIL if index >= total else _STEP_TAIL
    return (
        f"The conversation was compacted while you were on step {index} of {total} of the "
        f"approved plan: {text}\n"
        "Continue this step from where the summary and the recent messages leave off; do not "
        f"redo work that already succeeded.\n{_budget_line(budget_lines, estimated_lines)}{tail}"
    )


def timeout_resume_step_note(
    index: int,
    total: int,
    text: str,
    *,
    seconds: float,
    attempt: int,
    rate_tok_s: float | None,
    budget_lines: int | None = None,
    estimated_lines: int | None = None,
) -> str:
    """段の中で read タイムアウトが続いたとき、ハーネスが「続けて」の代わりに段を再開する文。"""
    tail = _LAST_STEP_TAIL if index >= total else _STEP_TAIL
    limit = f"~{budget_lines} lines" if budget_lines else chunking.LINES
    if rate_tok_s is not None:
        tokens = int(rate_tok_s * seconds)
        lines = chunking.lines_estimate(tokens)
        ratio = lines / budget_lines if budget_lines else 0.0
        against = (
            f", about {ratio:.0f}× the limit of {limit} per call"
            if ratio >= 2
            else f" — the limit is {limit} per call"
        )
        size = (
            f"At the ~{rate_tok_s:.0f} tokens/s measured in this conversation, each cancelled "
            f"response had already produced at least ~{tokens:,} tokens — roughly {lines} lines "
            f"inside one tool call{against}. "
        )
    else:
        size = (
            f"A tool call that takes {seconds:.0f} seconds to produce is far above the limit of "
            f"{limit} per call. "
        )
    return (
        f"Your response was cancelled again: nothing arrived for {seconds:.0f} seconds — that is "
        f"{attempt} cancelled responses in a row in step {index} of {total}. {size}"
        "The tool arguments never arrived, so nothing from those responses reached the "
        "workspace: files are exactly as they were before the cancelled responses.\n"
        f"Resuming step {index} of {total} of the approved plan: {text}\n"
        f"Start with ONE call of at most {limit}: write_file if the file does not exist yet, "
        "append_file if it does. Do not try to finish the whole step in one call. "
        f"{_budget_line(budget_lines, estimated_lines)}{tail}"
    )


def retry_step_note(index: int, total: int, text: str) -> str:
    """段を渡したのにツールを 1 回も呼ばずに終えたとき、同じ段をもう一度促す文。"""
    return (
        f"You ended step {index} of {total} without calling any tool, so nothing changed. "
        "If that step was genuinely already done, say so in one line and stop. Otherwise "
        f"call the first tool for it now: {text}"
    )
