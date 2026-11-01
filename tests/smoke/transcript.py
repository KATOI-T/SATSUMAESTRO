"""REPL の画面の転写を検査する。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ESC2 = re.compile(r"\x1b[=>()78][0-9A-Za-z]?")
_SPINNER = re.compile(r"^\s*[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")

_PANEL_TOP = re.compile(r"^╭─+ (?P<title>.+?) ─+╮\s*$")
_PANEL_BODY = re.compile(r"^│ ?(?P<body>.*?) ?│\s*$")
_PANEL_BOTTOM = re.compile(r"^╰─+╯\s*$")

_DECLARED = re.compile(
    r"（(\d+) 段に分けて 1 段ずつ実行します）|\(Runs in (\d+) steps, one at a time\)"
)
_STEP_START = re.compile(r"^— (?:段|Step) (\d+)/(\d+): ", re.MULTILINE)
_STEP_DONE = re.compile(r"^— (?:段|Step) (\d+)/(\d+) (?:完了|done): ", re.MULTILINE)
_SENDBACK = re.compile(r"^— (?:計画を差し戻し|Plan sent back): (.*)$", re.MULTILINE)
_BUDGET = re.compile(r"（1 段の上限 \d+ 行 ＝|\(step limit \d+ lines =")
_DELETE_NOTE = re.compile(
    r"(?:このセッションで作成したファイルを消します|Deletes files created in this session): ([^│]*)"
)
_TOKENS = re.compile(
    r"^— (?:[0-9.,]+K? tok|(?:出力|output) ~?[0-9.]+[KM]?(?: · .*)?)\s*$",
    re.MULTILINE,
)

FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"⚠ ターンを最後まで進められませんでした"),
    re.compile(r"⚠ The turn could not be finished"),
    re.compile(r"⚠ ターンが失敗しました"),
    re.compile(r"reached max_iterations=\d+"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"秒応答がなかったので中断"),
    re.compile(r"No response for \d+ s, so it was stopped"),
    re.compile(r"— 応答が途中で切れました"),
    re.compile(r"— The response was cut off"),
    re.compile(r"\[fake\] no rule matched"),
)

_EMPTY_BODIES = frozenset({"", "...", "…", "......"})
_NO_PLAN_FILE = ("計画ファイルがありません", "Noplanfile")


def strip_ansi(raw: bytes | str) -> str:
    """ANSI（CSI / OSC / 2 バイト列）を剥ぎ、改行を ``\\n`` に揃える。"""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    text = _OSC.sub("", text)
    text = _CSI.sub("", text)
    text = _ESC2.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_lines(text: str) -> list[str]:
    """右の空白を落とし、スピナー行を除き、**その後で**連続する同じ行を 1 つに畳む。"""
    out: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        if _SPINNER.match(line):
            continue
        if out and out[-1] == line:
            continue
        out.append(line)
    while out and out[-1] == "":
        out.pop()
    return out


def flat(text: str) -> str:
    """空白をすべて除く。"""
    return re.sub(r"\s+", "", text)


def panels(lines: list[str], title_re: str) -> list[str]:
    """題名が ``title_re`` に合う Rich の Panel の本文（枠の中の行を ``\\n`` で結合）を返す。"""
    found: list[str] = []
    pattern = re.compile(title_re)
    body: list[str] | None = None
    for line in lines:
        top = _PANEL_TOP.match(line)
        if top is not None:
            body = [] if pattern.search(top.group("title")) else None
            continue
        if body is None:
            continue
        if _PANEL_BOTTOM.match(line):
            found.append("\n".join(body))
            body = None
            continue
        inner = _PANEL_BODY.match(line)
        if inner is not None:
            body.append(inner.group("body"))
    return found


def approval_cards(lines: list[str]) -> list[str]:
    return panels(lines, r"プラン承認|Plan approval")


def permission_cards(lines: list[str], tool: str) -> list[str]:
    return panels(lines, rf"(?:権限確認|Permission): {re.escape(tool)} \(")


def is_empty_card(body: str) -> bool:
    """承認カードの本文が空か（``...`` だけ、計画ファイル無しの案内も空とする）。"""
    compact = flat(body)
    return compact in _EMPTY_BODIES or any(marker in compact for marker in _NO_PLAN_FILE)


def declared_steps(text: str) -> int | None:
    m = _DECLARED.search(text)
    return int(m.group(1) or m.group(2)) if m else None


def step_starts(text: str) -> list[tuple[int, int]]:
    return [(int(k), int(n)) for k, n in _STEP_START.findall(text)]


def completed_steps(text: str) -> list[int]:
    """完了した段の番号（重複を除いて昇順）。"""
    return sorted({int(k) for k, _ in _STEP_DONE.findall(text)})


def sendbacks(text: str) -> list[str]:
    return _SENDBACK.findall(text)


def has_budget_line(text: str) -> bool:
    return _BUDGET.search(text) is not None


def delete_notes(text: str) -> list[str]:
    return [m.strip() for m in _DELETE_NOTE.findall(text)]


def turn_ends(text: str) -> int:
    """ターンの終端（トークンの内訳行 ``— 出力 N …``）の行数。"""
    return len(_TOKENS.findall(text))


def failure_lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if any(p.search(line) for p in FAILURE_PATTERNS)]


@dataclass
class Report:
    """``check`` の結果。"""

    problems: list[str] = field(default_factory=list)
    declared: int | None = None
    completed: list[int] = field(default_factory=list)
    cards: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def check(text: str) -> Report:
    """plan → 承認 → 段 → 完走 の転写を検査し、違反を固定のコードで返す。"""
    lines = clean_lines(text)
    joined = "\n".join(lines)
    report = Report(
        declared=declared_steps(joined),
        completed=completed_steps(joined),
        cards=approval_cards(lines),
        failures=failure_lines(joined),
    )
    if not report.cards:
        report.problems.append("no-approval-card")
    for body in report.cards:
        if is_empty_card(body):
            report.problems.append("approval-card-empty")
    if report.declared is None:
        report.problems.append("no-step-declaration")
    elif report.completed != list(range(1, report.declared + 1)):
        report.problems.append(
            f"steps-mismatch: declared {report.declared}, completed {report.completed}"
        )
    report.problems.extend(f"failure: {line.strip()}" for line in report.failures)
    return report


class Transcript:
    """pty から読んだ生バイト列と、その剥いだ本文（読み足すたびに作り直す）。"""

    def __init__(self) -> None:
        self.raw = bytearray()
        self._text = ""
        self._text_len = -1

    def append(self, data: bytes) -> None:
        self.raw.extend(data)

    @property
    def text(self) -> str:
        if self._text_len != len(self.raw):
            self._text = "\n".join(clean_lines(strip_ansi(bytes(self.raw))))
            self._text_len = len(self.raw)
        return self._text

    @property
    def plain(self) -> str:
        """ANSI を剥いだだけの本文（畳まない）。"""
        return strip_ansi(bytes(self.raw))

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")

    def tail(self, n: int = 40) -> str:
        return "\n".join(self.lines[-n:])
