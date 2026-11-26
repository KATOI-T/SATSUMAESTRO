"""ファイル変更の差分を、行番号・記号・背景色と、変わった語の強調つきで描く。"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

from satsumaestro.cli.markdown import CODE_THEME
from satsumaestro.i18n.text import ui

CONTEXT_LINES = 3
_ADDED = Style(bgcolor="rgb(34,92,43)")
_REMOVED = Style(bgcolor="rgb(122,41,54)")
_ADDED_WORD = Style(bgcolor="rgb(56,166,96)")
_REMOVED_WORD = Style(bgcolor="rgb(179,89,107)")
_DIM = Style(dim=True)
_BOLD_NUMBER = re.compile(r"\d[\d,]*")
# 対の 2 行の文字のうち、変わった割合がこれを超えたら語を強調しない
_WORD_DIFF_MAX = 0.4
# 対の 2 行の字数の和がこれを超えたら語を強調しない（同じ語の並ぶ長い行は比べるのに時間がかかる）
_WORD_DIFF_MAX_CHARS = 2000
# 色付けは出す範囲の前のこの行数から始める（開いた複数行の文字列の色を拾う）
_HIGHLIGHT_LEAD = 200
_HIGHLIGHT_LIMIT = 100_000

Split = Callable[[str], list[str]]


@dataclass
class _Row:
    number: int | None  # None は離れた箇所の間の ``...``
    marker: str  # "+" / "-" / " "（文脈）/ ""（新規ファイルの行）
    code: Text = field(default_factory=Text)


class _DiffRows:
    """見出しの行・差分の行・末尾の注記（差分の行は背景を右端まで塗り、折り返しは本文に揃える）。"""

    def __init__(self, head: Text | None, rows: list[_Row], tail: Text | None) -> None:
        self._head = head
        self._rows = rows
        self._tail = tail
        numbers = [r.number for r in rows if r.number is not None]
        self._digits = len(str(max(numbers, default=1)))

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if self._head is not None:
            yield self._head
        for row in self._rows:
            if row.number is None:
                yield Text("...", style=_DIM)
                continue
            yield from self._render_row(console, options, row)
        if self._tail is not None:
            yield self._tail

    def _render_row(
        self, console: Console, options: ConsoleOptions, row: _Row
    ) -> Iterator[Segment]:
        bg = _ADDED if row.marker == "+" else _REMOVED if row.marker == "-" else None
        gutter = f" {row.number:>{self._digits}} {row.marker}"
        width = len(gutter)
        body = options.update(width=max(1, options.max_width - width))
        lines = console.render_lines(row.code, body, style=bg or Style.null(), pad=bg is not None)
        gutter_style = bg if bg is not None else _DIM
        for index, line in enumerate(lines):
            yield Segment(gutter if index == 0 else " " * width, gutter_style)
            yield from line
            yield Segment.line()


def diff_view(
    before: str, after: str, path: str, *, max_rows: int, split: Split = str.splitlines
) -> _DiffRows | Text:
    """``before`` から ``after`` への差分（行は ``split`` で出す。伏せても行の数は変わらない）。"""
    a, b = before.splitlines(), after.splitlines()
    sa, sb = (a, b) if split is str.splitlines else (split(before), split(after))
    groups = list(difflib.SequenceMatcher(None, a, b).get_grouped_opcodes(CONTEXT_LINES))
    # 色付けは変更後の、出す範囲とその前だけ（削除の行は色付けしない）
    start = max(0, min((g[0][3] for g in groups), default=0) - _HIGHLIGHT_LEAD)
    end = max((g[-1][4] for g in groups), default=0)
    code = _highlighted(sb[start:end], path)
    rows: list[_Row] = []
    added = removed = 0
    for n, group in enumerate(groups):
        if n:
            rows.append(_Row(None, ""))
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for i, j in zip(range(i1, i2), range(j1, j2), strict=True):
                    line = shown_context(a[i], sa[i], sb[j])
                    rows.append(_Row(j + 1, " ", code[j - start] if line == sb[j] else Text(line)))
                continue
            olds = [_Row(i + 1, "-", Text(sa[i])) for i in range(i1, i2)]
            news = [_Row(j + 1, "+", code[j - start].copy()) for j in range(j1, j2)]
            shown = max(0, max_rows - len(rows))  # 語の強調は出す行だけ
            for old, new in zip(olds[:shown], news, strict=False):
                _mark_changed_words(old.code, new.code)
            rows.extend([*olds, *news])
            removed += len(olds)
            added += len(news)
    if not rows:
        return Text(ui("cli.render.no_changes"), style="dim")
    return _DiffRows(_summary(added, removed), *_cut(rows, max_rows))


def new_file_view(
    content: str, path: str, *, max_rows: int, split: Split = str.splitlines
) -> _DiffRows:
    """新規ファイルの「N 行を書き込み」と先頭の行。"""
    lines = split(content)
    shown = lines[:max_rows]
    code = _highlighted(shown, path)
    rows = [_Row(i + 1, "", code[i]) for i in range(len(shown))]
    head = _bold_numbers(ui("cli.render.wrote_lines", n=len(lines)))
    hidden = len(lines) - len(shown)
    return _DiffRows(head, rows, _more(hidden) if hidden else None)


def shown_context(real: str, before: str, after: str) -> str:
    """変わらない行は、前と後のうち伏せた方を出す。"""
    return before if before != real else after


def _more(hidden: int) -> Text:
    return Text(ui("cli.render.more_rows", n=hidden), style=_DIM)


def _cut(rows: list[_Row], max_rows: int) -> tuple[list[_Row], Text | None]:
    if len(rows) <= max_rows:
        return rows, None
    return rows[:max_rows], _more(len(rows) - max_rows)


def _summary(added: int, removed: int) -> Text:
    parts = []
    if added:
        parts.append(ui("cli.render.diff_added", n=added))
    if removed and added:
        parts.append(ui("cli.render.diff_removed", n=removed))
    elif removed:
        parts.append(ui("cli.render.diff_removed_only", n=removed))
    return _bold_numbers(ui("cli.render.diff_join").join(parts))


def _bold_numbers(message: str) -> Text:
    text = Text(message)
    text.highlight_regex(_BOLD_NUMBER, "bold")
    return text


def _highlighted(lines: list[str], path: str) -> list[Text]:
    """``lines`` をまとめて色付けしてから行に分ける（複数行の文字列の色が切れない）。"""
    source = "\n".join(lines)
    lexer = Syntax.guess_lexer(path, code=source) if len(source) <= _HIGHLIGHT_LIMIT else "default"
    if lexer in ("default", "text"):
        return [Text(line) for line in lines]
    text = Syntax("", lexer, theme=CODE_THEME).highlight(source)
    parts = text.split("\n", allow_blank=True)[: len(lines)]
    if [p.plain for p in parts] != lines:  # 行の分け方が合わない（\r など）
        return [Text(line) for line in lines]
    return list(parts)


def _tokens(line: str) -> list[str]:
    """語（文字・数字・``_`` の並び）、空白の並び、それ以外の 1 文字に分ける。"""
    out: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        kind = _kind(ch)
        j = i + 1
        if kind != "other":
            while j < len(line) and _kind(line[j]) == kind:
                j += 1
        out.append(line[i:j])
        i = j
    return out


def _kind(ch: str) -> str:
    if ch.isalnum() or ch == "_":
        return "word"
    return "space" if ch.isspace() else "other"


def _mark_changed_words(old: Text, new: Text) -> None:
    """対になる削除と追加の行の、変わった語に明るい背景を付ける。"""
    if len(old.plain) + len(new.plain) > _WORD_DIFF_MAX_CHARS:
        return
    a, b = _tokens(old.plain), _tokens(new.plain)
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    total = len(old.plain) + len(new.plain)
    changed = sum(
        sum(map(len, a[i1:i2])) + sum(map(len, b[j1:j2]))
        for tag, i1, i2, j1, j2 in ops
        if tag != "equal"
    )
    if total == 0 or changed / total > _WORD_DIFF_MAX:
        return
    offsets_a = _offsets(a)
    offsets_b = _offsets(b)
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("replace", "delete") and i1 < i2:
            old.stylize(_REMOVED_WORD, offsets_a[i1], offsets_a[i2])
        if tag in ("replace", "insert") and j1 < j2:
            new.stylize(_ADDED_WORD, offsets_b[j1], offsets_b[j2])


def _offsets(tokens: list[str]) -> list[int]:
    out = [0]
    for tok in tokens:
        out.append(out[-1] + len(tok))
    return out
