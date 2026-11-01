"""edit_file の不一致を、次の 1 回で直せる材料つきで返す。"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

FLAG_MISMATCH = "edit-mismatch"
FLAG_REINDENT = "edit-reindent"
MIN_SIMILARITY = 0.5
MAX_SHOWN_LINES = 20
MAX_SHOWN_MATCHES = 10
MAX_FILE_LINES = 5000
MAX_OLD_LINES = 200
_TOP_WINDOWS = 30
_CHAR_RANKED = 6
_CHAR_BUDGET = 4_000_000
_APPLIED_MIN_CHARS = 20
_LINE_CLIP = 200


@dataclass(frozen=True)
class ClosestMatch:
    """``old`` に最も似たファイルの区間（行番号は 1 始まり・両端を含む）。"""

    start: int
    end: int
    similarity: float
    whitespace_only: bool


def _norm(line: str) -> str:
    return " ".join(line.split())


def closest_match(text: str, old: str) -> ClosestMatch | None:
    """``text`` の中で ``old`` に最も似た連続区間。"""
    old_lines = old.splitlines()
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    lines = text.splitlines()
    if not old_lines or not lines or len(lines) > MAX_FILE_LINES:
        return None
    if len(old_lines) > MAX_OLD_LINES:
        return None
    old_normed = [_norm(ln) for ln in old_lines]
    target = "\n".join(old_normed)
    normed = [_norm(ln) for ln in lines]
    n = len(old_lines)
    by_lines = SequenceMatcher(autojunk=False)
    by_lines.set_seq2(old_normed)
    scored: list[tuple[float, int, int]] = []
    for i in _candidate_starts(normed, old_normed):
        for size in (n - 1, n, n + 1):
            if size < 1 or i + size > len(lines):
                continue
            by_lines.set_seq1(normed[i : i + size])
            scored.append((by_lines.ratio(), i, size))
    scored.sort(key=lambda s: -s[0])
    by_chars = SequenceMatcher(autojunk=False)
    by_chars.set_seq2(target)
    best: tuple[float, int, int] | None = None
    budget = _CHAR_BUDGET
    for rank, (_, i, size) in enumerate(scored[:_CHAR_RANKED]):
        window = "\n".join(normed[i : i + size])
        cost = len(window) * len(target)
        if rank and cost > budget:
            break
        budget -= cost
        by_chars.set_seq1(window)
        ratio = by_chars.ratio()
        if best is None or ratio > best[0]:
            best = (ratio, i, size)
    if best is None or best[0] < MIN_SIMILARITY:
        return None
    ratio, i, size = best
    window = "\n".join(normed[i : i + size])
    return ClosestMatch(
        start=i + 1, end=i + size, similarity=ratio, whitespace_only=window == target
    )


def _candidate_starts(normed: list[str], old_normed: list[str]) -> list[int]:
    """精査する窓の開始行を先に絞る。"""
    n = len(old_normed)
    total = len(normed)
    wanted = {ln for ln in old_normed if ln}
    hits = [1 if ln in wanted and ln else 0 for ln in normed]
    size = min(n, total)
    scores: list[tuple[int, int]] = []
    running = sum(hits[:size])
    for i in range(total - size + 1):
        if i:
            running += hits[i + size - 1] - hits[i - 1]
        if running:
            scores.append((running, i))
    if scores:
        scores.sort(key=lambda s: (-s[0], s[1]))
        seeds = [i for _, i in scores[:_TOP_WINDOWS]]
    else:
        edges = [(old_normed[0], 0), (old_normed[-1], n - 1)]
        ranked: list[tuple[float, int]] = []
        for edge, back in edges:
            if not edge:
                continue
            m = SequenceMatcher(autojunk=False, b=edge)
            for i, ln in enumerate(normed):
                m.set_seq1(ln)
                ranked.append((m.quick_ratio(), i - back))
        ranked.sort(key=lambda s: (-s[0], s[1]))
        seeds = [i for _, i in ranked[:_TOP_WINDOWS]]
    starts = [j for i in seeds for j in (i, i - 1, i + 1) if 0 <= j < total]
    return list(dict.fromkeys(starts))


def _numbered(lines: list[str], start: int, end: int) -> list[str]:
    """``start``〜``end`` 行（1 始まり）を ``  45 | …`` の形で返す（上限つき）。"""
    width = len(str(end))
    shown_end = min(end, start + MAX_SHOWN_LINES - 1)
    out = []
    for no in range(start, shown_end + 1):
        body = lines[no - 1]
        if len(body) > _LINE_CLIP:
            body = body[:_LINE_CLIP] + " …"
        out.append(f"  {no:>{width}} | {body}")
    if shown_end < end:
        out.append(f"  … ({end - shown_end} more lines)")
    return out


def not_found_message(path: str, text: str, old: str, new: str) -> str:
    """一致 0 件のときにモデルへ返す文。"""
    head = f"old string not found in {path}."
    if len(new.strip()) >= _APPLIED_MIN_CHARS and new in text and old not in new:
        return (
            f"{head} The file already contains the `new` text, so this edit may have been "
            f"applied already. read_file {path} to check before editing again."
        )
    match = closest_match(text, old)
    if match is None:
        return (
            f"{head} No similar text was found. read_file {path} and copy the exact lines "
            "you want to change into `old`."
        )
    lines = text.splitlines()
    span = f"line {match.start}" if match.start == match.end else f"lines {match.start}-{match.end}"
    if match.whitespace_only:
        lead = f"The file has the same text at {span}, but the whitespace or indentation differs:"
    else:
        percent = min(99, int(match.similarity * 100))
        lead = f"The closest text in the file is at {span} ({percent}% similar):"
    tail = (
        "Use these lines verbatim as `old` (copy them exactly, including indentation), "
        "then apply your change to them in `new`."
    )
    body = [head, lead, *_numbered(lines, match.start, match.end)]
    diff = first_difference(lines[match.start - 1 : match.end], old, match.start) or (
        _edge_blank_difference(lines, match, old)
    )
    if diff:
        body.append(diff)
    return "\n".join([*body, tail])


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _describe_indent(ws: str) -> str:
    if not ws:
        return "no indentation"
    if set(ws) == {" "}:
        return f"{len(ws)} space{'s' if len(ws) > 1 else ''}"
    if set(ws) == {"\t"}:
        return f"{len(ws)} tab{'s' if len(ws) > 1 else ''}"
    return f"a mix of tabs and spaces ({ws!r})"


def _clip(line: str) -> str:
    return line if len(line) <= _LINE_CLIP else line[:_LINE_CLIP] + " …"


def first_difference(window: list[str], old: str, first_line: int) -> str | None:
    """候補区間と ``old`` の**最初に違う行**を 1 文で書く。"""
    old_lines = old.splitlines()
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    ops = SequenceMatcher(autojunk=False, a=window, b=old_lines).get_opcodes()
    for tag, i1, _i2, j1, _j2 in ops:
        if tag == "equal":
            continue
        no = first_line + i1
        if tag == "replace":
            f, o = window[i1], old_lines[j1]
            if f.strip() == o.strip() and _indent(f) != _indent(o):
                return (
                    f"First difference: line {no} is indented with "
                    f"{_describe_indent(_indent(f))} in the file, but with "
                    f"{_describe_indent(_indent(o))} in your `old`."
                )
            return f"First difference at line {no}:\n  file: {_clip(f)}\n  old:  {_clip(o)}"
        if tag == "delete":
            return f"First difference: your `old` skips line {no} of the file: {_clip(window[i1])}"
        return (
            f"First difference: your `old` has a line that is not in the file before line {no}: "
            f"{_clip(old_lines[j1])}"
        )
    return None


def _edge_blank_difference(lines: list[str], match: ClosestMatch, old: str) -> str | None:
    """本文は同じで、``old`` の前後にファイルには無い空行があるときの説明。"""
    raw = old.split("\n")
    if old.endswith("\n"):
        raw.pop()
    lead = len(raw) - len("\n".join(raw).lstrip("\n").split("\n")) if raw else 0
    trail = 0
    while raw and len(raw) > trail and not raw[-1 - trail].strip():
        trail += 1
    after = lines[match.end] if match.end < len(lines) else None
    if trail and (after is None or after.strip()):
        where = f"line {match.end + 1} of the file is: {_clip(after)}" if after else "the file ends"
        return (
            f"First difference: your `old` ends with {trail} empty line"
            f"{'s' if trail > 1 else ''} after line {match.end}, but {where}"
        )
    before = lines[match.start - 2] if match.start >= 2 else None
    if lead and (before is None or before.strip()):
        where = f"line {match.start - 1} of the file is: {_clip(before)}" if before else "it is"
        return (
            f"First difference: your `old` starts with {lead} empty line"
            f"{'s' if lead > 1 else ''} before line {match.start}, but {where}"
        )
    return None


@dataclass(frozen=True)
class Reindent:
    """字下げを合わせて適用する編集（:func:`reindent_edit`）。"""

    old: str
    new: str
    shift: int
    start: int
    end: int


def reindent_edit(text: str, old: str, new: str) -> Reindent | None:
    """字下げが一様にずれているだけなら、ファイルに合わせた ``old`` / ``new`` を返す。"""
    ends_nl = old.endswith("\n")
    old_lines = old.split("\n")
    if ends_nl:
        old_lines.pop()
    if not old_lines or not any(ln.strip() for ln in old_lines):
        return None
    if any("\t" in _indent(ln) for ln in old_lines + new.split("\n")):
        return None
    lines = text.split("\n")
    n = len(old_lines)
    found: list[tuple[int, int]] = []
    for i in range(len(lines) - n + 1):
        shift = _uniform_shift(lines[i : i + n], old_lines)
        if shift is not None:
            found.append((i, shift))
            if len(found) > 1:
                return None
    if len(found) != 1:
        return None
    i, shift = found[0]
    exact = "\n".join(lines[i : i + n]) + ("\n" if ends_nl else "")
    if text.count(exact) != 1:
        return None
    shifted: list[str] = []
    for ln in new.split("\n"):
        if not ln.strip():
            shifted.append(ln)
        elif shift > 0:
            shifted.append(" " * shift + ln)
        elif len(_indent(ln)) >= -shift:
            shifted.append(ln[-shift:])
        else:
            return None
    return Reindent(old=exact, new="\n".join(shifted), shift=shift, start=i + 1, end=i + n)


def _uniform_shift(window: list[str], old_lines: list[str]) -> int | None:
    """``window`` と ``old_lines`` が字下げの一様なずれだけで一致するなら、ファイル側へのずれ。"""
    shift: int | None = None
    for f, o in zip(window, old_lines, strict=True):
        if not o.strip():
            if f.strip():
                return None
            continue
        if f.lstrip(" ") != o.lstrip(" ") or "\t" in _indent(f):
            return None
        d = len(_indent(f)) - len(_indent(o))
        if shift is None:
            shift = d
        elif d != shift:
            return None
    return shift if shift else None


def reindent_note(r: Reindent) -> str:
    """字下げを合わせて適用したことを結果に添える（モデルの思い込みを正す）。"""
    span = f"line {r.start}" if r.start == r.end else f"lines {r.start}-{r.end}"
    moved = f"{abs(r.shift)} space{'s' if abs(r.shift) > 1 else ''}"
    way = "deeper" if r.shift > 0 else "shallower"
    return (
        f"\nnote: `old` did not match exactly. It matched {span} once every line was indented "
        f"{moved} {way}, so `old` and `new` were both shifted by that amount before replacing. "
        "The file's own indentation was kept; use it for later edits."
    )


def match_lines(text: str, old: str) -> list[int]:
    """``old`` の各出現が始まる行番号（1 始まり・重なりは数えない＝``str.count`` と同じ）。"""
    found: list[int] = []
    pos = text.find(old)
    while pos != -1:
        found.append(text.count("\n", 0, pos) + 1)
        pos = text.find(old, pos + max(1, len(old)))
    return found


def not_unique_message(path: str, text: str, old: str, count: int) -> str:
    """複数一致のときにモデルへ返す文。"""
    starts = match_lines(text, old)
    lines = text.splitlines()
    shown = starts[:MAX_SHOWN_MATCHES]
    listed = ", ".join(str(no) for no in shown) + (" …" if len(starts) > len(shown) else "")
    width = len(str(max(shown))) if shown else 1
    body = []
    for no in shown:
        line = lines[no - 1] if no - 1 < len(lines) else ""
        if len(line) > _LINE_CLIP:
            line = line[:_LINE_CLIP] + " …"
        body.append(f"  {no:>{width}} | {line}")
    return "\n".join(
        [
            f"old string is not unique in {path} ({count} matches, starting at lines {listed}):",
            *body,
            "Add neighbouring lines to `old` so it matches exactly one place, or set "
            "replace_all=true to change every occurrence.",
        ]
    )


def same_text_message(path: str) -> str:
    """``old`` と ``new`` が同じときにモデルへ返す文。"""
    return (
        f"old and new are the same, so {path} was not changed (replacing a text with itself does "
        "nothing). Put the current text in `old` and the changed text in `new`; read the file "
        "again if you are not sure what it contains now."
    )


def empty_old_message(path: str) -> str:
    """``old`` が空のときにモデルへ返す文。"""
    return (
        f"old is empty, so {path} was not changed (an empty `old` matches between every "
        "character). To insert text, put the 1-2 existing lines next to the insertion point in "
        "`old`, and write those same lines plus the new text in `new`. To replace the whole "
        "file, use write_file."
    )
