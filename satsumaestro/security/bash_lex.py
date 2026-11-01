"""bash コマンドのリダイレクト抽出（クォート状態を追う字句走査）。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

DEVICE_TARGETS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr"})

_WORD_END = frozenset(" \t\n|&;<>()")
_WORD_BREAK = frozenset(" \t\n;|&()")
_REDIRECT_TOKENS = ("&>>", "&>", ">>", ">|", "<>", ">", "<")


@dataclass
class _Heredoc:
    delim: str
    strip_tabs: bool
    quoted: bool


def iter_redirects(command: str) -> Iterator[tuple[str, str]]:
    """クォートの外にある、ファイルを対象とするリダイレクトを ``(演算子, 対象)`` で順に返す。"""
    s = command
    n = len(s)
    i = 0
    word_start = True
    closers: list[str] = []
    pending: list[_Heredoc] = []
    while i < n:
        c = s[i]
        if c == "\n":
            if closers:
                pending = []
                i += 1
            else:
                i = _skip_heredoc_bodies(s, i + 1, pending)
                pending = []
            word_start = True
            continue
        if c in " \t":
            i += 1
            word_start = True
            continue
        if c == "\\":
            i += 2
            word_start = False
            continue
        if c == "'":
            i = _skip_single(s, i)
            word_start = False
            continue
        if c == '"':
            i = _skip_double(s, i)
            word_start = False
            continue
        if c == "`":
            if closers and closers[-1] == "`":
                closers.pop()
            else:
                closers.append("`")
            i += 1
            word_start = False
            continue
        if c == "$":
            if s.startswith("$'", i):
                i = _skip_ansi_c(s, i + 1)
            elif s.startswith("$((", i):
                closers.append("))")
                i += 3
            elif s.startswith("$(", i):
                closers.append(")")
                i += 2
            elif s.startswith("${", i):
                closers.append("}")
                i += 2
            elif s.startswith("$[", i):
                closers.append("]")
                i += 2
            else:
                i += 1
            word_start = False
            continue
        if c == "#" and word_start and not closers:
            end = s.find("\n", i)
            i = n if end < 0 else end
            continue
        if closers and c in "(){}[]":
            i = _track_nesting(s, i, closers)
            word_start = c in "()"
            continue
        if c == "(" and s.startswith("((", i):
            closers.append("))")
            i += 2
            word_start = True
            continue
        if c == "&" and s.startswith("&>", i):
            op = "&>>" if s.startswith("&>>", i) else "&>"
            target, i, _ = _read_word(s, _skip_ws(s, i + len(op)), for_target=True)
            if target:
                yield op, target
            word_start = False
            continue
        if c not in "<>":
            i += 1
            word_start = c in _WORD_BREAK
            continue
        word_start = False
        if s.startswith("<<<", i):
            _, i, _ = _read_word(s, _skip_ws(s, i + 3))
            continue
        if s.startswith("<<", i):
            if closers:
                i += 2
                continue
            strip_tabs = s.startswith("<<-", i)
            delim, i, quoted = _read_word(s, _skip_ws(s, i + (3 if strip_tabs else 2)))
            pending.append(_Heredoc(delim, strip_tabs, quoted))
            continue
        if s.startswith("<(", i) or s.startswith(">(", i):
            if closers:
                closers.append(")")
            i += 2
            word_start = True
            continue
        if s.startswith("<&", i):
            _, i, _ = _read_word(s, _skip_ws(s, i + 2))
            continue
        if s.startswith(">&", i):
            target, i, _ = _read_word(s, _skip_ws(s, i + 2), for_target=True)
            if target and not (target == "-" or target.isdigit()):
                yield ">&", target
            continue
        op = next(o for o in _REDIRECT_TOKENS if s.startswith(o, i))
        target, i, _ = _read_word(s, _skip_ws(s, i + len(op)), for_target=True)
        if target:
            yield op, target


def _track_nesting(s: str, i: int, closers: list[str]) -> int:
    """文脈の中の括弧を数える。"""
    c = s[i]
    if c == "(":
        closers.append(")")
    elif c == "{":
        closers.append("}")
    elif c == "[":
        closers.append("]")
    elif c == ")":
        if closers[-1] == ")":
            closers.pop()
        elif closers[-1] == "))" and s.startswith("))", i):
            closers.pop()
            return i + 2
    elif c == closers[-1]:
        closers.pop()
    return i + 1


def _skip_single(s: str, i: int) -> int:
    """``s[i] == "'"`` の単引用符文字列の直後の位置。"""
    end = s.find("'", i + 1)
    if end < 0:
        raise ValueError("No closing quotation")
    return end + 1


def _skip_double(s: str, i: int) -> int:
    """``s[i] == '"'`` の二重引用符文字列の直後の位置。"""
    return _skip_escaped(s, i, '"')


def _skip_ansi_c(s: str, i: int) -> int:
    """``s[i] == "'"``（``$'`` の ``'``）の ANSI-C 文字列の直後の位置。"""
    return _skip_escaped(s, i, "'")


def _skip_escaped(s: str, i: int, quote: str) -> int:
    j = i + 1
    n = len(s)
    while j < n:
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == quote:
            return j + 1
        j += 1
    raise ValueError("No closing quotation")


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t":
        i += 1
    return i


def _read_word(s: str, i: int, *, for_target: bool = False) -> tuple[str, int, bool]:
    """``i`` から 1 語を読む。"""
    n = len(s)
    out: list[str] = []
    quoted = False
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            out.append(s[i + 1])
            i += 2
            quoted = True
        elif c == "$" and s.startswith("$'", i):
            if for_target:
                raise ValueError("ANSI-C quoted ($'...') redirect target cannot be checked")
            end = _skip_ansi_c(s, i + 1)
            out.append(s[i + 2 : end - 1])
            i = end
            quoted = True
        elif c == "'":
            end = _skip_single(s, i)
            out.append(s[i + 1 : end - 1])
            i = end
            quoted = True
        elif c == '"':
            end = _skip_double(s, i)
            out.append(s[i + 1 : end - 1].replace('\\"', '"'))
            i = end
            quoted = True
        elif c in _WORD_END:
            break
        else:
            out.append(c)
            i += 1
    return "".join(out), i, quoted


def _skip_heredoc_bodies(s: str, i: int, pending: list[_Heredoc]) -> int:
    """保留中のヒアドキュメントの本文を順に読み飛ばし、その直後の位置を返す。"""
    n = len(s)
    for doc in pending:
        logical = ""
        while i < n:
            end = s.find("\n", i)
            end = n if end < 0 else end
            line = s[i:end]
            i = end + 1
            if not doc.quoted and _continues(line):
                logical += line[:-1]
                continue
            logical += line
            if (logical.lstrip("\t") if doc.strip_tabs else logical) == doc.delim:
                break
            logical = ""
    return min(i, n)


def _continues(line: str) -> bool:
    """行末が行継続の ``\\`` か（``\\\\`` はエスケープされた ``\\`` なので継続しない）。"""
    trailing = len(line) - len(line.rstrip("\\"))
    return trailing % 2 == 1
