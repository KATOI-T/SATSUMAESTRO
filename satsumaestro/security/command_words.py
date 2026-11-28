"""シェルのコマンドを、引用を見分けて語と記号に分ける（守りの検査が共有する読み方）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple

# 長い物から当てる。引用やエスケープの中の記号は語の一部になる
_SEPARATORS = (";;&", ";;", ";&", "&&", "||", "|&", ";", "&", "|", "(", ")")
_REDIRECTS = ("&>>", "<<<", "<<-", "&>", ">>", ">|", ">&", "<<", "<&", "<>", ">", "<")
_OPERATORS = sorted((*_SEPARATORS, *_REDIRECTS), key=len, reverse=True)
_HEREDOCS = frozenset({"<<", "<<-"})
_BLANKS = " \t"
MAX_DEPTH = 8
_DQUOTE_ESCAPES = '"\\$`\n'
_BACKQUOTE_ESCAPE = re.compile(r"\\([\\`$])")
_BACKQUOTE_ESCAPE_IN_DQUOTES = re.compile(r"\\([\\`$\"])")
# コマンドの位置に来たときだけ予約語として読む語
_KEYWORDS = frozenset(
    {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "esac", "{", "}", "!"}
)
# 予約語の time が前に付く物（それ以外の time は包みのコマンドとして読む）
_TIMED = frozenset({"{", "!", "if", "while", "until", "for", "select", "case", "[[", "function"})
_CASE_ENDS = frozenset({";;", ";&", ";;&"})
_SEQUENCE = re.compile(
    r"(-?\d+)\.\.(-?\d+)(?:\.\.(-?\d+))?|([A-Za-z])\.\.([A-Za-z])(?:\.\.(-?\d+))?"
)
# ブレース展開で作る語の数と入れ子の上限（超えたら展開せずにそのまま読む）
_MAX_BRACE_WORDS = 1024
_MAX_BRACE_NESTING = 64


class Token(NamedTuple):
    """語・区切り（``op``）・リダイレクト・ヒアドキュメントの本文・語の中の置き換えの中身。"""

    text: str
    kind: str


def tokenize(command: str, *, complete_lines: bool = False) -> list[Token] | None:
    """語と記号の並び。引用が閉じなければ ``None``（``complete_lines`` ならその行より前の分）。"""
    # シェルは閉じない引用の行に来るまで、前の行を動かしている
    return _Lexer(command, complete_lines=complete_lines).run()


@dataclass
class Command:
    """単純コマンドの語と、ヒアドキュメントか ``<<<`` で渡す入力。"""

    words: list[str] = field(default_factory=list)
    stdin: str | None = None


def commands(tokens: list[Token], depth: int = 0) -> list[Command]:
    """単純コマンドごとの語と入力（予約語・リダイレクトとその行き先・for と case の語は除く）。"""
    return _Commands(tokens, depth).run()


def segments(tokens: list[Token], depth: int = 0) -> list[list[str]]:
    """単純コマンドごとの語。"""
    return [command.words for command in commands(tokens, depth)]


def command_start(words: list[str]) -> list[str]:
    """先頭の予約語（``then``・``do``・``{``・``!`` など）を除いた語。"""
    i = 0
    while i < len(words) and words[i] in _KEYWORDS:
        i += 1
    return words[i:]


def substitutions(text: str) -> list[str]:
    """``text`` の中の ``$(…)`` とバッククォートの中身（入れ子の内側は中身の側で読む）。"""
    found: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
        elif text.startswith("$(", i):
            end = _substitution_end(text, i + 2, 0)
            end = len(text) if end is None else end
            found.append(text[i + 2 : end])
            i = end + 1
        elif c == "`":
            end = _backquote_end(text, i + 1)
            found.append(_BACKQUOTE_ESCAPE.sub(r"\1", text[i + 1 : end]))
            i = end + 1
        else:
            i += 1
    return found


def _substitution_end(text: str, start: int, level: int) -> int | None:
    """``$(`` の後の ``start`` から中身を読み、閉じる ``)`` の位置。閉じなければ ``None``。"""
    if level >= MAX_DEPTH:
        end = _closing_paren(text, start)
        return None if end >= len(text) else end
    lexer = _Lexer(text, start=start, level=level + 1)
    lexer.run()
    return lexer.end


def _backquote_end(text: str, i: int) -> int:
    """``i`` から始まるバッククォートの中身を閉じる位置。無ければ末尾。"""
    while i < len(text) and text[i] != "`":
        i += 2 if text[i] == "\\" else 1
    return min(i, len(text))


def _closing_paren(text: str, i: int) -> int:
    """``i`` から始まる中身を閉じる ``)`` の位置（引用の中の括弧は数えない）。無ければ末尾。"""
    level = 0
    quote = ""
    while i < len(text):
        c = text[i]
        if c == "\\" and quote != "'":
            i += 2
            continue
        if quote:
            quote = "" if c == quote else quote
        elif c in "'\"":
            quote = c
        elif c == "(":
            level += 1
        elif c == ")":
            if level == 0:
                return i
            level -= 1
        i += 1
    return len(text)


class _TooManyWords(Exception):
    pass


_Chars = list[tuple[str, bool]]


def _brace_words(pieces: list[tuple[str, bool]]) -> list[str] | None:
    """引用していない ``{a,b}``・``{1..3}`` を展開した語。展開する物が無ければ ``None``。"""
    chars = [(c, quoted) for piece, quoted in pieces for c in piece]
    if not any(c == "{" and not quoted for c, quoted in chars):
        return None
    try:
        expanded = _expand_braces(chars, [_MAX_BRACE_WORDS], 0)
    except _TooManyWords:
        return None
    if expanded == [chars]:
        return None
    # 展開で空になった語は消える（引用した空の文字列を含む語は残る）
    keep_empty = any(piece == "" and quoted for piece, quoted in pieces)
    words = ["".join(c for c, _ in word) for word in expanded]
    return [word for word in words if word or keep_empty]


def _expand_braces(chars: _Chars, budget: list[int], nesting: int) -> list[_Chars]:
    found = _brace_at(chars)
    if found is None:
        budget[0] -= 1
        if budget[0] < 0:
            raise _TooManyWords
        return [chars]
    if nesting >= _MAX_BRACE_NESTING:
        raise _TooManyWords
    start, end, alternatives = found
    expanded: list[_Chars] = []
    for alternative in alternatives:
        word = chars[:start] + alternative + chars[end + 1 :]
        expanded.extend(_expand_braces(word, budget, nesting + 1))
    return expanded


def _brace_at(chars: _Chars) -> tuple[int, int, list[_Chars]] | None:
    """最初に展開する括弧の位置と、中の選択肢。``${…}`` と選択肢の無い括弧は飛ばす。"""
    i = 0
    while i < len(chars):
        if chars[i] != ("{", False):
            i += 1
            continue
        end = _matching_brace(chars, i)
        if end is not None and i and chars[i - 1] == ("$", False):
            i = end + 1
            continue
        alternatives = None if end is None else _alternatives(chars[i + 1 : end])
        if end is not None and alternatives is not None:
            return i, end, alternatives
        i += 1
    return None


def _matching_brace(chars: _Chars, i: int) -> int | None:
    depth = 0
    for j in range(i, len(chars)):
        if chars[j] == ("{", False):
            depth += 1
        elif chars[j] == ("}", False):
            depth -= 1
            if depth == 0:
                return j
    return None


def _alternatives(inner: _Chars) -> list[_Chars] | None:
    """括弧の中の、引用していない ``,`` で分けた選択肢か ``x..y`` の並び。"""
    parts: list[_Chars] = [[]]
    depth = 0
    for c, quoted in inner:
        if not quoted and c == "{":
            depth += 1
        elif not quoted and c == "}":
            depth -= 1
        elif not quoted and c == "," and depth == 0:
            parts.append([])
            continue
        parts[-1].append((c, quoted))
    if len(parts) > 1:
        return parts
    if any(quoted for _, quoted in inner):
        return None
    values = _sequence("".join(c for c, _ in inner))
    return None if values is None else [[(c, False) for c in value] for value in values]


def _sequence(text: str) -> list[str] | None:
    """``1..5``・``01..10..3``・``a..e`` の並び。並びでなければ ``None``。"""
    match = _SEQUENCE.fullmatch(text)
    if match is None:
        return None
    first, last, step = match[1] or match[4], match[2] or match[5], match[3] or match[6]
    letters = match[4] is not None
    low, high = (ord(first), ord(last)) if letters else (int(first), int(last))
    stride = abs(int(step or 1)) or 1
    if abs(high - low) // stride >= _MAX_BRACE_WORDS:
        raise _TooManyWords
    values = range(low, high + 1, stride) if low <= high else range(low, high - 1, -stride)
    if letters:
        return [chr(v) for v in values]
    padded = any(len(e.lstrip("-")) > 1 and e.lstrip("-").startswith("0") for e in (first, last))
    width = max(len(first), len(last)) if padded else 0
    return [str(v).zfill(width) for v in values]


def words(text: str) -> list[str]:
    """``text`` の語だけ（記号は捨てる）。読めなければ空。"""
    return [t.text for t in tokenize(text) or [] if t.kind == "word"]


class _Commands:
    """語と記号の並びを、予約語と for・case の作りを見分けて単純コマンドに分ける。"""

    def __init__(self, tokens: list[Token], depth: int) -> None:
        self.tokens = tokens
        self.depth = depth
        self.found: list[Command] = [Command()]
        self.awaiting: list[Command] = []
        self.start = True
        self.cases = 0
        self.pattern = False
        self.pattern_begins = False

    def run(self) -> list[Command]:
        redirect: str | None = None
        i = 0
        while i < len(self.tokens):
            text, kind = self.tokens[i]
            if self.pattern and self.in_pattern(text, kind):
                pass
            elif kind == "redirect":
                redirect = text
            elif kind == "op":
                redirect = None
                self.separate(text)
            elif kind == "word" and redirect is not None:
                if redirect == "<<<":
                    self.found[-1].stdin = text
                elif redirect in _HEREDOCS:
                    self.awaiting.append(self.found[-1])
                redirect = None
            elif kind == "word" and self.start:
                i = self.command_word(i)
            elif kind == "word":
                self.found[-1].words.append(text)
            elif kind in ("heredoc", "heredoc-literal"):
                if self.awaiting:
                    self.awaiting.pop(0).stdin = text
                if kind == "heredoc" and self.depth < MAX_DEPTH:
                    # 区切りを引用しないヒアドキュメントは、本文の $(…) と `…` を実行する
                    for inner in substitutions(text):
                        self.found.extend(self.inner(inner))
                    self.found.append(Command())
            elif kind == "substitution" and self.depth < MAX_DEPTH:
                # 語の中の置き換えは、そのコマンドより先に動く
                self.found[-1:-1] = self.inner(text)
            i += 1
        return [command for command in self.found if command.words]

    def inner(self, text: str) -> list[Command]:
        return commands(tokenize(text, complete_lines=True) or [], self.depth + 1)

    def closes_case_pattern(self) -> bool:
        """ここまでの並びが case の型を読み終えたところで、次の ``)`` が型を閉じるか。"""
        # 深さを上限にして作ると、置き換えとヒアドキュメントの中は読まない（形だけを見る）
        self.run()
        return self.pattern and not self.pattern_begins

    def separate(self, op: str) -> None:
        if op in _CASE_ENDS and self.cases:
            self.pattern = self.pattern_begins = True
        self.start = True
        self.found.append(Command())

    def in_pattern(self, text: str, kind: str) -> bool:
        """case の型の語と記号なら読み飛ばす。型でない物が来たらコマンドとして読み直す。"""
        if kind == "word" and text == "esac" and self.pattern_begins:
            self.cases -= 1
            self.pattern = False
            self.separate(text)
        elif kind == "word":
            self.pattern_begins = False
        elif kind == "op" and text == "\n":
            pass
        elif kind == "op" and text == "(" and self.pattern_begins:
            self.pattern_begins = False
        elif kind == "op" and text == "|" and not self.pattern_begins:
            pass
        elif kind == "op" and text == ")" and not self.pattern_begins:
            self.pattern = False
            self.separate(text)
        else:
            self.pattern = False
            return False
        return True

    def command_word(self, i: int) -> int:
        """コマンドの位置の語を読み、読み終えた最後の位置を返す。"""
        text = self.tokens[i].text
        if text in _KEYWORDS:
            if text == "esac" and self.cases:
                self.cases -= 1
            return i
        if text == "function" and self.word_at(i + 1):
            return i + 1
        if text == "coproc":
            return i + 1 if self.word_at(i + 1) and self.word_at(i + 2, "{") else i
        if text in ("for", "select") and self.word_at(i + 1):
            # for 名前 in 語… の語はコマンドでない。$( などの記号で読みをやめる
            if not self.word_at(i + 2, "in"):
                return i + 1
            end = i + 3
            while self.word_at(end):
                end += 1
            return end - 1
        if text == "case" and self.word_at(i + 1):
            end = i + 2
            while self.op_at(end, "\n"):
                end += 1
            if self.word_at(end, "in"):
                self.cases += 1
                self.pattern = self.pattern_begins = True
                return end
        if text == "time":
            end = i + 1
            while self.word_at(end, "-p") or self.word_at(end, "--"):
                end += 1
            if self.op_at(end, "(") or (self.word_at(end) and self.tokens[end].text in _TIMED):
                return end - 1
        self.found[-1].words.append(text)
        self.start = False
        return i

    def word_at(self, i: int, text: str | None = None) -> bool:
        if i >= len(self.tokens) or self.tokens[i].kind != "word":
            return False
        return text is None or self.tokens[i].text == text

    def op_at(self, i: int, text: str) -> bool:
        return i < len(self.tokens) and self.tokens[i] == Token(text, "op")


class _Lexer:
    def __init__(
        self, command: str, *, start: int = 0, level: int = 0, complete_lines: bool = False
    ) -> None:
        self.text = command
        self.i = start
        self.complete_lines = complete_lines
        self.complete = 0
        # level が 1 以上なら $( の中身を読み、閉じる ) の位置を end に置いて止まる
        self.level = level
        self.parens = 0
        self.end: int | None = None
        self.out: list[Token] = []
        self.word: list[tuple[str, bool]] = []
        self.subs: list[str] = []
        self.in_word = False
        self.quoted = False
        self.heredoc_next: str | None = None
        self.heredocs: list[tuple[str, bool, bool]] = []

    def run(self) -> list[Token] | None:
        text = self.text
        while self.i < len(text) and self.end is None:
            c = text[self.i]
            if c in _BLANKS:
                self.flush()
                self.i += 1
            elif c == "\n":
                self.flush()
                self.out.append(Token("\n", "op"))
                self.i += 1
                self.read_heredocs()
                self.complete = len(self.out)
            elif c == "#" and not self.in_word:
                end = text.find("\n", self.i)
                self.i = len(text) if end < 0 else end
            elif c == "\\":
                self.backslash()
            elif c == "'":
                if not self.single_quote(self.i + 1):
                    return self.unclosed()
            elif c == "$" and text.startswith("$'", self.i):
                if not self.ansi_quote():
                    return self.unclosed()
            elif c == '"':
                if not self.double_quote():
                    return self.unclosed()
            elif c == "`":
                closing = self.backquote(self.i + 1, _BACKQUOTE_ESCAPE)
                if closing is None:
                    return self.unclosed()
                self.word.append((text[self.i : closing + 1], True))
                self.in_word = True
                self.i = closing + 1
            elif (op := _operator_at(text, self.i)) is not None:
                self.operator(op)
            else:
                self.word.append((c, False))
                self.in_word = True
                self.i += 1
        self.flush()
        self.read_heredocs()
        return self.out

    def unclosed(self) -> list[Token] | None:
        return self.out[: self.complete] if self.complete_lines else None

    def flush(self) -> None:
        if not self.in_word:
            return
        text = "".join(piece for piece, _ in self.word)
        expanded = None
        if self.heredoc_next is not None:
            self.heredocs.append((text, self.heredoc_next == "<<-", self.quoted))
            self.heredoc_next = None
        else:
            expanded = _brace_words(self.word)
        self.out.extend(Token(word, "word") for word in ([text] if expanded is None else expanded))
        self.out.extend(Token(inner, "substitution") for inner in self.subs)
        self.word.clear()
        self.subs.clear()
        self.in_word = self.quoted = False

    def operator(self, op: str) -> None:
        # 2>file の 2 は語でなくリダイレクトの番号
        digits = "".join(piece for piece, _ in self.word).isdigit()
        if op in _REDIRECTS and self.in_word and not self.quoted and digits:
            self.word.clear()
            self.in_word = False
        self.flush()
        if self.level and op == "(":
            self.parens += 1
        elif self.level and op == ")":
            if self.parens:
                self.parens -= 1
            elif not _Commands(self.out, MAX_DEPTH).closes_case_pattern():
                self.end = self.i
                return
        kind = "redirect" if op in _REDIRECTS else "op"
        self.out.append(Token(op, kind))
        if op in _HEREDOCS:
            self.heredoc_next = op
        self.i += len(op)

    def backslash(self) -> None:
        nxt = self.text[self.i + 1 : self.i + 2]
        if nxt != "\n" and nxt:
            self.word.append((nxt, True))
            self.in_word = self.quoted = True
        self.i += 2

    def single_quote(self, start: int) -> bool:
        end = self.text.find("'", start)
        if end < 0:
            return False
        self.word.append((self.text[start:end], True))
        self.in_word = self.quoted = True
        self.i = end + 1
        return True

    def ansi_quote(self) -> bool:
        i = self.i + 2
        buf: list[str] = []
        while i < len(self.text) and self.text[i] != "'":
            if self.text[i] == "\\" and i + 1 < len(self.text):
                buf.append(self.text[i + 1])
                i += 2
                continue
            buf.append(self.text[i])
            i += 1
        if i >= len(self.text):
            return False
        self.word.append(("".join(buf), True))
        self.in_word = self.quoted = True
        self.i = i + 1
        return True

    def double_quote(self) -> bool:
        text = self.text
        i = self.i + 1
        buf: list[str] = []
        while i < len(text) and text[i] != '"':
            if text[i] == "\\" and text[i + 1 : i + 2] in tuple(_DQUOTE_ESCAPES):
                if text[i + 1] != "\n":
                    buf.append(text[i + 1])
                i += 2
                continue
            # 二重引用の中でも $(…) と `…` は動く。中の引用は中身の側で閉じる
            end: int | None = i
            if text.startswith("$(", i):
                end = _substitution_end(text, i + 2, self.level)
                if end is not None:
                    self.subs.append(text[i + 2 : end])
            elif text[i] == "`":
                end = self.backquote(i + 1, _BACKQUOTE_ESCAPE_IN_DQUOTES)
            if end is None:
                return False
            buf.append(text[i : end + 1])
            i = end + 1
        if i >= len(text):
            return False
        self.word.append(("".join(buf), True))
        self.in_word = self.quoted = True
        self.i = i + 1
        return True

    def backquote(self, start: int, escapes: re.Pattern[str]) -> int | None:
        """``start`` から始まるバッククォートの中身を置き換えとして控え、閉じる位置を返す。"""
        end = _backquote_end(self.text, start)
        if end >= len(self.text):
            return None
        self.subs.append(escapes.sub(r"\1", self.text[start:end]))
        return end

    def read_heredocs(self) -> None:
        """改行の後に続くヒアドキュメントの本文を、区切りの行まで 1 つの語として除けておく。"""
        for delimiter, strip_tabs, literal in self.heredocs:
            lines: list[str] = []
            while self.i < len(self.text):
                end = self.text.find("\n", self.i)
                end = len(self.text) if end < 0 else end
                line = self.text[self.i : end]
                self.i = end + 1
                if (line.lstrip("\t") if strip_tabs else line) == delimiter:
                    break
                lines.append(line)
            kind = "heredoc-literal" if literal else "heredoc"
            self.out.append(Token("\n".join(lines), kind))
        self.heredocs.clear()


def _operator_at(text: str, i: int) -> str | None:
    return next((op for op in _OPERATORS if text.startswith(op, i)), None)
