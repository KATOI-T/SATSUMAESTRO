"""シェルのコマンドを、引用を見分けて語と記号に分ける（守りの検査が共有する読み方）。"""

from __future__ import annotations

from typing import NamedTuple

# 長い物から当てる。引用やエスケープの中の記号は語の一部になる
_SEPARATORS = (";;&", ";;", ";&", "&&", "||", "|&", ";", "&", "|", "(", ")")
_REDIRECTS = ("&>>", "<<<", "<<-", "&>", ">>", ">|", ">&", "<<", "<&", "<>", ">", "<")
_OPERATORS = sorted((*_SEPARATORS, *_REDIRECTS), key=len, reverse=True)
_HEREDOCS = frozenset({"<<", "<<-"})
_BLANKS = " \t"
_MAX_DEPTH = 8
_DQUOTE_ESCAPES = '"\\$`\n'
# コマンドの位置に来たときだけ予約語として読む語
_KEYWORDS = frozenset(
    {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "esac", "{", "}", "!"}
)
# 予約語の time が前に付く物（それ以外の time は包みのコマンドとして読む）
_TIMED = frozenset({"{", "!", "if", "while", "until", "for", "select", "case", "[[", "function"})
_CASE_ENDS = frozenset({";;", ";&", ";;&"})


class Token(NamedTuple):
    """語（``word``）・区切り（``op``）・リダイレクト（``redirect``）・ヒアドキュメントの本文。"""

    text: str
    kind: str


def tokenize(command: str) -> list[Token] | None:
    """語と記号の並び。引用が閉じていなければ ``None``。"""
    return _Lexer(command).run()


def segments(tokens: list[Token], depth: int = 0) -> list[list[str]]:
    """単純コマンドごとの語（予約語・リダイレクトとその行き先・for と case の語は除く）。"""
    return _Commands(tokens, depth).run()


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
            end = _closing_paren(text, i + 2)
            found.append(text[i + 2 : end])
            i = end + 1
        elif c == "`":
            end = i + 1
            while end < len(text) and text[end] != "`":
                end += 2 if text[end] == "\\" else 1
            found.append(text[i + 1 : end])
            i = end + 1
        else:
            i += 1
    return found


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


def words(text: str) -> list[str]:
    """``text`` の語だけ（記号は捨てる）。読めなければ空。"""
    return [t.text for t in tokenize(text) or [] if t.kind == "word"]


class _Commands:
    """語と記号の並びを、予約語と for・case の作りを見分けて単純コマンドに分ける。"""

    def __init__(self, tokens: list[Token], depth: int) -> None:
        self.tokens = tokens
        self.depth = depth
        self.found: list[list[str]] = [[]]
        self.start = True
        self.cases = 0
        self.pattern = False
        self.pattern_begins = False

    def run(self) -> list[list[str]]:
        target_next = False
        i = 0
        while i < len(self.tokens):
            text, kind = self.tokens[i]
            if self.pattern and self.in_pattern(text, kind):
                pass
            elif kind == "redirect":
                target_next = True
            elif kind == "op":
                target_next = False
                self.separate(text)
            elif kind == "word" and target_next:
                target_next = False
            elif kind == "word" and self.start:
                i = self.command_word(i)
            elif kind == "word":
                self.found[-1].append(text)
            elif kind == "heredoc" and self.depth < _MAX_DEPTH:
                # 区切りを引用しないヒアドキュメントは、本文の $(…) と `…` を実行する
                for inner in substitutions(text):
                    self.found.extend(segments(tokenize(inner) or [], self.depth + 1))
                self.found.append([])
            i += 1
        return [words for words in self.found if words]

    def separate(self, op: str) -> None:
        if op in _CASE_ENDS and self.cases:
            self.pattern = self.pattern_begins = True
        self.start = True
        self.found.append([])

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
        self.found[-1].append(text)
        self.start = False
        return i

    def word_at(self, i: int, text: str | None = None) -> bool:
        if i >= len(self.tokens) or self.tokens[i].kind != "word":
            return False
        return text is None or self.tokens[i].text == text

    def op_at(self, i: int, text: str) -> bool:
        return i < len(self.tokens) and self.tokens[i] == Token(text, "op")


class _Lexer:
    def __init__(self, command: str) -> None:
        self.text = command
        self.i = 0
        self.out: list[Token] = []
        self.word: list[str] = []
        self.in_word = False
        self.quoted = False
        self.heredoc_next: str | None = None
        self.heredocs: list[tuple[str, bool, bool]] = []

    def run(self) -> list[Token] | None:
        text = self.text
        while self.i < len(text):
            c = text[self.i]
            if c in _BLANKS:
                self.flush()
                self.i += 1
            elif c == "\n":
                self.flush()
                self.out.append(Token("\n", "op"))
                self.i += 1
                self.read_heredocs()
            elif c == "#" and not self.in_word:
                end = text.find("\n", self.i)
                self.i = len(text) if end < 0 else end
            elif c == "\\":
                self.backslash()
            elif c == "'":
                if not self.single_quote(self.i + 1):
                    return None
            elif c == "$" and text.startswith("$'", self.i):
                if not self.ansi_quote():
                    return None
            elif c == '"':
                if not self.double_quote():
                    return None
            elif (op := _operator_at(text, self.i)) is not None:
                self.operator(op)
            else:
                self.word.append(c)
                self.in_word = True
                self.i += 1
        self.flush()
        self.read_heredocs()
        return self.out

    def flush(self) -> None:
        if not self.in_word:
            return
        text = "".join(self.word)
        if self.heredoc_next is not None:
            self.heredocs.append((text, self.heredoc_next == "<<-", self.quoted))
            self.heredoc_next = None
        self.out.append(Token(text, "word"))
        self.word.clear()
        self.in_word = self.quoted = False

    def operator(self, op: str) -> None:
        # 2>file の 2 は語でなくリダイレクトの番号
        if op in _REDIRECTS and self.in_word and not self.quoted and "".join(self.word).isdigit():
            self.word.clear()
            self.in_word = False
        self.flush()
        kind = "redirect" if op in _REDIRECTS else "op"
        self.out.append(Token(op, kind))
        if op in _HEREDOCS:
            self.heredoc_next = op
        self.i += len(op)

    def backslash(self) -> None:
        nxt = self.text[self.i + 1 : self.i + 2]
        if nxt != "\n" and nxt:
            self.word.append(nxt)
            self.in_word = self.quoted = True
        self.i += 2

    def single_quote(self, start: int) -> bool:
        end = self.text.find("'", start)
        if end < 0:
            return False
        self.word.append(self.text[start:end])
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
        self.word.append("".join(buf))
        self.in_word = self.quoted = True
        self.i = i + 1
        return True

    def double_quote(self) -> bool:
        i = self.i + 1
        buf: list[str] = []
        while i < len(self.text) and self.text[i] != '"':
            if self.text[i] == "\\" and self.text[i + 1 : i + 2] in tuple(_DQUOTE_ESCAPES):
                if self.text[i + 1] != "\n":
                    buf.append(self.text[i + 1])
                i += 2
                continue
            buf.append(self.text[i])
            i += 1
        if i >= len(self.text):
            return False
        self.word.append("".join(buf))
        self.in_word = self.quoted = True
        self.i = i + 1
        return True

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
