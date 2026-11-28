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


class Token(NamedTuple):
    """語（``word``）・区切り（``op``）・リダイレクト（``redirect``）・ヒアドキュメントの本文。"""

    text: str
    kind: str


def tokenize(command: str) -> list[Token] | None:
    """語と記号の並び。引用が閉じていなければ ``None``。"""
    return _Lexer(command).run()


def segments(tokens: list[Token], depth: int = 0) -> list[list[str]]:
    """区切りで分けた単純コマンドごとの語（リダイレクトとその行き先は除く）。"""
    found: list[list[str]] = [[]]
    target_next = False
    for token in tokens:
        if token.kind == "word":
            if target_next:
                target_next = False
            else:
                found[-1].append(token.text)
        elif token.kind == "redirect":
            target_next = True
        elif token.kind == "op":
            target_next = False
            found.append([])
        elif token.kind == "heredoc" and depth < _MAX_DEPTH:
            # 区切りを引用しないヒアドキュメントは、本文の $(…) と `…` を実行する
            for inner in substitutions(token.text):
                found.extend(segments(tokenize(inner) or [], depth + 1))
            found.append([])
    return [words for words in found if words]


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
