"""応答の Markdown を、背景を塗らない色付けと簡素な記号で描く。"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, ClassVar, cast

from markdown_it.token import Token
from rich import box
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.markdown import (
    BlockQuote,
    Heading,
    HorizontalRule,
    ListElement,
    ListItem,
    Markdown,
    MarkdownContext,
    MarkdownElement,
    TableElement,
    TextElement,
)
from rich.segment import Segment
from rich.style import Style
from rich.syntax import PygmentsSyntaxTheme, Syntax, TokenType
from rich.table import Table
from rich.text import Text

# インラインコードと、言語名の無いコードブロックの色
_INLINE_CODE = Style(color="rgb(177,185,249)")
_DIM = Style(dim=True)

_STYLES: dict[str, Style] = {
    "markdown.code": _INLINE_CODE,
    "markdown.h1": Style(bold=True, italic=True, underline=True),
    **{f"markdown.h{n}": Style(bold=True) for n in range(2, 7)},
    "markdown.block_quote": Style(italic=True),
}

# 英数字のほかに言語名に使える文字（``python{1,3}`` は ``python`` まで見る）
_LANG_CHARS = frozenset("_.+#-")


class _CodeTheme(PygmentsSyntaxTheme):
    """monokai の文字色だけを使い、背景と地の文字色は端末のままにする。"""

    _BASE = Style(color="#f8f8f2").color

    def get_style_for_token(self, token_type: TokenType) -> Style:
        s = super().get_style_for_token(token_type)
        color = None if s.color == self._BASE else s.color
        return Style(color=color, bold=s.bold, italic=s.italic, underline=s.underline)

    def get_background_style(self) -> Style:
        return Style.null()


CODE_THEME = _CodeTheme("monokai")


def code_block(
    code: str, lang: str | None, *, head: bool = True, context: Sequence[str] = ()
) -> RenderableType:
    """コードブロックの本体（``lang`` は ``""`` で言語名なし・``None`` で字下げのブロック）。"""
    if lang == "":
        return Text(code, style=_INLINE_CODE)
    lexer = _lexer_for(lang) if lang else None
    if lexer is None:
        body = Text(code)
        return Group(Text(lang, style=_DIM), body) if lang and head else body
    # context は色を続けるために前に付け、描かない行
    return Syntax(
        "\n".join([*context, code]),
        lexer,
        theme=CODE_THEME,
        word_wrap=True,
        line_range=(len(context) + 1, None) if context else None,
    )


@lru_cache(maxsize=64)
def _lexer_for(lang: str) -> str | None:
    """色付けできる言語名（そのままで無ければ、言語名に使える文字の手前までで探す）。"""
    cut = len(lang)
    for i, ch in enumerate(lang):
        if not (ch.isalnum() or ch in _LANG_CHARS):
            cut = i
            break
    for name in dict.fromkeys((lang, lang[:cut])):
        if name and Syntax("", name).lexer is not None:
            return name
    return None


class _Heading(Heading):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"
        yield text


class _CodeBlock(TextElement):
    style_name = "markdown.code_block"

    @classmethod
    def create(cls, markdown: Markdown, token: Token) -> _CodeBlock:
        info = (token.info or "").split()
        return cls(None if token.type == "code_block" else (info[0] if info else ""))

    def __init__(self, lang: str | None) -> None:
        self.lang = lang

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield code_block(str(self.text).rstrip(), self.lang)


class _BlockQuote(BlockQuote):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        render_options = options.update(width=options.max_width - 2)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bar = Segment("▎ ", _DIM)
        for line in lines:
            yield bar
            yield from line
            yield Segment.line()


class _Rule(HorizontalRule):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Text("---")
        yield Text()


class _List(ListElement):
    depth = 1

    def on_enter(self, context: MarkdownContext) -> None:
        self.depth = sum(isinstance(e, _List) for e in context.stack)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        first = 1 if self.list_start is None else self.list_start
        last = first + len(self.items) - 1
        for index, item in enumerate(self.items):
            if self.list_type == "bullet_list_open":
                marker = "- "
            else:
                marker = f"{_ordinal(self.depth, first + index, first, last)}. "
            assert isinstance(item, _Item)
            yield from item.render_marked(console, options, marker)


class _Item(ListItem):
    def render_marked(self, console: Console, options: ConsoleOptions, marker: str) -> RenderResult:
        width = cell_len(marker)
        render_options = options.update(width=options.max_width - width)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        padding = Segment(" " * width)
        for index, line in enumerate(lines):
            yield Segment(marker) if index == 0 else padding
            yield from line
            yield Segment.line()


def _ordinal(depth: int, number: int, first: int, last: int) -> str:
    """番号付きリストの番号（入れ子の 2 段目は英小文字、3 段目はローマ数字の小文字）。"""
    if depth == 2 and first >= 1:
        return _letters(number)
    if depth == 3 and first >= 1 and last <= 3999:
        return _roman(number)
    return str(number)


def _letters(n: int) -> str:
    out = ""
    while n > 0:
        n -= 1
        out = chr(ord("a") + n % 26) + out
        n //= 26
    return out


_ROMAN = (
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
    (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
)  # fmt: skip


def _roman(n: int) -> str:
    out = ""
    for value, digits in _ROMAN:
        while n >= value:
            out += digits
            n -= value
    return out


class _Table(TableElement):
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        table = Table(box=box.SQUARE, show_lines=True, header_style=Style(), padding=(0, 1))
        if self.header is not None and self.header.row is not None:
            for cell in self.header.row.cells:
                heading = cell.content.copy()
                heading.justify = "center"
                table.add_column(heading, min_width=3, vertical="middle")
        if self.body is not None:
            for row in self.body.rows:
                table.add_row(*(cell.content for cell in row.cells))
        yield table


class ReplyMarkdown(Markdown):
    """応答の Markdown（名前で引く色は console のテーマより :data:`_STYLES` を先に見る）。"""

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "heading_open": _Heading,
        "fence": _CodeBlock,
        "code_block": _CodeBlock,
        "blockquote_open": _BlockQuote,
        "hr": _Rule,
        "bullet_list_open": _List,
        "ordered_list_open": _List,
        "list_item_open": _Item,
        "table_open": _Table,
    }

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        # 最下部の領域は別のスレッドで描くので、console のテーマは積み替えない
        yield from super().__rich_console__(cast(Console, _StyledConsole(console)), options)


class _StyledConsole:
    """``get_style`` だけ :data:`_STYLES` を先に見て、ほかは元の console に任せる。"""

    def __init__(self, console: Console) -> None:
        self._console = console

    def get_style(self, name: str | Style, *, default: Style | str | None = None) -> Style:
        if isinstance(name, str) and name in _STYLES:
            return _STYLES[name]
        return self._console.get_style(name, default=default)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._console, attr)
