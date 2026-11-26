"""対話の REPL の起動の画面: 絵（桜島）と製品名、操作の案内の行。"""

from __future__ import annotations

from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment
from rich.style import Style
from rich.text import Text

from satsumaestro.i18n.text import ui

PRODUCT_NAME = "SATSUMAESTRO"
COLOR = "#fb5a2b"
# 字と枠の両方に色を付け、暗い背景でも明るい背景でも読めるようにする
COMMAND_KEY = "#f5a8f5 on #4d2550"
SHORTCUT_KEY = "#f2d36b on #4d4116"
# • は幅が曖昧で、全角で描く端末では 1 行に収まらない
_SEPARATOR = " ∙ "
_TIGHT_SEPARATOR = "  "
WIDTH = 60
# 絵の上の空き 1 行と、案内の下の空き 1 行・入力欄 4 行
_ABOVE = 1
_BELOW = 5

_ART = (
    "........................O.OOO...............................",
    ".......................OOOOOOO..............................",
    ".....................OOOOOOOOOOO............................",
    "....................OOO.OOOOO.OOO...........................",
    "...................OOO.OO.OOOO.OOOO.........................",
    ".................OOOO.OOO.OOOO.O.OOOO.......................",
    "................OOOO.OOO.OOOOOO.O.OOOO......................",
    "..............OOOOOO.OOO.OOOOOOO.O.OOOOOO...................",
    "............OOOOOOO.OOO.OOOOOOOO.OO..OOOOOO.................",
    "..........OOOOOOOO.OOOO.OOOOOOOOO.OOO.OOOOOOO...............",
    "......OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO.OOOOOOOOOO...........",
    "...OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO.......",
)

_GLYPHS = {
    "S": ("OOOO", "O...", "OOOO", "...O", "OOOO"),
    "A": (".OO.", "O..O", "OOOO", "O..O", "O..O"),
    "T": ("OOOO", ".OO.", ".OO.", ".OO.", ".OO."),
    "U": ("O..O", "O..O", "O..O", "O..O", "OOOO"),
    "M": ("O...O", "OO.OO", "O.O.O", "O...O", "O...O"),
    "E": ("OOOO", "O...", "OOO.", "O...", "OOOO"),
    "R": ("OOO.", "O..O", "OOO.", "O.O.", "O..O"),
    "O": ("OOOO", "O..O", "O..O", "O..O", "OOOO"),
}
_GAP = 2

_HALF = {("O", "O"): "█", ("O", "."): "▀", (".", "O"): "▄", (".", "."): " "}
# 点字の 1 字の中の点（行 0〜3・列 0〜1）のビット
_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
SPACED_NAME = "  ".join(PRODUCT_NAME)


def word_pixels(word: str) -> list[str]:
    """``word`` を字の画素の行（5 行）にする。"""
    rows = [".".join(_GLYPHS[ch][k] for ch in word) for k in range(5)]
    return [row.ljust(WIDTH, ".") for row in rows]


def banner_lines() -> list[str]:
    """大きな絵・空き・字の行（絵は字の幅の中央・行末の空白は落とす）。"""
    art = _centered(_to_half_blocks(list(_ART)), WIDTH)
    return [*art, *([""] * (_GAP // 2)), *_to_half_blocks(word_pixels(PRODUCT_NAME))]


def small_banner_lines() -> list[str]:
    """同じ絵を点字で縦横半分にした行（字間を空けた製品名の幅の中央）。"""
    return _centered(_to_braille(list(_ART)), len(SPACED_NAME))


def _to_half_blocks(rows: list[str]) -> list[str]:
    if len(rows) % 2:
        rows = [*rows, "." * len(rows[0])]
    return [
        "".join(_HALF[pair] for pair in zip(top, bottom, strict=True)).rstrip()
        for top, bottom in zip(rows[::2], rows[1::2], strict=True)
    ]


def _to_braille(rows: list[str]) -> list[str]:
    width = len(rows[0])
    rows = [*rows, *(["." * width] * (-len(rows) % 4))]
    lines = []
    for y in range(0, len(rows), 4):
        cells = []
        for x in range(0, width, 2):
            bits = sum(
                bit
                for dy, pair in enumerate(_DOTS)
                for dx, bit in enumerate(pair)
                if rows[y + dy][x + dx : x + dx + 1] == "O"
            )
            cells.append(chr(0x2800 + bits) if bits else " ")
        lines.append("".join(cells).rstrip())
    return lines


def _centered(lines: list[str], width: int) -> list[str]:
    """描かれた範囲が幅 ``width`` の中央に来るよう、行をずらす。"""
    drawn = [line for line in lines if line.strip()]
    left = min(len(line) - len(line.lstrip()) for line in drawn)
    shift = (width - (max(len(line) for line in drawn) - left)) // 2 - left
    if shift >= 0:
        return [" " * shift + line if line else line for line in lines]
    return [line[-shift:] for line in lines]


# これより狭い端末では製品名だけを出す
SMALL_WIDTH = max(len(SPACED_NAME), *(len(line) for line in small_banner_lines()))


def big_rows(width: int) -> int:
    """大きな絵の形で、絵の上から入力欄までが入る高さ。"""
    return _ABOVE + len(banner_lines()) + 1 + len(StartupHint().lines(width)) + _BELOW


def small_rows(width: int) -> int:
    """点字の絵の形で、絵の上から入力欄までが入る高さ。"""
    return _ABOVE + len(small_banner_lines()) + 3 + len(StartupHint().lines(width)) + _BELOW


class Banner:
    """起動の絵と製品名（描くときの幅と高さで形を選んで中央に置き、最後に 1 行あける）。"""

    def __init__(self, notice: str | None = None) -> None:
        self._notice = notice

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = options.max_width
        height = options.size.height - self._notice_rows(console, width)
        art = Style.parse(COLOR)
        name = Style.parse(f"bold {COLOR}")
        if width >= WIDTH and height >= big_rows(width):
            yield from _lines(banner_lines(), art, " " * ((width - WIDTH) // 2))
        elif width >= SMALL_WIDTH and height >= small_rows(width):
            pad = " " * ((width - len(SPACED_NAME)) // 2)
            yield from _lines([*small_banner_lines(), ""], art, pad)
            yield from _lines([SPACED_NAME], name, pad)
        else:  # 入らなければ切る（折り返さない）
            pad = " " * max(0, (width - len(PRODUCT_NAME)) // 2)
            yield Text(pad + PRODUCT_NAME, style=name, no_wrap=True, overflow="crop")
        yield Segment.line()

    def _notice_rows(self, console: Console, width: int) -> int:
        """案内の下に出す知らせの行数（その幅で折り返した数）。"""
        return len(Text(self._notice).wrap(console, width)) if self._notice else 0


def _hint_items(*, roomy: bool) -> list[Text]:
    """色の枠で示したキーと、その説明（``roomy`` が偽なら枠の余白を詰める）。"""
    keys = (
        ("/help", COMMAND_KEY, ui("cli.banner.help")),
        ("/exit", COMMAND_KEY, ui("cli.banner.quit")),
        ("Ctrl+C", SHORTCUT_KEY, ui("cli.banner.interrupt")),
        ("Shift+Tab", SHORTCUT_KEY, ui("cli.banner.switch_plan_code")),
    )
    space = " " if roomy else ""
    items = []
    for key, frame, label in keys:
        item = Text()
        item.append(f"{space}{key}{space}", frame)
        item.append(" ")
        item.append_text(Text.from_markup(label, style="dim"))
        items.append(item)
    return items


def _joined(items: list[Text], separator: str) -> Text:
    line = Text()
    for index, item in enumerate(items):
        if index:
            line.append(separator, "dim")
        line.append_text(item)
    return line


class StartupHint:
    """起動の画面の操作の案内（作ったときの言語のまま、幅に合わせて詰め方を選ぶ）。"""

    def __init__(self) -> None:
        self._roomy = _hint_items(roomy=True)
        self._tight = _hint_items(roomy=False)

    def lines(self, width: int) -> list[Text]:
        """幅に並べた行（入らなければ詰め、それでも入らなければ項目の切れ目で折る）。"""
        for items, separator in ((self._roomy, _SEPARATOR), (self._tight, _TIGHT_SEPARATOR)):
            line = _joined(items, separator)
            if line.cell_len <= width:
                return [line]
        rows: list[list[Text]] = [[]]
        for item in self._tight:
            if rows[-1] and _joined([*rows[-1], item], _TIGHT_SEPARATOR).cell_len > width:
                rows.append([])
            rows[-1].append(item)
        return [_joined(row, _TIGHT_SEPARATOR) for row in rows]

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for line in self.lines(options.max_width):
            line.justify = "center"
            yield line


def print_banner(console: Console, notice: str | None = None) -> bool:
    """端末なら絵と製品名を出す（``notice`` は案内の下に出す知らせで、高さに数える）。"""
    if not console.is_terminal:
        return False
    console.print()
    console.print(Banner(notice))
    return True


def _lines(lines: list[str], style: Style, pad: str) -> RenderResult:
    for line in lines:
        if line:
            yield Segment(pad)
            yield Segment(line, style)
        yield Segment.line()
