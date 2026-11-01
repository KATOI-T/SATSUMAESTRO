"""対話の REPL の起動のときに出す絵（桜島）と製品名。"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

PRODUCT_NAME = "SATSUMAESTRO"
COLOR = "#fb5a2b"
WIDTH = 60

_ART = (
    "..........................OOOOO.............................",
    ".......................OOOOO................................",
    "......................OOO...................................",
    ".....................OOO....................................",
    ".....................OOOOOOOOO..............................",
    "......................OOOOOOOOOO............................",
    "........................OOOOOOOOO...........................",
    "..............................OOOO..........................",
    "..............................OOO...........................",
    "............................OOOOO...........................",
    "...........................OOOO.............................",
    "..........................OO................................",
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


def word_pixels(word: str) -> list[str]:
    """``word`` を字の画素の行（5 行）にする。"""
    rows = [".".join(_GLYPHS[ch][k] for ch in word) for k in range(5)]
    return [row.ljust(WIDTH, ".") for row in rows]


def banner_pixels() -> list[str]:
    """絵・空き・製品名を縦に並べた画素の表。"""
    return [*_ART, *(["." * WIDTH] * _GAP), *word_pixels(PRODUCT_NAME)]


def banner_lines() -> list[str]:
    """画素の表を、縦 2 画素ずつ半ブロックの文字にした行（行末の空白は落とす）。"""
    rows = banner_pixels()
    if len(rows) % 2:
        rows.append("." * WIDTH)
    return [
        "".join(_HALF[pair] for pair in zip(top, bottom, strict=True)).rstrip()
        for top, bottom in zip(rows[::2], rows[1::2], strict=True)
    ]


def print_banner(console: Console) -> bool:
    """端末に十分な幅があれば絵と製品名を出す。"""
    if not console.is_terminal or console.width < WIDTH:
        return False
    for line in banner_lines():
        console.print(Text(line, style=COLOR), overflow="ignore", crop=False, no_wrap=True)
    console.print()
    return True
