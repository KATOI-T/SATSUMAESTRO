"""REPL の起動のときに出す絵（桜島）と製品名。"""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from satsumaestro.cli import banner


def _console(width: int, *, terminal: bool = True) -> Console:
    return Console(file=StringIO(), width=width, force_terminal=terminal, color_system="truecolor")


def _output(console: Console) -> str:
    file = console.file
    assert isinstance(file, StringIO)
    return file.getvalue()


def test_the_banner_is_sixty_columns_wide_and_sixteen_lines_tall() -> None:
    lines = banner.banner_lines()

    assert len(lines) == 16
    assert max(len(line) for line in lines) == banner.WIDTH
    assert set("".join(lines)) <= set("▀▄█ ")
    assert lines[12] == ""


def test_every_row_of_the_picture_spans_the_full_width() -> None:
    assert len(banner._ART) == 24
    assert {len(row) for row in banner._ART} == {banner.WIDTH}
    assert set("".join(banner._ART)) <= {"O", "."}


def test_the_name_is_drawn_glyph_by_glyph_with_one_pixel_between() -> None:
    rows = banner.word_pixels("SA")

    assert rows[0].startswith("OOOO..OO.")
    assert rows[2].startswith("OOOO.OOOO")
    assert {len(row) for row in banner.word_pixels(banner.PRODUCT_NAME)} == {banner.WIDTH}


def test_it_is_printed_in_vermilion_on_a_wide_terminal() -> None:
    console = _console(80)

    assert banner.print_banner(console)
    out = _output(console)
    assert "\x1b[38;2;251;90;43m" in out
    assert out.count("\n") == 17


def test_it_is_not_printed_on_a_terminal_narrower_than_the_picture() -> None:
    console = _console(banner.WIDTH - 1)

    assert not banner.print_banner(console)
    assert _output(console) == ""


def test_it_is_not_printed_when_the_output_is_not_a_terminal() -> None:
    console = _console(120, terminal=False)

    assert not banner.print_banner(console)
    assert _output(console) == ""
