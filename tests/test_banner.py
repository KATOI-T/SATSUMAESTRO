"""REPL の起動の画面: 絵（桜島）と製品名、操作の案内の行。"""

from __future__ import annotations

import unicodedata
from io import StringIO

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from satsumaestro.__about__ import __version__
from satsumaestro.cli import banner
from satsumaestro.i18n.text import pinned_ui_language

# 起動の画面がまるごと入る高さ（大きな絵を選ぶ）。
TALL = 40


def _console(
    width: int, *, height: int = TALL, terminal: bool = True, color: bool = True
) -> Console:
    """幅と高さを決めた Console（Windows でも同じ幅になるよう ``legacy_windows`` を切る）。"""
    return Console(
        file=StringIO(),
        width=width,
        height=height,
        force_terminal=terminal,
        color_system="truecolor" if color else None,
        legacy_windows=False,
    )


def _output(console: Console) -> str:
    file = console.file
    assert isinstance(file, StringIO)
    return file.getvalue()


def test_the_banner_is_sixty_columns_wide_and_ten_lines_tall() -> None:
    lines = banner.banner_lines()

    assert len(lines) == 10  # 絵 6 行・空き 1 行・字 3 行
    assert max(len(line) for line in lines) == banner.WIDTH
    assert set("".join(lines)) <= set("▀▄█ ")
    assert lines[6] == ""  # 絵と字の間の空き


def _drawn_span(lines: list[str]) -> tuple[int, int]:
    """空白を除いて描かれた範囲（左端の桁、右端の次の桁）。"""
    drawn = [line for line in lines if line.strip()]
    return min(len(line) - len(line.lstrip()) for line in drawn), max(len(line) for line in drawn)


def test_the_picture_is_centered_over_the_name() -> None:
    lines = banner.banner_lines()
    left, right = _drawn_span(lines[:6])
    assert _drawn_span(lines[7:]) == (0, banner.WIDTH)
    assert left == banner.WIDTH - right  # 左右の空きが同じ


def test_the_picture_is_the_mountain_without_the_smoke() -> None:
    assert len(banner._ART) == 12
    assert {len(row) for row in banner._ART} == {banner.WIDTH}
    assert set("".join(banner._ART)) <= {"O", "."}
    # 一番上の行は火口の窪み（左右に分かれた頂）で、その上に噴煙は無い。
    assert banner._ART[0].strip(".") == "O.OOO"


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
    assert out.count("\n") == 12  # 前に 1 行・10 行・後に 1 行
    assert out.startswith("\n")  # 起動したコマンドの行と絵の間


def _plain_lines(width: int, height: int = TALL) -> list[str]:
    console = _console(width, height=height, color=False)
    console.print(banner.Banner())
    return _output(console).split("\n")


@pytest.mark.parametrize(
    ("width", "height"), [(80, TALL), (80, 15), (banner.SMALL_WIDTH - 1, TALL)]
)
def test_the_version_is_not_shown(width: int, height: int) -> None:
    """バージョンは起動の画面に出さない（``/version`` で出す）。"""
    assert __version__ not in "\n".join(_plain_lines(width, height))


def _braille_pixels(lines: list[str]) -> list[str]:
    """点字の行を画素の表（``O`` / ``.``）に戻す（行末の空きは落とす）。"""
    width = max(len(line) for line in lines)
    rows = []
    for line in lines:
        cells = [ord(ch) - 0x2800 if ch != " " else 0 for ch in line.ljust(width)]
        for bits in ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80)):
            rows.append("".join("O" if c & b else "." for c in cells for b in bits))
    return rows


def test_the_small_picture_is_the_same_pixels_at_half_the_size() -> None:
    lines = banner.small_banner_lines()
    pixels = _braille_pixels(lines)
    art = list(banner._ART)

    assert len(lines) == 3  # 大きな絵の 6 行の半分
    # 中央に寄せた分（左の空き）を除けば、画素は大きな絵と同じ。
    shift = min(len(row) - len(row.lstrip(".")) for row in pixels) - min(
        len(row) - len(row.lstrip(".")) for row in art
    )
    assert [row[shift:].rstrip(".") for row in pixels] == [row.rstrip(".") for row in art]


def test_the_small_picture_is_centered_over_the_spaced_name() -> None:
    left, right = _drawn_span(banner.small_banner_lines())
    assert left == len(banner.SPACED_NAME) - right  # 左右の空きが同じ


def test_a_narrower_terminal_gets_the_picture_in_braille_and_the_name_spaced_out() -> None:
    width = banner.WIDTH - 1
    lines = _plain_lines(width)

    pad = " " * ((width - len(banner.SPACED_NAME)) // 2)
    small = [pad + line if line else line for line in banner.small_banner_lines()]
    assert lines[:6] == [*small, "", pad + "S  A  T  S  U  M  A  E  S  T  R  O", ""]
    assert banner.SMALL_WIDTH == len(banner.SPACED_NAME) == 34
    console = _console(banner.WIDTH - 1)
    assert banner.print_banner(console)
    assert f"\x1b[1;38;2;251;90;43m{banner.SPACED_NAME}" in _output(console)  # 太字の朱


def test_a_terminal_too_narrow_for_the_small_picture_gets_only_the_name() -> None:
    assert _plain_lines(banner.SMALL_WIDTH - 1)[:2] == [" " * 10 + banner.PRODUCT_NAME, ""]
    assert _plain_lines(8)[:2] == [banner.PRODUCT_NAME[:8], ""]  # 入らなければ切る（折り返さない）


def test_the_screen_needs_eighteen_rows_for_the_big_picture_and_thirteen_for_the_small() -> None:
    # 上の空き 1・絵と字 10・下の空き 1・案内 1・空き 1・入力欄 4
    assert banner.big_rows(80) == 18
    # 上の空き 1・点字の絵 3・空き 1・字 1・下の空き 1・案内 1・空き 1・入力欄 4
    assert banner.small_rows(80) == 13


def test_the_rows_count_the_hint_lines_at_the_width() -> None:
    # 日本語の案内は 63 桁未満で 2 行、37 桁未満で 3 行になる。
    assert banner.big_rows(62) == 19
    assert banner.small_rows(50) == 14
    assert banner.small_rows(banner.SMALL_WIDTH) == 15


@pytest.mark.parametrize(
    ("height", "form"),
    [
        (banner.big_rows(80), "big"),
        (banner.big_rows(80) - 1, "small"),
        (banner.small_rows(80), "small"),
        (banner.small_rows(80) - 1, "name"),
        (5, "name"),
    ],
)
def test_a_low_terminal_gets_a_smaller_form(height: int, form: str) -> None:
    out = "\n".join(_plain_lines(80, height))

    assert ("█" in out) is (form == "big")
    assert (banner.SPACED_NAME in out) is (form == "small")
    assert (banner.PRODUCT_NAME in out) is (form == "name")


# 前に確かめた更新の知らせ（80 桁では 2 行に折れる）
NOTICE = "SATSUMAESTRO 0.5.4 is available (you have 0.5.3). To upgrade: brew upgrade satsumaestro"


@pytest.mark.parametrize("notice", [None, NOTICE])
@pytest.mark.parametrize("language", ["ja", "en"])
@pytest.mark.parametrize("width", [34, 36, 37, 39, 40, 50, 59, 60, 62, 63, 67, 68, 80, 120])
def test_the_picture_and_everything_down_to_the_input_box_fit_the_terminal(
    language: str, width: int, notice: str | None
) -> None:
    """絵を描いたなら、絵の上から入力欄までが画面に入る（ちょうど入る高さから描く）。"""
    first: dict[str, int] = {}
    with pinned_ui_language(language):
        for height in range(5, 30):
            console = _console(width, height=height, color=False)
            banner.print_banner(console, notice)
            console.print(banner.StartupHint())
            if notice:
                console.print(Text(notice), justify="center")
            out = _output(console)
            rows = out.count("\n") + 1 + 4  # 案内の下の空き 1 行・入力欄 4 行
            form = "big" if "█" in out else "small" if banner.SPACED_NAME in out else "name"
            if form != "name":
                assert rows <= height, (height, out)
                if form not in first:
                    assert rows == height, (height, out)
                    first[form] = height
    assert "small" in first
    assert ("big" in first) is (width >= banner.WIDTH)


@pytest.mark.parametrize(("width", "rows"), [(120, 1), (80, 2)])
def test_the_update_notice_counts_its_wrapped_lines_toward_the_height(
    width: int, rows: int
) -> None:
    def form(height: int) -> str:
        console = _console(width, height=height, color=False)
        console.print(banner.Banner(NOTICE))
        return "big" if "█" in _output(console) else "smaller"

    assert form(banner.big_rows(width) + rows) == "big"
    assert form(banner.big_rows(width) + rows - 1) == "smaller"


def test_the_same_banner_is_laid_out_again_for_a_new_width() -> None:
    picture = banner.Banner()
    console = _console(80, color=False)
    console.print(picture)
    console.width = 40
    console.print(picture)

    out = _output(console)
    assert out.count(banner.banner_lines()[-1]) == 1  # 大きな字は 1 回目だけ
    assert out.count(banner.SPACED_NAME) == 1  # 字間を空けた製品名は 2 回目だけ


def test_it_is_not_printed_when_the_output_is_not_a_terminal() -> None:
    console = _console(120, terminal=False)

    assert not banner.print_banner(console)
    assert _output(console) == ""


def _margins(lines: list[str], width: int) -> tuple[int, int]:
    """描かれた範囲の左右の空き（桁）。"""
    left, right = _drawn_span(lines)
    return left, width - right


def _drawn(lines: list[str]) -> list[str]:
    """空でない行。"""
    return [line for line in lines if line.strip()]


@pytest.mark.parametrize("width", [banner.WIDTH, banner.WIDTH + 1, 80, 121, 200])
def test_the_wide_banner_is_centered_on_the_terminal(width: int) -> None:
    lines = _drawn(_plain_lines(width))
    left, right = _margins(lines[6:], width)  # 字の 3 行（絵は字の幅の中央）
    assert left == (width - banner.WIDTH) // 2 and abs(left - right) <= 1
    art_left, art_right = _margins(lines[:6], width)
    assert abs(art_left - art_right) <= 1


@pytest.mark.parametrize("width", [banner.SMALL_WIDTH, 45, banner.WIDTH - 1])
def test_the_small_banner_is_centered_on_the_terminal(width: int) -> None:
    lines = _drawn(_plain_lines(width))
    name_left, name_right = _margins(lines[-1:], width)
    assert abs(name_left - name_right) <= 1
    art_left, art_right = _margins(lines[:-1], width)
    assert abs(art_left - art_right) <= 1


@pytest.mark.parametrize("width", [13, 20, banner.SMALL_WIDTH - 1])
def test_the_name_alone_is_centered_on_the_terminal(width: int) -> None:
    left, right = _margins(_plain_lines(width)[:1], width)
    assert abs(left - right) <= 1


# ── 操作の案内 ──
ROOMY = " /help  ヘルプ ∙  /exit  終了 ∙  Ctrl+C  中断 ∙  Shift+Tab  plan/code 切替"
TIGHT = "/help ヘルプ  /exit 終了  Ctrl+C 中断  Shift+Tab plan/code 切替"
KEYS = ("/help", "/exit", "Ctrl+C", "Shift+Tab")


def _hint(width: int) -> str:
    console = _console(width, color=False)
    console.print(banner.StartupHint())
    return _output(console)


def _hint_lines(width: int) -> list[str]:
    return [line.plain for line in banner.StartupHint().lines(width)]


def test_the_hint_shows_the_keys_in_one_line() -> None:
    assert _hint_lines(120) == [ROOMY]
    assert _hint(120).strip() == ROOMY.strip()
    assert _hint(120).count("\n") == 1


def test_the_hint_tightens_before_it_wraps() -> None:
    roomy = cell_len(ROOMY)
    assert _hint_lines(roomy) == [ROOMY]
    assert _hint_lines(roomy - 1) == [TIGHT]  # 1 行のまま詰める
    assert _hint(roomy - 1).strip() == TIGHT
    tight = cell_len(TIGHT)
    assert _hint_lines(tight - 1) == [  # 入らなければ項目の切れ目で折る
        "/help ヘルプ  /exit 終了  Ctrl+C 中断",
        "Shift+Tab plan/code 切替",
    ]
    assert _hint(tight - 1).count("\n") == 2


@pytest.mark.parametrize("language", ["ja", "en"])
@pytest.mark.parametrize("width", range(26, 80))
def test_a_wrapped_hint_keeps_each_key_with_its_label(language: str, width: int) -> None:
    """折っても、キーとその説明を別の行に分けない（行の頭に説明だけが来ない）。"""
    with pinned_ui_language(language):
        lines = _hint_lines(width)
    for line in lines:
        assert cell_len(line) <= width
        assert line.split()[0] in KEYS, line  # 行の頭は必ずキー
        for key in KEYS:
            if key in line:  # キーの後ろには、同じ行にその説明が続く
                after = line.split(key, 1)[1].split()
                assert after and after[0] not in KEYS, line
    assert sum(line.count("/help") for line in lines) == 1


def test_the_hint_uses_no_character_of_ambiguous_width() -> None:
    """幅の曖昧な字（• など）は、全角で描く端末で 1 行からはみ出すので使わない。"""
    for language in ("ja", "en"):
        with pinned_ui_language(language):
            text = "".join(_hint_lines(200))
        assert all(unicodedata.east_asian_width(ch) != "A" for ch in text), text


def test_the_hint_keeps_the_language_it_was_made_in() -> None:
    """描き直すときに画面の言語が替わっていても、案内は出したときの言語のまま。"""
    with pinned_ui_language("en"):
        hint = banner.StartupHint()
    console = _console(120, color=False)
    console.print(hint)  # いまの画面の言語は日本語
    out = _output(console)
    assert "/help  help" in out and "ヘルプ" not in out


@pytest.mark.parametrize("width", [80, 100, 121])
def test_the_hint_is_centered_on_the_terminal(width: int) -> None:
    line = _hint(width).split("\n")[0]
    left = line.index("/help") - 1  # 枠の左の余白 1 桁の前まで
    right = width - (left + cell_len(ROOMY))
    assert left > 0 and abs(left - right) <= 1


def test_the_keys_are_framed_in_color() -> None:
    console = _console(120)
    console.print(banner.StartupHint())
    out = _output(console)

    # スラッシュコマンドは紫の枠、キーの組み合わせは黄の枠。
    assert "\x1b[38;2;245;168;245;48;2;77;37;80m /help \x1b[0m" in out
    assert "\x1b[38;2;242;211;107;48;2;77;65;22m Shift+Tab \x1b[0m" in out
    assert "Ctrl-D" not in out
