"""ファイル変更の差分: 要約の行・行番号と記号・右端までの背景・語の強調・前後 3 行。"""

from __future__ import annotations

import io
import re

import pytest
from rich.console import Console, RenderableType
from rich.text import Text

from satsumaestro.cli import diff_view as diff_view_module
from satsumaestro.cli.diff_view import diff_view, new_file_view
from satsumaestro.i18n.text import pinned_ui_language

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_ADDED_BG = "48;2;34;92;43"
_REMOVED_BG = "48;2;122;41;54"
_ADDED_WORD = "48;2;56;166;96"
_REMOVED_WORD = "48;2;179;89;107"


def _render(renderable: RenderableType, *, color: bool = False, width: int = 40) -> str:
    out = io.StringIO()
    Console(
        file=out, force_terminal=True, width=width, color_system="truecolor" if color else None
    ).print(renderable)
    return out.getvalue()


def _lines(renderable: RenderableType, width: int = 40) -> list[str]:
    return [ln.rstrip() for ln in _render(renderable, width=width).splitlines()]


def _line_with(out: str, needle: str) -> str:
    return next(ln for ln in out.splitlines() if needle in _ANSI.sub("", ln))


def test_rows_have_line_numbers_markers_and_a_summary() -> None:
    view = diff_view("a\nb\nc\n", "a\nB\nc\n", "x.txt", max_rows=40)
    assert _lines(view) == ["1 行を追加、1 行を削除", " 1  a", " 2 -b", " 2 +B", " 3  c"]


def test_summary_wording_in_english() -> None:
    with pinned_ui_language("en"):
        both = _lines(diff_view("a\n", "b\nc\n", "x.txt", max_rows=40))[0]
        only_removed = _lines(diff_view("a\nb\n", "a\n", "x.txt", max_rows=40))[0]
        only_added = _lines(diff_view("a\n", "a\nb\nc\n", "x.txt", max_rows=40))[0]
    assert both == "Added 2 lines, removed 1 line"
    assert only_removed == "Removed 1 line"
    assert only_added == "Added 2 lines"


def test_summary_numbers_are_bold() -> None:
    out = _render(diff_view("a\n", "b\n", "x.txt", max_rows=40), color=True)
    assert "\x1b[1m1\x1b[0m 行を追加" in out


def test_changed_rows_are_painted_to_the_right_edge() -> None:
    out = _render(diff_view("a\nb\n", "a\nB\n", "x.txt", max_rows=40), color=True, width=30)
    removed, added = _line_with(out, " 2 -b"), _line_with(out, " 2 +B")
    assert _REMOVED_BG in removed and _ADDED_BG in added
    assert len(_ANSI.sub("", removed)) == 30 and len(_ANSI.sub("", added)) == 30
    context = _line_with(out, " 1  a")
    assert "48;" not in context and "\x1b[2m 1" in context  # 文脈の行番号は灰色


def test_only_the_changed_word_is_highlighted() -> None:
    out = _render(
        diff_view("    return a - b\n", "    return a + b\n", "x.py", max_rows=40), color=True
    )
    removed, added = _line_with(out, " 1 -"), _line_with(out, " 1 +")
    assert f"{_REMOVED_WORD}m-" in removed and _ADDED_WORD in added
    assert _ANSI.sub("", removed).count("-") == 2  # 記号と、強調した 1 語だけ


def test_mostly_rewritten_lines_get_no_word_highlight() -> None:
    out = _render(diff_view("alpha beta\n", "gamma delta\n", "x.txt", max_rows=40), color=True)
    assert _REMOVED_WORD not in out and _ADDED_WORD not in out


def test_long_lines_keep_the_background_but_get_no_word_highlight() -> None:
    # 同じ語の並ぶ長い 1 行（JSON の配列など）は、語を比べると止まって見えるほど時間がかかる
    before = "[" + ",".join(["0"] * 20_000) + "]\n"
    after = before.replace("0]", "1]")
    out = _render(diff_view(before, after, "a.json", max_rows=40), color=True, width=200)
    assert _REMOVED_BG in out and _ADDED_BG in out
    assert _REMOVED_WORD not in out and _ADDED_WORD not in out


def test_words_are_compared_only_on_shown_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = diff_view_module._mark_changed_words

    def counting(old: Text, new: Text) -> None:
        calls.append(old.plain)
        real(old, new)

    monkeypatch.setattr(diff_view_module, "_mark_changed_words", counting)
    before = "".join(f"line {i}\n" for i in range(50))
    after = "".join(f"line {i}!\n" for i in range(50))
    diff_view(before, after, "x.txt", max_rows=10)
    assert calls == [f"line {i}" for i in range(10)]


def test_added_and_context_rows_are_highlighted_but_removed_rows_are_not() -> None:
    before = "def f():\n    return 1\n"
    after = "def f():\n    return 2\n"
    out = _render(diff_view(before, after, "m.py", max_rows=40), color=True, width=60)
    assert "38;2;102;217;239mdef" in _line_with(out, " 1  def")  # 文脈の行は色付けする
    assert "38;2;102;217;239;48;2;34;92;43mreturn" in _line_with(out, " 2 +")
    assert "38;2;" not in _line_with(out, " 2 -")


def test_far_apart_changes_are_separated_by_dots_with_three_context_lines() -> None:
    before = "".join(f"l{i}\n" for i in range(1, 31))
    after = before.replace("l5\n", "L5\n").replace("l25\n", "L25\n")
    lines = _lines(diff_view(before, after, "x.txt", max_rows=40))
    assert "..." in lines
    assert "  2  l2" in lines and "  1  l1" not in lines  # 前後 3 行（番号は 2 桁の幅）
    assert "  8  l8" in lines and "  9  l9" not in lines


def test_long_rows_wrap_under_the_text_and_keep_the_background() -> None:
    out = _render(diff_view("x\n", "y " * 30 + "\n", "x.txt", max_rows=40), color=True, width=30)
    rows = [ln for ln in out.splitlines() if _ADDED_BG in ln]
    assert len(rows) >= 2
    assert _ANSI.sub("", rows[1]).startswith("    y")  # 2 行目は記号の幅だけ下げる


def test_rows_beyond_the_limit_are_counted() -> None:
    lines = _lines(diff_view("", "".join(f"n{i}\n" for i in range(50)), "x.txt", max_rows=10))
    assert lines[-1] == "… +40 行"


def test_no_changes() -> None:
    assert _lines(diff_view("a\n", "a\n", "x.txt", max_rows=40)) == ["（変更なし）"]


def test_new_file_shows_the_first_rows_with_dim_numbers() -> None:
    content = "".join(f"line {i}\n" for i in range(1, 13))
    view = new_file_view(content, "a.txt", max_rows=10)
    lines = _lines(view)
    assert lines[0] == "12 行を書き込み"
    assert lines[1] == "  1 line 1" and lines[10] == " 10 line 10"
    assert lines[-1] == "… +2 行"
    assert "\x1b[2m  1 " in _render(view, color=True)


def test_a_change_inside_a_multiline_string_keeps_the_string_color() -> None:
    before = 'doc = """\n' + "".join(f"text {i}\n" for i in range(20)) + '"""\n'
    after = before.replace("text 15\n", "TEXT 15\n")
    out = _render(diff_view(before, after, "m.py", max_rows=40), color=True, width=60)
    assert "38;2;230;219;116" in _line_with(out, "+TEXT 15")  # 文字列の色（開きは表の外）
