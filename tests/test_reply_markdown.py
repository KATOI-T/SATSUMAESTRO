"""応答の Markdown: 左寄せの見出し・``-`` と ``1.``・背景の無いコード・罫線の表・``---``・``▎``。"""

from __future__ import annotations

import io
import re

from rich.console import Console

from satsumaestro.cli.markdown import ReplyMarkdown, _letters, _roman

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SGR = re.compile(r"\x1b\[([0-9;]*)m")


def _has_background(s: str) -> bool:
    """SGR の中に背景色の指定（40〜47・48;…・100〜107）があるか。38;2;r;g;b の値は飛ばす。"""
    for m in _SGR.finditer(s):
        params = [int(p) for p in m.group(1).split(";") if p]
        i = 0
        while i < len(params):
            p = params[i]
            if p == 48 or 40 <= p <= 47 or 100 <= p <= 107:
                return True
            if p == 38:
                i += 5 if params[i + 1 : i + 2] == [2] else 3
                continue
            i += 1
    return False


def _render(md: str, *, color: bool = False, width: int = 60) -> str:
    out = io.StringIO()
    Console(
        file=out, force_terminal=True, width=width, color_system="truecolor" if color else None
    ).print(ReplyMarkdown(md))
    return out.getvalue()


def _lines(md: str) -> list[str]:
    """描いた行（行末の空白を落とす）。先頭がリストか表のとき Rich が置く空行は除く。"""
    lines = [ln.rstrip() for ln in _render(md).splitlines()]
    return lines[1:] if lines and not lines[0] else lines


def test_headings_are_left_aligned_without_hashes() -> None:
    lines = _lines("# Title\n\n## Section\n\ntext")
    assert lines[0] == "Title" and lines[2] == "Section"
    out = _render("# Title\n\n## Section", color=True)
    assert "\x1b[1;3;4mTitle" in out  # h1: 太字・斜体・下線
    assert "\x1b[1mSection" in out  # h2: 太字だけ


def test_list_markers() -> None:
    lines = _lines("- a\n- b\n  - nested\n\n1. one\n2. two\n   1. sub\n      1. deep")
    body = [ln for ln in lines if ln]
    assert body[:3] == ["- a", "- b", "  - nested"]
    assert body[3:] == ["1. one", "2. two", "   a. sub", "      i. deep"]


def test_ordinal_helpers() -> None:
    assert [_letters(n) for n in (1, 26, 27)] == ["a", "z", "aa"]
    assert [_roman(n) for n in (1, 4, 9, 14, 3999)] == ["i", "iv", "ix", "xiv", "mmmcmxcix"]


def test_list_item_continuation_is_indented_by_the_marker_width() -> None:
    lines = _lines("10. " + "word " * 20)
    assert lines[0].startswith("10. word")
    assert lines[1].startswith("    word")


def test_code_block_has_no_background_and_no_padding() -> None:
    out = _render("before\n\n```python\ndef f():\n    return 1\n```\n\nafter", color=True)
    assert not _has_background(out)
    plain = [ln.rstrip() for ln in _ANSI.sub("", out).splitlines()]
    assert plain == ["before", "", "def f():", "    return 1", "", "after"]
    assert "\x1b[38;2;102;217;239mdef" in out  # 色付けはする


def test_code_block_without_language_uses_the_inline_code_color() -> None:
    out = _render("```\nplain words\n```", color=True)
    assert "\x1b[38;2;177;185;249mplain words" in out


def test_unknown_language_is_named_on_a_dim_line() -> None:
    out = _render("```zig9\nx := 1\n```", color=True)
    plain = [ln.rstrip() for ln in _ANSI.sub("", out).splitlines()]
    assert plain == ["zig9", "x := 1"]
    assert "\x1b[2mzig9" in out


def test_language_with_attributes_still_highlights() -> None:
    out = _render("```python{1,3}\ndef f(): pass\n```", color=True)
    assert "python" not in _ANSI.sub("", out)
    assert "\x1b[38;2;102;217;239mdef" in out


def test_inline_code_is_colored_without_bold_or_background() -> None:
    out = _render("use `make check` now", color=True)
    assert "\x1b[38;2;177;185;249mmake check\x1b[0m" in out
    assert not _has_background(out)


def test_table_is_a_box_with_centered_header_and_row_separators() -> None:
    lines = _lines("| File | Lines |\n|---|---|\n| app.py | 12 |\n| b | 3 |")
    assert lines == [
        "┌────────┬───────┐",
        "│  File  │ Lines │",
        "├────────┼───────┤",
        "│ app.py │ 12    │",
        "├────────┼───────┤",
        "│ b      │ 3     │",
        "└────────┴───────┘",
    ]


def test_rule_and_quote() -> None:
    lines = _lines("a\n\n---\n\nb\n\n> quoted")
    assert "---" in lines and lines[lines.index("---") - 1] == ""
    assert "▎ quoted" in lines
