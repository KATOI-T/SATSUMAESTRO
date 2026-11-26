"""出した内容を覚え、端末の幅が変わったら新しい幅で描き直す Console（``ScrollbackConsole``）。

鍵・ネットワーク不要。

- 覚えるのは ``keep_scrollback`` の後の、端末への出力だけ。最下部の描き直し領域（``Control``）は
  覚えない
- 描き直しは画面とスクロールバックを消してから、覚えた出力を今の幅で出し直す。行数を渡すと、末尾から
  その行数を埋める分だけを出す（全部を出すまでは「描き直しが要る」のまま）
- カードの答え（``input``）は問いの後ろに続けて覚える。パスワードは覚えない
- 覚える数には上限があり、古いものから捨てる
"""

from __future__ import annotations

import io

import pytest
import rich.console
from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from satsumaestro.cli import scrollback
from satsumaestro.cli.live_region import LiveRegion
from satsumaestro.cli.scrollback import ScrollbackConsole

_CLEAR_ALL = "\x1b[2J\x1b[3J\x1b[H"


def _console(width: int = 40, *, terminal: bool = True) -> tuple[ScrollbackConsole, io.StringIO]:
    out = io.StringIO()
    console = ScrollbackConsole(
        file=out, width=width, force_terminal=terminal, color_system=None, legacy_windows=False
    )
    return console, out


def _redrawn(console: Console, out: io.StringIO) -> str:
    out.seek(0)
    out.truncate()
    assert isinstance(console, ScrollbackConsole)
    console.redraw()
    return out.getvalue()


def test_redraw_clears_the_screen_and_prints_again_at_the_new_width() -> None:
    console, out = _console(40)
    console.keep_scrollback()
    console.print(Rule())
    console.print("word " * 12)
    console.width = 20

    again = _redrawn(console, out)
    assert again.startswith(_CLEAR_ALL)
    lines = again.removeprefix(_CLEAR_ALL).split("\n")
    assert lines[0] == "─" * 20
    assert max(len(line) for line in lines) <= 20 and len(lines) > 3  # 本文も 20 桁で折り返す


def test_a_redraw_with_rows_prints_only_the_tail_that_fills_them() -> None:
    console, out = _console()
    console.keep_scrollback()
    for n in range(10):
        console.print(f"line {n}\nmore {n}")  # 1 回の出力が 2 行
    console.width = 30

    out.seek(0)
    out.truncate()
    assert not console.redraw(rows=5)
    assert out.getvalue() == _CLEAR_ALL + "".join(f"line {n}\nmore {n}\n" for n in (7, 8, 9))
    assert console.needs_redraw()  # 全部を出すまでは要る
    assert console.redraw(rows=100)  # 埋まらなければ全部
    assert not console.needs_redraw()


def test_only_output_after_keep_scrollback_is_redrawn() -> None:
    console, out = _console()
    console.print("before")
    console.keep_scrollback()
    console.print("after")

    assert _redrawn(console, out) == _CLEAR_ALL + "after\n"


def test_nothing_is_kept_when_the_output_is_not_a_terminal() -> None:
    console, _ = _console(terminal=False)
    console.keep_scrollback()
    console.print("x")
    console.width = 20

    assert not console.needs_redraw()


def test_a_redraw_is_needed_only_after_the_width_changed() -> None:
    console, _ = _console(40)
    console.keep_scrollback()
    assert not console.needs_redraw()  # 何も出していない
    console.print("x")
    assert not console.needs_redraw()
    console.width = 30
    assert console.needs_redraw()
    console.redraw()
    assert not console.needs_redraw()


def test_the_live_region_at_the_bottom_is_not_kept() -> None:
    console, out = _console()
    console.keep_scrollback()
    region = LiveRegion(console, fps=0)
    region.start()
    region.update(Text("status line"))
    region.refresh()
    console.print("done")
    region.stop()

    again = _redrawn(console, out)
    assert "status line" not in again
    assert again.endswith("done\n")


def test_a_card_answer_is_kept_after_its_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda *_: "y")  # 端末が映す答え（改行は含まない）
    console, out = _console()
    console.keep_scrollback()
    assert console.input("Allow? ") == "y"

    assert _redrawn(console, out) == _CLEAR_ALL + "Allow? y\n"


def test_a_password_answer_is_not_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*_: object, **__: object) -> str:
        return "secret"

    monkeypatch.setattr("getpass.getpass", fake)
    # Rich 15 より前は読み込みの時に getpass を取り込む
    monkeypatch.setattr(rich.console, "getpass", fake, raising=False)
    console, out = _console()
    console.keep_scrollback()
    assert console.input("Key: ", password=True) == "secret"

    assert "secret" not in _redrawn(console, out)


def test_old_output_is_dropped_beyond_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scrollback, "MAX_RECORDS", 3)
    console, out = _console()
    console.keep_scrollback()
    for n in range(5):
        console.print(f"line {n}")

    assert _redrawn(console, out) == _CLEAR_ALL + "line 2\nline 3\nline 4\n"
