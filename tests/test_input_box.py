"""REPL の入力欄（複数行・罫線・ヒント行・送信後の残し方）。"""

from __future__ import annotations

import asyncio
import io
import re
import signal
import sys
from typing import Any

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput, Output
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console
from rich.rule import Rule

from satsumaestro.cli import input_box as input_box_mod
from satsumaestro.cli import key_protocol
from satsumaestro.cli.commands import parse_slash
from satsumaestro.cli.input_box import InputBox
from satsumaestro.cli.modes import mode_hint_fragments
from satsumaestro.cli.scrollback import ScrollbackConsole
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import ModeState

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _box(inp: Any, *, console: Console | None = None, **kw: Any) -> InputBox:
    output: Output = kw.pop("output", DummyOutput())
    return InputBox(
        console=console or Console(file=io.StringIO(), width=80),
        hint=lambda: mode_hint_fragments(ModeState()),
        status=lambda: "mock-model",
        fallback_prompt=lambda: "> ",
        input=inp,
        output=output,
        **kw,
    )


async def _read(keys: str, **kw: Any) -> str:
    with create_pipe_input() as inp:
        box = _box(inp, **kw)
        inp.send_text(keys)
        return await box.prompt_async()


async def test_enter_submits() -> None:
    assert await _read("hello\r") == "hello"


@pytest.mark.parametrize(
    "newline",
    [
        pytest.param("\x1b[27;2;13~", id="shift-enter-modify-other-keys"),
        pytest.param("\x1b[13;2u", id="shift-enter-csi-u"),
        # 端末の設定で Shift+Enter に ESC+CR を送らせた場合（ターン中に打つと CR は \n で届く）。
        pytest.param("\x1b\r", id="shift-enter-as-esc-cr"),
        pytest.param("\x1b\n", id="shift-enter-as-esc-cr-typed-ahead"),
    ],
)
async def test_newline_keys_insert_a_line_break(newline: str) -> None:
    assert await _read(f"first{newline}second\r") == "first\nsecond"


async def test_enter_typed_ahead_still_submits() -> None:
    # 入力欄の外（ターン中）で押した Enter は、端末が \n（Ctrl+J）に変えて溜める。
    assert await _read("next request\n") == "next request"


@pytest.mark.parametrize("text", ["a\\b", "a\\"])
async def test_backslash_is_plain_text(text: str) -> None:
    # 行末の「\」のあとの Enter も、改行にせず送る。
    assert await _read(text + "\r") == text


async def test_multi_line_paste_is_one_input() -> None:
    pasted = "\x1b[200~line 1\r\nline 2\nline 3\x1b[201~"
    assert await _read(pasted + "\r") == "line 1\nline 2\nline 3"


# ── 端末に区別して送らせている間（kitty keyboard protocol）──


def _vt100_output(raw: io.StringIO) -> Output:
    return Vt100_Output(raw, lambda: Size(rows=20, columns=60), enable_cpr=False)


async def test_protocol_is_requested_only_while_the_box_is_open() -> None:
    raw = io.StringIO()
    assert await _read("hi\r", output=_vt100_output(raw)) == "hi"
    written = raw.getvalue()
    assert written.count(key_protocol.ENABLE) == written.count(key_protocol.DISABLE) == 1
    assert written.index(key_protocol.ENABLE) < written.rindex(key_protocol.DISABLE)


async def test_protocol_is_released_after_ctrl_c() -> None:
    raw = io.StringIO()
    with pytest.raises(KeyboardInterrupt):
        await _read("draft\x1b[99;5u", output=_vt100_output(raw))
    assert raw.getvalue().count(key_protocol.DISABLE) == 1


async def test_keys_sent_as_csi_u_still_work() -> None:
    calls: list[str] = []
    kb = KeyBindings()

    @kb.add(Keys.BackTab)
    def _cycle(event: KeyPressEvent) -> None:
        calls.append("cycle")

    # Shift+Tab・Option+¥（日本語配列の \）・Ctrl+B（1 字戻る）が CSI u で届く。
    keys = "\x1b[9;2ua\x1b[92;3ub\x1b[98;5u!\r"
    assert await _read(keys, key_bindings=kb) == "a\\!b"
    assert calls == ["cycle"]


async def test_ctrl_d_sent_as_csi_u_ends_on_empty_input() -> None:
    with pytest.raises(EOFError):
        await _read("\x1b[100;5u")


async def test_ctrl_c_interrupts() -> None:
    with pytest.raises(KeyboardInterrupt):
        await _read("draft\x03")


async def test_ctrl_c_clears_the_draft() -> None:
    with create_pipe_input() as inp:
        box = _box(inp)
        inp.send_text("draft\x03")
        with pytest.raises(KeyboardInterrupt):
            await box.prompt_async()
        inp.send_text("\r")
        assert await asyncio.wait_for(box.prompt_async(), 5) == ""


async def test_ctrl_c_while_searching_leaves_the_next_box_editable() -> None:
    with create_pipe_input() as inp:
        box = _box(inp)
        inp.send_text("first request\r")
        assert await box.prompt_async() == "first request"
        inp.send_text("\x12fir\x03")
        with pytest.raises(KeyboardInterrupt):
            await box.prompt_async()
        # 検索の欄に残ると Enter が送信にならない
        inp.send_text("next\r")
        assert await asyncio.wait_for(box.prompt_async(), 5) == "next"
        inp.send_text("\x12first\r\r")
        assert await asyncio.wait_for(box.prompt_async(), 5) == "first request"


async def test_ctrl_d_on_empty_input_ends() -> None:
    with pytest.raises(EOFError):
        await _read("\x04")


async def test_ctrl_d_with_text_deletes_a_character() -> None:
    # Ctrl+B で 1 字戻り、Ctrl+D でカーソルの字（b）を消す。
    assert await _read("ab\x02\x04\r") == "a"


async def test_shift_tab_reaches_the_callers_binding() -> None:
    calls: list[str] = []
    kb = KeyBindings()

    @kb.add(Keys.BackTab)
    def _cycle(event: KeyPressEvent) -> None:
        calls.append("cycle")

    assert await _read("\x1b[Zgo\r", key_bindings=kb) == "go"
    assert calls == ["cycle"]


async def test_up_arrow_recalls_the_previous_input() -> None:
    with create_pipe_input() as inp:
        box = _box(inp)
        inp.send_text("first request\r")
        assert await box.prompt_async() == "first request"
        inp.send_text("\x1b[A\r")
        assert await box.prompt_async() == "first request"


async def test_ctrl_r_searches_the_history_and_enter_accepts_the_match() -> None:
    with create_pipe_input() as inp:
        box = _box(inp)
        for line in ("first request", "second"):
            inp.send_text(line + "\r")
            assert await box.prompt_async() == line
        # Ctrl+R で「first」を探し、Enter で検索を確定（送信しない）→ もう一度 Enter で送信。
        inp.send_text("\x12first\r\r")
        assert await box.prompt_async() == "first request"


async def test_submitted_text_is_echoed_with_a_marker() -> None:
    out = io.StringIO()
    console = Console(file=out, width=40, color_system=None)
    assert await _read("one\x1b[13;2utwo\r", console=console) == "one\ntwo"
    lines = out.getvalue().rstrip("\n").split("\n")
    assert [ln.rstrip() for ln in lines] == ["> one", "  two"]


async def test_long_paste_echo_keeps_the_head() -> None:
    out = io.StringIO()
    console = Console(file=out, width=40, color_system=None)
    body = "\n".join(f"row {i}" for i in range(1, 16))
    assert await _read(f"\x1b[200~{body}\x1b[201~\r", console=console) == body
    text = out.getvalue()
    assert "row 12" in text and "row 13" not in text
    assert "… ほか 3 行" in text


async def test_empty_input_is_not_echoed() -> None:
    out = io.StringIO()
    assert await _read("  \r", console=Console(file=out)) == "  "
    assert out.getvalue() == ""


async def test_layout_has_rules_and_hint_but_no_placeholder() -> None:
    # 空の欄に案内文を出すと、日本語入力の変換中（アプリに届かない文字）に重なって残る。
    raw = io.StringIO()
    assert await _read("\r", output=_vt100_output(raw)) == ""
    screen = _ANSI.sub("", raw.getvalue())
    assert "─" * 20 in screen
    assert "依頼を入力" not in screen
    assert "code モード" in screen and "mock-model" in screen
    assert "shift+tab" not in screen  # 切り替えの案内はヒント行に出さない


async def test_non_terminal_input_falls_back_to_a_single_line_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    class _Session:
        def __init__(self, **kw: Any) -> None:
            seen.append(kw)

        async def prompt_async(self, message: Any) -> str:
            seen.append(message())
            return "piped"

    monkeypatch.setattr(input_box_mod.sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(input_box_mod, "PromptSession", _Session)
    box = InputBox(
        console=Console(file=io.StringIO()),
        hint=lambda: [],
        status=lambda: "",
        fallback_prompt=lambda: "[code] satsuma> ",
    )
    assert await box.prompt_async() == "piped"
    assert seen[-1] == "[code] satsuma> "


async def test_non_terminal_input_uses_the_given_one_line_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[Any] = []

    class _Session:
        app = object()

        async def prompt_async(self, message: Any) -> str:
            return str(message())

    def make() -> Any:
        made.append(_Session())
        return made[-1]

    monkeypatch.setattr(input_box_mod.sys.stdin, "isatty", lambda: False, raising=False)
    box = InputBox(
        console=Console(file=io.StringIO()),
        hint=lambda: [],
        status=lambda: "",
        fallback_prompt=lambda: "[code] satsuma> ",
        fallback=make,
    )
    assert box.app is made[0].app
    assert await box.prompt_async() == "[code] satsuma> "
    assert await box.prompt_async() == "[code] satsuma> "
    assert len(made) == 1


def test_mode_hint_fragments() -> None:
    assert mode_hint_fragments(ModeState(main="plan")) == [("fg:ansigreen bold", "plan")]
    assert mode_hint_fragments(ModeState()) == [("fg:ansiblue bold", "code")]
    ro = ModeState(main="code", sub=PermissionMode.READ_ONLY)
    assert mode_hint_fragments(ro) == [("fg:ansiblue bold", "code"), ("class:hint", ":read-only")]


def test_multi_line_input_and_slash_commands() -> None:
    assert parse_slash("一行目\n二行目") is None
    assert parse_slash("/Users/alice/x を消して\n理由: 古い") is None
    cmd = parse_slash("/help\nmore")
    assert cmd is not None and cmd.name == "help"


# ── 幅が変わったとき ──

_SYNC = re.compile(r"\x1b\[\?2026[hl]")
_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="SIGWINCH が無い")


class _SizedOutput(DummyOutput):
    """幅を変えられ、生の書き出し（同期出力の印）を覚える出力。"""

    def __init__(self, columns: int, rows: int = 40) -> None:
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.raw: list[str] = []

    def get_size(self) -> Size:
        return Size(rows=self.rows, columns=self.columns)

    def write_raw(self, data: str) -> None:
        self.raw.append(data)


def _scrollback(width: int, lines: int = 1) -> tuple[ScrollbackConsole, io.StringIO]:
    out = io.StringIO()
    console = ScrollbackConsole(file=out, width=width, force_terminal=True, color_system=None)
    console.keep_scrollback()
    console.print(Rule())
    for n in range(lines):
        console.print(f"line {n}")
    return console, out


async def _resize(console: Console, output: _SizedOutput, width: int) -> None:
    """端末の幅を変えたときと同じく、SIGWINCH で入力欄に知らせる。"""
    output.columns = console.width = width
    signal.raise_signal(signal.SIGWINCH)
    await asyncio.sleep(0.02)


@_posix_only
async def test_a_width_change_while_typing_redraws_the_screen_at_once() -> None:
    console, out = _scrollback(80)
    output = _SizedOutput(80)
    with create_pipe_input() as inp:
        box = _box(inp, console=console, output=output)
        task = asyncio.create_task(box.prompt_async())
        await asyncio.sleep(0.05)
        await _resize(console, output, 50)
        # 待たずに、画面を消して出力と入力欄を新しい幅で 1 枚に描く
        redrawn = out.getvalue().split("\x1b[3J")
        assert len(redrawn) == 2
        assert redrawn[1].startswith("\x1b[H" + "─" * 50 + "\n")
        assert _SYNC.findall("".join(output.raw)) == ["\x1b[?2026h", "\x1b[?2026l"]
        await asyncio.sleep(input_box_mod._RESIZE_SETTLE_S + 0.1)
        assert out.getvalue().count("\x1b[3J") == 1  # 全部が画面に入ったので出し直さない
        inp.send_text("go\r")
        assert await task == "go"


@_posix_only
async def test_a_long_history_shows_its_tail_first_and_the_rest_after_the_width_settles() -> None:
    console, out = _scrollback(80, lines=30)
    output = _SizedOutput(80, rows=10)
    with create_pipe_input() as inp:
        box = _box(inp, console=console, output=output)
        task = asyncio.create_task(box.prompt_async())
        await asyncio.sleep(0.05)
        await _resize(console, output, 60)
        await _resize(console, output, 50)
        tails = out.getvalue().split("\x1b[3J")[1:]
        assert len(tails) == 2 and all("line 0\n" not in t for t in tails)  # まず見えている分
        assert "line 29\n" in tails[-1]
        await asyncio.sleep(input_box_mod._RESIZE_SETTLE_S + 0.2)
        redrawn = out.getvalue().split("\x1b[3J")
        assert len(redrawn) == 4  # 変わり終わってから 1 回だけ全部
        assert redrawn[-1].startswith("\x1b[H" + "─" * 50 + "\nline 0\n")
        inp.send_text("go\r")
        assert await task == "go"


async def test_a_width_change_during_a_turn_is_redrawn_before_the_next_input() -> None:
    console, out = _scrollback(80)
    console.width = 50
    with create_pipe_input() as inp:
        box = _box(inp, console=console, output=_SizedOutput(50))
        inp.send_text("go\r")
        assert await box.prompt_async() == "go"
    assert out.getvalue().count("\x1b[3J") == 1
    assert not console.needs_redraw()


@_posix_only
async def test_a_height_change_only_redraws_the_input_box() -> None:
    console, out = _scrollback(80)
    output = _SizedOutput(80)
    with create_pipe_input() as inp:
        box = _box(inp, console=console, output=output)
        task = asyncio.create_task(box.prompt_async())
        await asyncio.sleep(0.05)
        output.rows = 20
        signal.raise_signal(signal.SIGWINCH)
        await asyncio.sleep(0.05)
        inp.send_text("go\r")
        assert await task == "go"
    assert "\x1b[3J" not in out.getvalue()
