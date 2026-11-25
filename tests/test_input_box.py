"""REPL の入力欄（複数行・罫線・ヒント行・送信後の残し方）。"""

from __future__ import annotations

import io
import re
from typing import Any

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.output import DummyOutput, Output
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.console import Console

from satsumaestro.cli import input_box as input_box_mod
from satsumaestro.cli import key_protocol
from satsumaestro.cli.commands import parse_slash
from satsumaestro.cli.input_box import InputBox
from satsumaestro.cli.modes import mode_hint_fragments
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
