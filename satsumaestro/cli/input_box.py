"""REPL の入力欄: 上下の罫線で挟んだ複数行の入力と、その下のヒント行。"""

from __future__ import annotations

import sys
from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input, create_input
from prompt_toolkit.key_binding import (
    KeyBindings,
    KeyBindingsBase,
    KeyPressEvent,
    merge_key_bindings,
)
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import Output
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import is_dumb_terminal
from prompt_toolkit.widgets import SearchToolbar
from rich.console import Console
from rich.table import Table
from rich.text import Text

from satsumaestro.cli import key_protocol
from satsumaestro.i18n.text import ui

# 入力欄の高さの上限（端末の高さに対する割合と最低の行数）。超えた分は欄の中でスクロールする
_MAX_HEIGHT_RATIO = 0.5
_MIN_MAX_HEIGHT = 3
# 送信後に残す本文の行数。長い貼り付けは先頭だけ残し、残りの行数を添える
_ECHO_MAX_LINES = 12

_STYLE = Style.from_dict(
    {
        "rule": "fg:ansibrightblack",
        "hint": "fg:ansibrightblack",
        "marker": "bold",
    }
)


class InputBox:
    """REPL の入力欄（端末でない入力とダム端末では 1 行入力の ``PromptSession`` に戻す）。"""

    def __init__(
        self,
        *,
        console: Console,
        hint: Callable[[], StyleAndTextTuples],
        status: Callable[[], str],
        fallback_prompt: Callable[[], AnyFormattedText],
        key_bindings: KeyBindings | None = None,
        fallback: Callable[[], PromptSession[str]] | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self._console = console
        self._hint = hint
        self._status = status
        self._fallback_prompt = fallback_prompt
        self._extra_bindings = key_bindings
        self._make_fallback = fallback
        self._history = InMemoryHistory()
        self._fallback: PromptSession[str] | None = None
        self._buffer = Buffer(multiline=True, history=self._history)
        self._app: Application[str] | None = None
        # 解析器を替えられない入力では、端末に区別して送るよう頼まない（届いた列を読めない）
        self._distinguish_keys = False
        if input is not None or (sys.stdin.isatty() and not is_dumb_terminal()):
            inp = input if input is not None else create_input()
            self._distinguish_keys = key_protocol.install(inp)
            self._app = self._build_app(inp, output)

    @property
    def app(self) -> Application[str]:
        """入力を受ける Application（入力欄か、戻した 1 行入力の物）。"""
        if self._app is not None:
            return self._app
        return self._one_line().app

    async def prompt_async(self) -> str:
        """1 回分の入力を受け取る（Ctrl+C と空での Ctrl+D は例外にする）。"""
        if self._app is None:
            return await self._one_line().prompt_async(self._fallback_prompt)
        self._buffer.reset()
        # 区別して送らせるのは入力欄の間だけ。ターン中も続けると Ctrl+C が割り込みにならない
        output = self._app.output
        if self._distinguish_keys:
            output.write_raw(key_protocol.ENABLE)
            output.flush()
        try:
            text = await self._app.run_async()
        finally:
            if self._distinguish_keys:
                output.write_raw(key_protocol.DISABLE)
                output.flush()
        self._echo(text)
        return text

    def _one_line(self) -> PromptSession[str]:
        if self._fallback is None:
            if self._make_fallback is not None:
                self._fallback = self._make_fallback()
            else:
                self._fallback = PromptSession(
                    key_bindings=self._extra_bindings, history=self._history
                )
        return self._fallback

    def _build_app(self, input: Input, output: Output | None) -> Application[str]:
        buffer = self._buffer
        search = SearchToolbar()
        body = Window(
            # 空の欄に案内文を出さない: 日本語入力の変換中の文字はアプリに届かず、案内文が残る
            BufferControl(buffer, search_buffer_control=search.control),
            height=self._body_height,
            dont_extend_height=True,
            wrap_lines=True,
            get_line_prefix=_line_prefix,
        )
        hint_row = VSplit(
            [
                Window(FormattedTextControl(self._hint_fragments), height=1),
                Window(
                    FormattedTextControl(lambda: [("class:hint", f"{self._status()}  ")]),
                    height=1,
                    dont_extend_width=True,
                ),
            ]
        )
        layout = Layout(
            HSplit([_rule(), body, search, _rule(), hint_row]),
            focused_element=body,
        )
        return Application(
            layout=layout,
            key_bindings=self._bindings(),
            style=_STYLE,
            erase_when_done=True,
            input=input,
            output=output,
        )

    def _body_height(self) -> Dimension:
        rows = self._app.output.get_size().rows if self._app is not None else 0
        return Dimension(max=max(_MIN_MAX_HEIGHT, int(rows * _MAX_HEIGHT_RATIO)))

    def _hint_fragments(self) -> StyleAndTextTuples:
        mode = self._hint()
        return [("", "  "), *mode, ("class:hint", ui("cli.input_box.mode_hint"))]

    def _bindings(self) -> KeyBindingsBase:
        kb = KeyBindings()
        buffer = self._buffer
        # Ctrl+R の検索中（検索の行にフォーカス）は、Enter を検索の確定に残す
        editing = has_focus(buffer)

        @kb.add("enter", filter=editing)
        def _submit(event: KeyPressEvent) -> None:
            buffer.append_to_history()
            event.app.exit(result=buffer.text)

        # Shift+Enter は Esc+Enter に読み替えて届く。Ctrl+J（\n）は送信のまま: ターン中に
        # 押した Enter は端末が \n に変えて溜めるので、改行にすると先に打った依頼が送れない
        @kb.add("escape", "enter", filter=editing)
        @kb.add("escape", "c-j", filter=editing)
        def _newline(event: KeyPressEvent) -> None:
            buffer.newline(copy_margin=False)

        @kb.add("c-c")
        @kb.add("<sigint>")
        def _interrupt(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d", filter=editing & Condition(lambda: not buffer.text))
        def _end(event: KeyPressEvent) -> None:
            event.app.exit(exception=EOFError())

        if self._extra_bindings is None:
            return kb
        return merge_key_bindings([kb, self._extra_bindings])

    def _echo(self, text: str) -> None:
        """送った本文を ``> 本文`` の形で残す（折り返しも本文の位置に揃える）。"""
        body = text.strip()
        if not body:
            return
        lines = body.split("\n")
        hidden = len(lines) - _ECHO_MAX_LINES
        shown = Text("\n".join(lines[:_ECHO_MAX_LINES]))
        if hidden > 0:
            shown.append("\n" + ui("cli.input_box.more_lines", n=hidden), style="dim")
        grid = Table.grid()
        grid.add_column(width=2, no_wrap=True)
        grid.add_column()
        grid.add_row(Text(">", style="bold"), shown)
        self._console.print(grid)


def _rule() -> Window:
    return Window(char="─", height=1, style="class:rule")


def _line_prefix(line: int, wrap: int) -> StyleAndTextTuples:
    """1 行目の先頭だけ ``> ``、続きの行と折り返しは同じ幅の空白。"""
    if line == 0 and wrap == 0:
        return [("class:marker", "> ")]
    return [("", "  ")]
