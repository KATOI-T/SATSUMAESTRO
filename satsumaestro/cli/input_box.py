"""REPL の入力欄: 上下の罫線で挟んだ複数行の入力と、その下のヒント行。"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.filters import Condition, Filter, has_focus, is_done
from prompt_toolkit.formatted_text import AnyFormattedText, StyleAndTextTuples
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input, create_input
from prompt_toolkit.key_binding import (
    KeyBindings,
    KeyBindingsBase,
    KeyPressEvent,
    merge_key_bindings,
)
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import AnyContainer
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
from satsumaestro.cli.completion_menu import SuggestionMenu, menu_bindings, menu_window
from satsumaestro.cli.scrollback import ScrollbackConsole
from satsumaestro.i18n.text import ui

# 入力欄の高さの上限（端末の高さに対する割合と最低の行数）。超えた分は欄の中でスクロールする
_MAX_HEIGHT_RATIO = 0.5
_MIN_MAX_HEIGHT = 3
# 送信後に残す本文の行数。長い貼り付けは先頭だけ残し、残りの行数を添える
_ECHO_MAX_LINES = 12
# 幅が変わり終わったとみなすまでの秒数。スクロールバックの全部はそれから出し直す
_RESIZE_SETTLE_S = 0.5
# 同期出力: 消してから描き終えるまでを 1 枚で見せる（応じない端末は無視する）
_SYNC_BEGIN = "\x1b[?2026h"
_SYNC_END = "\x1b[?2026l"

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
        completer: Completer | None = None,
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
        # 1 行で / から始まる間だけ、打ちながら候補を出す
        self._buffer = Buffer(
            multiline=True,
            history=self._history,
            completer=completer,
            complete_while_typing=Condition(self._is_slash_line),
        )
        self._app: _BoxApplication | None = None
        self._search: SearchToolbar | None = None
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
        self._leave_search()
        # ターン中に幅が変わっていたら、入力欄を出す前に描き直す
        console = self._console
        if isinstance(console, ScrollbackConsole) and console.needs_redraw():
            console.redraw()
        # 区別して送らせるのは入力欄の間だけ。ターン中も続けると Ctrl+C が割り込みにならない
        output = self._app.output
        self._app.columns = output.get_size().columns
        if self._distinguish_keys:
            output.write_raw(key_protocol.ENABLE)
            output.flush()
        try:
            text = await self._app.run_async()
        finally:
            self._app.finish()
            if self._distinguish_keys:
                output.write_raw(key_protocol.DISABLE)
                output.flush()
        self._echo(text)
        return text

    def _leave_search(self) -> None:
        """Ctrl+R の検索の途中で閉じた欄も、本文の欄から始め直す。"""
        if self._app is None or self._search is None:
            return
        self._app.layout.search_links.clear()
        self._search.search_buffer.reset()
        self._app.layout.focus(self._buffer)

    def _one_line(self) -> PromptSession[str]:
        if self._fallback is None:
            if self._make_fallback is not None:
                self._fallback = self._make_fallback()
            else:
                self._fallback = PromptSession(
                    key_bindings=self._extra_bindings, history=self._history
                )
        return self._fallback

    def _build_app(self, input: Input, output: Output | None) -> _BoxApplication:
        buffer = self._buffer
        search = self._search = SearchToolbar()
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
        # 候補の一覧は、開いている間だけヒント行の代わりに出す
        menu = SuggestionMenu(buffer)
        shown = Condition(menu.is_open) & has_focus(buffer) & ~is_done
        rows: list[AnyContainer] = [
            _rule(),
            body,
            search,
            _rule(),
            ConditionalContainer(hint_row, filter=~shown),
            ConditionalContainer(menu_window(menu), filter=shown),
        ]
        layout = Layout(HSplit(rows), focused_element=body)
        console = self._console
        return _BoxApplication(
            repaint=console.redraw if isinstance(console, ScrollbackConsole) else None,
            layout=layout,
            key_bindings=self._bindings(menu, shown),
            style=_STYLE,
            erase_when_done=True,
            input=input,
            output=output,
        )

    def _body_height(self) -> Dimension:
        rows = self._app.output.get_size().rows if self._app is not None else 0
        return Dimension(max=max(_MIN_MAX_HEIGHT, int(rows * _MAX_HEIGHT_RATIO)))

    def _is_slash_line(self) -> bool:
        text = self._buffer.text
        return text.startswith("/") and "\n" not in text

    def _hint_fragments(self) -> StyleAndTextTuples:
        mode = self._hint()
        return [("", "  "), *mode, ("class:hint", ui("cli.input_box.mode_hint"))]

    def _bindings(self, menu: SuggestionMenu, shown: Filter) -> KeyBindingsBase:
        kb = KeyBindings()
        buffer = self._buffer
        # Ctrl+R の検索中（検索の行にフォーカス）は、Enter を検索の確定に残す
        editing = has_focus(buffer)

        def submit(event: KeyPressEvent) -> None:
            buffer.append_to_history()
            event.app.exit(result=buffer.text)

        kb.add("enter", filter=editing)(submit)

        # Shift+Enter は区別して届けば NEWLINE_KEY、端末の設定によっては Esc+Enter。Ctrl+J（\n）は
        # 送信のまま: ターン中に押した Enter は端末が \n に変えて溜めるので、改行にすると送れない
        @kb.add(key_protocol.NEWLINE_KEY, filter=editing)
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

        # 一覧のキーを最後に重ね、出ている間は Enter・上下・Tab・Esc を一覧が受ける
        extra = [self._extra_bindings] if self._extra_bindings is not None else []
        return merge_key_bindings([kb, *extra, menu_bindings(menu, shown, submit=submit)])

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


class _BoxApplication(Application[str]):
    """幅が変わったら、入力欄を描く前にそれまでの出力を新しい幅で出し直す Application。"""

    def __init__(self, *, repaint: Callable[[int | None], bool] | None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # repaint(rows) は画面を消して末尾の rows 行分（None は全部）を出し、全部を出したかを返す
        self._repaint_above = repaint
        self.columns = 0
        self._settle: asyncio.TimerHandle | None = None
        self._synced = False
        self.after_render += self._end_sync

    def _on_resize(self) -> None:
        # 端末が出した行を折り返し直すので、入力欄の行数だけ戻って消すと罫線の切れ端が残る
        size = self.output.get_size()
        if self._repaint_above is None or size.columns == self.columns:
            super()._on_resize()
            return
        self.columns = size.columns
        if self._settle is not None:
            self._settle.cancel()
            self._settle = None
        if not self._repaint(size.rows):
            self._settle = asyncio.get_running_loop().call_later(
                _RESIZE_SETTLE_S, self._repaint_all
            )

    def finish(self) -> None:
        """入力欄を閉じた後に、待っている描き直しを捨てて同期出力を閉じる。"""
        if self._settle is not None:
            self._settle.cancel()
            self._settle = None
        self._end_sync(self)

    def _repaint_all(self) -> None:
        self._settle = None
        if self.is_running and not self.is_done:
            self._repaint(None)

    def _repaint(self, rows: int | None) -> bool:
        assert self._repaint_above is not None
        self.output.write_raw(_SYNC_BEGIN)
        self._synced = True
        # 文字の属性と自動折り返しを戻す（画面ごと消すので消す位置のずれは構わない）
        self.renderer.erase(leave_alternate_screen=False)
        complete = self._repaint_above(rows)
        self.renderer.request_absolute_cursor_position()
        self._redraw()
        return complete

    def _end_sync(self, _: object) -> None:
        if self._synced:
            self._synced = False
            self.output.write_raw(_SYNC_END)
            self.output.flush()


def _rule() -> Window:
    return Window(char="─", height=1, style="class:rule")


def _line_prefix(line: int, wrap: int) -> StyleAndTextTuples:
    """1 行目の先頭だけ ``> ``、続きの行と折り返しは同じ幅の空白。"""
    if line == 0 and wrap == 0:
        return [("class:marker", "> ")]
    return [("", "  ")]
