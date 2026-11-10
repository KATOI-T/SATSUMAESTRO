"""入力欄の下に出す候補の一覧（選んでいる行に色をつけ、Tab で埋め、Enter で実行する）。"""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition, Filter, has_focus, is_done
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, Window
from prompt_toolkit.layout.controls import GetLinePrefixCallable, UIContent, UIControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.utils import get_cwidth

MENU_ROWS = 8
_INDENT = 2
_GAP = 2
_CURRENT = "fg:ansicyan bold"
_CURRENT_META = "fg:ansicyan"
_META = "fg:ansibrightblack"


def _fit(text: str, width: int) -> str:
    """表示の幅が ``width`` を超えたら、末尾を ``…`` にして詰める。"""
    if get_cwidth(text) <= width:
        return text
    kept, used = [], 0
    for char in text:
        used += get_cwidth(char)
        if used > width - 1:
            break
        kept.append(char)
    return "".join(kept) + "…"


class SuggestionMenu:
    """候補と、選んでいる行（入力欄の文字は、埋めるまで変えない）。"""

    def __init__(self, buffer: Buffer, *, rows: int = MENU_ROWS) -> None:
        self.buffer = buffer
        self.rows = rows
        self._state: CompletionState | None = None
        self._index: int | None = None
        self._top = 0
        self._last_text = buffer.text
        buffer.on_text_changed += self._reopen_after_delete

    def completions(self) -> list[Completion]:
        state = self.buffer.complete_state
        if state is None:
            return []
        if state is not self._state:
            typed = state.original_document.text_before_cursor
            self._state, self._top = state, 0
            self._index = None if typed[-1:].isspace() else 0
        return state.completions

    def is_open(self) -> bool:
        return bool(self.completions())

    def chosen(self) -> Completion | None:
        """選んでいる候補（引数をまだ打っていない間は、どれも選ばない）。"""
        completions = self.completions()
        if not completions or self._index is None:
            return None
        return completions[min(self._index, len(completions) - 1)]

    def has_choice(self) -> bool:
        return self.chosen() is not None

    def move(self, step: int) -> None:
        completions = self.completions()
        if not completions:
            return
        if self._index is None:
            self._index = 0 if step > 0 else len(completions) - 1
        else:
            self._index = (self._index + step) % len(completions)

    def fill(self, *, then_suggest: bool) -> None:
        """選んでいる候補（無ければ先頭）で埋める（``then_suggest`` なら続きの候補を出す）。"""
        completions = self.completions()
        completion = self.chosen() or (completions[0] if completions else None)
        if completion is None:
            return
        buffer = self.buffer
        cursor = buffer.cursor_position
        start = cursor + completion.start_position
        text = buffer.text[:start] + completion.text + buffer.text[cursor:]
        buffer.complete_state = None
        buffer.document = Document(text, start + len(completion.text))
        if then_suggest:
            buffer.start_completion()

    def close(self) -> None:
        self.buffer.complete_state = None

    def lines(self, width: int, height: int | None = None) -> list[StyleAndTextTuples]:
        """見えている行（選んでいる行が見えるように送る）。"""
        completions = self.completions()
        shown = min(self.rows if height is None else height, self.rows, len(completions))
        if shown <= 0:
            return []
        index = None if self._index is None else min(self._index, len(completions) - 1)
        if index is not None and index < self._top:
            self._top = index
        elif index is not None and index >= self._top + shown:
            self._top = index - shown + 1
        self._top = max(0, min(self._top, len(completions) - shown))
        name_width = min(
            max(get_cwidth(c.display_text) for c in completions),
            max(8, (width - _INDENT) // 2),
        )
        room = width - _INDENT - name_width - _GAP - 1
        rows: list[StyleAndTextTuples] = []
        for i in range(self._top, self._top + shown):
            completion = completions[i]
            current = i == index
            name = _fit(completion.display_text, name_width)
            row: StyleAndTextTuples = [("", " " * _INDENT), (_CURRENT if current else "", name)]
            meta = completion.display_meta_text
            if meta and room > 1:
                pad = " " * (name_width - get_cwidth(name) + _GAP)
                row += [("", pad), (_CURRENT_META if current else _META, _fit(meta, room))]
            rows.append(row)
        return rows

    def _reopen_after_delete(self, buffer: Buffer) -> None:
        text, last = buffer.text, self._last_text
        self._last_text = text
        deleted = text != last and last.startswith(text)
        if deleted and text.startswith("/") and buffer.complete_state is None:
            buffer.start_completion()


class _MenuControl(UIControl):
    def __init__(self, menu: SuggestionMenu) -> None:
        self._menu = menu

    def is_focusable(self) -> bool:
        return False

    def preferred_width(self, max_available_width: int) -> int | None:
        return max_available_width

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: GetLinePrefixCallable | None,
    ) -> int | None:
        return min(len(self._menu.completions()), self._menu.rows)

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self._menu.lines(width, height)
        return UIContent(get_line=lambda i: lines[i], line_count=len(lines))


def menu_bindings(menu: SuggestionMenu, shown: Filter) -> KeyBindings:
    """一覧が出ている間のキー（上下で選び、Tab で埋め、Enter で選んだ行を実行し、Esc で閉じる）。"""
    bindings = KeyBindings()

    @bindings.add("down", filter=shown)
    @bindings.add("c-n", filter=shown)
    def _next(event: KeyPressEvent) -> None:
        menu.move(1)

    @bindings.add("up", filter=shown)
    @bindings.add("c-p", filter=shown)
    def _previous(event: KeyPressEvent) -> None:
        menu.move(-1)

    @bindings.add("tab", filter=shown)
    def _fill(event: KeyPressEvent) -> None:
        menu.fill(then_suggest=True)

    @bindings.add("enter", filter=Condition(menu.has_choice) & shown)
    def _run(event: KeyPressEvent) -> None:
        menu.fill(then_suggest=False)
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", filter=shown, eager=True)
    def _close(event: KeyPressEvent) -> None:
        menu.close()

    return bindings


def install_menu(session: PromptSession[str]) -> SuggestionMenu | None:
    """標準の候補の箱を入力欄の下の一覧に差し替える（差し替えられなければ ``None``）。"""
    if not isinstance(session, PromptSession):
        return None
    buffer = session.default_buffer
    for container in session.layout.walk():
        if not isinstance(container, FloatContainer):
            continue
        for i, fl in enumerate(container.floats):
            if not isinstance(fl.content, CompletionsMenu):
                continue
            menu = SuggestionMenu(buffer)
            shown = Condition(menu.is_open) & has_focus(buffer) & ~is_done
            window = Window(_MenuControl(menu), dont_extend_height=True)
            container.floats[i] = Float(
                content=ConditionalContainer(window, filter=shown),
                left=0,
                right=0,
                ycursor=True,
                transparent=True,
            )
            own = session.key_bindings or KeyBindings()
            session.key_bindings = merge_key_bindings([own, menu_bindings(menu, shown)])
            return menu
    return None
