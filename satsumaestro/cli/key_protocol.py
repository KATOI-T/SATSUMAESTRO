"""Shift+Enter を Enter と区別して受け取る（kitty keyboard protocol）。"""

from __future__ import annotations

import re

from prompt_toolkit.input import Input
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.keys import Keys

# 旗 1（曖昧なキーを区別する）を積む／1 段おろす
ENABLE = "\x1b[>1u"
DISABLE = "\x1b[<u"
# prompt_toolkit に Shift+Enter のキーが無いので、Esc+Enter と分けるため F24 の列で渡す
NEWLINE = "\x1b[24;2~"
NEWLINE_KEY = Keys.F24

# 問い合わせへの返事・CSI 符号[:別キー][;修飾[:種別][;文字]] u・
# modifyOtherKeys（CSI 27;修飾;符号 ~）
_SEQUENCE = re.compile(
    r"""
    \x1b\[(?:
        \?\d*u
      | (\d+)(?::\d*)*(?:;(\d*)(?::\d+)?(?:;[\d:]*)?)?u
      | 27;(\d+);(\d+)~
    )
    """,
    re.VERBOSE,
)
# 終端の字がまだ届いていない列は、次の読み込みと合わせて読む
_INCOMPLETE = re.compile(r"\x1b(?:\[[\d;:?]*)?\Z")

_SHIFT, _ALT, _CTRL = 1, 2, 4
_LOCKS = 64 | 128  # Caps Lock・Num Lock
_ENTER, _TAB, _ESCAPE, _BACKSPACE = 13, 9, 27, 127
_KEYPAD_ENTER = 57414
_PRIVATE_USE = 57344  # F13 以降・テンキー・メディアキーなど
# Num Lock を切ったテンキーの移動のキーと、ふつうのキーの列の終わり
_KEYPAD = {
    57417: "D", 57418: "C", 57419: "A", 57420: "B", 57421: "5~", 57422: "6~",
    57423: "H", 57424: "F", 57425: "2~", 57426: "3~",
}  # fmt: skip


def to_legacy(text: str) -> str:
    """``CSI … u`` と modifyOtherKeys の列を、従来の端末が送る列に置き換える。"""
    return _SEQUENCE.sub(_replace, text)


def install(inp: Input) -> bool:
    """``inp`` の解析を :class:`LegacyKeyParser` に替え、替えた（替えてある）なら True を返す。"""
    for holder, name in _parser_slots(inp):
        parser = getattr(holder, name, None)
        if isinstance(parser, Vt100Parser):
            if not isinstance(parser, LegacyKeyParser):
                setattr(holder, name, LegacyKeyParser(parser.feed_key_callback))
            return True
    return False


def _parser_slots(inp: Input) -> tuple[tuple[object, str], ...]:
    # POSIX は入力そのもの、Windows は VT の読み手が解析器を持つ
    return ((inp, "vt100_parser"), (getattr(inp, "console_input_reader", None), "_vt100_parser"))


class LegacyKeyParser(Vt100Parser):
    """届いた文字列を :func:`to_legacy` に通してから解析する。"""

    _held = ""
    _passing = False

    def feed(self, data: str) -> None:
        if self._passing:  # 貼り付けの終わりより後ろを、親が渡し直すとき（変換済み）
            super().feed(data)
            return
        text = self._held + data
        match = _INCOMPLETE.search(text)
        cut = match.start() if match else len(text)
        self._held = text[cut:]
        self._pass(to_legacy(text[:cut]))

    def flush(self) -> None:
        held, self._held = self._held, ""
        self._pass(held)
        super().flush()

    def _pass(self, data: str) -> None:
        self._passing = True
        try:
            super().feed(data)
        finally:
            self._passing = False


def _replace(match: re.Match[str]) -> str:
    if match.group(1) is not None:
        code, mods = match.group(1), match.group(2)
    elif match.group(4) is not None:
        code, mods = match.group(4), match.group(3)
    else:  # 問い合わせへの返事
        return ""
    return _legacy_key(int(code), max(int(mods or 1) - 1, 0) & ~_LOCKS)


def _legacy_key(code: int, mods: int) -> str:
    shift, alt, ctrl = mods & _SHIFT, mods & _ALT, mods & _CTRL
    if code in (_ENTER, _KEYPAD_ENTER):
        # Shift / Alt 付きは入力欄の改行。Ctrl+Enter は送信
        return NEWLINE if (shift or alt) and not ctrl else "\r"
    if code == _TAB:
        return "\x1b[Z" if shift else "\t"
    if code == _ESCAPE:
        return "\x1b"
    if code == _BACKSPACE:
        return "\x7f"
    if code in _KEYPAD and not mods & ~(_SHIFT | _ALT | _CTRL):
        return _keypad_key(_KEYPAD[code], mods)
    if code >= _PRIVATE_USE or mods & ~(_SHIFT | _ALT | _CTRL):
        return ""  # 入力欄で使わないキー（Cmd 付きなど）
    char = chr(code)
    if ctrl:
        control = _control_char(code)
        return ("\x1b" + control if alt else control) if control else ""
    if alt:
        # 英数字は Meta（Esc+字）、ほかは Option で組んだ字（日本語配列の \ など）
        if char.isascii() and char.isalnum():
            return "\x1b" + (char.upper() if shift else char)
        return char
    return char.upper() if shift else char


def _keypad_key(end: str, mods: int) -> str:
    """テンキーの矢印・Home・Delete などを、ふつうのキーの列にする（読めない組は捨てる）。"""
    if not mods:
        return f"\x1b[{end}"
    if end.endswith("~"):
        sequence = f"\x1b[{end[:-1]};{mods + 1}~"
    else:
        sequence = f"\x1b[1;{mods + 1}{end}"
    return sequence if sequence in ANSI_SEQUENCES else ""


def _control_char(code: int) -> str:
    if code == ord(" "):
        return "\x00"
    if code == ord("/"):
        return "\x1f"
    if ord("@") <= code <= ord("~"):
        return chr(code & 0x1F)
    return ""
