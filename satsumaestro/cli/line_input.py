"""問いを端末の行のモードで 1 行読む。"""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager, nullcontext

from rich.console import Console


def ask_line(console: Console, prompt: str) -> str:
    """問いの答えを 1 行読む（端末が raw のまま残っていても Enter で終わる）。"""
    with _line_mode():
        return console.input(prompt, markup=False)


def _line_mode() -> AbstractContextManager[None]:
    """標準入力が端末なら、その間だけ ICRNL・ICANON・ECHO・ISIG・IEXTEN を立て、抜けると戻す。"""
    if sys.platform == "win32" or not _stdin_is_terminal():
        return nullcontext()
    from prompt_toolkit.input.vt100 import cooked_mode

    return cooked_mode(sys.stdin.fileno())


def _stdin_is_terminal() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


__all__ = ["ask_line"]
