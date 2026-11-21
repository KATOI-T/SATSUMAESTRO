"""問いは端末の行のモードで読む（raw のまま残った端末でも Enter で答えられる）。"""

from __future__ import annotations

import io
import os
import sys
import termios
from collections.abc import Iterator
from typing import Any

import pytest

from satsumaestro.cli.line_input import ask_line

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="termios が無い")

_IFLAG, _LFLAG = 0, 3


@pytest.fixture
def left_raw(monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    """標準入力を、issue のモード（``-icrnl -ixon -isig -iexten``）の pty にする。"""
    master, slave = os.openpty()
    attrs = termios.tcgetattr(slave)
    attrs[_IFLAG] &= ~(termios.ICRNL | termios.IXON)
    attrs[_LFLAG] &= ~(termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    stdin = open(slave, encoding="utf-8", closefd=False)  # noqa: SIM115
    monkeypatch.setattr(sys, "stdin", stdin)
    try:
        yield slave
    finally:
        stdin.close()
        os.close(slave)
        os.close(master)


class _Console:
    """``input`` の間の端末のモードを控える。"""

    def __init__(self, fd: int | None = None) -> None:
        self.fd = fd
        self.seen: list[Any] = []
        self.prompts: list[str] = []

    def input(self, prompt: str = "", *, markup: bool = True) -> str:
        self.prompts.append(prompt)
        if self.fd is not None:
            self.seen = termios.tcgetattr(self.fd)
        return "y"


def test_a_question_reads_in_line_mode_and_puts_the_mode_back(left_raw: int) -> None:
    before = termios.tcgetattr(left_raw)
    console = _Console(left_raw)
    assert ask_line(console, "Approve? ") == "y"  # type: ignore[arg-type]
    assert console.prompts == ["Approve? "]
    iflag, lflag = console.seen[_IFLAG], console.seen[_LFLAG]
    assert iflag & termios.ICRNL
    for flag in (termios.ICANON, termios.ECHO, termios.ISIG, termios.IEXTEN):
        assert lflag & flag
    assert termios.tcgetattr(left_raw) == before


def test_without_a_terminal_the_question_is_read_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    console = _Console()
    assert ask_line(console, "Approve? ") == "y"  # type: ignore[arg-type]
    assert console.prompts == ["Approve? "]
