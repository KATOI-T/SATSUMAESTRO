"""kitty keyboard protocol の列を、従来の端末が送る列に戻す。"""

from __future__ import annotations

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyPress
from prompt_toolkit.keys import Keys

from satsumaestro.cli.key_protocol import NEWLINE, NEWLINE_KEY, LegacyKeyParser, install, to_legacy


@pytest.mark.parametrize(
    ("sequence", "legacy"),
    [
        pytest.param("\x1b[13;2u", NEWLINE, id="shift-enter"),
        pytest.param("\x1b[27;2;13~", NEWLINE, id="shift-enter-modify-other-keys"),
        pytest.param("\x1b[13;2:1u", NEWLINE, id="shift-enter-with-event-type"),
        pytest.param("\x1b[13;3u", NEWLINE, id="alt-enter"),
        pytest.param("\x1b[13;5u", "\r", id="ctrl-enter"),
        pytest.param("\x1b[57414;2u", NEWLINE, id="shift-keypad-enter"),
        pytest.param("\x1b[99;5u", "\x03", id="ctrl-c"),
        pytest.param("\x1b[99;69u", "\x03", id="ctrl-c-with-caps-lock"),
        pytest.param("\x1b[100;5u", "\x04", id="ctrl-d"),
        pytest.param("\x1b[32;5u", "\x00", id="ctrl-space"),
        pytest.param("\x1b[98;7u", "\x1b\x02", id="ctrl-alt-b"),
        pytest.param("\x1b[27u", "\x1b", id="escape"),
        pytest.param("\x1b[9;2u", "\x1b[Z", id="shift-tab"),
        pytest.param("\x1b[127;2u", "\x7f", id="shift-backspace"),
        pytest.param("\x1b[98;3u", "\x1bb", id="alt-b"),
        pytest.param("\x1b[98;4u", "\x1bB", id="alt-shift-b"),
        pytest.param("\x1b[92;3u", "\\", id="option-yen-on-jis"),
        pytest.param("\x1b[8747;3u", "∫", id="option-composed"),
        pytest.param("\x1b[57376u", "", id="f13"),
        pytest.param("\x1b[99;9u", "", id="cmd-c"),
        pytest.param("\x1b[?1u", "", id="query-reply"),
    ],
)
def test_sequences_become_legacy_keys(sequence: str, legacy: str) -> None:
    assert to_legacy(f"a{sequence}b") == f"a{legacy}b"


@pytest.mark.parametrize(
    "text",
    ["\x1b[A", "\x1b[1;5A", "\x1b[3~", "\x1bOP", "\x1b[Z", "\x1b[200~x\x1b[201~", "日本語"],
)
def test_legacy_sequences_are_left_alone(text: str) -> None:
    assert to_legacy(text) == text


def _parse(*chunks: str, flush: bool = False) -> list[KeyPress]:
    keys: list[KeyPress] = []
    parser = LegacyKeyParser(keys.append)
    for chunk in chunks:
        parser.feed(chunk)
    if flush:
        parser.flush()
    return keys


def test_sequence_split_across_reads_is_joined() -> None:
    keys = _parse("a\x1b[13;", "2ub")
    assert [k.key for k in keys] == ["a", NEWLINE_KEY, "b"]


def test_lone_escape_is_released_on_flush() -> None:
    assert _parse("\x1b") == []
    assert [k.key for k in _parse("\x1b", flush=True)] == [Keys.Escape]


def test_keys_after_a_paste_are_converted_once() -> None:
    keys = _parse("\x1b[200~x\\y\x1b[201~\x1b[99;5u")
    assert [(k.key, k.data) for k in keys] == [
        (Keys.BracketedPaste, "x\\y"),
        (Keys.ControlC, "\x03"),
    ]


def test_install_replaces_the_parser_once() -> None:
    with create_pipe_input() as inp:
        install(inp)
        parser = inp.vt100_parser
        install(inp)
        assert isinstance(parser, LegacyKeyParser) and inp.vt100_parser is parser
