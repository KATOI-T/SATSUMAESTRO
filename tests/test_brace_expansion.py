"""守りの検査が、引用していない括弧をシェルと同じに展開してから語を読む。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from satsumaestro.security import deletion_guard, self_invocation_guard, state_write_guard
from satsumaestro.security.command_words import tokenize


def _words(command: str) -> list[str]:
    tokens = tokenize(command)
    assert tokens is not None
    return [t.text for t in tokens if t.kind == "word"]


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    (work / ".satsuma").mkdir(parents=True)
    (work / ".satsuma" / "policy.md").write_text("rules\n")
    (work / "src").mkdir()
    return work


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("{stro,} repl", ["stro", "repl"]),
        ("{rm,-rf,..}", ["rm", "-rf", ".."]),
        ("s{a,}tsumaestro", ["satsumaestro", "stsumaestro"]),
        ("{a,b}{c,d}", ["ac", "ad", "bc", "bd"]),
        ("{a,b{c,d}}", ["a", "bc", "bd"]),
        ("x{1..3}y", ["x1y", "x2y", "x3y"]),
        ("{3..1}", ["3", "2", "1"]),
        ("{x..z}", ["x", "y", "z"]),
        ("{,} a", ["a"]),
        ("{a}{b,c}", ["{a}b", "{a}c"]),
        ("{a{b,c}", ["{ab", "{ac"]),
        ("''{a,}", ["a", ""]),
        ("{a,'b,c'}", ["a", "b,c"]),
    ],
)
def test_unquoted_braces_are_expanded(command: str, expected: list[str]) -> None:
    assert _words(command) == expected


@pytest.mark.parametrize(
    "command",
    ["'{a,b}'", "\\{a,b}", '"{a,b}"', "{}", "{a}", "{a,b", "@{u}..HEAD", "${x:-{a,b}}"],
)
def test_braces_the_shell_leaves_alone_are_kept(command: str) -> None:
    assert len(_words(command)) == 1


def test_a_heredoc_delimiter_is_not_expanded() -> None:
    tokens = tokenize("cat <<{a,b}\nsatsumaestro repl\n{a,b}\necho done")
    assert tokens is not None
    assert [t.text for t in tokens if t.kind == "word"] == ["cat", "{a,b}", "echo", "done"]


def test_too_many_words_are_left_unexpanded_and_read_quickly() -> None:
    started = time.monotonic()
    assert _words("echo " + "{a,b}" * 30) == ["echo", "{a,b}" * 30]
    assert _words("echo " + "{" * 500 + "a,b" + "}" * 500)[1:] == [
        "{" * 499 + "a" + "}" * 499,
        "{" * 499 + "b" + "}" * 499,
    ]
    assert len(_words("echo " + "{a," * 300 + "b" + "}" * 300)) == 2
    assert time.monotonic() - started < 2


@pytest.mark.parametrize("command", ["{stro,} repl", "s{a,}tsumaestro repl", "{satsumaestro,repl}"])
def test_self_invocation_built_by_brace_expansion_is_found(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is not None


@pytest.mark.parametrize("command", ["{rm,-rf,..}", "rm -rf {src,..}", "rm -rf .{s,}atsuma"])
def test_deletions_built_by_brace_expansion_are_found(ws: Path, command: str) -> None:
    assert deletion_guard.check_bash(command, ws) is not None


def test_a_state_write_built_by_brace_expansion_is_found(ws: Path) -> None:
    assert state_write_guard.check_bash("cp x .satsuma/{policy.md,}", ws) is not None


@pytest.mark.parametrize("command", ["echo '{stro,} repl'", "find . -name '*.{py,md}'"])
def test_quoted_braces_are_not_taken_for_self_invocation(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is None
