"""守りの検査が、引用・改行・ヒアドキュメントをシェルと同じに読み分ける。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import satsumaestro.engine  # noqa: F401
from satsumaestro.security import deletion_guard, self_invocation_guard, state_write_guard
from satsumaestro.security.bash_pipeline import review_read_only
from satsumaestro.security.command_words import segments, substitutions, tokenize
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.remember_guard import remember_blocker


def _segments(command: str) -> list[list[str]]:
    tokens = tokenize(command)
    assert tokens is not None
    return segments(tokens)


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
        ("a b; c", [["a", "b"], ["c"]]),
        ("a\nb", [["a"], ["b"]]),
        ("a && b || c | d & e", [["a"], ["b"], ["c"], ["d"], ["e"]]),
        ("echo ';' x", [["echo", ";", "x"]]),
        ("echo \\; x", [["echo", ";", "x"]]),
        ("echo \"a;b\" '(c)'", [["echo", "a;b", "(c)"]]),
        (
            "find . \\( -name x \\) -exec rm {} \\;",
            [["find", ".", "(", "-name", "x", ")", "-exec", "rm", "{}", ";"]],
        ),
        ("cmd > out.txt 2>&1", [["cmd"]]),
        ("cmd 2> err.txt < in.txt", [["cmd"]]),
        ("cmd &> all.txt", [["cmd"]]),
        ("echo a\\\nb", [["echo", "ab"]]),
        ("echo hi # rm -rf ..", [["echo", "hi"]]),
        ("echo a#b", [["echo", "a#b"]]),
        ("echo $'a\\'b'", [["echo", "a'b"]]),
        ("FOO='x y' cmd", [["FOO=x y", "cmd"]]),
        ('echo "a\\"b"', [["echo", 'a"b']]),
    ],
)
def test_commands_are_split_by_the_shell_rules(command: str, expected: list[list[str]]) -> None:
    assert _segments(command) == expected


def test_an_unclosed_quote_is_unreadable() -> None:
    assert tokenize("echo 'a") is None
    assert tokenize('echo "a') is None


def test_a_heredoc_body_is_not_read_as_commands() -> None:
    command = "cat <<EOF > notes.md\nrm -rf ..\nsatsumaestro repl\nEOF\necho done"
    assert _segments(command) == [["cat"], ["echo", "done"]]


def test_tabs_before_the_heredoc_end_are_stripped_with_a_dash() -> None:
    command = "cat <<-EOF\n\trm -rf ..\n\tEOF\necho done"
    assert _segments(command) == [["cat"], ["echo", "done"]]


def test_substitutions_in_an_unquoted_heredoc_are_read() -> None:
    command = "cat <<EOF\n$(satsumaestro repl)\n`rm -rf ..`\nEOF"
    assert _segments(command) == [["cat"], ["satsumaestro", "repl"], ["rm", "-rf", ".."]]
    assert _segments("cat <<'EOF'\n$(satsumaestro repl)\nEOF") == [["cat"]]


def test_substitutions_skip_quoted_parentheses() -> None:
    assert substitutions("a $(b ')' c) `d` e") == ["b ')' c", "d"]


@pytest.mark.parametrize(
    "command",
    [
        "echo hi\nsatsumaestro repl",
        "true\n\nstro prompt x",
        "cat <(satsumaestro repl)",
        "cat <<EOF\n$(satsumaestro repl)\nEOF",
    ],
)
def test_self_invocation_is_found_on_later_lines_and_in_heredocs(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "echo ';' satsumaestro",
        "grep -r 'satsumaestro repl' .",
        "echo hi # satsumaestro repl",
        "cat <<EOF > notes.md\nsatsumaestro repl\nEOF",
        "cat <<'EOF'\n$(satsumaestro repl)\nEOF",
    ],
)
def test_quoted_and_heredoc_text_is_not_taken_for_self_invocation(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "echo hi\nrm -rf ..",
        "find . \\( -name '*.md' \\) -delete",
        "find . '(' -name '*.md' ')' -delete",
        "find / -exec true \\; -delete",
        "find . -name x -exec true \\; -exec rm -rf / \\;",
        "cat <<EOF\n$(rm -rf ..)\nEOF",
    ],
)
def test_deletions_are_found_on_later_lines_and_past_quoted_punctuation(
    ws: Path, command: str
) -> None:
    assert deletion_guard.check_bash(command, ws) is not None


@pytest.mark.parametrize(
    "command",
    [
        "echo 'rm -rf ..'",
        "cat <<EOF > notes.md\nrm -rf ..\nEOF",
        "printf '%s' 'a;rm -rf ..'",
        "find . -name '*.pyc' -exec rm {} \\;",
    ],
)
def test_quoted_and_heredoc_text_is_not_taken_for_deletions(ws: Path, command: str) -> None:
    assert deletion_guard.check_bash(command, ws) is None


@pytest.mark.parametrize(
    "command",
    [
        "echo hi\necho x > .satsuma/policy.md",
        "cat <<EOF\n$(echo x > .satsuma/policy.md)\nEOF",
    ],
)
def test_state_writes_are_found_on_later_lines_and_in_heredocs(ws: Path, command: str) -> None:
    assert state_write_guard.check_bash(command, ws) is not None


def test_heredoc_text_is_not_taken_for_a_state_write(ws: Path) -> None:
    command = "cat <<EOF > notes.md\necho x > .satsuma/policy.md\nEOF"
    assert state_write_guard.check_bash(command, ws) is None


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("if a; then b x; elif c; then d; else e; fi", [["a"], ["b", "x"], ["c"], ["d"], ["e"]]),
        ("while a; do b; done", [["a"], ["b"]]),
        ("until a; do b; done", [["a"], ["b"]]),
        ("for i in a b; do c $i; done", [["c", "$i"]]),
        ("for i do a; done", [["a"]]),
        ("select x in a b; do c; done", [["c"]]),
        ("{ a; b; }", [["a"], ["b"]]),
        ("! a", [["a"]]),
        ("case $x in a|b) c;; (d) e;; *) f\nesac", [["c"], ["e"], ["f"]]),
        ("case $x in a) case $y in b) c;; esac;; d) e;; esac; g", [["c"], ["e"], ["g"]]),
        ("case x\nin\n  x) a ;;\nesac", [["a"]]),
        ("case x in x) a; esac; b", [["a"], ["b"]]),
        ("function f { a; }", [["a"]]),
        ("function f() { a; }", [["a"]]),
        ("time { a; }", [["a"]]),
        ("time -p if a; then b; fi", [["a"], ["b"]]),
        ("time -p a", [["time", "-p", "a"]]),
        ("coproc w { a; }", [["a"]]),
        ("coproc a b", [["a", "b"]]),
        ("echo if then { ! }", [["echo", "if", "then", "{", "!", "}"]]),
        ("FOO=1 if x", [["FOO=1", "if", "x"]]),
    ],
)
def test_reserved_words_are_read_only_where_a_command_starts(
    command: str, expected: list[list[str]]
) -> None:
    assert _segments(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "case x in $(a)) b;; esac",
        "case $(a) in x) b;; esac",
        "for x in $(a); do b; done",
    ],
)
def test_a_substitution_in_for_words_or_case_patterns_is_still_read(command: str) -> None:
    assert ["a"] in _segments(command)


@pytest.mark.parametrize(
    "command",
    [
        "if true; then satsumaestro repl; fi",
        "if satsumaestro repl; then :; fi",
        "while true; do satsumaestro repl; done",
        "until false; do stro prompt x; done",
        "for i in 1; do satsumaestro repl; done",
        "{ satsumaestro repl; }",
        "! satsumaestro repl",
        "case x in x) satsumaestro repl;; esac",
        "time { satsumaestro repl; }",
        "function f { satsumaestro repl; }; f",
        "coproc satsumaestro repl",
    ],
)
def test_self_invocation_is_found_inside_compound_commands(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "for satsumaestro in a b; do echo $satsumaestro; done",
        "case $1 in satsumaestro|stro) echo self;; esac",
        "case $1 in (stro) echo self;; esac",
    ],
)
def test_loop_words_and_case_patterns_are_not_taken_for_self_invocation(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "if true; then rm -rf ..; fi",
        "{ rm -rf ..; }",
        "! rm -rf ..",
        "while true; do rm -rf ..; done",
        "case x in *) rm -rf ..;; esac",
    ],
)
def test_deletions_are_found_inside_compound_commands(ws: Path, command: str) -> None:
    assert deletion_guard.check_bash(command, ws) is not None


def test_loop_words_are_not_taken_for_a_deletion(ws: Path) -> None:
    assert deletion_guard.check_bash("for x in rm -rf ..; do echo $x; done", ws) is None


def test_a_find_inside_a_compound_command_is_read_by_the_state_write_guard(ws: Path) -> None:
    command = "if true; then find . -name '*.md' -exec sed -i s/a/b/ {} +; fi"
    assert state_write_guard.check_bash(command, ws) is not None


def test_read_only_mode_reads_commands_inside_compound_commands() -> None:
    verdict = review_read_only("if true; then rm notes.txt; fi", PermissionMode.READ_ONLY)
    assert verdict.kind != "allow"
    assert review_read_only("for f in *.md; do cat $f; done", PermissionMode.READ_ONLY).kind == (
        "allow"
    )


def test_a_deletion_inside_a_compound_command_cannot_be_remembered() -> None:
    assert remember_blocker("bash", {"command": "{ rm -rf build; }"}) is not None


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("echo `a b` c", [["a", "b"], ["echo", "`a b`", "c"]]),
        ('echo "$(a b)" c', [["a", "b"], ["echo", "$(a b)", "c"]]),
        ('echo "`a`"', [["a"], ["echo", "`a`"]]),
        ('echo "$(echo "q u") x"', [["echo", "q u"], ["echo", '$(echo "q u") x']]),
        ('echo "x$(a "$(b)")y"', [["b"], ["a", "$(b)"], ["echo", 'x$(a "$(b)")y']]),
        ("echo `echo \\`a\\``", [["a"], ["echo", "`a`"], ["echo", "`echo \\`a\\``"]]),
        ('echo "`echo \\"a\\"`"', [["echo", "a"], ["echo", '`echo \\"a\\"`']]),
        ("echo '$(a)' \"\\$(b)\" '`c`'", [["echo", "$(a)", "$(b)", "`c`"]]),
        ("cat <(a) >(b)", [["cat"], ["a"], ["b"]]),
    ],
)
def test_substitutions_in_words_are_read_before_their_command(
    command: str, expected: list[list[str]]
) -> None:
    assert _segments(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        'echo "$(case x in x) a;; esac) and"',
        "cat <<EOF\n$(case x in x) a;; esac)\nEOF",
    ],
)
def test_a_case_pattern_does_not_end_a_substitution(command: str) -> None:
    assert ["a"] in _segments(command)


@pytest.mark.parametrize(
    "command",
    [
        'x="$(cat <<EOF\n))\nEOF\n)"; b',
        'echo "$(echo a # )\n)"; b',
    ],
)
def test_heredocs_and_comments_inside_a_substitution_do_not_end_it(command: str) -> None:
    assert _segments(command)[-1] == ["b"]


def test_an_unclosed_substitution_is_unreadable() -> None:
    assert tokenize("echo `a") is None
    assert tokenize('echo "$(a"') is None
    assert substitutions("$(case x in x) y;; esac) z") == ["case x in x) y;; esac"]


@pytest.mark.parametrize(
    "command",
    [
        "echo `satsumaestro repl`",
        'echo "$(satsumaestro repl)"',
        'echo "`stro prompt x`"',
        'x="$(satsumaestro repl)"',
        'echo "$(echo "$(satsumaestro repl)")"',
        'echo "$(case x in x) satsumaestro repl;; esac)"',
    ],
)
def test_self_invocation_is_found_in_backquotes_and_double_quoted_substitutions(
    command: str,
) -> None:
    assert self_invocation_guard.check_bash(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "echo '`satsumaestro repl`'",
        'echo "\\$(satsumaestro repl)"',
        "echo '$(stro prompt x)'",
    ],
)
def test_text_that_the_shell_does_not_run_is_not_taken_for_self_invocation(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is None


@pytest.mark.parametrize("command", ['echo "$(rm -rf ..)"', "echo `rm -rf ..`"])
def test_deletions_are_found_in_backquotes_and_double_quoted_substitutions(
    ws: Path, command: str
) -> None:
    assert deletion_guard.check_bash(command, ws) is not None


@pytest.mark.parametrize(
    "command", ['echo "$(echo x > .satsuma/policy.md)"', "echo `cp x .satsuma/policy.md`"]
)
def test_state_writes_are_found_in_backquotes_and_double_quoted_substitutions(
    ws: Path, command: str
) -> None:
    assert state_write_guard.check_bash(command, ws) is not None


def test_single_quoted_substitution_text_is_not_taken_for_a_state_write(ws: Path) -> None:
    command = "echo '$(echo x > .satsuma/policy.md)' > notes.md"
    assert state_write_guard.check_bash(command, ws) is None


def test_deeply_nested_substitutions_are_read_quickly(ws: Path) -> None:
    command = "echo " + '"$(' * 400 + "satsumaestro repl" + ')"' * 400
    started = time.monotonic()
    self_invocation_guard.check_bash(command)
    deletion_guard.check_bash(command, ws)
    state_write_guard.check_bash(command, ws)
    assert time.monotonic() - started < 2
