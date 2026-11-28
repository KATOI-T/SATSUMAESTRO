"""守りの検査が、ほかのコマンドが動かすコマンド（sh -c・eval・xargs・find -exec・包み）も読む。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.security import deletion_guard, self_invocation_guard, state_write_guard
from satsumaestro.security.command_words import Command, commands, tokenize


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
        ("bash <<'EOF'\nsatsumaestro repl\nEOF", [Command(["bash"], "satsumaestro repl")]),
        ("bash <<EOF -s\na\nEOF\nb", [Command(["bash", "-s"], "a"), Command(["b"])]),
        ("sh <<< 'a b'", [Command(["sh"], "a b")]),
        ("cat x", [Command(["cat", "x"])]),
    ],
)
def test_each_command_keeps_the_text_given_to_it_as_input(
    command: str, expected: list[Command]
) -> None:
    tokens = tokenize(command)
    assert tokens is not None
    assert commands(tokens) == expected


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'satsumaestro repl'",
        "sh -c 'stro prompt x'",
        "bash -lc 'satsumaestro repl'",
        'zsh -c "satsumaestro repl"',
        "dash -ec 'satsumaestro repl' name",
        "bash --norc -c 'satsumaestro repl'",
        "bash -o pipefail -c 'satsumaestro repl'",
        "fish --command='satsumaestro repl'",
        "bash -c \"sh -c 'satsumaestro repl'\"",
        "sudo -u x bash -c 'satsumaestro repl'",
        "eval 'satsumaestro repl'",
        "eval satsumaestro repl",
        "echo x | xargs satsumaestro",
        "xargs -I{} satsumaestro {}",
        "xargs -n 1 stro prompt",
        "find . -maxdepth 0 -exec satsumaestro repl \\;",
        "find . -maxdepth 0 -exec sh -c 'satsumaestro repl' \\;",
        "timeout 60 satsumaestro repl",
        "env satsumaestro repl",
        "env -i PATH=/usr/bin satsumaestro repl",
        "nice -n 5 satsumaestro repl",
        "nohup satsumaestro repl &",
        "stdbuf -oL satsumaestro repl",
        "time -p satsumaestro repl",
        "caffeinate -i satsumaestro repl",
        "setsid satsumaestro repl",
        "bash <<'EOF'\nsatsumaestro repl\nEOF",
        "sh <<< 'satsumaestro repl'",
        "bash -s <<EOF\nstro prompt x\nEOF",
    ],
)
def test_self_invocation_is_found_in_commands_that_other_commands_run(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        'bash -c "$X"',
        "bash script.sh <<'EOF'\nsatsumaestro repl\nEOF",
        "grep -r 'bash -c satsumaestro' .",
        "echo 'eval satsumaestro repl'",
        "cat <<'EOF'\nbash -c 'satsumaestro repl'\nEOF",
    ],
)
def test_text_that_no_shell_runs_is_not_taken_for_self_invocation(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'rm -rf ..'",
        "eval rm -rf ..",
        "xargs rm -rf ..",
        "timeout 5 bash -c 'rm -rf ..'",
        "bash <<'EOF'\nrm -rf ..\nEOF",
        "find . -maxdepth 0 -exec sh -c 'rm -rf ..' \\;",
        "bash -c 'cd src && rm -rf ../..'",
    ],
)
def test_deletions_are_found_in_commands_that_other_commands_run(ws: Path, command: str) -> None:
    assert deletion_guard.check_bash(command, ws) is not None


def test_a_cd_inside_a_shell_does_not_move_the_commands_after_it(ws: Path) -> None:
    assert deletion_guard.check_bash("bash -c 'cd /'; rm -rf src", ws) is None


@pytest.mark.parametrize(
    "command",
    [
        "bash -c 'echo x > .satsuma/policy.md'",
        "timeout 5 sh -c 'echo x > .satsuma/policy.md'",
        "eval 'echo x > .satsuma/policy.md'",
        "sh <<'EOF'\ncp x .satsuma/policy.md\nEOF",
        "bash -c 'find . -name \"*.md\" -exec sed -i s/a/b/ {} +'",
    ],
)
def test_state_writes_are_found_in_commands_that_other_commands_run(ws: Path, command: str) -> None:
    assert state_write_guard.check_bash(command, ws) is not None


@pytest.mark.parametrize(
    "command",
    ["FOO=/tmp satsumaestro repl", "A=1 B+=/x stro prompt x", "env PATH=/usr/bin satsumaestro"],
)
def test_assignments_with_a_slash_are_stepped_over(command: str) -> None:
    assert self_invocation_guard.check_bash(command) is not None


def test_a_deletion_after_an_assignment_with_a_slash_is_found(ws: Path) -> None:
    assert deletion_guard.check_bash("FOO=/x rm -rf ..", ws) is not None


@pytest.mark.parametrize("command", ['bash -c "$X"', "grep x <<< .satsuma/policy.md"])
def test_input_text_is_not_taken_for_a_state_write(ws: Path, command: str) -> None:
    assert state_write_guard.check_bash(command, ws) is None
