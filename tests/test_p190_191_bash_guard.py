"""bash の実行前の検査: read-only は単純コマンドを全部見る・展開するリダイレクト先は止める。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import satsumaestro.engine  # noqa: F401
from satsumaestro.security.bash_pipeline import review_command, review_read_only
from satsumaestro.security.deletion_guard import check_bash, simple_commands
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.tools.bash_validation import validate_bash
from satsumaestro.util.errors import PathEscapeError

RO = PermissionMode.READ_ONLY
WRITES = "writes to the filesystem"


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("sudo sudo rm x", WRITES),
        ("sudo -u root rm x", WRITES),
        ("sudo -nu root rm x", WRITES),
        ("sudo --user root rm x", WRITES),
        ("sudo -- rm x", WRITES),
        ("doas -u root rm x", WRITES),
        ("sudo -e notes.txt", "'sudoedit' writes"),
        ("sudo -i", "could not tell which command 'sudo' runs"),
        ("env rm x", WRITES),
        ("env FOO=1 rm x", WRITES),
        ("nice -n 5 rm x", WRITES),
        ("timeout 5 rm x", WRITES),
        ("xargs rm", WRITES),
        ("command rm x", WRITES),
        ("nohup rm x", WRITES),
        ("exec rm x", WRITES),
        ("ls; rm x", WRITES),
        ("ls && rm x", WRITES),
        ("ls || rm x", WRITES),
        ("ls | xargs rm", WRITES),
        ("cat f | sed -i s/a/b/ g", "sed -i rewrites the file in place"),
        ("ls && git -C sub push", "'git push' changes the repository"),
        ("sh -c 'rm x'", WRITES),
        ("eval rm x", WRITES),
        ("cat <(rm x)", WRITES),
        ("cat a >(rm x)", WRITES),
        ("cat <(echo (x))", "could not parse a substitution"),
        ("find . -delete", "'find' deletes files here"),
        ("find . -exec rm {} ;", "'find' deletes files here"),
        ("find . -exec sh -c 'rm x' ;", WRITES),
    ],
)
def test_read_only_blocks_every_simple_command(command: str, message: str) -> None:
    verdict = review_read_only(command, RO)
    assert verdict.kind == "block" and message in verdict.message, verdict


@pytest.mark.parametrize(
    "command",
    [
        "sudo sudo ls",
        "sudo ls",
        "sudo cat /var/log/system.log",
        "diff <(ls) <(ls a)",
        "ls | grep x",
        "timeout 5 cat x",
        "env FOO=1 ls",
        "bash -c 'ls -la'",
        "find . -name '*.py'",
        "git log --oneline",
        "git -C sub status",
        "echo x 2>&1",
        "cat a > /dev/null",
        "grep -r foo .",
    ],
)
def test_read_only_still_allows_reading(command: str) -> None:
    assert review_read_only(command, RO).kind == "allow"


def test_other_modes_are_untouched() -> None:
    for mode in (PermissionMode.WORKSPACE_WRITE, PermissionMode.DANGER_FULL_ACCESS):
        assert review_read_only("ls; rm x", mode).kind == "allow"


@pytest.mark.parametrize(
    "command",
    ["env " * 300 + "ls", "xargs " * 300 + "rm", "eval " * 300 + "ls", "nice -n 5 " * 200 + "ls"],
)
def test_deep_nesting_stops_quickly(command: str) -> None:
    started = time.perf_counter()
    verdict = review_read_only(command, RO)
    assert verdict.kind == "block" and "too deeply" in verdict.message
    assert time.perf_counter() - started < 1.0


def test_review_command_uses_the_same_read_only_check(tmp_path: Path) -> None:
    assert review_command("ls && rm x", RO, tmp_path).kind == "block"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("sudo -nu root rm x", (True, "rm", ["x"])),
        ("sudo -uroot rm x", (True, "rm", ["x"])),
        ("sudo --user root rm x", (True, "rm", ["x"])),
        ("sudo --user=root rm x", (True, "rm", ["x"])),
        ("sudo -h host ls", (True, "ls", [])),
        ("sudo -- rm x", (True, "rm", ["x"])),
        ("sudo -ne notes.txt", (True, "sudoedit", ["notes.txt"])),
        ("sudo -u root -e notes.txt", (True, "sudoedit", ["notes.txt"])),
        ("doas -u root rm x", (True, "rm", ["x"])),
        ("sudo -i", (True, "", [])),
    ],
)
def test_privilege_options_are_skipped_with_their_values(
    command: str, expected: tuple[bool, str, list[str]]
) -> None:
    assert simple_commands(command) == [expected]


def test_sudo_deletion_with_a_clustered_user_option_is_prohibited(tmp_path: Path) -> None:
    reason = check_bash("sudo -nu root rm notes.txt", tmp_path)
    assert reason is not None and "sudo" in reason


@pytest.mark.parametrize(
    "command",
    [
        'echo x > "$HOME/x"',
        "echo x > ~/x",
        "echo x 2>~/err.log",
        "echo x >> $OUT",
        "cat < ~/notes.txt",
    ],
)
def test_expanded_redirect_targets_are_refused(tmp_path: Path, command: str) -> None:
    with pytest.raises(PathEscapeError, match="expanded by the shell"):
        validate_bash(command, WorkspaceJail(tmp_path))


@pytest.mark.parametrize("command", ["echo x > out.txt", "ls 2>/dev/null", "echo x 2>&1"])
def test_plain_redirect_targets_still_pass(tmp_path: Path, command: str) -> None:
    assert validate_bash(command, WorkspaceJail(tmp_path)).kind == "allow"
