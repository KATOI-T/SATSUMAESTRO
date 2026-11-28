"""包みの語（env・timeout・nice など）を挟んだ削除にも、削除の絶対禁止を当てる。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.security.deletion_guard import check_bash, simple_commands


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "work"
    (ws / ".satsuma").mkdir(parents=True)
    (ws / "build").mkdir()
    return ws


@pytest.mark.parametrize(
    "command",
    [
        "env rm -rf ..",
        "env -i rm -rf ..",
        "env - rm -rf ..",
        "env -u HOME rm -rf ..",
        "env --unset HOME rm -rf ..",
        "env -C /tmp -P /bin rm -rf ..",
        "env FOO=1 BAR=2 rm -rf ..",
        "env -S 'rm -rf ..'",
        "env -S'-i rm -rf ..'",
        "env --split-string='rm -rf ..'",
        "env -iS 'rm -rf ..'",
        "timeout 5 rm -rf ..",
        "timeout -s KILL -k 1 5 rm -rf ..",
        "timeout --signal=KILL --preserve-status 5s rm -rf ..",
        "timeout --kill-after 1 5 rm -rf ..",
        "nice -n 5 rm -rf ..",
        "nice -n5 rm -rf ..",
        "nice -5 rm -rf ..",
        "nice --adjustment 5 rm -rf ..",
        "command -p rm -rf ..",
        "time -p rm -rf ..",
        "time -o out.txt rm -rf ..",
        "exec -a name rm -rf ..",
        "stdbuf -o L rm -rf ..",
        "nice -n 5 env timeout 5 rm -rf ..",
        "cd build && env rm -rf ../..",
        "timeout 5 find .. -delete",
        "env rm -rf .satsuma",
        "nice -n 5 unlink .satsuma/policy.md",
        "env -C / rm -rf tmp",
        "env --chdir=/ rm -rf tmp",
        "env -iC / rm -rf tmp",
        "env -C .. rm -rf work",
        "cd build && env -C .. rm -rf .satsuma",
        "env -a name rm -rf ..",
        "env -ia name rm -rf ..",
        "env --argv0 name rm -rf ..",
        "env --argv0=name rm -rf ..",
        "env --unse HOME rm -rf ..",
        "env --split='rm -rf ..'",
        "env --ch / rm -rf tmp",
        "env --ch=/ rm -rf tmp",
        "timeout --sig KILL 5 rm -rf ..",
        "timeout --kill 1 5 rm -rf ..",
        "nice --adj 5 rm -rf ..",
        "stdbuf --out L rm -rf ..",
        "time --form %e rm -rf ..",
        "sudo --us root rm -rf build",
    ],
)
def test_wrapped_deletions_are_prohibited(workspace: Path, command: str) -> None:
    assert check_bash(command, workspace) is not None


@pytest.mark.parametrize(
    "command",
    [
        "env rm -rf build",
        "timeout 5 rm -rf build",
        "nice -n 5 rm -rf build",
        "time -p rm -rf build",
        "timeout 5 grep -r rm ..",
        "env grep rm ..",
        "nice -n 5 make clean",
        "command -v rm",
        "env -S 'ls ..'",
        "timeout --foreground 5 rm -rf build",
        "timeout --sig=KILL 5 rm -rf build",
        "env --ignore-environment rm -rf build",
        "env --ch=build rm -rf sub",
    ],
)
def test_wrapped_commands_that_do_not_delete_outside_are_allowed(
    workspace: Path, command: str
) -> None:
    assert check_bash(command, workspace) is None


def test_sudo_inside_a_wrapper_is_still_privileged(workspace: Path) -> None:
    assert check_bash("env sudo rm -rf build", workspace) is not None
    assert check_bash("sudo -u root timeout 5 rm -rf build", workspace) is not None


def test_the_shared_reading_of_wrappers_is_unchanged() -> None:
    """自動許可などが使う読み方は前のまま（包みの後ろを広く見るのは各検査の側）。"""
    assert simple_commands("timeout 5 ls") == [(False, "timeout", ["5", "ls"])]
    assert simple_commands("nice -n 5 ls") == [(False, "-n", ["5", "ls"])]
    assert simple_commands("env -C / rm -rf tmp")[0][1] == "env"
