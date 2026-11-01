"""bash コマンドの実行前の内容検査のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.security.bash_pipeline import (
    CommandKind,
    command_kind,
    hazard_warning,
    leading_command,
    review_command,
    review_for_mode,
    review_paths,
    review_read_only,
    review_sed,
)
from satsumaestro.security.enforcer import PermissionMode

_WS = Path("/srv/checkout")
_RO = PermissionMode.READ_ONLY
_WW = PermissionMode.WORKSPACE_WRITE


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf dist/cache",
        "cp notes.md notes.bak",
        "npm install left-pad",
        "git push upstream feature/login",
        "kill -TERM 4242",
        "systemctl stop postgresql",
        "su - admin",
        "fdisk -l /dev/nvme0n1",
        "iptables -X",
        "snap install htop",
        "wipefs -a /dev/sdc",
        "docker rm -f cache-redis",
        "date > stamp.txt",
        "date >'stamp'",
        "date &>log",
        "date >&log",
        "grep -c '<li>' menu.html > total.txt",
        'printf "unterminated',
    ],
)
def test_read_only_stops_state_changes(command: str) -> None:
    verdict = review_read_only(command, _RO)
    assert verdict.kind == "block"
    assert verdict.message


def test_read_only_messages_distinguish_files_from_system() -> None:
    assert "writes to the filesystem" in review_read_only("cp notes.md notes.bak", _RO).message
    assert "writes to the filesystem" in review_read_only("rm -rf dist/cache", _RO).message
    assert "changes system state" in review_read_only("pkill -f vite", _RO).message
    assert "changes system state" in review_read_only("brew upgrade jq", _RO).message


@pytest.mark.parametrize(
    "command",
    [
        "ls -lh src",
        "cat /etc/os-release",
        "grep -rn TODO src",
        "git status --short",
        "ps -ef",
        "journalctl -u nginx -n 50",
        "curl -sI https://www.python.org/",
        "grep -c '</ul>' menu.html",
        "grep -n 'count > 10' stats.py",
        "make test 2>&1",
        "ls missing 2>/dev/null",
        "sort < names.txt",
    ],
)
def test_read_only_passes_reads(command: str) -> None:
    assert review_read_only(command, _RO).kind == "allow"


def test_read_only_check_is_inert_in_other_modes() -> None:
    assert review_read_only("rm -rf dist/cache", _WW).kind == "allow"
    assert review_read_only("rm -rf dist/cache", PermissionMode.DANGER_FULL_ACCESS).kind == "allow"


def test_git_options_with_values_are_skipped() -> None:
    assert review_read_only("git -C vendor/lib status", _RO).kind == "allow"
    assert review_read_only("git -c core.pager=cat log", _RO).kind == "allow"
    assert command_kind("git -C vendor/lib log --oneline") is CommandKind.READ
    assert review_read_only("git -C status push upstream", _RO).kind == "block"
    assert review_read_only("git -C vendor/lib push upstream main", _RO).kind == "block"
    assert review_read_only("git", _RO).kind == "allow"


def test_sudo_is_judged_by_its_inner_command() -> None:
    verdict = review_read_only("sudo rm -rf /var/cache/app", _RO)
    assert verdict.kind == "block"
    assert "'rm'" in verdict.message
    assert review_read_only("sudo -n cat /var/log/syslog", _RO).kind == "allow"


def test_sed_in_place_is_stopped_in_read_only() -> None:
    verdict = review_sed("sed -i 's/http:/https:/' links.md", _RO)
    assert verdict.kind == "block"
    assert "sed -i" in verdict.message
    assert review_sed("sed -i.orig 's/v1/v2/' setup.cfg", _RO).kind == "block"
    assert review_sed("sed --in-place 's/v1/v2/' setup.cfg", _RO).kind == "block"
    assert review_sed("sed 's/http:/https:/' links.md", _RO).kind == "allow"
    assert review_sed("sed 's/ -i / -n /' flags.txt", _RO).kind == "allow"
    assert review_sed("sed -i 's/v1/v2/' setup.cfg", _WW).kind == "allow"


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf ~",
        "dd if=/dev/urandom of=/dev/nvme0n1",
        ":(){ :|:& };:",
        "shred -n 3 /dev/sdb",
        "wipefs -a /dev/nvme1n1",
        "chmod -R 777 /opt/app",
    ],
)
def test_hazards_are_flagged(command: str) -> None:
    verdict = hazard_warning(command)
    assert verdict.kind == "warn"
    assert verdict.message.startswith(("This command", "'"))


def test_recursive_forced_rm_is_flagged_when_the_flags_are_separate() -> None:
    for command in ("rm -r -f out/", "rm -f -R out/", "rm -rv -f out/"):
        assert hazard_warning(command).kind == "warn", command
        assert "rm with both -r and -f" in hazard_warning(command).message
    for command in (
        "rm -rf out/",
        "rm -fr out/",
        "rm -r out/",
        "rm -f old.log",
        "rm old.log",
        "ls -lh src",
        "grep -rf words.txt .",
    ):
        assert hazard_warning(command).kind == "allow", command


def test_workspace_write_flags_system_paths_only_for_writers() -> None:
    verdict = review_for_mode("cp nginx.conf /etc/nginx/nginx.conf", _WW)
    assert verdict.kind == "warn"
    assert "system path" in verdict.message
    assert review_for_mode("cp nginx.conf ./conf/", _WW).kind == "allow"
    assert review_for_mode("cat /etc/os-release", _WW).kind == "allow"
    assert (
        review_for_mode(
            "cp nginx.conf /etc/nginx/nginx.conf", PermissionMode.DANGER_FULL_ACCESS
        ).kind
        == "allow"
    )


def test_paths_outside_workspace_are_flagged() -> None:
    verdict = review_paths("cat ../../shared/config.yml", _WS)
    assert verdict.kind == "warn"
    assert "'../'" in verdict.message
    verdict = review_paths("cat ~/.aws/credentials", _WS)
    assert verdict.kind == "warn"
    assert "'~/' or $HOME" in verdict.message
    assert review_paths(f"cat {_WS}/docs/../README.md", _WS).kind == "allow"
    assert review_paths("cat README.md", _WS).kind == "allow"


def test_command_kind() -> None:
    assert command_kind("ls -lh src") is CommandKind.READ
    assert command_kind("cp notes.md notes.bak") is CommandKind.MODIFY
    assert command_kind("rm -rf dist/cache") is CommandKind.DESTROY
    assert command_kind("curl -sI https://www.python.org/") is CommandKind.NETWORK
    assert command_kind("pkill -f vite") is CommandKind.PROCESS
    assert command_kind("ps -ef") is CommandKind.PROCESS
    assert command_kind("brew upgrade jq") is CommandKind.PACKAGE
    assert command_kind("sudo mount /dev/sdb1 /mnt") is CommandKind.ADMIN
    assert command_kind("docker ps -a") is CommandKind.ADMIN
    assert command_kind("sed -i 's/v1/v2/' setup.cfg") is CommandKind.MODIFY
    assert command_kind("sed 's/v1/v2/' setup.cfg") is CommandKind.READ
    assert command_kind("git status --short") is CommandKind.READ
    assert command_kind("git push upstream feature/login") is CommandKind.MODIFY
    assert command_kind("frobnicate --all") is CommandKind.UNKNOWN
    assert command_kind("") is CommandKind.UNKNOWN


def test_leading_command_skips_assignments() -> None:
    assert leading_command("LANG=C sort names.txt") == "sort"
    assert leading_command("TZ=UTC LC_ALL=C date +%F") == "date"
    assert leading_command('GREETING="hi there" printf ok') == "printf"
    assert leading_command("grep -rn TODO src") == "grep"
    assert leading_command("_DEBUG=1 make") == "make"
    assert leading_command("DEBUG=1") == ""
    assert leading_command("") == ""
    assert leading_command('printf "unterminated') == ""
    assert leading_command("1=foo rm -rf dist/cache") == "1=foo"


def test_review_command_returns_the_first_failure() -> None:
    verdict = review_command("rm -rf /", _RO, _WS)
    assert verdict.kind == "block"
    assert "read-only" in verdict.message
    verdict = review_command("rm -rf /etc/nginx", _WW, _WS)
    assert verdict.kind == "warn"
    assert "system path" in verdict.message
    verdict = review_command("rm -r -f out/", PermissionMode.DANGER_FULL_ACCESS, _WS)
    assert verdict.kind == "warn"
    assert "rm with both -r and -f" in verdict.message
    assert review_command("ls -lh src", _RO, _WS).kind == "allow"
