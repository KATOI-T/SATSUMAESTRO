"""bash コマンドの実行前の内容検査。"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from satsumaestro.security.bash_lex import DEVICE_TARGETS, iter_redirects
from satsumaestro.security.deletion_guard import deletes, simple_commands
from satsumaestro.security.enforcer import PermissionMode


class Verdict(BaseModel):
    """検査 1 回の結果。"""

    model_config = ConfigDict(frozen=True)

    kind: Literal["allow", "block", "warn"]
    message: str = ""


class CommandKind(StrEnum):
    """コマンドが何をするかの大まかな分類（警告と診断のため。権限には使わない）。"""

    READ = "read"
    MODIFY = "modify"
    DESTROY = "destroy"
    NETWORK = "network"
    PROCESS = "process"
    PACKAGE = "package"
    ADMIN = "admin"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CommandProfile:
    """コマンド語 1 つの性質。"""

    kind: CommandKind
    mutates: bool


def _group(names: str, kind: CommandKind, *, mutates: bool) -> dict[str, CommandProfile]:
    profile = CommandProfile(kind, mutates)
    return dict.fromkeys(names.split(), profile)


_PROFILES: dict[str, CommandProfile] = {
    **_group(
        "ls cat head tail less more wc sort uniq grep egrep fgrep find which whereis whatis "
        "man info file stat du df free uptime uname hostname whoami id groups env printenv "
        "echo printf date cal bc expr test true false pwd tree diff cmp md5sum sha256sum "
        "sha1sum xxd od hexdump strings readlink realpath basename dirname seq yes tput "
        "column jq yq xargs tr cut paste awk sed",
        CommandKind.READ,
        mutates=False,
    ),
    **_group(
        "cp mv mkdir rmdir touch chmod chown chgrp ln install tee truncate mkfifo mknod dd "
        "sudoedit",
        CommandKind.MODIFY,
        mutates=True,
    ),
    **_group("rm shred wipefs", CommandKind.DESTROY, mutates=True),
    **_group(
        "curl wget ssh scp rsync ftp sftp nc ncat telnet ping traceroute dig nslookup host "
        "whois ifconfig ip netstat ss nmap",
        CommandKind.NETWORK,
        mutates=False,
    ),
    **_group(
        "ps top htop bg fg jobs nohup disown wait nice renice", CommandKind.PROCESS, mutates=False
    ),
    **_group("kill pkill killall", CommandKind.PROCESS, mutates=True),
    **_group(
        "apt apt-get yum dnf pacman brew pip pip3 npm yarn pnpm bun cargo gem go rustup snap "
        "flatpak",
        CommandKind.PACKAGE,
        mutates=True,
    ),
    **_group("lsblk blkid journalctl dmesg", CommandKind.ADMIN, mutates=False),
    **_group("sudo", CommandKind.ADMIN, mutates=False),
    **_group(
        "su chroot mount umount fdisk parted systemctl service docker modprobe insmod rmmod "
        "iptables ufw firewall-cmd sysctl crontab at useradd userdel usermod groupadd groupdel "
        "passwd visudo reboot shutdown halt poweroff",
        CommandKind.ADMIN,
        mutates=True,
    ),
}

_GIT_READ_VERBS = frozenset(
    "status log diff show branch tag stash remote fetch ls-files ls-tree cat-file rev-parse "
    "describe shortlog blame bisect reflog config".split()
)

GIT_OPTIONS_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace"})

_EXEC_WRAPPERS = frozenset(
    {"env", "timeout", "xargs", "nice", "nohup", "time", "command", "exec", "builtin", "watch"}
)
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
_FIND_EXEC = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
_RUNNERS = _EXEC_WRAPPERS | _SHELLS | {"eval", "find"}
_SUBSTITUTION = re.compile(r"[$<>]\(([^()]*)\)|`([^`]*)`")
_MAX_NESTING = 8

_WRITING_REDIRECTS = frozenset({">", ">>", ">|", "<>", "&>", "&>>", ">&"})

_SYSTEM_PREFIXES = (
    "/etc/",
    "/usr/",
    "/var/",
    "/boot/",
    "/sys/",
    "/proc/",
    "/dev/",
    "/sbin/",
    "/lib/",
    "/opt/",
)


@dataclass(frozen=True)
class Hazard:
    """誰が見ても危ない形。"""

    trigger: str
    effect: str


_HAZARDS: tuple[Hazard, ...] = (
    Hazard(
        "rm -rf /", "removes everything under the filesystem root and leaves the system unbootable"
    ),
    Hazard("rm -rf ~", "removes the whole home directory, including dotfiles and keys"),
    Hazard("rm -rf *", "removes every file and folder in the current directory"),
    Hazard("rm -rf .", "removes the current directory and everything below it"),
    Hazard("mkfs", "formats a device, erasing all data stored on it"),
    Hazard("dd if=", "copies raw bytes to its target and can overwrite a disk or partition"),
    Hazard("> /dev/sd", "writes directly to a disk device"),
    Hazard("chmod -R 777", "makes an entire tree readable and writable by every user"),
    Hazard("chmod -R 000", "strips all permissions from an entire tree"),
    Hazard(":(){ :|:& };:", "spawns processes without limit until the machine stops responding"),
)

_PASS = Verdict(kind="allow")


def _stop(reason: str) -> Verdict:
    return Verdict(kind="block", message=reason)


def _flag(message: str) -> Verdict:
    return Verdict(kind="warn", message=message)


def _tokens(command: str) -> list[str]:
    """シェルの単語に分ける。"""
    try:
        return shlex.split(command, comments=False, posix=True)
    except ValueError:
        return []


def _is_assignment(token: str) -> bool:
    """`NAME=value` の形か。"""
    name, sep, _ = token.partition("=")
    return (
        bool(sep)
        and bool(name)
        and (name[0].isalpha() or name[0] == "_")
        and all(ch.isalnum() or ch == "_" for ch in name)
    )


def leading_command(command: str) -> str:
    """先頭の `NAME=value` の前置を飛ばした、最初のコマンド語。"""
    for token in _tokens(command):
        if not _is_assignment(token):
            return token
    return ""


def _git_verb(command: str) -> str | None:
    """`git` の後ろでオプションを飛ばした最初の語（サブコマンド）。"""
    return _git_verb_of(command.split()[1:])


def _git_verb_of(words: list[str]) -> str | None:
    """git の引数から、オプション（値を取るものは値ごと）を飛ばした最初の語。"""
    i = 0
    while i < len(words):
        word = words[i]
        if not word.startswith("-"):
            return word
        i += 2 if word in GIT_OPTIONS_WITH_VALUE else 1
    return None


def _profile(command: str) -> CommandProfile | None:
    return _PROFILES.get(leading_command(command))


def _has_inplace_flag(command: str) -> bool:
    return _inplace(_tokens(command)[1:])


def _inplace(args: list[str]) -> bool:
    return any(t.startswith("-i") or t == "--in-place" for t in args)


def review_read_only(command: str, mode: PermissionMode) -> Verdict:
    """read-only モードで、状態を変えるコマンド・書き込みのリダイレクト・変更系の git を止める。"""
    if mode is not PermissionMode.READ_ONLY:
        return _PASS
    return _review_read_only(command, 0)


def _review_read_only(command: str, depth: int) -> Verdict:
    if depth > _MAX_NESTING:
        return _stop(_TOO_DEEP)
    commands = simple_commands(command)
    if commands is None:
        return _stop("could not parse command: unbalanced quotes")
    for privileged, name, args in commands:
        if privileged and not name:
            return _stop(
                "could not tell which command 'sudo' runs; read-only mode does not allow it"
            )
        verdict = _review_simple(name, args, depth)
        if verdict.kind != "allow":
            return verdict
    inner = list(_SUBSTITUTION.finditer(command))
    opened = command.count("<(") + command.count(">(") + command.count("$(")
    if opened > sum(1 for m in inner if m.group(1) is not None):
        return _stop(
            "could not parse a substitution like '<(…)' in this command; read-only mode does "
            "not allow it"
        )
    for match in inner:
        verdict = _review_read_only(match.group(1) or match.group(2) or "", depth + 1)
        if verdict.kind != "allow":
            return verdict
    try:
        redirects = list(iter_redirects(command))
    except ValueError as e:
        return _stop(f"could not parse command: {e}")
    for op, target in redirects:
        if op in _WRITING_REDIRECTS and target not in DEVICE_TARGETS:
            return _stop(
                f"the redirection '{op}' would write to a file; read-only mode does not allow it"
            )
    return _PASS


_TOO_DEEP = "the command nests other commands too deeply to check; read-only mode does not allow it"


def _review_simple(name: str, args: list[str], depth: int) -> Verdict:
    """単純コマンド 1 つ（``simple_commands`` が前置きを剥がした名前と引数）の read-only 検査。"""
    if depth > _MAX_NESTING:
        return _stop(_TOO_DEEP)
    profile = _PROFILES.get(name)
    if profile is not None and profile.mutates:
        if profile.kind in (CommandKind.MODIFY, CommandKind.DESTROY):
            return _stop(
                f"'{name}' writes to the filesystem; read-only mode allows only commands that "
                "leave files untouched"
            )
        return _stop(
            f"'{name}' changes system state (packages, services, processes or users); "
            "read-only mode does not allow it"
        )
    if deletes(name, args):
        return _stop(
            f"'{name}' deletes files here; read-only mode allows only commands that leave files "
            "untouched"
        )
    if name == "sed" and _inplace(args):
        return _stop(
            "sed -i rewrites the file in place; read-only mode does not allow it "
            "(drop -i to print to stdout instead)"
        )
    if name == "git":
        verb = _git_verb_of(args)
        if verb is not None and verb not in _GIT_READ_VERBS:
            return _stop(
                f"'git {verb}' changes the repository; read-only mode allows only git commands "
                "that read"
            )
    if name in _SHELLS and "-c" in args:
        at = args.index("-c") + 1
        return _review_read_only(args[at], depth + 1) if at < len(args) else _PASS
    if name == "eval":
        return _review_read_only(" ".join(args), depth + 1)
    if name == "find":
        for i, word in enumerate(args[:-1]):
            if word in _FIND_EXEC:
                verdict = _review_simple(os.path.basename(args[i + 1]), args[i + 2 :], depth + 1)
                if verdict.kind != "allow":
                    return verdict
    if name in _EXEC_WRAPPERS or name.startswith("-"):
        for i, word in enumerate(args):
            if word.startswith("-"):
                continue
            base = os.path.basename(word)
            verdict = _review_simple(base, args[i + 1 :], depth + 1)
            if verdict.kind != "allow" or base in _RUNNERS:
                return verdict
    return _PASS


def hazard_warning(command: str) -> Verdict:
    """誰が見ても危ない形（root の削除・ディスクへの直書き・fork bomb …）を確認に回す。"""
    for hazard in _HAZARDS:
        if hazard.trigger in command:
            return _flag(f"This command {hazard.effect}. Check the target before allowing it.")
    first = leading_command(command)
    profile = _PROFILES.get(first)
    if profile is not None and profile.kind is CommandKind.DESTROY and first != "rm":
        return _flag(f"'{first}' destroys data by design; there is no undo once it runs.")
    if _is_recursive_forced_rm(command):
        return _flag(
            "rm with both -r and -f removes a whole tree without asking. "
            "Check that the path is the one you mean."
        )
    return _PASS


def _is_recursive_forced_rm(command: str) -> bool:
    """`rm` に再帰（-r / -R）と強制（-f）の旗が別々に付いているか。"""
    words = command.split()
    if "rm" not in words:
        return False
    flags = [
        w for w in words[words.index("rm") + 1 :] if w.startswith("-") and not w.startswith("--")
    ]
    recursive = any(w.startswith(("-r", "-R")) for w in flags)
    forced = any(w.startswith("-f") for w in flags)
    return recursive and forced


def review_for_mode(command: str, mode: PermissionMode) -> Verdict:
    """モードごとの検査。"""
    if mode is PermissionMode.READ_ONLY:
        return review_read_only(command, mode)
    if mode is PermissionMode.WORKSPACE_WRITE:
        profile = _profile(command)
        if profile is not None and profile.mutates and any(p in command for p in _SYSTEM_PREFIXES):
            first = leading_command(command)
            return _flag(
                f"'{first}' targets a system path outside the workspace; this needs permission "
                "beyond workspace-write"
            )
    return _PASS


def review_sed(command: str, mode: PermissionMode) -> Verdict:
    """read-only モードで `sed -i`（その場で書き換える）を止める。"""
    if (
        mode is PermissionMode.READ_ONLY
        and leading_command(command) == "sed"
        and _has_inplace_flag(command)
    ):
        return _stop(
            "sed -i rewrites the file in place; read-only mode does not allow it "
            "(drop -i to print to stdout instead)"
        )
    return _PASS


def review_paths(command: str, workspace: Path) -> Verdict:
    """作業ディレクトリの外を指す手掛かり（`../` / `~/` / `$HOME`）を確認に回す。"""
    if "../" in command and str(workspace) not in command:
        return _flag(
            "the path climbs out of the current directory with '../'; make sure it still "
            "resolves inside the workspace"
        )
    if "~/" in command or "$HOME" in command:
        return _flag(
            "the command points at the home directory ('~/' or $HOME); make sure it stays "
            "inside the workspace scope"
        )
    return _PASS


def command_kind(command: str) -> CommandKind:
    """先頭のコマンド語から分類する（警告と診断のため）。"""
    first = leading_command(command)
    if first == "sed":
        return CommandKind.MODIFY if _has_inplace_flag(command) else CommandKind.READ
    if first == "git":
        verb = _git_verb(command)
        return CommandKind.READ if verb is None or verb in _GIT_READ_VERBS else CommandKind.MODIFY
    profile = _PROFILES.get(first)
    return profile.kind if profile is not None else CommandKind.UNKNOWN


def review_command(command: str, mode: PermissionMode, workspace: Path) -> Verdict:
    """全部の検査を順に掛け、最初に通らなかった結果を返す（全部通れば allow）。"""
    for verdict in (
        review_for_mode(command, mode),
        review_sed(command, mode),
        hazard_warning(command),
        review_paths(command, workspace),
    ):
        if verdict.kind != "allow":
            return verdict
    return _PASS
