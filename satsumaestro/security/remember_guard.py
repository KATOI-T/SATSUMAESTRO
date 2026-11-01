"""永続許可（``a``）できない操作の判定。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from satsumaestro.i18n.text import ui
from satsumaestro.security.bash_pipeline import GIT_OPTIONS_WITH_VALUE, hazard_warning
from satsumaestro.security.deletion_guard import deletes, simple_commands
from satsumaestro.security.state_write_guard import AGENT_CONFIG_FILES, tool_path_field

DELETION = "deletion"
PRIVILEGE = "privilege"
DESTRUCTIVE = "destructive"
UNPARSABLE = "unparsable"
AGENT_CONFIG = "agent-config"
GIT_PREFIX = "git:"

_PRIVILEGE_COMMANDS = frozenset({"sudo", "doas", "su"})
_DESTRUCTIVE_COMMANDS = frozenset({"shred", "wipefs", "dd"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
_EXEC_WRAPPERS = frozenset(
    {"env", "timeout", "xargs", "nice", "nohup", "time", "command", "exec", "builtin", "watch"}
)
_FIND_EXEC = frozenset({"-exec", "-execdir", "-ok", "-okdir"})
_SUBSTITUTION = re.compile(r"[$<>]\(([^()]*)\)|`([^`]*)`")


def blocker_note(blocker: str) -> str:
    """権限カードに添える 1 行（``a`` を出さない理由）。"""
    if blocker == DELETION:
        return ui("security.remember_guard.blocked_deletion")
    if blocker == PRIVILEGE:
        return ui("security.remember_guard.blocked_privilege")
    if blocker == DESTRUCTIVE:
        return ui("security.remember_guard.blocked_destructive")
    if blocker == AGENT_CONFIG:
        return ui("security.remember_guard.blocked_agent_config")
    if blocker.startswith(GIT_PREFIX):
        sub = blocker.removeprefix(GIT_PREFIX)
        return ui("security.remember_guard.blocked_git", sub=sub)
    return ui("security.remember_guard.this_cannot_be_remembered_the")


def remember_blocker(tool_name: str, tool_input: Mapping[str, object]) -> str | None:
    """永続許可できない呼び出しなら理由の種類を、できれば ``None`` を返す。"""
    if tool_name == "delete_path":
        return DELETION
    if tool_name == "bash":
        command = tool_input.get("command")
        if not isinstance(command, str):
            return None
        if _mentions_config_file(command):
            return AGENT_CONFIG
        return _command_blocker(command)
    field = tool_path_field(tool_name)
    if field is not None and _is_config_name(str(tool_input.get(field) or "")):
        return AGENT_CONFIG
    return None


def _is_config_name(path: str) -> bool:
    return os.path.basename(path.rstrip("/")).lower() in AGENT_CONFIG_FILES


def _mentions_config_file(command: str) -> bool:
    return any(_is_config_name(word) for word in re.split(r"[\s'\"<>|;&()=]+", command) if word)


def _command_blocker(command: str) -> str | None:
    commands = simple_commands(command)
    if commands is None:
        return UNPARSABLE
    for privileged, name, args in commands:
        if privileged:
            return PRIVILEGE
        reason = _simple_blocker(name, args)
        if reason is not None:
            return reason
    for match in _SUBSTITUTION.finditer(command):
        reason = _command_blocker(match.group(1) or match.group(2) or "")
        if reason is not None:
            return reason
    if hazard_warning(command).kind == "warn":
        return DESTRUCTIVE
    return None


def _simple_blocker(name: str, args: list[str]) -> str | None:
    """単純コマンド 1 つの判定（包みの内側へ再帰する）。"""
    if name in _PRIVILEGE_COMMANDS:
        return PRIVILEGE
    if deletes(name, args):
        return DELETION
    if name in _DESTRUCTIVE_COMMANDS or name.startswith("mkfs"):
        return DESTRUCTIVE
    if name == "git":
        return _git_blocker(args)
    if name in _SHELLS and "-c" in args:
        index = args.index("-c") + 1
        return _command_blocker(args[index]) if index < len(args) else None
    if name == "eval":
        return _command_blocker(" ".join(args))
    if name == "find":
        for i, word in enumerate(args[:-1]):
            if word in _FIND_EXEC:
                reason = _simple_blocker(os.path.basename(args[i + 1]), args[i + 2 :])
                if reason is not None:
                    return reason
        return None
    if name in _EXEC_WRAPPERS:
        for i, word in enumerate(args):
            reason = _simple_blocker(os.path.basename(word), args[i + 1 :])
            if reason is not None:
                return reason
    return None


def _git_blocker(args: list[str]) -> str | None:
    """取り消せない git 操作か。"""
    sub, rest = _git_subcommand(args)
    if sub is None:
        return None
    if sub == "rm":
        return None if "--cached" in rest else DELETION
    if _git_irreversible(sub, rest):
        return f"{GIT_PREFIX}{sub}"
    return None


def _git_subcommand(args: list[str]) -> tuple[str | None, list[str]]:
    """グローバルフラグ（``-C path`` などは値ごと）を飛ばした最初の語と、その後ろ。"""
    i = 0
    while i < len(args):
        if not args[i].startswith("-"):
            return args[i], args[i + 1 :]
        i += 2 if args[i] in GIT_OPTIONS_WITH_VALUE else 1
    return None, []


def _git_irreversible(sub: str, rest: list[str]) -> bool:
    def flag(long: str, short: str | None = None) -> bool:
        return long in rest or (short is not None and _short(rest, short))

    positional = [tok for tok in rest if not tok.startswith("-")]
    if sub == "push":
        return (
            any(tok.startswith("--force") for tok in rest)
            or _short(rest, "f")
            or flag("--delete", "d")
            or flag("--mirror")
            or flag("--prune")
            or any(spec.startswith(("+", ":")) for spec in positional)
        )
    if sub == "reset":
        return flag("--hard")
    if sub == "clean":
        return flag("--force", "f") and not flag("--dry-run", "n")
    if sub == "branch":
        return _short(rest, "D") or (flag("--delete", "d") and flag("--force", "f"))
    if sub == "checkout":
        return "--" in rest or "." in rest or flag("--force", "f")
    if sub == "switch":
        return flag("--discard-changes") or flag("--force", "f")
    if sub == "restore":
        return flag("--worktree", "W") or not flag("--staged", "S")
    if sub == "stash":
        return bool(positional) and positional[0] in ("drop", "clear")
    return False


def _short(rest: list[str], letter: str) -> bool:
    """短いフラグの束（``-fd`` など）に ``letter`` が入っているか。"""
    return any(
        tok.startswith("-") and not tok.startswith("--") and letter in tok[1:] for tok in rest
    )
