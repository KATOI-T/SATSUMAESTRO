"""削除の絶対禁止。"""

from __future__ import annotations

import glob
import os
import shlex
import stat
from collections.abc import Iterator
from pathlib import Path

from satsumaestro.util.paths import AGENT_DIRNAME, default_config_home, in_state_dir

PROHIBITED_DELETION_PREFIX = "deletion prohibited:"

PROHIBITED_DELETION_GUIDANCE = (
    "This deletion is prohibited by SATSUMAESTRO and cannot be performed by any tool, "
    "regardless of permissions or user approval. Do not retry it or rephrase it with another "
    "command, flag, or path spelling. Tell the user it cannot be done here; if it is really "
    "needed, the user must do it themselves."
)

_DELETE_COMMANDS = frozenset({"rm", "rmdir", "unlink"})
_PRIVILEGE_COMMANDS = frozenset({"sudo", "doas"})
_WRAPPERS = frozenset({"command", "builtin", "exec", "nice", "nohup", "time"})
_PRIVILEGE_VALUE_SHORT = frozenset("aCDghprRtTUu")
_PRIVILEGE_VALUE_LONG = frozenset(
    {
        "--chdir",
        "--chroot",
        "--close-from",
        "--command-timeout",
        "--group",
        "--host",
        "--other-user",
        "--prompt",
        "--role",
        "--type",
        "--user",
    }
)
_GLOB_CHARS = frozenset("*?[")


def check_path(path: str, cwd: Path) -> str | None:
    """``path``（作業ディレクトリ基準）の削除が禁止なら理由（1 行目の文言）を返す。"""
    target = _target(path, cwd)
    return None if target is None else _forbidden(target, cwd.resolve())


def check_bash(command: str, cwd: Path) -> str | None:
    """bash コマンド中の削除が禁止なら理由を返す。"""
    workspace = cwd.resolve()
    for privileged, name, targets in _deletions(command, workspace):
        for target in targets:
            reason = _forbidden(target, workspace, allow_workspace=name == "find")
            if reason is not None:
                return reason
            if privileged:
                return f"{PROHIBITED_DELETION_PREFIX} cannot delete with sudo ({target})"
        if privileged and not targets:
            return f"{PROHIBITED_DELETION_PREFIX} cannot delete with sudo"
    return None


def path_targets(path: str, cwd: Path) -> list[Path]:
    """``delete_path`` の削除対象（絶対パス）。"""
    target = _target(path, cwd.resolve())
    return [] if target is None else [target]


def bash_targets(command: str, cwd: Path) -> list[Path]:
    """bash コマンドが消すパス（絶対パス）。"""
    return [t for _, _, targets in _deletions(command, cwd.resolve()) for t in targets]


def simple_commands(command: str) -> list[tuple[bool, str, list[str]]] | None:
    """区切り記号で分けた単純コマンドごとの (特権付きか, コマンド名, 引数)。"""
    tokens = _tokenize(command)
    if tokens is None:
        return None
    return [_unwrap(words) for words in _segments(tokens)]


def deletes(name: str, args: list[str]) -> bool:
    """単純コマンド ``name args`` がファイルを消すか（``rm`` / ``find -delete`` など）。"""
    return _deletion_targets(name, args) is not None


def _deletions(command: str, workspace: Path) -> Iterator[tuple[bool, str, list[Path]]]:
    """削除コマンドごとに (特権付きか, コマンド名, 対象の絶対パス) を返す。"""
    commands = simple_commands(command)
    if commands is None:
        return
    here = workspace
    for privileged, name, args in commands:
        if name == "cd":
            here = _cd(args, here)
            continue
        paths = _deletion_targets(name, args)
        if paths is None:
            continue
        yield privileged, name, [t for raw in paths for t in _expand(raw, here)]


def _forbidden(target: Path, workspace: Path, *, allow_workspace: bool = False) -> str | None:
    root = Path(target.anchor or "/")
    if target == root or target.parent == root:
        return (
            f"{PROHIBITED_DELETION_PREFIX} cannot delete a path directly under the filesystem "
            f"root ({target})"
        )
    if (target == workspace and not allow_workspace) or (
        target != workspace and workspace.is_relative_to(target)
    ):
        return (
            f"{PROHIBITED_DELETION_PREFIX} cannot delete the working directory itself or a "
            f"directory above it ({target})"
        )
    if _in_agent_state(target, workspace):
        return (
            f"{PROHIBITED_DELETION_PREFIX} cannot delete the agent's state directory "
            f"({AGENT_DIRNAME}/): {target}"
        )
    if _needs_privilege(target):
        return f"{PROHIBITED_DELETION_PREFIX} cannot delete a location that needs sudo ({target})"
    return None


def _in_agent_state(target: Path, workspace: Path) -> bool:
    """作業ディレクトリの ``.satsuma`` か ``~/.satsuma``（とその中身）か。"""
    if target.is_relative_to(workspace) and in_state_dir(target.relative_to(workspace)):
        return True
    home = default_config_home().expanduser()
    home = home.parent.resolve() / home.name if home.name else home.resolve()
    if workspace.is_relative_to(home):
        return False
    return target == home or target.is_relative_to(home)


def _needs_privilege(target: Path) -> bool:
    """現在のユーザーでは消せない（削除に root 権限が要る）か。"""
    try:
        info = target.lstat()
        parent = target.parent.stat()
    except (OSError, ValueError):
        return False
    if not os.access(target.parent, os.W_OK | os.X_OK):
        return True
    uid = os.geteuid()
    return bool(parent.st_mode & stat.S_ISVTX) and uid not in (info.st_uid, parent.st_uid)


def _target(path: str, base: Path) -> Path | None:
    """削除対象の絶対パス。"""
    if not path:
        return None
    try:
        expanded = os.path.expanduser(path.replace("${HOME}", "~").replace("$HOME", "~"))
        if "$" in expanded:
            return None
        if "\x00" in expanded:
            return None
        joined = Path(
            os.path.normpath(Path(expanded) if os.path.isabs(expanded) else base / expanded)
        )
        if joined.name in ("", ".", ".."):
            return joined.resolve()
        return joined.parent.resolve() / joined.name
    except ValueError:
        return None


def _expand(raw: str, base: Path) -> list[Path]:
    if any(c in raw for c in _GLOB_CHARS):
        try:
            pattern = os.path.expanduser(raw)
            matches = glob.glob(pattern if os.path.isabs(pattern) else str(base / pattern))
        except ValueError:
            matches = []
        if matches:
            return [t for m in matches if (t := _target(m, base)) is not None]
    target = _target(raw, base)
    return [] if target is None else [target]


def _tokenize(command: str) -> list[str] | None:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return None


def _segments(tokens: list[str]) -> list[list[str]]:
    """区切り記号（``;`` ``&&`` ``|`` など）で単純コマンドへ分ける。"""
    segments: list[list[str]] = [[]]
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok and all(c in "();<>|&" for c in tok):
            if "<" in tok or ">" in tok:
                skip_next = not tok.endswith("&")
                continue
            segments.append([])
            continue
        segments[-1].append(tok)
    return [s for s in segments if s]


def _unwrap(words: list[str]) -> tuple[bool, str, list[str]]:
    """``FOO=1`` / ``sudo -u x`` / ``command`` などを剥がし、(特権付きか, 名前, 引数) を返す。"""
    privileged = False
    i = 0
    while i < len(words):
        word = words[i]
        name = os.path.basename(word)
        if "=" in word and not word.startswith("-") and name == word:
            i += 1
        elif name in _PRIVILEGE_COMMANDS:
            privileged = True
            i += 1
            while i < len(words) and words[i].startswith("-"):
                word = words[i]
                if word == "--":
                    i += 1
                    break
                if _privilege_edits(word):
                    return True, "sudoedit", words[i + 1 :]
                i += _privilege_option_width(word)
        elif name in _WRAPPERS:
            i += 1
        else:
            return privileged, name, words[i + 1 :]
    return privileged, "", []


def _privilege_option_width(word: str) -> int:
    """sudo / doas のオプション 1 つが占める語の数（値を次の語で取るなら 2）。"""
    if word.startswith("--"):
        return 2 if word in _PRIVILEGE_VALUE_LONG else 1
    for at, ch in enumerate(word[1:], start=1):
        if ch in _PRIVILEGE_VALUE_SHORT:
            return 2 if at == len(word) - 1 else 1
    return 1


def _privilege_edits(word: str) -> bool:
    """sudo の ``-e`` / ``--edit``（束ねた形も。値を取る文字より後ろは値なので見ない）。"""
    if word.startswith("--"):
        return word == "--edit"
    for ch in word[1:]:
        if ch == "e":
            return True
        if ch in _PRIVILEGE_VALUE_SHORT:
            return False
    return False


def _deletion_targets(name: str, args: list[str]) -> list[str] | None:
    """削除コマンドなら対象パスの一覧、削除コマンドでなければ ``None``。"""
    if name in _DELETE_COMMANDS:
        paths: list[str] = []
        after_dashdash = False
        for tok in args:
            if after_dashdash or not tok.startswith("-") or tok == "-":
                paths.append(tok)
            elif tok == "--":
                after_dashdash = True
        return paths
    if name == "find" and ("-delete" in args or _exec_deletes(args)):
        roots: list[str] = []
        for tok in args:
            if tok.startswith("-") or tok in ("(", "!", ")"):
                break
            roots.append(tok)
        return roots or ["."]
    return None


def _exec_deletes(args: list[str]) -> bool:
    """``find … -exec rm …`` / ``-execdir rm …`` か。"""
    return any(
        tok in ("-exec", "-execdir")
        and i + 1 < len(args)
        and os.path.basename(args[i + 1]) in _DELETE_COMMANDS
        for i, tok in enumerate(args)
    )


def _cd(args: list[str], here: Path) -> Path:
    targets = [a for a in args if not a.startswith("-")]
    if len(targets) != 1:
        return Path(os.path.expanduser("~")).resolve() if not targets else here
    target = _target(targets[0], here)
    if target is None:
        return here
    try:
        return target.resolve()
    except ValueError:
        return here
