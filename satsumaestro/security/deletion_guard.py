"""削除の絶対禁止。"""

from __future__ import annotations

import glob
import os
import shlex
import stat
from collections.abc import Iterator
from pathlib import Path

from satsumaestro.util.paths import (
    AGENT_DIRNAME,
    default_config_home,
    in_state_dir,
    inside_entry,
    names_component,
)

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
# 削除の判定で剥がす包みと、値を次の語で取るオプション（短い形の字・長い形）
_WRAPPER_OPTIONS: dict[str, tuple[str, frozenset[str]]] = {
    "builtin": ("", frozenset()),
    "command": ("", frozenset()),
    "env": ("uCPS", frozenset({"--unset", "--chdir", "--split-string"})),
    "exec": ("a", frozenset()),
    "nice": ("n", frozenset({"--adjustment"})),
    "nohup": ("", frozenset()),
    "stdbuf": ("ioe", frozenset({"--input", "--output", "--error"})),
    "time": ("fo", frozenset({"--format", "--output"})),
    "timeout": ("sk", frozenset({"--signal", "--kill-after"})),
}
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
    tokens = _tokenize(command)
    if tokens is None:
        return
    here = workspace
    for words in _segments(tokens):
        moves: list[str] = []
        privileged, name, args = _unwrap(words, moves=moves)
        if name == "cd":
            here = _cd(args, here)
            continue
        paths = _deletion_targets(name, args)
        if paths is None:
            continue
        base = here
        for move in moves:
            base = _cd([move], base)
        yield privileged, name, [t for raw in paths for t in _expand(raw, base)]


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
    if target.is_relative_to(workspace) and in_state_dir(target.relative_to(workspace), workspace):
        return True
    home = default_config_home().expanduser()
    home = home.parent.resolve() / home.name if home.name else home.resolve()
    if _within(workspace, home):
        return False
    if _within(target, home):
        return True
    # 置き場所の名前を別の書き方（``~/.SATSUMA`` など）で書いても、同じ場所として守る
    parent = home.parent
    if not home.name or target == parent or not target.is_relative_to(parent):
        return False
    return names_component(Path(target.relative_to(parent).parts[0]), home.name, parent)


def _within(path: Path, home: Path) -> bool:
    """``path`` が ``home`` そのものかその中か（``home`` がリンクなら指す先でも見る）。"""
    if path == home or path.is_relative_to(home):
        return True
    top = Path(path.anchor)
    return home.is_symlink() and inside_entry(path.relative_to(top), home, top)


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


def _unwrap(words: list[str], *, moves: list[str] | None = None) -> tuple[bool, str, list[str]]:
    """``FOO=1`` / ``sudo -u x`` / ``command`` などを剥がし、(特権付きか, 名前, 引数) を返す。"""
    # moves を渡すと、包みのオプションも正しく飛ばし、env -C の移り先をそこへ足す
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
                i += _option_width(word, _PRIVILEGE_VALUE_SHORT, _PRIVILEGE_VALUE_LONG)
        elif moves is not None and name in _WRAPPER_OPTIONS:
            words, i = _skip_wrapper(name, words, i + 1, moves)
        elif name in _WRAPPERS:
            i += 1
        else:
            return privileged, name, words[i + 1 :]
    return privileged, "", []


def _skip_wrapper(name: str, words: list[str], i: int, moves: list[str]) -> tuple[list[str], int]:
    """包み ``name`` のオプション（timeout は時間も）を飛ばした語の並びと、中のコマンドの位置。"""
    short, long = _WRAPPER_OPTIONS[name]
    while i < len(words) and words[i].startswith("-") and (words[i] != "-" or name == "env"):
        word = words[i]
        if word == "--":
            i += 1
            break
        if name == "env":
            # -S は値を語に分けて env の引数として読み直す。-C は作業場所を移す
            split = _option_value(word, words, i, "S", "--split-string", short)
            if split is not None:
                words = [*words[:i], *(_tokenize(split[0]) or []), *words[split[1] :]]
                continue
            move = _option_value(word, words, i, "C", "--chdir", short)
            if move is not None:
                moves.append(move[0])
        i += _option_width(word, short, long)
    if name == "timeout" and i < len(words):
        i += 1
    return words, i


def _option_value(
    word: str, words: list[str], i: int, letter: str, long: str, value_letters: str
) -> tuple[str, int] | None:
    """``word`` が字 ``letter``（長い形 ``long``）のオプションなら、その値と次の語の位置。"""
    if word.startswith("--"):
        if word == long:
            return (words[i + 1], i + 2) if i + 1 < len(words) else ("", i + 1)
        if word.startswith(f"{long}="):
            return word.partition("=")[2], i + 1
        return None
    at = next((n for n, ch in enumerate(word[1:], start=1) if ch in value_letters), None)
    if at is None or word[at] != letter:
        return None
    if at < len(word) - 1:
        return word[at + 1 :], i + 1
    return (words[i + 1], i + 2) if i + 1 < len(words) else ("", i + 1)


def _option_width(word: str, short: frozenset[str] | str, long: frozenset[str]) -> int:
    """オプション 1 つが占める語の数（値を次の語で取るなら 2）。"""
    if word.startswith("--"):
        return 2 if word in long else 1
    for at, ch in enumerate(word[1:], start=1):
        if ch in short:
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
