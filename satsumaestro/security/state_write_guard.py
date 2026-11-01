"""エージェント自身の状態ディレクトリへの書き込みの絶対禁止。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from satsumaestro.security.auto_allow import (
    _AUTO_ALLOW_COMMANDS,
    _RULES,
    _git_candidates,
    split_command_segments,
)
from satsumaestro.security.deletion_guard import (
    _cd,
    _expand,
    _in_agent_state,
    _tokenize,
    _unwrap,
)
from satsumaestro.tools.mkdir_hint import pure_mkdir_targets
from satsumaestro.util.paths import AGENT_DIRNAME

PROHIBITED_STATE_WRITE_PREFIX = "write prohibited:"

PROHIBITED_STATE_WRITE_GUIDANCE = (
    f"{AGENT_DIRNAME}/ holds this agent's own settings, permission policy and session records. No "
    "tool or command can change it, regardless of permissions or user approval. Do not retry "
    "with another command, flag, or path spelling. If the user wants a setting changed, tell "
    "them which file to edit themselves."
)

PLAN_DIR_WRITE_GUIDANCE = (
    "{plan_dir}/ only takes this conversation's plan file, `{plan_file}`. The path above is not "
    "that file (compare the file name character by character). Write the plan to `{plan_file}`."
)

AGENT_CONFIG_FILES = frozenset({f"{AGENT_DIRNAME}.json", ".env"})

_CANNOT_WRITE = frozenset({"grep", "egrep", "fgrep"})

_WRITE_TOOL_PATH_FIELDS = {
    "write_file": "path",
    "edit_file": "path",
    "append_file": "path",
    "notebook_edit": "notebook_path",
}


def check_tool(name: str, path: str, cwd: Path, *, visible: Iterable[Path] = ()) -> str | None:
    """ファイル書き込みツールの対象が状態ディレクトリなら理由を返す。"""
    if name not in _WRITE_TOOL_PATH_FIELDS or not path:
        return None
    workspace = cwd.resolve()
    for target in _expand(path, workspace):
        if _guarded(target, workspace, visible):
            return _reason(target)
    return None


def plan_dir_guidance(name: str, path: str, cwd: Path, plan_file: str) -> str | None:
    """書き込みツールの対象が計画ファイルの置き場の中なら、:data:`PLAN_DIR_WRITE_GUIDANCE`。"""
    if name not in _WRITE_TOOL_PATH_FIELDS or not path:
        return None
    workspace = cwd.resolve()
    plan_dir = Path(plan_file).parent
    resolved = (workspace / plan_dir).resolve()
    if not any(target.parent == resolved for target in _expand(path, workspace)):
        return None
    return PLAN_DIR_WRITE_GUIDANCE.format(plan_dir=plan_dir.as_posix(), plan_file=plan_file)


def tool_path_field(name: str) -> str | None:
    """書き込みツールのパス入力の名前（対象外のツールは ``None``）。"""
    return _WRITE_TOOL_PATH_FIELDS.get(name)


def check_bash(command: str, cwd: Path, *, visible: Iterable[Path] = ()) -> str | None:
    """bash が状態ディレクトリを書き換え得るなら理由を返す。"""
    workspace = cwd.resolve()
    touched = [t for t in _mentioned_paths(command, workspace) if _guarded(t, workspace, visible)]
    if not touched or _read_only(command, workspace):
        return None
    made = pure_mkdir_targets(command, workspace)
    if made is not None and all(t.is_dir() for t in made):
        return None
    return _reason(touched[0])


def config_file_tool(name: str, path: str, cwd: Path) -> Path | None:
    """ファイル書き込みツールの対象がエージェントの設定ファイルなら、そのパス。"""
    if name not in _WRITE_TOOL_PATH_FIELDS or not path:
        return None
    return next((t for t in _expand(path, cwd.resolve()) if is_config_file(t)), None)


def config_file_bash(command: str, cwd: Path) -> Path | None:
    """bash がエージェントの設定ファイルを書き換え得るなら、そのパス。"""
    workspace = cwd.resolve()
    touched = [t for t in _mentioned_paths(command, workspace) if is_config_file(t)]
    if not touched or _read_only(command, workspace):
        return None
    return touched[0]


def is_config_file(target: Path) -> bool:
    """エージェントの設定ファイル（リンクならリンク先も見る）か。"""
    names = {target.name.lower(), target.resolve().name.lower()}
    return bool(names & AGENT_CONFIG_FILES)


def _reason(target: Path) -> str:
    return (
        f"{PROHIBITED_STATE_WRITE_PREFIX} the agent's state directory ({AGENT_DIRNAME}/) "
        f"cannot be changed: {target}"
    )


def _guarded(target: Path, workspace: Path, visible: Iterable[Path]) -> bool:
    return _in_agent_state(target, workspace) and target not in {v.resolve() for v in visible}


def _mentioned_paths(command: str, workspace: Path) -> list[Path]:
    """コマンドに出てくるパスらしい語（引数・リダイレクト先・``of=`` の右辺）を絶対パスにする。"""
    tokens = _tokenize(command)
    if tokens is None:
        return []
    paths: list[Path] = []
    here = workspace
    words: list[str] = []

    def flush() -> None:
        nonlocal here
        if not words:
            return
        _, name, args = _unwrap(words)
        for arg in args:
            raw = arg.split("=", 1)[1] if "=" in arg and not arg.startswith("/") else arg
            if raw and not raw.startswith("-"):
                paths.extend(_expand(raw, here))
        if name == "cd":
            here = _cd(args, here)
        words.clear()

    redirect_next = False
    for tok in tokens:
        if tok and all(c in "();<>|&" for c in tok):
            if "<" in tok or ">" in tok:
                redirect_next = not tok.endswith("&")
                continue
            flush()
            continue
        if redirect_next:
            redirect_next = False
            paths.extend(_expand(tok, here))
            continue
        words.append(tok)
    flush()
    return paths


def _read_only(command: str, workspace: Path) -> bool:
    """bash 自動許可と同じ語彙・規則で、読み取り専用と言えるか（パスの検査はしない）。"""
    segments = split_command_segments(command)
    if isinstance(segments, str):
        return False
    for seg in segments:
        head = seg.words[0]
        if "/" in head or "=" in head or head not in _AUTO_ALLOW_COMMANDS:
            return False
        args = seg.words[1:]
        rule = None if head in _CANNOT_WRITE else _RULES.get(head)
        if rule is not None and rule.violation(args, workspace) is not None:
            return False
        if head == "git" and _git_candidates(args)[1] is not None:
            return False
    return True
