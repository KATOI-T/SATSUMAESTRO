"""ワークスペースルート探索・セッションディレクトリ算出。"""

from __future__ import annotations

import os
from pathlib import Path, PurePath

from satsumaestro.util.environ import ENVIRON

AGENT_DIRNAME = ".satsuma"


def names_component(relative: PurePath, name: str, root: Path | None = None) -> bool:
    """``relative`` の成分が ``name`` か（大小は区別せず、``root`` があれば同じ実体かも見る）。"""
    folded = name.casefold()
    if any(part.casefold() == folded for part in relative.parts):
        return True
    if root is None:
        return False
    here = str(root)
    for part in relative.parts:
        sibling = os.path.join(here, name)
        here = os.path.join(here, part)
        try:
            # たいていの親には ``name`` が無いので、先にそちらを見て 1 回の stat で次へ進む
            target = os.stat(sibling)
            if os.path.samestat(os.stat(here), target):
                return True
        except (OSError, ValueError):
            continue
    return False


def inside_entry(relative: PurePath, entry: Path, root: Path) -> bool:
    """``root/relative`` が ``entry``（リンクは先を辿る）と同じ実体か、その中か。"""
    try:
        target = os.stat(entry)
    except (OSError, ValueError):
        return False
    here = str(root)
    for part in relative.parts:
        here = os.path.join(here, part)
        try:
            if os.path.samestat(os.stat(here), target):
                return True
        except (OSError, ValueError):
            return False
    return False


def in_state_dir(relative: Path, root: Path | None = None) -> bool:
    """ワークスペース相対パスが状態ディレクトリ配下（またはそれ自体）か。"""
    if names_component(relative, AGENT_DIRNAME, root):
        return True
    if root is None or not (root / AGENT_DIRNAME).is_symlink():
        return False
    return inside_entry(relative, root / AGENT_DIRNAME, root)


def resolve_from(path: str, root: Path) -> Path | None:
    """``path`` を ``root`` 起点で解決した絶対パス（リンクも辿る。解決できなければ ``None``）。"""
    try:
        candidate = Path(path)
        return (candidate if candidate.is_absolute() else root / candidate).resolve()
    except (OSError, ValueError, RuntimeError):
        return None


def rule_form(path: str, root: Path) -> str:
    """policy のパスの規則と照らす形（``root`` からの相対パス。外は絶対パス・区切りは ``/``）。"""
    resolved = resolve_from(path, root)
    if resolved is None:
        return path
    if resolved == root:
        return "."
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    head = _root_written_otherwise(resolved, root)
    if head is not None:
        return resolved.relative_to(head).as_posix()
    return resolved.as_posix()


def _root_written_otherwise(resolved: Path, root: Path) -> Path | None:
    """``resolved`` の先頭が ``root`` と同じ実体（大小などの書き方だけ違う）ならその先頭。"""
    if len(resolved.parts) < len(root.parts):
        return None
    head = Path(*resolved.parts[: len(root.parts)])
    try:
        return head if os.path.samestat(os.stat(head), os.stat(root)) else None
    except (OSError, ValueError):
        return None


def default_config_home() -> Path:
    """ユーザー設定の置き場所（``SATSUMA_CONFIG_HOME`` → ``~/.satsuma``）。"""
    if ENVIRON.get("SATSUMA_CONFIG_HOME"):
        return Path(ENVIRON["SATSUMA_CONFIG_HOME"])
    return Path.home() / AGENT_DIRNAME


def is_within(path: str | Path, root: str | Path) -> bool:
    """``path`` が ``root`` そのものか、その下にあるか。"""
    return _resolved(path).is_relative_to(_resolved(root))


def _resolved(path: str | Path) -> Path:
    p = Path(path).expanduser()
    try:
        return p.resolve()
    except OSError:
        return p


def find_git_root(start: Path) -> Path | None:
    """``start`` から親を辿り、``.git`` を持つ最初のディレクトリを返す。"""
    start = start.resolve()
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return None


def find_workspace_root(start: Path) -> Path:
    """``start`` を起点にワークスペースルートを探索して返す。"""
    start = start.resolve()
    git_root = find_git_root(start)
    if git_root is not None:
        return git_root
    config_home = _resolved(default_config_home())
    for parent in (start, *start.parents):
        state_dir = parent / AGENT_DIRNAME
        if (state_dir.is_dir() and state_dir != config_home) or (
            parent / f"{AGENT_DIRNAME}.json"
        ).is_file():
            return parent
    return start


def project_dirs(cwd: Path) -> list[Path]:
    """作業ディレクトリから git の根までのディレクトリを、近い順に返す。"""
    start = cwd.resolve()
    stop = find_git_root(start)
    if stop is None:
        return [start]
    dirs: list[Path] = []
    for parent in (start, *start.parents):
        dirs.append(parent)
        if parent == stop:
            break
    return dirs


def find_project_file(cwd: Path, relative: str | Path) -> Path | None:
    """``relative`` を作業ディレクトリから git の根まで探し、一番近いものを返す。"""
    for directory in project_dirs(cwd):
        candidate = directory / relative
        if candidate.exists():
            return candidate
    return None


_STATE_GITIGNORE = """\
# Local state written by SATSUMAESTRO. Do not commit it.
sessions/
plans/
background/
evals/
benchmarks/
rag.db
rag.db-*
settings.local.json
orchestration-state.json
capacity.json
trace.jsonl
"""


def ensure_state_gitignore(root: Path) -> None:
    """``root/.satsuma/.gitignore`` が無ければ作る。"""
    state_dir = root / AGENT_DIRNAME
    if _resolved(state_dir) == _resolved(default_config_home()):
        return
    path = state_dir / ".gitignore"
    if path.exists():
        return
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(_STATE_GITIGNORE, encoding="utf-8")
    except OSError:
        return


def sessions_dir(root: Path) -> Path:
    """セッション JSONL を格納するディレクトリを返す（実 I/O は session/store.py）。"""
    return root / AGENT_DIRNAME / "sessions"
