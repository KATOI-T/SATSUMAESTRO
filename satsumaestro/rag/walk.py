"""インジェスト対象の走査。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pathspec

from satsumaestro.util.paths import AGENT_DIRNAME

_ALWAYS_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        AGENT_DIRNAME,
    }
)
_LARGEST_INDEXED_FILE = 1_000_000
_BINARY_SNIFF_BYTES = 8_000


def load_ignore_spec(root: Path) -> pathspec.PathSpec[pathspec.Pattern]:
    """``root`` 直下の ``.gitignore`` を読み ``PathSpec`` を返す（無ければ空）。"""
    lines: list[str] = []
    gitignore = root / ".gitignore"
    if gitignore.is_file():
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
    return pathspec.PathSpec.from_lines("gitignore", lines)


def is_ignored(spec: pathspec.PathSpec[pathspec.Pattern], rel_posix: str, *, is_dir: bool) -> bool:
    """``rel_posix``（root 相対・posix 区切り）が索引・一覧から省く対象か。"""
    name = rel_posix.rstrip("/").rsplit("/", 1)[-1]
    if is_dir and name in _ALWAYS_SKIP_DIRS:
        return True
    if is_dir and not rel_posix.endswith("/"):
        rel_posix += "/"
    return bool(spec.match_file(rel_posix))


def iter_source_files(root: Path, globs: list[str] | None = None) -> Iterator[Path]:
    """索引対象のファイルを決定順（ソート済み）で列挙する。"""
    spec = load_ignore_spec(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part in _ALWAYS_SKIP_DIRS for part in rel.parts):
            continue
        if spec.match_file(rel.as_posix()):
            continue
        if globs and not any(rel.match(g) for g in globs):
            continue
        if not _is_indexable(path):
            continue
        yield path


def _is_indexable(path: Path) -> bool:
    """サイズ上限とバイナリ判定（NUL バイト混入）でふるいにかける。"""
    try:
        if path.stat().st_size > _LARGEST_INDEXED_FILE:
            return False
        with path.open("rb") as f:
            head = f.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return False
    return b"\x00" not in head
