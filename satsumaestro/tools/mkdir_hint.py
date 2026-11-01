"""書き込みの前の ``mkdir`` を、確認カードに回す前に「要らない」と返す。"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

FLAG_MKDIR_SKIPPED = "mkdir-skipped"

MKDIR_NOT_NEEDED = (
    "Not run: this mkdir is not needed. write_file and append_file create any missing parent "
    "directories themselves, so write the file directly (for example write_file with path "
    "`{example}`). If you really need an empty directory, run the same mkdir again and the "
    "user will be asked to approve it."
)

_FLAGS = frozenset({"-p", "-v", "-pv", "-vp", "--parents", "--verbose"})


def pure_mkdir_targets(command: str, workspace: Path) -> list[Path] | None:
    """``command`` が作業ディレクトリの中へ ``mkdir`` するだけなら、作る場所を返す。"""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    root = workspace.resolve()
    here = root
    if len(tokens) >= 3 and tokens[0] == "cd" and tokens[2] == "&&":
        moved = _inside(tokens[1], root, root)
        if moved is None:
            return None
        here = moved
        tokens = tokens[3:]
    if len(tokens) < 2 or tokens[0] != "mkdir":
        return None
    targets: list[Path] = []
    for word in tokens[1:]:
        if word.startswith("-"):
            if word not in _FLAGS:
                return None
            continue
        target = _inside(word, here, root)
        if target is None or target == root:
            return None
        targets.append(target)
    return targets or None


def _inside(raw: str, base: Path, root: Path) -> Path | None:
    if any(c in raw for c in "$*?[{~`;|&<>()"):
        return None
    path = Path(os.path.normpath(raw if os.path.isabs(raw) else base / raw))
    resolved = _resolve_existing(path)
    return resolved if resolved.is_relative_to(root) else None


def _resolve_existing(path: Path) -> Path:
    existing = path
    rest: list[str] = []
    while not existing.exists() and existing != existing.parent:
        rest.append(existing.name)
        existing = existing.parent
    return existing.resolve().joinpath(*reversed(rest))


def not_needed_message(targets: list[Path], workspace: Path) -> str:
    rel = targets[0].relative_to(workspace.resolve())
    return MKDIR_NOT_NEEDED.format(example=f"{rel.as_posix()}/<file>")
