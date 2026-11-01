"""JSONL 読み書き・原子的保存。"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from typing import Any


def read(path: Path) -> list[dict[str, Any]]:
    """JSONL を 1 行 1 dict として読み込む。"""
    lines: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            lines.append(json.loads(raw))
    return lines


def write_atomic(path: Path, lines: list[dict[str, Any]]) -> None:
    """JSONL を原子的に書き出す。"""
    payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
    _replace_atomic(path, payload + "\n")


def write_text_atomic(path: Path, text: str) -> None:
    """任意テキストを原子的に書き出す（write_file/edit_file 用）。"""
    _replace_atomic(path, text)


def _replace_atomic(path: Path, text: str) -> None:
    """``text`` を一時ファイルに書いてから ``path`` に置き換える。"""
    if path.is_dir():
        raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
