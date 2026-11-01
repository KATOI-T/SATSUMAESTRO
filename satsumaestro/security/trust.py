"""信頼したディレクトリの記録（``~/.satsuma/trusted.json``）。"""

from __future__ import annotations

import json
from pathlib import Path

from satsumaestro.util.jsonl import write_text_atomic
from satsumaestro.util.paths import default_config_home, find_git_root, is_within

TRUST_FILE = "trusted.json"
_VERSION = 1


def trust_root(cwd: Path) -> Path:
    """``cwd`` を信頼するときに記録する根: git の根、無ければ ``cwd`` そのもの。"""
    return find_git_root(cwd) or cwd.resolve()


class TrustStore:
    """``trusted.json`` の読み書き。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else default_config_home() / TRUST_FILE

    def roots(self) -> list[Path]:
        """記録した根。"""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        raw = data.get("roots") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        return [Path(r) for r in raw if isinstance(r, str) and Path(r).is_absolute()]

    def is_trusted(self, path: Path) -> bool:
        """``path`` が記録した根のどれかか、その配下か。"""
        return any(is_within(path, root) for root in self.roots())

    def trust(self, root: Path) -> None:
        """``root`` を記録に足す（既にあれば何もしない）。"""
        roots = self.roots()
        root = root.resolve()
        if root not in roots:
            self._write([*roots, root])

    def untrust(self, root: Path) -> bool:
        """``root`` を記録から外す。"""
        roots = self.roots()
        root = root.resolve()
        if root not in roots:
            return False
        self._write([r for r in roots if r != root])
        return True

    def _write(self, roots: list[Path]) -> None:
        body = {"version": _VERSION, "roots": [str(r) for r in roots]}
        write_text_atomic(self.path, json.dumps(body, ensure_ascii=False, indent=2) + "\n")
