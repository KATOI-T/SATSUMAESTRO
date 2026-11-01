"""ワークスペース jail。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from satsumaestro.util.errors import PathEscapeError, StateDirError
from satsumaestro.util.paths import AGENT_DIRNAME, in_state_dir

PATH_ESCAPE_PREFIX = "path escape: "
OUTSIDE_WORKSPACE = "not under the workspace:"
STATE_DIR_HIDDEN = (
    f"{AGENT_DIRNAME}/ holds this agent's own state (past sessions, plans, settings) "
    "and is not accessible:"
)
STATE_DIR_UNDELETABLE = f"{AGENT_DIRNAME}/ holds this agent's own state and is not deletable:"


class WorkspaceJail:
    """ファイル操作を ``root`` 配下に限定する jail。"""

    def __init__(self, root: Path, *, visible_state: Iterable[str | Path] = ()) -> None:
        self.root = root.resolve()
        self._visible_state = frozenset(self._absolute(v) for v in visible_state)

    def _absolute(self, p: str | Path) -> Path:
        candidate = Path(p)
        try:
            if candidate.is_absolute():
                return candidate.resolve()
            return (self.root / candidate).resolve()
        except ValueError as e:
            raise PathEscapeError(f"path cannot be resolved: {p!r} ({e})") from e

    def hides(self, p: str | Path) -> bool:
        """``p`` がエージェントの状態ディレクトリとして隠す対象か。"""
        try:
            cand = self._absolute(p)
        except PathEscapeError:
            return False
        if not cand.is_relative_to(self.root):
            return False
        return in_state_dir(cand.relative_to(self.root)) and cand not in self._visible_state

    def resolve(self, p: str | Path, *, allow_hidden: bool = False) -> Path:
        """``p`` を絶対パスへ解決し、jail 内であることを保証して返す。"""
        cand = self._absolute(p)
        if cand != self.root and not cand.is_relative_to(self.root):
            raise PathEscapeError(
                f"{OUTSIDE_WORKSPACE} {p} (the workspace is {self.root}). If the user asked for "
                "this location, tell them it is outside the workspace (satsumaestro has to be "
                "started in that directory, or with --cwd) instead of writing somewhere else."
            )
        if not allow_hidden and self.hides(cand):
            raise StateDirError(f"{STATE_DIR_HIDDEN} {p}")
        return cand

    def is_inside(self, p: str | Path) -> bool:
        """``p`` が jail 内なら ``True``。"""
        try:
            self.resolve(p)
        except PathEscapeError:
            return False
        return True
