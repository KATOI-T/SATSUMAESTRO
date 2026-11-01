"""このセッションで作った成果物を、頼まれていないのに消させない。"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from satsumaestro.i18n.text import ui

UNREQUESTED_DELETION_PREFIX = "deletion stopped:"

UNREQUESTED_DELETION_GUIDANCE = (
    "These files were created in this session as the user's deliverables, and the user has not "
    "asked to delete anything. Keep them: do not delete, move, or empty them to reset, clean "
    "up, or re-run a check, and do not retry with another command or path spelling. If you "
    "think cleanup is needed, say so in your reply and let the user decide."
)

_REQUEST_JA = re.compile(
    r"削除|消し|消す|消せ|消去|片付|片づ|捨て|除去|掃除|クリーン|リセット|元に戻|初期状態"
)
_REQUEST_EN = re.compile(
    r"\b(delete|remove|rm|clean\s*up|cleanup|clear|erase|wipe|discard|revert|undo|reset)\b",
    re.IGNORECASE,
)

_NAMES_SHOWN = 3


def requests_deletion(texts: Iterable[str]) -> bool:
    """ユーザーの発言のどれかが削除を頼んでいるか。"""
    return any(_REQUEST_JA.search(t) or _REQUEST_EN.search(t) for t in texts)


class SessionOutputs:
    """このセッションで新しく作ったファイルの記録（作った順）。"""

    def __init__(self) -> None:
        self._created: dict[Path, None] = {}

    def __len__(self) -> int:
        return len(self._created)

    def add(self, path: Path) -> None:
        self._created[path] = None

    def affected(self, targets: Iterable[Path]) -> list[Path]:
        """``targets`` を消すと失われる成果物（対象そのもの・対象ディレクトリの中）。"""
        roots = list(targets)
        return [p for p in self._created if any(p == t or p.is_relative_to(t) for t in roots)]

    def forget(self, targets: Iterable[Path]) -> None:
        """消えた成果物を記録から外す。"""
        for p in self.affected(targets):
            del self._created[p]


def blocked_message(files: list[Path], workspace: Path) -> str:
    """止めたときのツール結果（1 行目は画面向け・2 行目以降はモデル向け）。"""
    return (
        f"{UNREQUESTED_DELETION_PREFIX} {_names(files, workspace)} "
        f"{'was' if len(files) == 1 else 'were'} created in this session, and the user did not "
        f"ask to delete anything\n{UNREQUESTED_DELETION_GUIDANCE}"
    )


def card_note(files: list[Path], workspace: Path) -> str:
    """確認カードに添える 1 行（画面の言語）。"""
    names = _names(files, workspace, screen=True)
    return ui("security.output_guard.deletes_files_created_in_this", names=names)


def _names(files: list[Path], workspace: Path, *, screen: bool = False) -> str:
    """名前を ``_NAMES_SHOWN`` 件まで並べる。"""
    shown = [_label(p, workspace, screen=screen) for p in files[:_NAMES_SHOWN]]
    rest = len(files) - len(shown)
    if not screen:
        return ", ".join(shown) + (f" and {rest} more" if rest else "")
    joined = ui("security.output_guard.list_separator").join(shown)
    if not rest:
        return joined
    return joined + ui("security.output_guard.and_more", rest=rest)


def _label(path: Path, workspace: Path, *, screen: bool) -> str:
    name = str(path.relative_to(workspace)) if path.is_relative_to(workspace) else str(path)
    if not screen:
        return name
    try:
        with path.open("rb") as f:
            count = sum(1 for _ in f)
    except OSError:
        return name
    return ui("security.output_guard.lines", n=count, name=name)
