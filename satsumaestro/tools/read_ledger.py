"""ファイルを最後に読んだ時点の記録と、edit_file の不一致に添える「読み直せ」の一文。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

FLAG_STALE = "edit-stale"
OVERWRITE_REJECT_LIMIT = 2


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass
class _Entry:
    read_hash: str
    writes_since_read: int = 0
    last_write_hash: str | None = None


@dataclass
class ReadLedger:
    """会話ごとの、ファイル単位の読み書きの記録。"""

    _entries: dict[Path, _Entry] = field(default_factory=dict)
    written: set[Path] = field(default_factory=set)
    _related: set[tuple[str, str]] = field(default_factory=set)
    _subclass_checked: set[Path] = field(default_factory=set)
    written_by: dict[Path, list[str]] = field(default_factory=dict)
    overwrite_pending: dict[Path, str] = field(default_factory=dict)
    overwrite_rejections: dict[Path, int] = field(default_factory=dict)

    def note_read(self, path: Path) -> None:
        """read_file の成功時に呼ぶ（その時点のファイル全体を記録し、書いた回数を戻す）。"""
        text = _read(path)
        if text is not None:
            self._entries[path] = _Entry(read_hash=_digest(text))

    def note_write(self, path: Path, content: str, tool: str = "") -> None:
        """書き込みツールの成功時に呼ぶ（書いた後の全体と、書いたツール名）。"""
        self.written.add(path)
        if tool:
            self.written_by.setdefault(path, []).append(tool)
        self.overwrite_pending.pop(path, None)
        self.overwrite_rejections.pop(path, None)
        entry = self._entries.get(path)
        if entry is not None:
            entry.writes_since_read += 1
            entry.last_write_hash = _digest(content)

    def note_overwrite_rejected(self, path: Path, first_line: str) -> None:
        """上書きの門が差し戻した。"""
        self.overwrite_pending[path] = first_line
        self.overwrite_rejections[path] = self.overwrite_rejections.get(path, 0) + 1

    def overwrite_override(self, path: Path, first_line: str) -> bool:
        """差し戻しの後の同じパスへの write_file をそのまま書くか。"""
        pending = self.overwrite_pending.get(path)
        if pending is None:
            return False
        return (
            pending == first_line
            or self.overwrite_rejections.get(path, 0) >= OVERWRITE_REJECT_LIMIT
        )

    def first_subclass_check(self, path: Path) -> bool:
        """このファイルの派生クラスを調べるのが、この会話で初めてなら True（以後は False）。"""
        if path in self._subclass_checked:
            return False
        self._subclass_checked.add(path)
        return True

    def related_shown(self, base: str, rel: str) -> bool:
        return (base, rel) in self._related

    def mark_related(self, base: str, rel: str) -> None:
        self._related.add((base, rel))

    def stale_note(self, path: Path, rel: str, current: str) -> str | None:
        """edit_file が一致しなかったときに添える一文。"""
        entry = self._entries.get(path)
        if entry is None:
            return (
                f"note: no read_file of {rel} is recorded in this session. If you wrote `old` "
                f"from memory, read_file {rel} first and copy the lines from it."
            )
        now = _digest(current)
        if now == entry.read_hash:
            return None
        if entry.writes_since_read and now == entry.last_write_hash:
            times = "edit" if entry.writes_since_read == 1 else "edits"
            return (
                f"note: {rel} has changed since you last read it: your own "
                f"{entry.writes_since_read} {times} since then modified it. `old` must match the "
                f"file as it is now, so read_file the part you want to change again first."
            )
        return (
            f"note: {rel} was modified since you last read it (not only by your own edits). "
            f"read_file it again before editing."
        )


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
