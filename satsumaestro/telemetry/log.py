"""テレメトリの記録帳。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceEntry:
    """記録の 1 件。"""

    session: str
    seq: int
    event: str
    at_ms: int
    data: dict[str, object] = field(default_factory=dict)

    def as_json_line(self) -> str:
        row = {
            "session": self.session,
            "seq": self.seq,
            "event": self.event,
            "at_ms": self.at_ms,
            "data": self.data,
        }
        return json.dumps(row, ensure_ascii=False, default=str)


class TraceLog:
    """記録帳。"""

    def __init__(self, file: Path | None = None) -> None:
        self.entries: list[TraceEntry] = []
        self.file = file
        if file is not None:
            try:
                file.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                _log.debug("trace log directory not created: %s", file, exc_info=True)

    def add(self, entry: TraceEntry) -> None:
        self.entries.append(entry)
        if self.file is None:
            return
        try:
            with self.file.open("a", encoding="utf-8") as out:
                out.write(entry.as_json_line() + "\n")
        except OSError:
            _log.debug("trace log not written: %s", self.file, exc_info=True)
