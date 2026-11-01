"""この会話で書いたファイルを途中の部分で丸ごと置き換える ``write_file`` の差し戻し。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from satsumaestro.util.paths import AGENT_DIRNAME

FLAG_REJECT = "overwrite-reject"
FLAG_OVERRIDE = "overwrite-override"


def first_line(text: str) -> str:
    """最初の空でない行（前後の空白を除く。無ければ空文字）。"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def is_fragment(current: str, content: str) -> bool:
    """``content`` が ``current`` の全体の新しい版ではなく、途中の部分に見えるか。"""
    stripped = current.strip()
    if not stripped or stripped in content:
        return False
    return first_line(current) != first_line(content)


def is_guarded(real: Path, root: Path, written: Iterable[Path]) -> bool:
    """この会話で書いたファイルで、状態ディレクトリ（計画ファイル）の外か。"""
    return real in set(written) and not real.is_relative_to(root / AGENT_DIRNAME)


def _count_lines(text: str) -> int:
    return len(text.splitlines())


def _size(text: str) -> int:
    """大きさ。"""
    return len(text)


def _tools_summary(tools: Iterable[str]) -> str:
    """``write_file + 2 append_file`` の形（最初に出た順）。"""
    counts = Counter(tools)
    return " + ".join(f"{n} {tool}" if n > 1 else tool for tool, n in counts.items())


def reject_message(path: str, current: str, tools: Iterable[str]) -> str:
    """差し戻しの tool result。"""
    lines = _count_lines(current)
    size = _size(current)
    by = _tools_summary(tools)
    origin = f" ({by})" if by else ""
    return (
        f"write_file rejected: would delete {lines} lines of {path}\n"
        f"{path} already has {lines} lines ({size} bytes) from this conversation{origin}.\n"
        f"The new content does not start the way the file starts (`{first_line(current)}`), so it "
        "looks like a later part of the file, not a new version of the whole file; write_file "
        f"replaces the whole file. {path} was left unchanged.\n"
        "- To add it at the end of the file, call append_file with the same content.\n"
        "- To insert or change part of the file, use edit_file (read_file that part first).\n"
        "- To really replace the whole file with this content, call write_file again with the "
        "same path and the same content; that second call is written (different content is "
        "checked again)."
    )


def replaced_note(current: str) -> str:
    """置き換えた前の大きさ `` (replaced N lines, M bytes)``。"""
    if not current:
        return ""
    return f" (replaced {_count_lines(current)} lines, {_size(current)} bytes)"
