"""プロジェクトの指示ファイルを選び、system の動的後半に載せる文にする。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from satsumaestro.util.paths import AGENT_DIRNAME, find_workspace_root
from satsumaestro.util.tokens import estimate_upper

INSTRUCTION_NAMES = ("SATSUMA.md", "CLAUDE.md", "AGENTS.md")
MAX_INSTRUCTION_CHARS = 16_000
WINDOW_DIVISOR = 8


@dataclass(frozen=True)
class _Loaded:
    path: Path
    text: str
    truncated: bool


def search_dirs(cwd: Path) -> list[Path]:
    """作業場所の根と ``cwd``（根が先。同じなら 1 つ）。"""
    cwd = cwd.resolve()
    root = find_workspace_root(cwd)
    return [cwd] if root == cwd else [root, cwd]


def candidates(base: Path) -> list[Path]:
    """``base`` で指示ファイルとして探すパス（この順。最初に見つかった 1 つだけを読む）。"""
    paths: list[Path] = []
    for name in INSTRUCTION_NAMES:
        paths.append(base / AGENT_DIRNAME / name)
        paths.append(base / name)
    return paths


def _read_head(path: Path, limit: int) -> str | None:
    """先頭から ``limit`` 文字まで読む（巨大なファイルを丸ごと読まない）。"""
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return None


def _fingerprint(text: str) -> str:
    """空白の違いを無視した内容の指紋（同じ内容のコピーを 1 回だけ載せる）。"""
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def instruction_tokens(window: int) -> int | None:
    """窓から決めた指示ファイルの量（トークン）。"""
    return window // WINDOW_DIVISOR if window > 0 else None


def _fit_tokens(text: str, tokens: int) -> int:
    """先頭から何文字までなら見積もりが ``tokens`` に収まるか（見積もりは先頭から増える一方）。"""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_upper(text[:middle]) <= tokens:
            low = middle
        else:
            high = middle - 1
    return low


def _cut(text: str, limit: int, tokens: int | None = None) -> str:
    """``limit`` 文字（と ``tokens`` トークン）に収める。"""
    end = min(len(text), limit)
    if tokens is not None:
        end = min(end, _fit_tokens(text, tokens))
    if end >= len(text):
        return text
    head = text[:end]
    newline = head.rfind("\n")
    return head[: newline + 1] if newline > 0 else head


def _pick(base: Path, limit: int) -> tuple[Path, str] | None:
    """``base`` の候補のうち、最初に見つかった空でないファイル（解決済みのパスと先頭の本文）。"""
    for candidate in candidates(base):
        if not candidate.is_file():
            continue
        text = _read_head(candidate, limit)
        if text is not None and text.strip():
            return candidate.resolve(), text
    return None


def discover_instructions(
    cwd: Path, budget: int = MAX_INSTRUCTION_CHARS, tokens: int | None = None
) -> str:
    """指示ファイルを選び、``# Instructions from <パス>`` の節を並べた文にする（無ければ空）。"""
    seen_paths: set[Path] = set()
    seen_texts: set[str] = set()
    loaded: list[_Loaded] = []
    skipped: list[Path] = []
    remaining = budget
    remaining_tokens = tokens
    for base in search_dirs(cwd):
        picked = _pick(base, budget + 1)
        if picked is None:
            continue
        path, text = picked
        fingerprint = _fingerprint(text)
        if path in seen_paths or fingerprint in seen_texts:
            continue
        seen_paths.add(path)
        seen_texts.add(fingerprint)
        if remaining <= 0 or (remaining_tokens is not None and remaining_tokens <= 0):
            skipped.append(path)
            continue
        shown = _cut(text, remaining, remaining_tokens)
        if not shown.strip():
            skipped.append(path)
            remaining = 0
            continue
        cut = len(shown) < len(text)
        loaded.append(_Loaded(path, shown, cut))
        remaining -= len(shown)
        if remaining_tokens is not None:
            remaining_tokens -= estimate_upper(shown)
        if cut:
            remaining = 0
    chunks = [_section(item) for item in loaded]
    if skipped:
        chunks.append(_skipped_section(skipped, budget, tokens))
    return "\n\n".join(chunks)


def _section(item: _Loaded) -> str:
    body = item.text.rstrip("\n")
    if not item.truncated:
        return f"# Instructions from {item.path}\n{body}"
    return (
        f"# Instructions from {item.path}\n{body}\n"
        f"[truncated: only the first {len(item.text):,} characters of this file are shown here. "
        f"Read {item.path} with read_file for the rest.]"
    )


def _skipped_section(paths: list[Path], budget: int, tokens: int | None) -> str:
    listed = "\n".join(f"- {p}" for p in paths)
    size = f"{budget:,} characters" + ("" if tokens is None else f" or about {tokens:,} tokens")
    return (
        "# Instruction files not loaded\n"
        f"The instruction budget ({size}) is used up, so these files were not "
        "loaded. Read them with read_file when they matter for the task.\n"
        f"{listed}"
    )
