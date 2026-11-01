"""ファイル → チャンク分割。"""

from __future__ import annotations

import hashlib

from satsumaestro.rag.models import Chunk
from satsumaestro.util.tokens import estimate_tokens

DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP = 0.15


def content_hash(text: str) -> str:
    """本文の内容ハッシュ（増分判定・チャンク同一性の鍵）を返す。"""
    return hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=16).hexdigest()


def chunk_file(
    path: str,
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: float = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """行境界を尊重しつつ ~``max_tokens``・``overlap`` 重複で分割する。"""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0.0, 1.0)")

    lines = text.splitlines()
    if not lines:
        return []

    costs = [estimate_tokens(line) + 1 for line in lines]
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = start
        used = 0
        while end < len(lines) and (end == start or used + costs[end] <= max_tokens):
            used += costs[end]
            end += 1

        body = "\n".join(lines[start:end])
        if body.strip():
            chunks.append(
                Chunk(
                    path=path,
                    start_line=start + 1,
                    end_line=end,
                    text=body,
                    content_hash=content_hash(f"{path}:{start + 1}:{body}"),
                )
            )
        if end >= len(lines):
            break
        start = _next_start(start, end, costs, max_tokens, overlap)
    return chunks


def _next_start(start: int, end: int, costs: list[int], max_tokens: int, overlap: float) -> int:
    """末尾から ``overlap`` 割のトークン分だけ巻き戻した次チャンクの開始 index を返す。"""
    budget = max_tokens * overlap
    back = end
    acc = 0.0
    while back > start and acc + costs[back - 1] <= budget:
        acc += costs[back - 1]
        back -= 1
    return back if back > start else end
