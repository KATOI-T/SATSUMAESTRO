"""RAGService。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from satsumaestro.config.models import PermissionConfig, RagConfig
from satsumaestro.rag.chunk import chunk_file, content_hash
from satsumaestro.rag.embed import Embedder, EmbedderLike
from satsumaestro.rag.models import Chunk, IngestReport
from satsumaestro.rag.store import VectorStore
from satsumaestro.rag.walk import iter_source_files
from satsumaestro.security.auto_allow import SecretMatcher
from satsumaestro.util.errors import RagError

K_MAX = 32
_SNIPPET_CHARS = 1_200


class RagBackend(Protocol):
    """索引アクセスの抽象（インプロセス / 将来の別プロセス HTTP）。"""

    async def search(self, query: str, k: int) -> list[Chunk]: ...

    async def stats(self) -> dict[str, object]: ...

    async def aclose(self) -> None: ...


class LocalBackend:
    """インプロセス実装: ``Embedder.embed(query)`` → ``VectorStore.query``。"""

    def __init__(self, store: VectorStore, embedder: EmbedderLike) -> None:
        self.store = store
        self.embedder = embedder

    async def search(self, query: str, k: int) -> list[Chunk]:
        vectors = await self.embedder.embed([query])
        if not vectors:
            raise RagError("embedding for the query was empty")
        return await asyncio.to_thread(self.store.query, vectors[0], k)

    async def stats(self) -> dict[str, object]:
        return await asyncio.to_thread(self.store.stats)

    async def aclose(self) -> None:
        await self.embedder.aclose()
        await asyncio.to_thread(self.store.close)


class RAGService:
    """インジェスト・検索・引用整形のファサード。"""

    def __init__(
        self,
        cfg: RagConfig,
        workspace_root: Path,
        embedder: EmbedderLike | None = None,
        store: VectorStore | None = None,
        secrets: SecretMatcher | None = None,
    ) -> None:
        self.cfg = cfg
        self.workspace_root = workspace_root
        self.secrets = secrets or SecretMatcher.from_config(PermissionConfig())
        db_path = Path(cfg.db_path)
        self.db_path = db_path if db_path.is_absolute() else workspace_root / db_path
        self.store = store if store is not None else VectorStore(self.db_path)
        self.embedder: EmbedderLike = embedder or Embedder(
            base_url=cfg.embed_base_url, model=cfg.embed_model
        )
        self.backend: RagBackend = LocalBackend(self.store, self.embedder)

    async def ingest(self, root: Path, globs: list[str] | None = None) -> IngestReport:
        """``root`` 配下を走査し、変更のあったファイルだけを索引へ反映する。"""
        report = IngestReport()
        seen: set[str] = set()
        await asyncio.to_thread(self.store.reset_if_model_changed, self.cfg.embed_model)
        for path in iter_source_files(root, globs):
            rel = path.relative_to(root).as_posix()
            if self.secrets.matches(rel):
                report.files_secret_skipped += 1
                continue
            report.files_scanned += 1
            seen.add(rel)
            try:
                await self._ingest_file(path, rel, report)
            except RagError as e:
                report.errors.append(f"{rel}: {e}")
            except OSError as e:
                report.errors.append(f"{rel}: {e}")
        await self._prune_deleted(seen, report)
        return report

    async def _ingest_file(self, path: Path, rel: str, report: IngestReport) -> None:
        mtime_ns = path.stat().st_mtime_ns
        state = await asyncio.to_thread(self.store.file_state, rel)
        if state is not None and state[0] == mtime_ns:
            report.chunks_skipped += 1
            return

        text = path.read_text(encoding="utf-8", errors="replace")
        file_hash = content_hash(text)
        if state is not None and state[1] == file_hash:
            await asyncio.to_thread(self.store.touch_file, rel, mtime_ns, file_hash)
            report.chunks_skipped += 1
            return

        chunks = chunk_file(rel, text)
        if not chunks:
            await asyncio.to_thread(self.store.delete_path, rel)
            await asyncio.to_thread(self.store.touch_file, rel, mtime_ns, file_hash)
            return

        vectors = await self.embedder.embed([c.text for c in chunks])
        await self._ensure_index(vectors)
        added = await asyncio.to_thread(self.store.upsert, chunks, vectors, mtime_ns, file_hash)
        report.files_indexed += 1
        report.chunks_added += added

    async def _ensure_index(self, vectors: list[list[float]]) -> None:
        """埋め込み次元に索引を合わせる（モデル変更時は作り直し）。"""
        if not vectors:
            return
        await asyncio.to_thread(self.store.ensure_index, self.cfg.embed_model, len(vectors[0]))

    async def _prune_deleted(self, seen: set[str], report: IngestReport) -> None:
        """走査で見つからなくなった（削除・除外された）パスを索引から落とす。"""
        indexed = await asyncio.to_thread(self.store.indexed_paths)
        for stale in sorted(indexed - seen):
            await asyncio.to_thread(self.store.delete_path, stale)
            report.files_removed += 1

    async def search(self, query: str, k: int = 8) -> list[Chunk]:
        """cosine top-k のチャンクを返す。"""
        return await self.backend.search(query, max(1, min(k, K_MAX)))

    async def query(self, question: str, k: int = 8) -> str:
        """検索結果を引用形式に整形して返す（``rag_search`` / CLI 共通）。"""
        return format_citations(await self.search(question, k))

    async def stats(self) -> dict[str, object]:
        """索引サマリ（``rag stats`` / ``doctor`` 用）。"""
        return await self.backend.stats()

    async def aclose(self) -> None:
        """バックエンド（HTTP クライアント・DB 接続）を閉じる。"""
        await self.backend.aclose()


def format_citations(chunks: list[Chunk]) -> str:
    """``path:開始行-終了行`` と score 付きの引用テキストへ整形する。"""
    if not chunks:
        return "No matching chunks were found in the workspace index."
    blocks: list[str] = []
    for chunk in chunks:
        body = chunk.text[:_SNIPPET_CHARS]
        if len(chunk.text) > _SNIPPET_CHARS:
            body += "\n… (truncated)"
        blocks.append(
            f"[{chunk.path}:{chunk.start_line}-{chunk.end_line}] (score {chunk.score:.3f})\n{body}"
        )
    return "\n\n".join(blocks)
