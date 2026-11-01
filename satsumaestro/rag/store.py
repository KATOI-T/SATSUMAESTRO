"""SQLite ベクトルストア。"""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path

from satsumaestro.rag.models import Chunk
from satsumaestro.util.paths import AGENT_DIRNAME, ensure_state_gitignore

_STORE_FORMAT = "1"


class VectorStore:
    """チャンク本体・ベクトル・増分メタを持つ SQLite 索引。"""

    def __init__(self, db_path: Path, enable_vec: bool = True) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.parent.name == AGENT_DIRNAME:
            ensure_state_gitignore(db_path.parent.parent)
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._vec_enabled = enable_vec and _load_vec_extension(self._db)
        self._dim: int | None = None
        self._init_schema()

    @property
    def vec_enabled(self) -> bool:
        """``sqlite-vec`` 拡張が有効か（False なら総当りフォールバック）。"""
        return self._vec_enabled

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            -- for incremental ingest (mtime first, then content_hash)
            CREATE TABLE IF NOT EXISTS files (
                path         TEXT PRIMARY KEY,
                mtime_ns     INTEGER NOT NULL,
                content_hash TEXT NOT NULL
            );
            -- chunk bodies (used to rebuild citations)
            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY,
                path         TEXT NOT NULL,
                start_line   INTEGER NOT NULL,
                end_line     INTEGER NOT NULL,
                text         TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding    BLOB           -- float32 array for the brute-force fallback
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            """
        )
        self._db.commit()
        if self._get_meta("schema_version") is None:
            self._set_meta("schema_version", _STORE_FORMAT)
        self._dim = _parse_int(self._get_meta("dim"))

    def _get_meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._db.commit()

    def ensure_index(self, model: str, dim: int) -> bool:
        """埋め込みモデル・次元に索引を合わせる。"""
        with self._lock:
            prev_model = self._get_meta("embed_model")
            prev_dim = _parse_int(self._get_meta("dim"))
            rebuilt = False
            if prev_model is not None and (prev_model != model or prev_dim != dim):
                self.clear()
                rebuilt = True
            self._set_meta("embed_model", model)
            self._set_meta("dim", str(dim))
            self._dim = dim
            if self._vec_enabled:
                self._db.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
                    f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{dim}] distance_metric=cosine)"
                )
                self._db.commit()
            return rebuilt

    def reset_if_model_changed(self, model: str) -> bool:
        """埋め込みモデルが変わっていれば索引を空にする（作り直した場合 ``True``）。"""
        with self._lock:
            prev_model = self._get_meta("embed_model")
            if prev_model is None or prev_model == model:
                return False
            self.clear()
            self._set_meta("embed_model", model)
            return True

    def clear(self) -> None:
        """索引を空にする（モデル変更時の作り直し・``rag reset`` 用）。"""
        with self._lock:
            self._db.execute("DELETE FROM chunks")
            self._db.execute("DELETE FROM files")
            if self._vec_enabled:
                self._db.execute("DROP TABLE IF EXISTS vec_chunks")
            self._db.commit()

    def file_state(self, path: str) -> tuple[int, str] | None:
        """``(mtime_ns, content_hash)`` を返す。"""
        with self._lock:
            row = self._db.execute(
                "SELECT mtime_ns, content_hash FROM files WHERE path = ?", (path,)
            ).fetchone()
            return (int(row["mtime_ns"]), str(row["content_hash"])) if row else None

    def touch_file(self, path: str, mtime_ns: int, content_hash: str) -> None:
        """内容不変（hash 一致）のとき ``mtime`` だけ更新する。"""
        with self._lock:
            self._db.execute(
                "INSERT INTO files(path, mtime_ns, content_hash) VALUES (?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET mtime_ns = excluded.mtime_ns, "
                "content_hash = excluded.content_hash",
                (path, mtime_ns, content_hash),
            )
            self._db.commit()

    def indexed_paths(self) -> set[str]:
        """索引済みパスの集合（走査で消えたファイルの掃除に使う）。"""
        with self._lock:
            return {str(r["path"]) for r in self._db.execute("SELECT path FROM files")}

    def delete_path(self, path: str) -> None:
        """指定パスのチャンクとメタを削除する（孤児チャンク防止）。"""
        with self._lock:
            rows = self._db.execute("SELECT id FROM chunks WHERE path = ?", (path,)).fetchall()
            if rows and self._vec_ready():
                self._db.executemany(
                    "DELETE FROM vec_chunks WHERE chunk_id = ?", [(int(r["id"]),) for r in rows]
                )
            self._db.execute("DELETE FROM chunks WHERE path = ?", (path,))
            self._db.execute("DELETE FROM files WHERE path = ?", (path,))
            self._db.commit()

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        mtime_ns: int,
        file_hash: str,
    ) -> int:
        """1 ファイル分のチャンクを入れ替える（旧チャンク削除 → 追加）。"""
        with self._lock:
            if len(chunks) != len(vectors):
                raise ValueError("chunks and vectors must have the same length")
            if not chunks:
                return 0
            path = chunks[0].path
            self.delete_path(path)
            for chunk, vector in zip(chunks, vectors, strict=True):
                blob = _pack(vector)
                cur = self._db.execute(
                    "INSERT INTO chunks(path, start_line, end_line, text, content_hash, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.text,
                        chunk.content_hash,
                        blob,
                    ),
                )
                if self._vec_ready():
                    self._db.execute(
                        "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                        (cur.lastrowid, blob),
                    )
            self._db.execute(
                "INSERT INTO files(path, mtime_ns, content_hash) VALUES (?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET mtime_ns = excluded.mtime_ns, "
                "content_hash = excluded.content_hash",
                (path, mtime_ns, file_hash),
            )
            self._db.commit()
            return len(chunks)

    def query(self, vector: list[float], k: int) -> list[Chunk]:
        """cosine 類似度の top-k を返す（``score`` は 1.0 が最類似）。"""
        with self._lock:
            if k <= 0:
                return []
            if self._vec_ready():
                return self._query_vec(vector, k)
            return self._query_bruteforce(vector, k)

    def _query_vec(self, vector: list[float], k: int) -> list[Chunk]:
        rows = self._db.execute(
            "SELECT c.path, c.start_line, c.end_line, c.text, c.content_hash, v.distance "
            "FROM (SELECT chunk_id, distance FROM vec_chunks "
            "      WHERE embedding MATCH ? AND k = ?) AS v "
            "JOIN chunks AS c ON c.id = v.chunk_id ORDER BY v.distance",
            (_pack(vector), k),
        ).fetchall()
        return [_row_to_chunk(r, 1.0 - float(r["distance"])) for r in rows]

    def _query_bruteforce(self, vector: list[float], k: int) -> list[Chunk]:
        """拡張不達環境の総当り cosine（純 Python・numpy 非依存）。"""
        norm_q = _norm(vector)
        if norm_q == 0.0:
            return []
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in self._db.execute(
            "SELECT path, start_line, end_line, text, content_hash, embedding FROM chunks"
        ):
            blob = row["embedding"]
            if not blob:
                continue
            other = _unpack(blob)
            if len(other) != len(vector):
                continue
            norm_o = _norm(other)
            if norm_o == 0.0:
                continue
            dot = sum(a * b for a, b in zip(vector, other, strict=True))
            scored.append((dot / (norm_q * norm_o), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_row_to_chunk(row, score) for score, row in scored[:k]]

    def stats(self) -> dict[str, object]:
        """``doctor`` / ``rag stats`` 用の索引サマリ。"""
        with self._lock:
            chunks = int(self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
            files = int(self._db.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"])
            return {
                "chunks": chunks,
                "files": files,
                "backend": "sqlite-vec" if self._vec_enabled else "bruteforce",
                "embed_model": self._get_meta("embed_model") or "",
                "dim": self._dim or 0,
            }

    def close(self) -> None:
        """接続を閉じる（多重呼び出し安全）。"""
        with self._lock:
            self._db.close()

    def _vec_ready(self) -> bool:
        """``vec0`` 経路が使えるか（拡張が有効かつ ``ensure_index`` 済み）。"""
        if not self._vec_enabled:
            return False
        row = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
        ).fetchone()
        return row is not None


def _load_vec_extension(db: sqlite3.Connection) -> bool:
    """``sqlite-vec`` をロードする。"""
    try:
        import sqlite_vec  # type: ignore[import-untyped]

        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    except Exception:
        return False
    return True


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def _norm(vector: list[float]) -> float:
    return float(sum(x * x for x in vector) ** 0.5)


def _row_to_chunk(row: sqlite3.Row, score: float) -> Chunk:
    return Chunk(
        path=str(row["path"]),
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        text=str(row["text"]),
        content_hash=str(row["content_hash"]),
        score=score,
    )


def _parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
