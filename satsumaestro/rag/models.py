"""RAG のデータモデル。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    """索引の最小単位。"""

    model_config = ConfigDict(extra="forbid")

    path: str
    start_line: int
    end_line: int
    text: str
    content_hash: str
    score: float = 0.0


class IngestReport(BaseModel):
    """``RAGService.ingest`` の結果。"""

    model_config = ConfigDict(extra="forbid")

    files_scanned: int = 0
    files_indexed: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_skipped: int = 0
    files_secret_skipped: int = 0
    errors: list[str] = []
