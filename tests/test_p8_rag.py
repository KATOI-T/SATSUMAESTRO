"""RAG 学習の結線テスト。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import pytest

from satsumaestro.config.models import RagConfig, RuntimeConfig
from satsumaestro.rag.chunk import chunk_file
from satsumaestro.rag.embed import Embedder
from satsumaestro.rag.models import Chunk
from satsumaestro.rag.service import K_MAX, RAGService, format_citations
from satsumaestro.rag.store import VectorStore
from satsumaestro.rag.walk import iter_source_files
from satsumaestro.security.enforcer import PermissionMode, classify
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.rag_search import RagSearch
from satsumaestro.util.errors import RagError
from tests.fixtures.mock_embedder import MockEmbedder


def _vec_available() -> bool:
    """このプラットフォームで ``sqlite-vec`` 拡張がロードできるか。"""
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp) / "probe.db")
        try:
            return store.vec_enabled
        finally:
            store.close()


_HAS_VEC = _vec_available()


def _workspace(tmp_path: Path) -> Path:
    """検索対象が明確に分かれた小さなワークスペースを作る。"""
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "auth.py").write_text(
        "def login(user, password):\n    token = issue_token(user)\n    return token\n",
        encoding="utf-8",
    )
    (root / "src" / "payment.py").write_text(
        "def charge(invoice, amount):\n"
        "    receipt = capture_payment(invoice, amount)\n"
        "    return receipt\n",
        encoding="utf-8",
    )
    return root


def _service(
    root: Path, embedder: MockEmbedder | None = None, enable_vec: bool = True
) -> RAGService:
    cfg = RagConfig(db_path=str(root / ".satsuma" / "rag.db"))
    store = VectorStore(root / ".satsuma" / "rag.db", enable_vec=enable_vec)
    return RAGService(cfg, root, embedder=embedder or MockEmbedder(), store=store)


def test_chunking_keeps_line_ranges_and_overlaps() -> None:
    text = "\n".join(f"line {i} alpha beta gamma delta" for i in range(1, 201))
    chunks = chunk_file("big.txt", text, max_tokens=100, overlap=0.15)

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 200
    for chunk in chunks:
        assert chunk.start_line <= chunk.end_line
        assert chunk.text.splitlines()[0].startswith(f"line {chunk.start_line} ")
    assert chunks[1].start_line <= chunks[0].end_line


def test_chunking_single_huge_line_makes_progress() -> None:
    """1 行が上限を超えても前進する（無限ループしない）。"""
    text = "x" * 10_000 + "\nshort tail\n"
    chunks = chunk_file("min.js", text, max_tokens=50)

    assert len(chunks) == 2
    assert chunks[0].start_line == chunks[0].end_line == 1


def test_chunking_skips_blank_only_file() -> None:
    assert chunk_file("empty.txt", "\n\n   \n") == []


def test_walk_respects_gitignore_and_skips_binary(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    (root / ".gitignore").write_text("ignored/\n*.log\n", encoding="utf-8")
    (root / "ignored").mkdir()
    (root / "ignored" / "secret.py").write_text("nope", encoding="utf-8")
    (root / "debug.log").write_text("noise", encoding="utf-8")
    (root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "lib.js").write_text("skip", encoding="utf-8")

    found = {p.relative_to(root).as_posix() for p in iter_source_files(root)}

    assert found == {".gitignore", "src/auth.py", "src/payment.py"}


@pytest.mark.parametrize("enable_vec", [True, False])
async def test_ingest_then_search_ranks_expected_file(tmp_path: Path, enable_vec: bool) -> None:
    """sqlite-vec 経路・総当り経路のどちらでも期待 path が top-1 に来る。"""
    root = _workspace(tmp_path)
    service = _service(root, enable_vec=enable_vec)
    try:
        report = await service.ingest(root)
        assert report.files_indexed == 2
        assert report.chunks_added == 2
        assert report.errors == []

        hits = await service.search("issue_token login password", k=2)
        assert hits[0].path == "src/auth.py"
        assert hits[0].score > hits[-1].score
        assert hits[0].start_line == 1 and hits[0].end_line == 3
    finally:
        await service.aclose()


async def test_search_backends_agree_on_order(tmp_path: Path) -> None:
    """拡張の有無で結果順が変わらない（フォールバックが等価であること）。"""
    if not _HAS_VEC:
        pytest.skip("sqlite-vec が利用できない環境")
    root = _workspace(tmp_path)
    ranked: list[list[str]] = []
    for enable_vec in (True, False):
        db = tmp_path / f"idx-{enable_vec}.db"
        cfg = RagConfig(db_path=str(db))
        service = RAGService(
            cfg, root, embedder=MockEmbedder(), store=VectorStore(db, enable_vec=enable_vec)
        )
        try:
            await service.ingest(root)
            ranked.append([c.path for c in await service.search("capture_payment invoice", k=2)])
        finally:
            await service.aclose()
    assert ranked[0] == ranked[1] == ["src/payment.py", "src/auth.py"]


async def test_search_clamps_k_to_max(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    embedder = MockEmbedder()
    service = _service(root, embedder)
    try:
        await service.ingest(root)
        hits = await service.search("token", k=1_000)
        assert len(hits) <= K_MAX
    finally:
        await service.aclose()


async def test_ingest_is_incremental(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    embedder = MockEmbedder()
    service = _service(root, embedder)
    try:
        first = await service.ingest(root)
        assert first.chunks_added == 2
        embed_calls_after_first = len(embedder.calls)

        second = await service.ingest(root)
        assert second.chunks_added == 0
        assert second.chunks_skipped == 2
        assert len(embedder.calls) == embed_calls_after_first

        target = root / "src" / "auth.py"
        os.utime(target, (1_800_000_000, 1_800_000_000))
        third = await service.ingest(root)
        assert third.chunks_added == 0
        assert third.chunks_skipped == 2
        assert len(embedder.calls) == embed_calls_after_first

        target.write_text("def login(user):\n    return refresh_session(user)\n", encoding="utf-8")
        fourth = await service.ingest(root)
        assert fourth.files_indexed == 1
        assert fourth.chunks_skipped == 1
        assert len(embedder.calls) == embed_calls_after_first + 1
    finally:
        await service.aclose()


async def test_ingest_replaces_chunks_and_prunes_deleted(tmp_path: Path) -> None:
    """短くなったファイルの旧チャンクが孤児として残らない／削除ファイルは索引から落ちる。"""
    root = _workspace(tmp_path)
    long_file = root / "src" / "long.py"
    long_file.write_text("\n".join(f"row_{i} = compute_metric({i})" for i in range(400)))
    service = _service(root)
    try:
        await service.ingest(root)
        before = await service.stats()

        long_file.write_text("row_0 = compute_metric(0)\n", encoding="utf-8")
        await service.ingest(root)
        after_shrink = await service.stats()
        assert int(after_shrink["chunks"]) < int(before["chunks"])

        long_file.unlink()
        report = await service.ingest(root)
        assert report.files_removed == 1
        remaining = {c.path for c in await service.search("compute_metric", k=K_MAX)}
        assert "src/long.py" not in remaining
    finally:
        await service.aclose()


async def test_ingest_degrades_when_embeddings_unreachable(tmp_path: Path) -> None:
    """埋め込み不達でも例外を投げず errors[] に記録して走査を続ける。"""
    root = _workspace(tmp_path)
    service = _service(root, MockEmbedder(fail=True))
    try:
        report = await service.ingest(root)
        assert report.files_scanned == 2
        assert report.chunks_added == 0
        assert len(report.errors) == 2
        assert all("offline" in e for e in report.errors)
    finally:
        await service.aclose()


async def test_search_raises_rag_error_when_embeddings_unreachable(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    service = _service(root, MockEmbedder(fail=True))
    try:
        with pytest.raises(RagError):
            await service.search("anything")
    finally:
        await service.aclose()


async def test_index_rebuilds_when_embedding_dim_changes(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    db = tmp_path / "idx.db"
    cfg = RagConfig(db_path=str(db), embed_model="mock-64")
    service = RAGService(cfg, root, embedder=MockEmbedder(dim=64), store=VectorStore(db))
    try:
        await service.ingest(root)
        assert int((await service.stats())["dim"]) == 64
    finally:
        await service.aclose()

    cfg2 = RagConfig(db_path=str(db), embed_model="mock-32")
    service2 = RAGService(cfg2, root, embedder=MockEmbedder(dim=32), store=VectorStore(db))
    try:
        await service2.ingest(root)
        stats = await service2.stats()
        assert int(stats["dim"]) == 32
        assert int(stats["chunks"]) == 2
        hits = await service2.search("issue_token", k=2)
        assert hits and hits[0].path == "src/auth.py"
    finally:
        await service2.aclose()


def test_rag_search_spec_is_read_only_and_parallel_safe() -> None:
    assert RagSearch.spec.required_mode is PermissionMode.READ_ONLY
    assert RagSearch.spec.parallel_safe is True
    assert classify("rag_search", {}, PermissionMode.DANGER_FULL_ACCESS) is PermissionMode.READ_ONLY


async def test_rag_search_tool_returns_citations(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    service = _service(root)
    try:
        await service.ingest(root)
        ctx = ToolContext(
            workspace_root=root,
            jail=WorkspaceJail(root),
            config=RuntimeConfig(),
            rag=service,
        )
        result = await RagSearch().run(RagSearch.Input(query="issue_token", k=1), ctx)
        assert result.is_error is False
        assert "[src/auth.py:1-3]" in result.output
        assert "issue_token" in result.output
    finally:
        await service.aclose()


async def test_rag_search_tool_is_error_when_rag_disabled(tmp_path: Path) -> None:
    ctx = ToolContext(
        workspace_root=tmp_path, jail=WorkspaceJail(tmp_path), config=RuntimeConfig(), rag=None
    )
    result = await RagSearch().run(RagSearch.Input(query="anything"), ctx)

    assert result.is_error is True
    assert "rag ingest" in result.output


async def test_rag_search_tool_degrades_on_backend_failure(tmp_path: Path) -> None:
    """埋め込み不達でも例外を漏らさず is_error で返す（会話は継続する）。"""
    root = _workspace(tmp_path)
    service = _service(root, MockEmbedder(fail=True))
    try:
        ctx = ToolContext(
            workspace_root=root, jail=WorkspaceJail(root), config=RuntimeConfig(), rag=service
        )
        result = await RagSearch().run(RagSearch.Input(query="anything"), ctx)
        assert result.is_error is True
        assert "RAG unavailable" in result.output
    finally:
        await service.aclose()


def test_format_citations_without_hits() -> None:
    assert "No matching chunks" in format_citations([])


def test_format_citations_truncates_long_chunks() -> None:
    chunk = Chunk(
        path="a.py", start_line=1, end_line=999, text="x" * 5_000, content_hash="h", score=0.5
    )
    output = format_citations([chunk])
    assert "(truncated)" in output
    assert len(output) < 2_000


async def test_embedder_reorders_by_index_and_batches() -> None:
    """応答順を仮定せず ``index`` で並べ直す。"""
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        seen.append(payload)
        items = [
            {"index": i, "embedding": [float(i), 0.0]}
            for i in reversed(range(len(payload["input"])))
        ]
        return httpx.Response(200, json={"data": items})

    embedder = Embedder(
        base_url="http://embed.test/v1",
        model="bge-m3",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://embed.test/v1"
        ),
    )
    try:
        vectors = await embedder.embed(["a", "b", "c"])
    finally:
        await embedder.aclose()

    assert vectors == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    assert seen[0]["model"] == "bge-m3"
    assert seen[0]["input"] == ["a", "b", "c"]


async def test_embedder_retries_5xx_then_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="warming up")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    embedder = Embedder(
        base_url="http://embed.test/v1",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://embed.test/v1"
        ),
    )
    try:
        assert await embedder.embed(["q"]) == [[1.0]]
    finally:
        await embedder.aclose()
    assert attempts == 2


async def test_embedder_does_not_retry_client_error() -> None:
    """4xx（モデル名誤り等）は再試行しても直らないので即座に RagError。"""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404, text="model not found")

    embedder = Embedder(
        base_url="http://embed.test/v1",
        model="missing",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://embed.test/v1"
        ),
    )
    try:
        with pytest.raises(RagError, match="404"):
            await embedder.embed(["q"])
    finally:
        await embedder.aclose()
    assert attempts == 1


async def test_embedder_falls_back_to_single_input_batches() -> None:
    """バッチ入力を受け付けないサーバでは 1 件ずつ送り直す（degrade）。"""
    batch_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)
        batch_sizes.append(len(payload["input"]))
        if len(payload["input"]) > 1:
            return httpx.Response(400, text="input must be a string")
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    embedder = Embedder(
        base_url="http://embed.test/v1",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://embed.test/v1"
        ),
    )
    try:
        vectors = await embedder.embed(["a", "b"])
    finally:
        await embedder.aclose()

    assert vectors == [[1.0, 2.0], [1.0, 2.0]]
    assert batch_sizes == [2, 1, 1]


async def test_embedder_reports_unreachable_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    embedder = Embedder(
        base_url="http://embed.test/v1",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://embed.test/v1"
        ),
    )
    try:
        with pytest.raises(RagError, match="unreachable"):
            await embedder.embed(["q"])
    finally:
        await embedder.aclose()
