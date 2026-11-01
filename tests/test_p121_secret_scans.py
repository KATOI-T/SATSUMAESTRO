"""走査する読み取り（grep_search・RAG の索引）で秘密ファイルを読まずに飛ばす。"""

from __future__ import annotations

import os
from pathlib import Path

from satsumaestro.config.models import PermissionConfig, RagConfig, RuntimeConfig
from satsumaestro.rag.service import RAGService
from satsumaestro.rag.store import VectorStore
from satsumaestro.security.auto_allow import SecretMatcher
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.grep_search import GrepSearch
from tests.fixtures.mock_embedder import MockEmbedder


def _tree(root: Path) -> None:
    (root / ".env").write_text("API_TOKEN=abc123\n", encoding="utf-8")
    (root / ".env.example").write_text("API_TOKEN=\n", encoding="utf-8")
    (root / "app.py").write_text("token = read('API_TOKEN')\n", encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / ".env.local").write_text("API_TOKEN=local999\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "util.py").write_text("API_TOKEN_NAME = 'x'\n", encoding="utf-8")


async def _grep(ctx: ToolContext, **kwargs: object) -> str:
    result = await GrepSearch().run(GrepSearch.Input(**kwargs), ctx)  # type: ignore[arg-type]
    assert not result.is_error
    return result.output


async def test_grep_skips_secret_files_and_says_so(tool_ctx: ToolContext) -> None:
    _tree(tool_ctx.workspace_root)
    out = await _grep(tool_ctx, pattern="API_TOKEN")
    assert "abc123" not in out and "local999" not in out
    assert "app.py:1:" in out and "src/util.py:1:" in out
    assert out.splitlines()[-1] == "(skipped 3 secret files: .env, .env.example, config/.env.local)"


async def test_grep_notes_skips_even_without_a_match(tool_ctx: ToolContext) -> None:
    _tree(tool_ctx.workspace_root)
    out = await _grep(tool_ctx, pattern="no-such-text")
    assert out == "(skipped 3 secret files: .env, .env.example, config/.env.local)"


async def test_grep_without_secrets_has_no_note(tool_ctx: ToolContext) -> None:
    _tree(tool_ctx.workspace_root)
    out = await _grep(tool_ctx, pattern="API_TOKEN", path="src")
    assert out == "src/util.py:1:API_TOKEN_NAME = 'x'"


async def test_grep_other_modes_skip_too(tool_ctx: ToolContext) -> None:
    _tree(tool_ctx.workspace_root)
    files = await _grep(tool_ctx, pattern="API_TOKEN", output_mode="files_with_matches")
    assert files.splitlines()[:2] == ["app.py", "src/util.py"]
    count = await _grep(tool_ctx, pattern="API_TOKEN", output_mode="count")
    assert ".env" not in count.splitlines()[0] and count.endswith("config/.env.local)")


async def test_grep_skips_links_by_target_and_by_name(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    _tree(root)
    os.symlink(root / ".env", root / "plain.txt")
    os.symlink(root / "app.py", root / "id_rsa")
    out = await _grep(tool_ctx, pattern="abc123|token =")
    assert "abc123" not in out and "id_rsa:" not in out
    assert "plain.txt" in out.splitlines()[-1] and "id_rsa" in out.splitlines()[-1]


async def test_grep_follows_secret_patterns_remove(tool_ctx: ToolContext) -> None:
    _tree(tool_ctx.workspace_root)
    permission = PermissionConfig(secret_patterns_remove=[".env.example"])
    ctx = tool_ctx.model_copy(update={"config": RuntimeConfig(permission=permission)})
    out = await _grep(ctx, pattern="API_TOKEN=$")
    assert out.splitlines() == [
        ".env.example:1:API_TOKEN=",
        "(skipped 2 secret files: .env, config/.env.local)",
    ]


async def test_grep_lists_at_most_five_names(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    for i in range(7):
        (root / f"k{i}.pem").write_text("pem\n", encoding="utf-8")
    out = await _grep(tool_ctx, pattern="pem")
    assert out == "(skipped 7 secret files: k0.pem, k1.pem, k2.pem, k3.pem, k4.pem, …)"


def _service(root: Path, secrets: SecretMatcher | None = None) -> RAGService:
    db = root.parent / "rag.db"
    return RAGService(
        RagConfig(db_path=str(db)),
        root,
        embedder=MockEmbedder(),
        store=VectorStore(db),
        secrets=secrets,
    )


async def test_rag_ingest_leaves_out_secret_files(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    _tree(root)
    service = _service(root)
    try:
        report = await service.ingest(root)
        assert report.files_secret_skipped == 3 and report.files_scanned == 2
        indexed = service.store.indexed_paths()
        assert indexed == {"app.py", "src/util.py"}
    finally:
        await service.aclose()


async def test_rag_ingest_drops_secrets_indexed_before(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    _tree(root)
    everything = SecretMatcher([])
    first = _service(root, everything)
    try:
        await first.ingest(root)
        assert ".env" in first.store.indexed_paths()
    finally:
        await first.aclose()
    second = _service(root)
    try:
        report = await second.ingest(root)
        assert report.files_removed == 3
        assert second.store.indexed_paths() == {"app.py", "src/util.py"}
    finally:
        await second.aclose()
