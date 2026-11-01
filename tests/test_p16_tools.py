"""ツール拡充・強化のテスト。"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from satsumaestro.config.models import RuntimeConfig, WebSearchConfig
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import web_search as web_search_mod
from satsumaestro.tools.builtins.glob_search import GlobSearch
from satsumaestro.tools.builtins.grep_search import GrepSearch
from satsumaestro.tools.builtins.notebook_edit import NotebookEdit
from satsumaestro.tools.builtins.read_file import _BYTE_CAP, ReadFile
from satsumaestro.tools.builtins.web_search import WebSearch
from satsumaestro.util.errors import ToolError


async def test_grep_context_lines(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "a.txt").write_text("l1\nl2\nMATCH1\nl4\nl5\nl6\nMATCH2\nl8\n", encoding="utf-8")
    result = await GrepSearch().run(
        GrepSearch.Input(pattern="MATCH", context_before=1, context_after=1), tool_ctx
    )
    assert result.output.splitlines() == [
        "a.txt-2-l2",
        "a.txt:3:MATCH1",
        "a.txt-4-l4",
        "--",
        "a.txt-6-l6",
        "a.txt:7:MATCH2",
        "a.txt-8-l8",
    ]


async def test_grep_context_overlap_merges(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "a.txt").write_text("MATCH1\nmid\nMATCH2\n", encoding="utf-8")
    result = await GrepSearch().run(
        GrepSearch.Input(pattern="MATCH", context_before=1, context_after=1), tool_ctx
    )
    assert result.output.splitlines() == [
        "a.txt:1:MATCH1",
        "a.txt-2-mid",
        "a.txt:3:MATCH2",
    ]


async def test_grep_output_modes(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "a.txt").write_text("x\nfoo\nfoo\n", encoding="utf-8")
    (root / "b.txt").write_text("foo\n", encoding="utf-8")

    content = await GrepSearch().run(GrepSearch.Input(pattern="foo"), tool_ctx)
    assert content.output.splitlines() == ["a.txt:2:foo", "a.txt:3:foo", "b.txt:1:foo"]

    files = await GrepSearch().run(
        GrepSearch.Input(pattern="foo", output_mode="files_with_matches"), tool_ctx
    )
    assert files.output.splitlines() == ["a.txt", "b.txt"]

    count = await GrepSearch().run(GrepSearch.Input(pattern="foo", output_mode="count"), tool_ctx)
    assert count.output.splitlines() == ["a.txt:2", "b.txt:1"]


async def test_grep_multiline_and_ignore_case(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "m.txt").write_text("def foo(\n    x\n):\nPASS\n", encoding="utf-8")

    multiline = await GrepSearch().run(
        GrepSearch.Input(pattern=r"foo\(.*?\):", multiline=True), tool_ctx
    )
    assert multiline.output.splitlines() == ["m.txt:1:def foo(", "m.txt:2:    x", "m.txt:3:):"]

    strict = await GrepSearch().run(GrepSearch.Input(pattern="pass"), tool_ctx)
    assert strict.output == ""
    loose = await GrepSearch().run(GrepSearch.Input(pattern="pass", ignore_case=True), tool_ctx)
    assert loose.output.splitlines() == ["m.txt:4:PASS"]


async def test_grep_truncation(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "big.txt").write_text("\n".join(["hit"] * 1200), encoding="utf-8")
    result = await GrepSearch().run(GrepSearch.Input(pattern="hit"), tool_ctx)
    lines = result.output.splitlines()
    assert len(lines) == 1001
    assert lines[-1] == "... (truncated at 1000 matches)"


Handler = Callable[[httpx.Request], httpx.Response]


def _web_ctx(tmp_path: Path, cfg: WebSearchConfig) -> ToolContext:
    jail = WorkspaceJail(tmp_path)
    return ToolContext(workspace_root=jail.root, jail=jail, config=RuntimeConfig(web_search=cfg))


def _install_mock(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    def make(timeout_s: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(web_search_mod, "_make_client", make)


async def test_web_search_stub_when_unconfigured(tmp_path: Path) -> None:
    ctx = _web_ctx(tmp_path, WebSearchConfig(provider=None))
    result = await WebSearch().run(WebSearch.Input(query="anything"), ctx)
    assert result.is_error is True
    assert "no search backend configured" in result.output


async def test_web_search_missing_base_url_degrades(tmp_path: Path) -> None:
    ctx = _web_ctx(tmp_path, WebSearchConfig(provider="searxng"))
    result = await WebSearch().run(WebSearch.Input(query="anything"), ctx)
    assert result.is_error is True
    assert "no search backend configured" in result.output


async def test_web_search_duckduckgo_parse_and_domain_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        '<a class="result__a" href="//duckduckgo.com/l/'
        '?uddg=https%3A%2F%2Fdocs.python.org%2F3%2F&amp;rut=abc">Python <b>Docs</b></a>'
        '<a class="result__a" href="https://example.com/page">Example</a>'
        '<a class="result__a" href="https://docs.python.org/3/">Duplicate</a>'
        '<a class="result__a" href="//duckduckgo.com/y.js?ad_domain=udemy.com&amp;u3=x">Ad</a>'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "html.duckduckgo.com"
        assert request.url.params["q"] == "python docs"
        return httpx.Response(200, text=body)

    _install_mock(monkeypatch, handler)
    ctx = _web_ctx(tmp_path, WebSearchConfig(provider="duckduckgo"))
    result = await WebSearch().run(
        WebSearch.Input(query="python docs", allowed_domains=["python.org"]), ctx
    )
    assert result.is_error is False
    assert result.output == "- [Python Docs](https://docs.python.org/3/)"

    unfiltered = await WebSearch().run(WebSearch.Input(query="python docs"), ctx)
    assert "y.js" not in unfiltered.output
    assert unfiltered.output.splitlines() == [
        "- [Python Docs](https://docs.python.org/3/)",
        "- [Example](https://example.com/page)",
    ]


async def test_web_search_searxng_blocked_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "results": [
            {"title": "A", "url": "https://a.com/1"},
            {"title": "B", "url": "https://b.com/2"},
            {"title": "Sub", "url": "https://sub.b.com/3"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json=payload)

    _install_mock(monkeypatch, handler)
    ctx = _web_ctx(tmp_path, WebSearchConfig(provider="searxng", base_url="http://searx.local"))
    result = await WebSearch().run(
        WebSearch.Input(query="anything", blocked_domains=["b.com"]), ctx
    )
    assert result.output == "- [A](https://a.com/1)"


async def test_web_search_generic_max_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"results": [{"title": f"T{i}", "url": f"https://site{i}.com/"} for i in range(10)]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).startswith("http://api.local/find?q=")
        return httpx.Response(200, json=payload)

    _install_mock(monkeypatch, handler)
    ctx = _web_ctx(
        tmp_path,
        WebSearchConfig(provider="generic", base_url="http://api.local/find", max_results=3),
    )
    result = await WebSearch().run(WebSearch.Input(query="anything"), ctx)
    assert result.output.splitlines() == [
        "- [T0](https://site0.com/)",
        "- [T1](https://site1.com/)",
        "- [T2](https://site2.com/)",
    ]


async def test_web_search_base_url_with_existing_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, text="<html></html>")

    _install_mock(monkeypatch, handler)
    for base_url, expect in [
        ("https://ddg.local/html/?kl=jp", "https://ddg.local/html/?kl=jp&q=anything"),
        ("https://ddg.local/html/?", "https://ddg.local/html/?q=anything"),
        ("https://ddg.local/html/", "https://ddg.local/html/?q=anything"),
    ]:
        ctx = _web_ctx(tmp_path, WebSearchConfig(provider="duckduckgo", base_url=base_url))
        await WebSearch().run(WebSearch.Input(query="anything"), ctx)
        assert seen[-1] == expect


async def test_web_search_network_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    _install_mock(monkeypatch, handler)
    ctx = _web_ctx(tmp_path, WebSearchConfig(provider="duckduckgo"))
    result = await WebSearch().run(WebSearch.Input(query="anything"), ctx)
    assert result.is_error is False
    assert "failed" in result.output


async def test_web_search_http_error_and_no_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_mock(monkeypatch, lambda request: httpx.Response(503, text="down"))
    ctx = _web_ctx(tmp_path, WebSearchConfig(provider="duckduckgo"))
    result = await WebSearch().run(WebSearch.Input(query="anything"), ctx)
    assert result.is_error is False
    assert "HTTP 503" in result.output

    _install_mock(monkeypatch, lambda request: httpx.Response(200, text="<html></html>"))
    empty = await WebSearch().run(WebSearch.Input(query="anything"), ctx)
    assert empty.is_error is False
    assert "No search results matched" in empty.output

    _install_mock(monkeypatch, lambda request: httpx.Response(202, text="<html>anomaly</html>"))
    blocked = await WebSearch().run(WebSearch.Input(query="anything"), ctx)
    assert blocked.is_error is False
    assert "as a bot" in blocked.output


async def test_read_offset_limit_large_file(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    lines = [f"line{i}" for i in range(1, 10_001)]
    (root / "big.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = await ReadFile().run(ReadFile.Input(path="big.txt", offset=5000, limit=3), tool_ctx)
    assert result.output == "line5000\nline5001\nline5002\n"


async def test_read_full_truncated_at_max_bytes(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "huge.txt").write_text("a" * (_BYTE_CAP + 100), encoding="utf-8")
    result = await ReadFile().run(ReadFile.Input(path="huge.txt"), tool_ctx)
    assert result.output.endswith(f"... (truncated at {_BYTE_CAP} bytes)")
    assert len(result.output) < _BYTE_CAP + 100


async def test_glob_exclude_and_limit(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("x", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "d.py").write_text("x", encoding="utf-8")
    (root / "dist").mkdir()
    (root / "dist" / "gen.py").write_text("x", encoding="utf-8")
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "node_modules" / "pkg" / "b.py").write_text("x", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "c.py").write_text("x", encoding="utf-8")
    for p in root.rglob("*"):
        os.utime(p, (1_000_000, 1_000_000))

    result = await GlobSearch().run(GlobSearch.Input(pattern="**/*.py"), tool_ctx)
    assert result.output.splitlines() == ["docs/d.py", "src/a.py"]

    result = await GlobSearch().run(
        GlobSearch.Input(pattern="**/*.py", exclude=["docs/*"]), tool_ctx
    )
    assert result.output.splitlines() == ["src/a.py"]

    for i in range(5):
        f = root / f"f{i}.txt"
        f.write_text("x", encoding="utf-8")
        os.utime(f, (1_000_000, 1_000_000))
    result = await GlobSearch().run(GlobSearch.Input(pattern="f*.txt", limit=2), tool_ctx)
    assert result.output.splitlines() == ["f0.txt", "f1.txt", "... (truncated at 2 entries)"]


def _write_notebook(root: Path) -> Path:
    nb = {
        "cells": [
            {
                "id": "c1",
                "cell_type": "code",
                "metadata": {},
                "source": "print(1)",
                "outputs": [],
                "execution_count": None,
            },
            {"id": "c2", "cell_type": "markdown", "metadata": {}, "source": "# hi"},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = root / "nb.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def _cells(path: Path) -> list[dict[str, object]]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    assert isinstance(cells, list)
    return cells


async def test_notebook_replace_insert_delete(tool_ctx: ToolContext) -> None:
    path = _write_notebook(tool_ctx.workspace_root)
    tool = NotebookEdit()

    await tool.run(
        NotebookEdit.Input(notebook_path="nb.ipynb", cell_id="c1", new_source="print(2)"),
        tool_ctx,
    )
    cells = _cells(path)
    assert len(cells) == 2
    assert cells[0]["source"] == "print(2)"
    assert cells[0]["cell_type"] == "code"

    await tool.run(
        NotebookEdit.Input(
            notebook_path="nb.ipynb",
            cell_id="c1",
            new_source="## new",
            cell_type="markdown",
            edit_mode="insert",
        ),
        tool_ctx,
    )
    cells = _cells(path)
    assert len(cells) == 3
    assert cells[1]["cell_type"] == "markdown"
    assert cells[1]["source"] == "## new"
    assert cells[1]["id"]

    await tool.run(
        NotebookEdit.Input(notebook_path="nb.ipynb", new_source="top", edit_mode="insert"),
        tool_ctx,
    )
    cells = _cells(path)
    assert len(cells) == 4
    assert cells[0]["source"] == "top"

    await tool.run(
        NotebookEdit.Input(notebook_path="nb.ipynb", cell_id="c2", edit_mode="delete"),
        tool_ctx,
    )
    cells = _cells(path)
    assert len(cells) == 3
    assert "c2" not in [c["id"] for c in cells]


async def test_notebook_edit_errors(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    _write_notebook(root)
    (root / "plain.txt").write_text("x", encoding="utf-8")
    tool = NotebookEdit()

    with pytest.raises(ToolError, match="ipynb"):
        await tool.run(
            NotebookEdit.Input(notebook_path="plain.txt", cell_id="c1", new_source="x"),
            tool_ctx,
        )
    with pytest.raises(ToolError, match="not found"):
        await tool.run(
            NotebookEdit.Input(notebook_path="nb.ipynb", cell_id="nope", new_source="x"),
            tool_ctx,
        )
    with pytest.raises(ToolError, match="new_source"):
        await tool.run(NotebookEdit.Input(notebook_path="nb.ipynb", cell_id="c1"), tool_ctx)


async def test_notebook_replace_with_explicit_empty_source(tool_ctx: ToolContext) -> None:
    path = _write_notebook(tool_ctx.workspace_root)
    await NotebookEdit().run(
        NotebookEdit.Input(notebook_path="nb.ipynb", cell_id="c1", new_source=""), tool_ctx
    )
    assert _cells(path)[0]["source"] == ""


def test_descriptions_nonempty_and_schema_stable() -> None:
    by_name = {e["name"]: e for e in default_registry().json_schemas()}
    assert "notebook_edit" in by_name

    for name in ("grep_search", "web_search", "read_file", "glob_search", "notebook_edit"):
        assert len(by_name[name]["description"]) > 80, name

    grep_props = by_name["grep_search"]["input_schema"]["properties"]
    assert {"output_mode", "context_before", "context_after", "multiline", "ignore_case"} <= set(
        grep_props
    )
    assert {"allowed_domains", "blocked_domains"} <= set(
        by_name["web_search"]["input_schema"]["properties"]
    )
    assert {"exclude", "limit"} <= set(by_name["glob_search"]["input_schema"]["properties"])
    assert set(by_name["notebook_edit"]["input_schema"]["required"]) == {"notebook_path"}
