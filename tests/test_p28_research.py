"""Web リサーチ強化のテスト。"""

from __future__ import annotations

import ipaddress

import httpx
import pytest

from satsumaestro.config.models import RuntimeConfig, WebFetchConfig, WebSearchConfig
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import web_fetch as web_fetch_mod
from satsumaestro.tools.builtins import web_search as web_search_mod
from satsumaestro.tools.builtins.web_fetch import WebFetch
from satsumaestro.tools.builtins.web_search import WebSearch
from satsumaestro.util.html_text import html_to_markdown

_PAGE = """
<html><head><title>Doc &amp; Guide</title><style>body{color:red}</style></head>
<body>
<nav><a href="/home">Home</a> | <a href="/about">About</a></nav>
<aside>Ad banner here</aside>
<article>
<h1>Getting Started</h1>
<p>Install the tool with <code>pip install satsuma</code> and read the
<a href="https://example.com/docs">documentation</a>.</p>
<h2>Steps</h2>
<ul><li>first step</li><li>second step</li></ul>
<ol><li>one</li><li>two</li></ol>
<pre>make check
make run</pre>
</article>
<footer>Copyright 2026</footer>
<script>alert("x")</script>
</body></html>
"""


def test_html_to_markdown_basic() -> None:
    md = html_to_markdown(_PAGE)
    assert md.startswith("# Doc & Guide")
    assert "# Getting Started" in md and "## Steps" in md
    assert "`pip install satsuma`" in md
    assert "[documentation](https://example.com/docs)" in md
    assert "- first step" in md and "- second step" in md
    assert "1. one" in md and "2. two" in md
    assert "```\nmake check\nmake run\n```" in md


def test_html_extract_drops_boilerplate() -> None:
    md = html_to_markdown(_PAGE)
    for noise in ("alert", "color:red", "Ad banner", "Copyright 2026", "Home"):
        assert noise not in md


def test_main_container_selection_without_article() -> None:
    html = (
        "<html><body>"
        '<div id="sidebar"><ul><li>nav1</li><li>nav2</li></ul></div>'
        '<div id="content"><p>' + "main body text " * 50 + "</p></div>"
        "</body></html>"
    )
    md = html_to_markdown(html)
    assert "main body text" in md and "nav1" not in md


def test_html_to_markdown_empty_returns_empty() -> None:
    assert html_to_markdown("<html><body><script>x()</script></body></html>") == ""


def _on(tool_ctx: ToolContext) -> ToolContext:
    """web_fetch を有効にした設定（既定は無効）。"""
    cfg = RuntimeConfig(web_fetch=WebFetchConfig(enabled=True))
    return tool_ctx.model_copy(update={"config": cfg})


def _mock_fetch(monkeypatch: pytest.MonkeyPatch, handler) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        web_fetch_mod,
        "_make_client",
        lambda timeout_s: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def public(host: str, port: int) -> list[ipaddress.IPv4Address]:
        return [ipaddress.IPv4Address("93.184.216.34")]

    monkeypatch.setattr(web_fetch_mod, "_resolve_host", public)


async def test_fetch_mode_raw_and_markdown(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_PAGE, headers={"content-type": "text/html"})

    _mock_fetch(monkeypatch, handler)
    md = await WebFetch().run(WebFetch.Input(url="https://x/doc"), _on(tool_ctx))
    assert "# Getting Started" in md.output and "<article>" not in md.output
    raw = await WebFetch().run(WebFetch.Input(url="https://x/doc", mode="raw"), _on(tool_ctx))
    assert "<article>" in raw.output


async def test_fetch_non_html_stays_raw(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text='{"key": "<b>value</b>"}', headers={"content-type": "application/json"}
        )

    _mock_fetch(monkeypatch, handler)
    res = await WebFetch().run(WebFetch.Input(url="https://x/api"), _on(tool_ctx))
    assert '{"key": "<b>value</b>"}' in res.output


async def test_fetch_empty_extraction_degrades_to_raw(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "<html><body><script>only_scripts()</script></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    _mock_fetch(monkeypatch, handler)
    res = await WebFetch().run(WebFetch.Input(url="https://x/empty"), _on(tool_ctx))
    assert not res.is_error and "only_scripts" in res.output


async def test_fetch_max_bytes_after_extract(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        html = "<html><body><article><p>" + "word " * 100 + "</p></article></body></html>"
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    _mock_fetch(monkeypatch, handler)
    monkeypatch.setattr(web_fetch_mod, "part_budget", lambda cfg: 10)
    res = await WebFetch().run(WebFetch.Input(url="https://x/long"), _on(tool_ctx))
    _, body, note = res.output.split("\n\n")
    assert len(body) == 40 and body.startswith("word")
    assert note.startswith("(showing characters 0-40 of ")


async def test_preferred_domains_soft_boost(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "results": [
            {"title": "blog", "url": "https://blog.example.net/a"},
            {"title": "official", "url": "https://docs.python.org/3/"},
            {"title": "forum", "url": "https://forum.example.org/t"},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        web_search_mod,
        "_make_client",
        lambda timeout_s: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    cfg = RuntimeConfig(
        web_search=WebSearchConfig(
            provider="generic",
            base_url="https://search.local",
            preferred_domains=["python.org"],
        )
    )
    ctx = tool_ctx.model_copy(update={"config": cfg})
    res = await WebSearch().run(WebSearch.Input(query="python docs"), ctx)
    lines = res.output.splitlines()
    assert lines[0] == "- [official](https://docs.python.org/3/)"
    assert "[blog]" in lines[1] and "[forum]" in lines[2]


def test_prompt_has_routing_guidance() -> None:
    cfg = RuntimeConfig(web_search=WebSearchConfig(provider="duckduckgo"))
    static = SystemPromptBuilder()._static_sections("m", cfg)
    assert "# Web search usage" in static and "time-sensitive" in static


def test_default_prompt_has_no_search_guidance() -> None:
    static = SystemPromptBuilder()._static_sections("m", RuntimeConfig())
    assert "# Web search usage" not in static


def test_prompt_omits_guidance_when_search_disabled() -> None:
    cfg = RuntimeConfig(web_search=WebSearchConfig(provider=None))
    static = SystemPromptBuilder()._static_sections("m", cfg)
    assert "# Web search usage" not in static
