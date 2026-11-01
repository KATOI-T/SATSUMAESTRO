"""web_fetch が返す本文を文脈に合わせて切り、続きを ``start`` で取れるようにする。"""

from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator

import httpx
import pytest

from satsumaestro.config.models import RuntimeConfig, WebFetchConfig
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import web_fetch as web_fetch_mod
from satsumaestro.tools.builtins.web_fetch import WebFetch, part_budget, part_end

_URL = "https://example.com/long"


class _Chunks(httpx.AsyncByteStream):
    """Content-Length の無い、小分けに届く本文。"""

    def __init__(self, parts: list[bytes]) -> None:
        self.parts = parts

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            yield part


@pytest.fixture
def pages(monkeypatch: pytest.MonkeyPatch) -> dict[str, httpx.Response]:
    """URL → 応答。"""
    routes: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url), httpx.Response(404, text="missing"))

    async def resolve(host: str, port: int) -> list[ipaddress.IPv4Address]:
        return [ipaddress.IPv4Address("93.184.216.34")]

    monkeypatch.setattr(
        web_fetch_mod,
        "_make_client",
        lambda timeout_s: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(web_fetch_mod, "_resolve_host", resolve)
    return routes


def _ctx(tool_ctx: ToolContext, threshold: int = 120_000, **web_fetch: object) -> ToolContext:
    cfg = RuntimeConfig(
        compact_token_threshold=threshold,
        web_fetch=WebFetchConfig(enabled=True, **web_fetch),
    )
    return tool_ctx.model_copy(update={"config": cfg})


async def _fetch(ctx: ToolContext, start: int = 0, mode: str = "raw") -> tuple[str, bool]:
    res = await WebFetch().run(WebFetch.Input(url=_URL, mode=mode, start=start), ctx)
    return res.output, res.is_error


@pytest.mark.parametrize(
    ("threshold", "budget"),
    [
        (120_000, 30_000),
        (26_214, 6_553),
        (4_000, 2_000),
        (0, 30_000),
    ],
)
def test_budget_is_a_quarter_of_the_compaction_threshold(threshold: int, budget: int) -> None:
    assert part_budget(RuntimeConfig(compact_token_threshold=threshold)) == budget


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("a" * 100, 0, 40),
        ("あ" * 100, 0, 10),
        ("a" * 20 + "あ" * 100, 0, 25),
        ("a" * 100, 50, 90),
        ("a" * 30, 0, 30),
    ],
)
def test_part_end_counts_ascii_as_a_quarter_token(text: str, start: int, end: int) -> None:
    assert part_end(text, start, 10) == end


async def test_long_content_is_returned_in_parts(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response]
) -> None:
    text = "".join(f"{i:07d}," for i in range(12_500))
    pages[_URL] = httpx.Response(200, text=text)
    ctx = _ctx(tool_ctx, threshold=40_000)

    first, err1 = await _fetch(ctx)
    second, err2 = await _fetch(ctx, start=40_000)
    last, err3 = await _fetch(ctx, start=80_000)

    assert not (err1 or err2 or err3)
    assert first.endswith(
        "\n\n(showing characters 0-40000 of 100000. To read on, call web_fetch with "
        'url="https://example.com/long", mode="raw", start=40000)'
    )
    assert second.endswith('mode="raw", start=80000)')
    assert last.endswith("\n\n(showing characters 80000-100000 of 100000, the end of the content)")
    parts = [out.split("\n\n")[1] for out in (first, second, last)]
    assert [len(p) for p in parts] == [40_000, 40_000, 20_000]
    assert "".join(parts) == text


async def test_an_english_page_that_fits_comes_whole_but_japanese_is_cut(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response]
) -> None:
    ctx = _ctx(tool_ctx, threshold=104_857)
    pages[_URL] = httpx.Response(200, text="word " * 19_400)
    english, _ = await _fetch(ctx)
    pages[_URL] = httpx.Response(200, text="天気" * 15_000)
    japanese, _ = await _fetch(ctx)

    assert "(showing" not in english and len(english) == len("HTTP 200\n\n") + 97_000
    assert japanese.endswith('mode="raw", start=26214)')


async def test_short_content_has_no_note(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response]
) -> None:
    pages[_URL] = httpx.Response(200, text="hello")
    assert await _fetch(_ctx(tool_ctx)) == ("HTTP 200\n\nhello", False)


async def test_start_past_the_end_is_an_error(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response]
) -> None:
    pages[_URL] = httpx.Response(200, text="hello")
    out, err = await _fetch(_ctx(tool_ctx), start=5)
    assert err
    assert out == "web_fetch: start=5 is past the end of the content (5 characters)."


async def test_an_http_error_is_shown_whatever_the_start(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response]
) -> None:
    pages[_URL] = httpx.Response(403, text="Please set a user-agent")
    out, err = await _fetch(_ctx(tool_ctx), start=29_647)
    assert err
    assert out == "HTTP 403\n\nPlease set a user-agent"


async def test_small_context_gets_smaller_parts(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response]
) -> None:
    pages[_URL] = httpx.Response(200, text="天気" * 10_000)
    out, _ = await _fetch(_ctx(tool_ctx, threshold=26_214))
    assert len(out.split("\n\n")[1]) == 6_553
    assert out.endswith('mode="raw", start=6553)')


async def test_parts_are_cut_from_the_extracted_text(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response]
) -> None:
    html = (
        "<html><body><nav>menu</nav><article>" + "<p>word</p>" * 5_000 + "</article></body></html>"
    )
    pages[_URL] = httpx.Response(200, text=html, headers={"content-type": "text/html"})
    ctx = _ctx(tool_ctx, threshold=8_000)

    out, err = await _fetch(ctx, mode="markdown")

    assert not err
    body, note = out.split("\n\n", 1)[1].rsplit("\n\n", 1)
    assert len(body) == 8_000 and body.startswith("word") and "menu" not in body
    total = int(note.split(" of ")[1].split(".")[0])
    assert total < len(html)
    assert note.endswith('mode="markdown", start=8000)')


async def test_download_cut_note_follows_the_part_note(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response]
) -> None:
    pages[_URL] = httpx.Response(200, stream=_Chunks([b"a" * 10_000] * 5))
    out, _ = await _fetch(_ctx(tool_ctx, threshold=8_000, max_bytes=40_000))
    assert out.endswith(
        "start=8000)\n\n(truncated at 40000 bytes, the user's setting web_fetch.max_bytes)"
    )


async def test_a_huge_timeout_is_lowered_to_the_cap(
    tool_ctx: ToolContext, pages: dict[str, httpx.Response], monkeypatch: pytest.MonkeyPatch
) -> None:
    pages[_URL] = httpx.Response(200, text="ok", headers={"content-type": "text/plain"})
    seen: list[float] = []
    make = web_fetch_mod._make_client

    def capture(timeout_s: float) -> httpx.AsyncClient:
        seen.append(timeout_s)
        return make(timeout_s)

    monkeypatch.setattr(web_fetch_mod, "_make_client", capture)
    ctx = _ctx(tool_ctx)
    await WebFetch().run(WebFetch.Input(url=_URL, timeout_s=100000), ctx)
    await WebFetch().run(WebFetch.Input(url=_URL, timeout_s=5), ctx)
    assert seen == [120.0, 5.0]
