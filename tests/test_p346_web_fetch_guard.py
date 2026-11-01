"""web_fetch の宛先と大きさの制限。"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import AsyncIterator

import httpx
import pytest

from satsumaestro.config.models import RuntimeConfig, WebFetchConfig
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import web_fetch as web_fetch_mod
from satsumaestro.tools.builtins.web_fetch import WebFetch, is_local_address

PUBLIC = "93.184.216.34"
_DNS = {
    "example.com": [PUBLIC],
    "docs.example.com": [PUBLIC],
    "intranet.example.com": ["10.0.0.5"],
    "mixed.example.com": [PUBLIC, "192.168.1.10"],
    "localhost": ["127.0.0.1", "::1"],
}


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "0.0.0.0",
        "100.64.0.1",
        "224.0.0.1",
        "::1",
        "fd00::1",
        "fe80::1",
        "::ffff:10.0.0.1",
        "::ffff:127.0.0.1",
    ],
)
def test_local_addresses(address: str) -> None:
    assert is_local_address(ipaddress.ip_address(address))


@pytest.mark.parametrize("address", [PUBLIC, "8.8.8.8", "2606:4700::1111", "::ffff:8.8.8.8"])
def test_public_addresses(address: str) -> None:
    assert not is_local_address(ipaddress.ip_address(address))


class _Server:
    """MockTransport の応答と、届いた URL の記録。"""

    def __init__(self, routes: dict[str, httpx.Response]) -> None:
        self.routes = routes
        self.seen: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(str(request.url))
        return self.routes.get(str(request.url), httpx.Response(404, text="missing"))


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _Server:
    srv = _Server({})
    monkeypatch.setattr(
        web_fetch_mod,
        "_make_client",
        lambda timeout_s: httpx.AsyncClient(transport=httpx.MockTransport(srv.handler)),
    )

    async def resolve(host: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        try:
            return [ipaddress.ip_address(host)]
        except ValueError:
            pass
        if host not in _DNS:
            raise socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided")
        return [ipaddress.ip_address(a) for a in _DNS[host]]

    monkeypatch.setattr(web_fetch_mod, "_resolve_host", resolve)
    return srv


def _ctx(tool_ctx: ToolContext, **web_fetch: object) -> ToolContext:
    """web_fetch を有効にした設定（既定は無効）。"""
    cfg = RuntimeConfig(web_fetch=WebFetchConfig(enabled=True, **web_fetch))  # type: ignore[arg-type]
    return tool_ctx.model_copy(update={"config": cfg})


async def _fetch(ctx: ToolContext, url: str) -> tuple[str, bool]:
    res = await WebFetch().run(WebFetch.Input(url=url, mode="raw"), ctx)
    return res.output, res.is_error


async def test_public_host_is_fetched(tool_ctx: ToolContext, server: _Server) -> None:
    server.routes["https://example.com/page"] = httpx.Response(200, text="hello")
    out, err = await _fetch(_ctx(tool_ctx), "https://example.com/page")
    assert not err and out == "HTTP 200\n\nhello"


@pytest.mark.parametrize(
    ("url", "address"),
    [
        ("http://127.0.0.1:11434/", "127.0.0.1"),
        ("http://169.254.169.254/latest/meta-data/", "169.254.169.254"),
        ("http://[::1]:8080/", "::1"),
        ("http://[fd00::1]/", "fd00::1"),
        ("http://0.0.0.0:8000/", "0.0.0.0"),
        ("http://localhost:3000/", "127.0.0.1"),
        ("https://intranet.example.com/wiki", "10.0.0.5"),
        ("https://mixed.example.com/", "192.168.1.10"),
    ],
)
async def test_local_destinations_are_refused(
    tool_ctx: ToolContext, server: _Server, url: str, address: str
) -> None:
    out, err = await _fetch(_ctx(tool_ctx), url)
    assert err and out.startswith("web_fetch refused:") and f"({address})" in out
    assert "Only the user can allow" in out and "not a tool argument" in out
    assert server.seen == []


async def test_unresolvable_host_is_refused(tool_ctx: ToolContext, server: _Server) -> None:
    out, err = await _fetch(_ctx(tool_ctx), "https://no-such-host.invalid/")
    assert err and out.startswith("web_fetch refused: could not resolve no-such-host.invalid")
    assert server.seen == []


@pytest.mark.parametrize("url", ["ftp://example.com/file", "file:///etc/passwd"])
async def test_only_http_and_https(tool_ctx: ToolContext, server: _Server, url: str) -> None:
    out, err = await _fetch(_ctx(tool_ctx), url)
    assert err and "only http and https" in out and server.seen == []


async def test_allow_private_hosts_skips_the_check(tool_ctx: ToolContext, server: _Server) -> None:
    server.routes["http://127.0.0.1:8000/x"] = httpx.Response(200, text="local")
    out, err = await _fetch(_ctx(tool_ctx, allow_private_hosts=True), "http://127.0.0.1:8000/x")
    assert not err and out.endswith("local")


async def test_redirect_to_metadata_is_refused(tool_ctx: ToolContext, server: _Server) -> None:
    server.routes["https://example.com/go"] = httpx.Response(
        302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
    )
    out, err = await _fetch(_ctx(tool_ctx), "https://example.com/go")
    assert err and "(169.254.169.254)" in out
    assert server.seen == ["https://example.com/go"]


async def test_redirects_are_followed_with_relative_locations(
    tool_ctx: ToolContext, server: _Server
) -> None:
    server.routes["https://example.com/a"] = httpx.Response(
        301, headers={"location": "https://docs.example.com/b"}
    )
    server.routes["https://docs.example.com/b"] = httpx.Response(
        307, headers={"location": "/c?x=1"}
    )
    server.routes["https://docs.example.com/c?x=1"] = httpx.Response(200, text="done")
    out, err = await _fetch(_ctx(tool_ctx), "https://example.com/a")
    assert not err and out.endswith("done") and len(server.seen) == 3


async def test_too_many_redirects(tool_ctx: ToolContext, server: _Server) -> None:
    for i in range(4):
        server.routes[f"https://example.com/{i}"] = httpx.Response(
            302, headers={"location": f"https://example.com/{i + 1}"}
        )
    out, err = await _fetch(_ctx(tool_ctx, max_redirects=2), "https://example.com/0")
    assert err and "more than 2 redirects" in out and len(server.seen) == 3


class _Explodes(httpx.AsyncByteStream):
    """読まれたら失敗する本文（Content-Length で読まずに断ることを確かめる）。"""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise AssertionError("the body was read")
        yield b""  # pragma: no cover


async def test_declared_size_over_the_limit_is_not_read(
    tool_ctx: ToolContext, server: _Server
) -> None:
    server.routes["https://example.com/big"] = httpx.Response(
        200, headers={"content-length": "5000000"}, stream=_Explodes()
    )
    out, err = await _fetch(_ctx(tool_ctx), "https://example.com/big")
    assert err and out == (
        "web_fetch refused: the response is 5000000 bytes, over the limit of 2000000 bytes "
        "(the user's setting web_fetch.max_bytes). Nothing was read."
    )


class _Chunks(httpx.AsyncByteStream):
    """Content-Length の無い、小分けに届く本文。"""

    def __init__(self, parts: list[bytes]) -> None:
        self.parts = parts
        self.sent = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self.parts:
            self.sent += 1
            yield part


async def test_stream_over_the_limit_is_cut(tool_ctx: ToolContext, server: _Server) -> None:
    body = _Chunks([b"a" * 6, b"b" * 6, b"c" * 6, b"d" * 6])
    server.routes["https://example.com/stream"] = httpx.Response(200, stream=body)
    out, err = await _fetch(_ctx(tool_ctx, max_bytes=10), "https://example.com/stream")
    assert not err
    assert out == (
        "HTTP 200\n\naaaaaabbbb\n\n(truncated at 10 bytes, the user's setting web_fetch.max_bytes)"
    )
    assert body.sent == 2


async def test_cut_in_the_middle_of_a_character(tool_ctx: ToolContext, server: _Server) -> None:
    server.routes["https://example.com/ja"] = httpx.Response(
        200,
        stream=_Chunks(["日本語".encode()]),
        headers={"content-type": "text/plain; charset=utf-8"},
    )
    out, err = await _fetch(_ctx(tool_ctx, max_bytes=4), "https://example.com/ja")
    assert not err and out.startswith("HTTP 200\n\n日�")


async def test_html_is_extracted_after_the_cut(tool_ctx: ToolContext, server: _Server) -> None:
    html = "<html><body><article><h1>Title</h1><p>body text</p></article></body></html>"
    server.routes["https://example.com/doc"] = httpx.Response(
        200, text=html, headers={"content-type": "text/html"}
    )
    res = await WebFetch().run(WebFetch.Input(url="https://example.com/doc"), _ctx(tool_ctx))
    assert not res.is_error and "# Title" in res.output and "<article>" not in res.output


async def test_disabled_by_default_does_not_connect(tool_ctx: ToolContext, server: _Server) -> None:
    server.routes["https://example.com/page"] = httpx.Response(200, text="hello")
    out, err = await _fetch(tool_ctx, "https://example.com/page")
    assert err and out == (
        "web_fetch is turned off. Only the user can turn it on, with the setting "
        "web_fetch.enabled (not a tool argument)."
    )
    assert server.seen == []
