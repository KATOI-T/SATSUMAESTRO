"""web_fetch の送り先の許可リスト（web_fetch.allowed_hosts）。"""

from __future__ import annotations

import ipaddress
from collections.abc import Callable

import httpx
import pytest
from pydantic import ValidationError

from satsumaestro.config.models import RuntimeConfig, WebFetchConfig
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import web_fetch as web_fetch_mod
from satsumaestro.tools.builtins.web_fetch import WebFetch
from satsumaestro.util.urls import host_matches

PUBLIC = "93.184.216.34"
_DNS = {
    "example.com": [PUBLIC],
    "docs.example.com": [PUBLIC],
    "notexample.com": [PUBLIC],
    "other.example.org": [PUBLIC],
    "intranet.example.com": ["10.0.0.5"],
}


class _Server:
    """MockTransport の応答と、届いた URL と、引いた名前の記録。"""

    def __init__(self) -> None:
        self.routes: dict[str, httpx.Response] = {}
        self.seen: list[str] = []
        self.resolved: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(str(request.url))
        return self.routes.get(str(request.url), httpx.Response(200, text="ok"))


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _Server:
    srv = _Server()
    monkeypatch.setattr(
        web_fetch_mod,
        "_make_client",
        lambda timeout_s: httpx.AsyncClient(transport=httpx.MockTransport(srv.handler)),
    )

    async def resolve(host: str, port: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        srv.resolved.append(host)
        try:
            return [ipaddress.ip_address(host)]
        except ValueError:
            return [ipaddress.ip_address(a) for a in _DNS[host]]

    monkeypatch.setattr(web_fetch_mod, "_resolve_host", resolve)
    return srv


def _ctx(tool_ctx: ToolContext, **web_fetch: object) -> ToolContext:
    cfg = RuntimeConfig(web_fetch=WebFetchConfig(enabled=True, **web_fetch))  # type: ignore[arg-type]
    return tool_ctx.model_copy(update={"config": cfg})


async def _fetch(ctx: ToolContext, url: str) -> tuple[str, bool]:
    res = await WebFetch().run(WebFetch.Input(url=url, mode="raw"), ctx)
    return res.output, res.is_error


def _refused_for(host: str) -> Callable[[str], bool]:
    def check(out: str) -> bool:
        return (
            out.startswith(f"web_fetch refused: {host} is not in web_fetch.allowed_hosts")
            and "Only the user can add hosts" in out
            and "not a tool argument" in out
        )

    return check


@pytest.mark.parametrize("url", ["https://example.com/p", "https://docs.example.com/p"])
async def test_a_listed_host_and_its_subdomains_are_fetched(
    tool_ctx: ToolContext, server: _Server, url: str
) -> None:
    out, err = await _fetch(_ctx(tool_ctx, allowed_hosts=["example.com"]), url)
    assert not err and out.endswith("ok") and server.seen == [url]


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("https://other.example.org/p", "other.example.org"),
        ("https://notexample.com/p", "notexample.com"),
        ("http://127.0.0.1:8000/p", "127.0.0.1"),
    ],
)
async def test_an_unlisted_host_is_refused_before_its_name_is_looked_up(
    tool_ctx: ToolContext, server: _Server, url: str, host: str
) -> None:
    out, err = await _fetch(_ctx(tool_ctx, allowed_hosts=["example.com"]), url)
    assert err and _refused_for(host)(out)
    assert server.resolved == [] and server.seen == []


async def test_the_list_applies_even_when_private_hosts_are_allowed(
    tool_ctx: ToolContext, server: _Server
) -> None:
    ctx = _ctx(tool_ctx, allowed_hosts=["example.com"], allow_private_hosts=True)
    out, err = await _fetch(ctx, "http://127.0.0.1:8000/p")
    assert err and _refused_for("127.0.0.1")(out) and server.seen == []
    ctx = _ctx(tool_ctx, allowed_hosts=["127.0.0.1"], allow_private_hosts=True)
    out, err = await _fetch(ctx, "http://127.0.0.1:8000/p")
    assert not err and out.endswith("ok")


async def test_a_redirect_to_an_unlisted_host_is_refused(
    tool_ctx: ToolContext, server: _Server
) -> None:
    server.routes["https://example.com/go"] = httpx.Response(
        302, headers={"location": "https://other.example.org/x"}
    )
    out, err = await _fetch(_ctx(tool_ctx, allowed_hosts=["example.com"]), "https://example.com/go")
    assert err and _refused_for("other.example.org")(out)
    assert server.seen == ["https://example.com/go"]
    assert "other.example.org" not in server.resolved


async def test_a_redirect_within_the_list_is_followed(
    tool_ctx: ToolContext, server: _Server
) -> None:
    server.routes["https://example.com/go"] = httpx.Response(
        302, headers={"location": "https://docs.example.com/x"}
    )
    out, err = await _fetch(_ctx(tool_ctx, allowed_hosts=["example.com"]), "https://example.com/go")
    assert not err and out.endswith("ok") and len(server.seen) == 2


async def test_an_empty_list_does_not_limit_the_hosts(
    tool_ctx: ToolContext, server: _Server
) -> None:
    out, err = await _fetch(_ctx(tool_ctx), "https://other.example.org/p")
    assert not err and out.endswith("ok")


async def test_a_listed_host_still_needs_a_public_address(
    tool_ctx: ToolContext, server: _Server
) -> None:
    ctx = _ctx(tool_ctx, allowed_hosts=["intranet.example.com"])
    out, err = await _fetch(ctx, "https://intranet.example.com/wiki")
    assert err and "(10.0.0.5)" in out and "web_fetch.allow_private_hosts" in out
    assert server.seen == []


@pytest.mark.parametrize("listed", ["Example.COM", "example.com.", ".example.com"])
async def test_listed_names_ignore_case_and_dots_at_the_ends(
    tool_ctx: ToolContext, server: _Server, listed: str
) -> None:
    out, err = await _fetch(_ctx(tool_ctx, allowed_hosts=[listed]), "https://docs.example.com/p")
    assert not err and out.endswith("ok")


@pytest.mark.parametrize(
    ("host", "domain", "expected"),
    [
        ("example.com", "example.com", True),
        ("docs.example.com", "example.com", True),
        ("notexample.com", "example.com", False),
        ("example.com", "docs.example.com", False),
        ("EXAMPLE.com.", "example.com", True),
        ("example.com", ".Example.com.", True),
    ],
)
def test_host_matches(host: str, domain: str, expected: bool) -> None:
    assert host_matches(host, domain) is expected


@pytest.mark.parametrize("bad", ["https://example.com", "example.com/path", "", "  "])
def test_urls_paths_and_empty_names_are_setting_errors(bad: str) -> None:
    with pytest.raises(ValidationError, match="allowed_hosts"):
        WebFetchConfig(allowed_hosts=[bad])
