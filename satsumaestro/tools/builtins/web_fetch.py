"""web_fetch 組み込みツール。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from typing import ClassVar, Literal

import httpx
from pydantic import BaseModel, Field

from satsumaestro.config.models import RuntimeConfig, WebFetchConfig
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.html_text import html_to_markdown

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

_MIN_PART_TOKENS = 2_000
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
TURNED_OFF = (
    "web_fetch is turned off. Only the user can turn it on, with the setting "
    "web_fetch.enabled (not a tool argument)."
)


def _make_client(timeout_s: float) -> httpx.AsyncClient:
    """httpx クライアントの生成点（テストが MockTransport で monkeypatch する）。"""
    return httpx.AsyncClient(timeout=timeout_s, follow_redirects=False)


async def _resolve_host(host: str, port: int) -> list[IPAddress]:
    """ホストを解決したアドレスの一覧（テストが monkeypatch する生成点）。"""
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return [ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]) for info in infos]


def part_budget(cfg: RuntimeConfig) -> int:
    """1 回で返す本文の見積もりトークン数の上限（圧縮の閾値の 1/4）。"""
    threshold = cfg.compact_token_threshold
    if threshold <= 0:
        threshold = RuntimeConfig.model_fields["compact_token_threshold"].default
    return max(_MIN_PART_TOKENS, threshold // 4)


def part_end(text: str, start: int, budget: int) -> int:
    """``start`` から見積もりで ``budget`` トークンに収まる末尾の位置。"""
    units = budget * 4
    for i in range(start, len(text)):
        units -= 1 if text[i] < "\x80" else 4
        if units < 0:
            return i
    return len(text)


def is_local_address(ip: IPAddress) -> bool:
    """インターネットの外の宛先か。"""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast


async def _refusal(url: httpx.URL, cfg: WebFetchConfig) -> str | None:
    """取りに行かない URL なら拒否の文。"""
    if url.scheme not in ("http", "https"):
        return f"web_fetch refused: only http and https URLs can be fetched: {url}"
    host = url.host
    if not host:
        return f"web_fetch refused: the URL has no host: {url}"
    if cfg.allow_private_hosts:
        return None
    port = url.port or (443 if url.scheme == "https" else 80)
    try:
        addresses = await _resolve_host(host, port)
    except (OSError, ValueError) as e:
        return f"web_fetch refused: could not resolve {host} ({e})"
    if not addresses:
        return f"web_fetch refused: could not resolve {host}"
    local = next((a for a in addresses if is_local_address(a)), None)
    if local is not None:
        return (
            f"web_fetch refused: {host} resolves to a private or local address ({local}). "
            "Only the user can allow these addresses, with the setting "
            "web_fetch.allow_private_hosts (not a tool argument)."
        )
    return None


def _redirect_target(resp: httpx.Response) -> httpx.URL | None:
    """リダイレクトなら次の宛先（相対の Location は今の URL から解決する）。"""
    location = resp.headers.get("location")
    if resp.status_code not in _REDIRECT_CODES or not location:
        return None
    return resp.url.join(location)


async def _read_body(resp: httpx.Response, limit: int) -> tuple[bytes, bool] | str:
    """本文を上限まで読む。"""
    declared = resp.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        return (
            f"web_fetch refused: the response is {int(declared)} bytes, over the limit of "
            f"{limit} bytes (the user's setting web_fetch.max_bytes). Nothing was read."
        )
    chunks: list[bytes] = []
    size = 0
    async for chunk in resp.aiter_bytes():
        if size + len(chunk) > limit:
            chunks.append(chunk[: limit - size])
            return b"".join(chunks), True
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks), False


def _looks_like_html(content_type: str, body: str) -> bool:
    """Content-Type で HTML か判定する（欠落時のみ先頭バイトでスニフ）。"""
    ct = content_type.lower()
    if ct:
        return "html" in ct or "xhtml" in ct
    return body.lstrip()[:1] == "<"


MAX_TIMEOUT_S = 120.0


class WebFetch(Tool):
    """URL を取得して本文テキストを返す。"""

    class Input(BaseModel):
        url: str = Field(description="Absolute http(s) URL to fetch.")
        timeout_s: float = Field(
            default=30.0,
            gt=0.0,
            description="Seconds to wait for the response (at most 120; larger values are "
            "lowered to 120).",
        )
        mode: Literal["markdown", "raw"] = Field(
            default="markdown",
            description='"markdown" extracts the main content of HTML pages as Markdown '
            '(default); "raw" returns the response body unmodified.',
        )
        start: int = Field(
            default=0,
            ge=0,
            description="Character offset to start from, in the content of the same mode. Long "
            "content is returned in parts; the result tells how to call for the next part.",
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="web_fetch",
        description=(
            "Fetch a URL and return its content. HTML pages are reduced to their main "
            'content as Markdown by default; pass mode="raw" for the unmodified body.'
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, WebFetch.Input)
        cfg = ctx.config.web_fetch
        if not cfg.enabled:
            return ToolResult(output=TURNED_OFF, is_error=True)
        try:
            url = httpx.URL(inp.url)
        except httpx.InvalidURL as e:
            return ToolResult(output=f"fetch failed: invalid URL: {e}", is_error=True)
        try:
            async with _make_client(min(inp.timeout_s, MAX_TIMEOUT_S)) as client:
                for _ in range(cfg.max_redirects + 1):
                    refusal = await _refusal(url, cfg)
                    if refusal is not None:
                        return ToolResult(output=refusal, is_error=True)
                    async with client.stream("GET", url) as resp:
                        target = _redirect_target(resp)
                        if target is None:
                            read = await _read_body(resp, cfg.max_bytes)
                            if isinstance(read, str):
                                return ToolResult(output=read, is_error=True)
                            data, truncated = read
                            text = data.decode(resp.encoding or "utf-8", errors="replace")
                            return _result(
                                resp,
                                text,
                                inp,
                                note=_cut_note(cfg.max_bytes) if truncated else "",
                                budget=part_budget(ctx.config),
                            )
                    url = target
        except httpx.HTTPError as e:
            return ToolResult(output=f"fetch failed: {e}", is_error=True)
        return ToolResult(
            output=f"fetch failed: more than {cfg.max_redirects} redirects (last: {url})",
            is_error=True,
        )


def _cut_note(limit: int) -> str:
    """取得の上限で切ったときの注記。"""
    return f"\n\n(truncated at {limit} bytes, the user's setting web_fetch.max_bytes)"


def _part(text: str, inp: WebFetch.Input, start: int, budget: int) -> tuple[str, str] | None:
    """``start`` 字目から見積もりで ``budget`` トークン分と、続きの取り方の注記。"""
    total = len(text)
    if start and start >= total:
        return None
    end = part_end(text, start, budget)
    if end < total:
        args = f'url={json.dumps(inp.url, ensure_ascii=False)}, mode="{inp.mode}", start={end}'
        note = (
            f"\n\n(showing characters {start}-{end} of {total}. To read on, call web_fetch "
            f"with {args})"
        )
    elif start:
        note = f"\n\n(showing characters {start}-{end} of {total}, the end of the content)"
    else:
        note = ""
    return text[start:end], note


def _result(
    resp: httpx.Response, body: str, inp: WebFetch.Input, *, note: str, budget: int
) -> ToolResult:
    """取得した本文をツールの結果にする（HTML は本文を抽出して Markdown に）。"""
    is_error = resp.status_code >= 400
    text = body
    if (
        inp.mode == "markdown"
        and not is_error
        and _looks_like_html(resp.headers.get("content-type", ""), body)
    ):
        text = html_to_markdown(body) or body
    part = _part(text, inp, 0 if is_error else inp.start, budget)
    if part is None:
        return ToolResult(
            output=f"web_fetch: start={inp.start} is past the end of the content "
            f"({len(text)} characters).",
            is_error=True,
        )
    shown, more = part
    return ToolResult(output=f"HTTP {resp.status_code}\n\n{shown}{more}{note}", is_error=is_error)
