"""web_search 組み込みツール。"""

from __future__ import annotations

import html
import re
import urllib.parse
from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from satsumaestro.config.models import WebSearchConfig
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema

_DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
TURNED_OFF = (
    "web_search is turned off. Only the user can turn it on, with the settings "
    "web_search.provider and web_search.base_url (not tool arguments)."
)
_DESCRIPTION = (
    "Search the web and return results as a Markdown list of '- [title](url)' "
    "(capped, default 8). Narrow results with allowed_domains/blocked_domains. "
    "This returns links only{fetch}. "
    "Use it for time-sensitive or post-cutoff facts (news, prices, versions, "
    "releases, current service specifics); skip it for closed knowledge you can "
    "answer deterministically."
)
DESCRIPTION_WITHOUT_FETCH = _DESCRIPTION.format(fetch="")

_DDG_RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
)
_TAG_RE = re.compile(r"<[^>]+>")


_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _make_client(timeout_s: float) -> httpx.AsyncClient:
    """httpx クライアントの生成点（テストが MockTransport で monkeypatch する）。"""
    return httpx.AsyncClient(
        timeout=timeout_s, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
    )


def _with_query(base: str, q: str) -> str:
    """``base`` へ ``q=`` を安全に連結する。"""
    base = base.rstrip("?&")
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}q={q}"


def is_ready(cfg: WebSearchConfig) -> bool:
    """検索の URL を組めるか（プロバイダが選ばれ、要る ``base_url`` がある）。"""
    return _query_url(cfg, "q") is not None


def _query_url(cfg: WebSearchConfig, query: str) -> str | None:
    """プロバイダ設定から検索 URL を組む。"""
    q = urllib.parse.quote_plus(query)
    if cfg.provider == "duckduckgo":
        return _with_query(cfg.base_url or _DUCKDUCKGO_URL, q)
    if cfg.provider == "searxng":
        if not cfg.base_url:
            return None
        return f"{cfg.base_url.rstrip('/')}/search?q={q}&format=json"
    if cfg.provider == "generic":
        if not cfg.base_url:
            return None
        return _with_query(cfg.base_url, q)
    return None


def _unwrap_ddg_redirect(href: str) -> str:
    """``//duckduckgo.com/l/?uddg=<encoded>`` 形式のリダイレクトを実 URL へ展開する。"""
    parsed = urllib.parse.urlparse(href)
    if parsed.path.startswith("/y.js"):
        return ""
    if parsed.path.startswith("/l/"):
        return urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
    if href.startswith("//"):
        return f"https:{href}"
    return href


def _parse_duckduckgo(body: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    for href, title_html in _DDG_RESULT_RE.findall(body):
        url = _unwrap_ddg_redirect(html.unescape(href))
        title = html.unescape(_TAG_RE.sub("", title_html)).strip()
        if url:
            results.append((title or url, url))
    return results


def _parse_results_json(payload: object) -> list[tuple[str, str]]:
    """searxng / generic 共通の ``{"results": [{"title","url"}]}`` を正規化する。"""
    if not isinstance(payload, dict):
        return []
    items = payload.get("results")
    if not isinstance(items, list):
        return []
    results: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        title = item.get("title")
        results.append((title if isinstance(title, str) and title else url, url))
    return results


def failed_engines(payload: object) -> list[tuple[str, str]]:
    """SearXNG の ``unresponsive_engines``（応答しなかったエンジンと、その理由の組）。"""
    if not isinstance(payload, dict):
        return []
    items = payload.get("unresponsive_engines")
    if not isinstance(items, list):
        return []
    failed: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, list) and item and isinstance(item[0], str):
            reason = item[1] if len(item) > 1 and isinstance(item[1], str) else ""
            if (item[0], reason) not in failed:
                failed.append((item[0], reason))
    return failed


def engines_text(failed: list[tuple[str, str]]) -> str:
    """``brave (Suspended: too many requests), duckduckgo (CAPTCHA)`` の形に並べる。"""
    return ", ".join(f"{name} ({reason})" if reason else name for name, reason in failed)


def _host_matches(host: str, domain: str) -> bool:
    domain = domain.lower().lstrip(".")
    return host == domain or host.endswith(f".{domain}")


def _filter_domains(
    results: list[tuple[str, str]], allowed: list[str], blocked: list[str]
) -> list[tuple[str, str]]:
    """ホスト一致（サフィックス一致含む）で allowed / blocked を適用する。"""
    kept: list[tuple[str, str]] = []
    for title, url in results:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
        if not host:
            continue
        if allowed and not any(_host_matches(host, d) for d in allowed):
            continue
        if any(_host_matches(host, d) for d in blocked):
            continue
        kept.append((title, url))
    return kept


class WebSearch(Tool):
    """Web 検索（設定されたプロバイダ経由）。"""

    class Input(BaseModel):
        query: str = Field(min_length=2, description="Search query in natural language.")
        allowed_domains: list[str] = Field(
            default_factory=list,
            description="Only keep results whose host matches one of these domains "
            '(suffix match, e.g. "python.org"). Empty means no restriction.',
        )
        blocked_domains: list[str] = Field(
            default_factory=list,
            description="Drop results whose host matches one of these domains (suffix match).",
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="web_search",
        description=_DESCRIPTION.format(
            fetch="; use web_fetch to read the content of a specific URL"
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, WebSearch.Input)
        cfg = ctx.config.web_search
        url = _query_url(cfg, inp.query)
        if url is None:
            return ToolResult(
                output="web_search has no search backend configured "
                "(see the web_search.provider and web_search.base_url settings).",
                is_error=True,
            )
        try:
            async with _make_client(cfg.timeout_s) as client:
                resp = await client.get(url)
        except httpx.HTTPError as e:
            return ToolResult(output=f"Fetching the search results failed: {e}")
        if resp.status_code >= 400:
            return ToolResult(output=f"Fetching the search results failed: HTTP {resp.status_code}")
        if cfg.provider == "duckduckgo" and resp.status_code == 202:
            return ToolResult(
                output="The search engine is temporarily blocking this client as a bot "
                "(HTTP 202). Try again later, or switch the web_search.provider setting."
            )
        failed: list[tuple[str, str]] = []
        if cfg.provider == "duckduckgo":
            found = _parse_duckduckgo(resp.text)
        else:
            try:
                payload = resp.json()
            except ValueError:
                return ToolResult(output="Could not read the search response (it is not JSON).")
            found = _parse_results_json(payload)
            failed = failed_engines(payload)
        if not found and failed:
            return ToolResult(
                output="The search failed: the engines of the search backend did not respond "
                f"({engines_text(failed)}). There are 0 results. This comes from the state of "
                "the user's search backend, not from the query, so rewording the query and "
                "searching again will not help.",
                is_error=True,
            )
        found = _filter_domains(found, inp.allowed_domains, inp.blocked_domains)
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for title, link in found:
            if link in seen:
                continue
            seen.add(link)
            unique.append((title, link))
        if not unique:
            return ToolResult(output="No search results matched.")
        if cfg.preferred_domains:
            unique.sort(
                key=lambda tu: (
                    0
                    if any(
                        _host_matches((urllib.parse.urlparse(tu[1]).hostname or "").lower(), d)
                        for d in cfg.preferred_domains
                    )
                    else 1
                )
            )
        lines = [f"- [{title}]({link})" for title, link in unique[: cfg.max_results]]
        return ToolResult(output="\n".join(lines))
