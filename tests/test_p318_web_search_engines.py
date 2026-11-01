"""SearXNG のエンジンが応答しなかった回を、本当に 0 件の回と分けて返す。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from satsumaestro.cli import app as cli_app
from satsumaestro.config.models import RuntimeConfig, WebSearchConfig
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import web_search as web_search_mod
from satsumaestro.tools.builtins.web_search import WebSearch, failed_engines

_STOPPED = [
    ["brave", "Suspended: too many requests"],
    ["duckduckgo", "CAPTCHA"],
    ["startpage", "Suspended: CAPTCHA"],
]


def _serve(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    def make(timeout_s: float) -> httpx.AsyncClient:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        return httpx.AsyncClient(transport=transport)

    monkeypatch.setattr(web_search_mod, "_make_client", make)


def _ctx(tmp_path: Path) -> ToolContext:
    jail = WorkspaceJail(tmp_path)
    cfg = WebSearchConfig(provider="searxng", base_url="http://searx.local")
    return ToolContext(workspace_root=jail.root, jail=jail, config=RuntimeConfig(web_search=cfg))


async def _search(tmp_path: Path, **kwargs: Any) -> Any:
    return await WebSearch().run(WebSearch.Input(query="明日の天気", **kwargs), _ctx(tmp_path))


async def test_no_results_with_stopped_engines_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve(monkeypatch, {"results": [], "unresponsive_engines": _STOPPED})

    result = await _search(tmp_path)

    assert result.is_error
    assert result.output == (
        "The search failed: the engines of the search backend did not respond "
        "(brave (Suspended: too many requests), duckduckgo (CAPTCHA), "
        "startpage (Suspended: CAPTCHA)). There are 0 results. This comes from the state of "
        "the user's search backend, not from the query, so rewording the query and "
        "searching again will not help."
    )


async def test_no_results_without_stopped_engines_is_a_plain_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _serve(monkeypatch, {"results": [], "unresponsive_engines": []})

    result = await _search(tmp_path)

    assert not result.is_error
    assert result.output == "No search results matched."


async def test_results_with_some_stopped_engines_are_returned_as_is(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "results": [{"title": "A", "url": "https://a.com/1"}],
        "unresponsive_engines": _STOPPED,
    }
    _serve(monkeypatch, payload)

    result = await _search(tmp_path)

    assert not result.is_error
    assert result.output == "- [A](https://a.com/1)"


async def test_results_dropped_by_the_domain_filter_are_a_plain_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "results": [{"title": "A", "url": "https://a.com/1"}],
        "unresponsive_engines": _STOPPED,
    }
    _serve(monkeypatch, payload)

    result = await _search(tmp_path, allowed_domains=["python.org"])

    assert not result.is_error
    assert result.output == "No search results matched."


@pytest.mark.parametrize(
    ("payload", "failed"),
    [
        ({"unresponsive_engines": [["brave", "CAPTCHA"]]}, [("brave", "CAPTCHA")]),
        ({"unresponsive_engines": [["brave"]]}, [("brave", "")]),
        (
            {"unresponsive_engines": [["ddg", "timeout"], ["ddg", "timeout"], ["ddg", "CAPTCHA"]]},
            [("ddg", "timeout"), ("ddg", "CAPTCHA")],
        ),
        ({"unresponsive_engines": [["brave", None], "x", [], [1, "y"]]}, [("brave", "")]),
        ({"unresponsive_engines": "brave"}, []),
        ({}, []),
        ([], []),
    ],
)
def test_failed_engines_reads_only_the_well_formed_entries(
    payload: object, failed: list[tuple[str, str]]
) -> None:
    assert failed_engines(payload) == failed


@pytest.mark.parametrize(
    ("payload", "status", "detail"),
    [
        (
            {"results": [], "unresponsive_engines": _STOPPED[:2]},
            "warn",
            "0 件。応答しないエンジン: brave (Suspended: too many requests), duckduckgo (CAPTCHA)",
        ),
        (
            {
                "results": [{"title": "t", "url": "https://e.com"}],
                "unresponsive_engines": [["brave", "CAPTCHA"]],
            },
            "ok",
            "疎通 OK（1 件）。応答しないエンジン: brave (CAPTCHA)",
        ),
        ({"results": [{"title": "t", "url": "https://e.com"}]}, "ok", "疎通 OK（1 件）"),
    ],
)
async def test_probe_shows_the_stopped_engines(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], status: str, detail: str
) -> None:
    _serve(monkeypatch, payload)

    got = await cli_app._probe_web_search("searxng", 5.0, "http://searx.local/search?q=x")

    assert got == (status, detail)
