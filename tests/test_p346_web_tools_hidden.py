"""設定で使えない web_fetch / web_search をモデルに見せない。"""

from __future__ import annotations

from pathlib import Path

import pytest

import satsumaestro.cli.app as cli_app
from satsumaestro.config.models import (
    HookConfig,
    RuntimeConfig,
    WebFetchConfig,
    WebSearchConfig,
)
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import TurnSummary
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionOutcome, default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, unavailable_tools
from satsumaestro.tools.builtins import web_fetch, web_search
from tests.fixtures.mock_provider import ScriptedFailureProvider
from tests.test_p153_missing_model import _ollama, _patch_provider, _route_http

SEARXNG = WebSearchConfig(provider="searxng", base_url="http://localhost:8080")


def test_both_are_unavailable_by_default() -> None:
    assert unavailable_tools(RuntimeConfig()) == {
        "web_fetch": web_fetch.TURNED_OFF,
        "web_search": web_search.TURNED_OFF,
    }


def test_configured_tools_are_available() -> None:
    cfg = RuntimeConfig(web_search=SEARXNG, web_fetch=WebFetchConfig(enabled=True))
    assert unavailable_tools(cfg) == {}


@pytest.mark.parametrize(
    ("search", "ready"),
    [
        (WebSearchConfig(provider="duckduckgo"), True),
        (WebSearchConfig(provider="searxng"), False),
        (WebSearchConfig(provider="generic"), False),
        (SEARXNG, True),
    ],
)
def test_web_search_needs_a_provider_and_its_base_url(search: WebSearchConfig, ready: bool) -> None:
    assert ("web_search" not in unavailable_tools(RuntimeConfig(web_search=search))) is ready


def test_unavailable_tools_are_not_registered() -> None:
    reg = default_registry(unavailable=["web_fetch", "web_search"])
    assert reg.get("web_fetch") is None and reg.get("web_search") is None
    assert {"web_fetch", "web_search"}.isdisjoint(reg.deferred_names())


def test_search_description_points_to_fetch_only_when_fetch_is_there() -> None:
    assert "use web_fetch" in default_registry().description("web_search")
    without = default_registry(unavailable=["web_fetch"])
    assert "web_fetch" not in without.description("web_search")
    assert "web_fetch" not in without.filtered(exclude={"agent"}).description("web_search")


class _Prompter:
    def __init__(self) -> None:
        self.calls = 0

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.calls += 1
        return PermissionOutcome(decision="allow", reason="y")


def _runtime(tmp_path: Path, name: str, args: dict[str, object]) -> ConversationRuntime:
    cfg = RuntimeConfig(max_iterations=4)
    steps: list[BaseException | list[AssistantEvent | BaseException]] = [
        [ToolUse(id="t1", name=name, input=args), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(unavailable=unavailable_tools(cfg)),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg,
    )


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("web_fetch", {"url": "https://example.com/"}, web_fetch.TURNED_OFF),
        ("WebFetch", {"url": "https://example.com/"}, web_fetch.TURNED_OFF),
        ("web_search", {"query": "weather tomorrow"}, web_search.TURNED_OFF),
    ],
)
async def test_calling_a_hidden_tool_names_the_setting(
    tmp_path: Path, name: str, args: dict[str, object], expected: str
) -> None:
    rt = _runtime(tmp_path, name, args)
    prompter = _Prompter()
    summary: TurnSummary = await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    [result] = [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]
    assert result.is_error and result.output == expected
    assert summary.unknown_tool_calls == 1 and summary.deferred_autoloads == 0
    assert prompter.calls == 0 and not rt._expanded


async def test_the_cli_does_not_register_them_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SATSUMA_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("SATSUMA_WEB_SEARCH_BASE_URL", raising=False)
    monkeypatch.delenv("SATSUMA_WEB_FETCH_ENABLED", raising=False)
    _patch_provider(monkeypatch, [])
    _route_http(monkeypatch, _ollama())
    rt, _, _ = await cli_app._assemble_runtime(
        "gpt-oss:120b", tmp_path, interactive=False, trusted=True
    )
    assert rt.tools.get("web_fetch") is None and rt.tools.get("web_search") is None
