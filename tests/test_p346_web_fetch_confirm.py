"""web_fetch の初回の確認と、web_search の既定（無効）。"""

from __future__ import annotations

import ipaddress
from pathlib import Path

import httpx
import pytest

from satsumaestro.config.models import (
    HookConfig,
    RuntimeConfig,
    WebFetchConfig,
    WebSearchConfig,
)
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter, TurnSummary
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import (
    PermissionOutcome,
    PermissionPolicy,
    PermissionRule,
    RuleMatch,
    default_policy,
)
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.builtins import web_fetch as web_fetch_mod
from tests.fixtures.mock_provider import ScriptedFailureProvider

WW = PermissionMode.WORKSPACE_WRITE


@pytest.fixture(autouse=True)
def _web(monkeypatch: pytest.MonkeyPatch) -> None:
    """取得は擬似の応答、名前の解決は公開のアドレス（ネットワークに出ない）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=f"page {request.url.path}")

    monkeypatch.setattr(
        web_fetch_mod,
        "_make_client",
        lambda timeout_s: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def public(host: str, port: int) -> list[ipaddress.IPv4Address]:
        return [ipaddress.IPv4Address("93.184.216.34")]

    monkeypatch.setattr(web_fetch_mod, "_resolve_host", public)


class _Prompter:
    def __init__(self, *decisions: PermissionOutcome) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._decisions = list(decisions) or [PermissionOutcome(decision="deny", reason="no")]

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.calls.append((str(call.input.get("url")), warning))
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


def _fetch(url: str, n: int) -> ToolUse:
    return ToolUse(id=f"f{n}", name="web_fetch", input={"url": url, "mode": "raw"})


def _runtime(
    tmp_path: Path,
    urls: list[str],
    *,
    policy: PermissionPolicy | PermissionResolver | None = None,
    web_fetch: WebFetchConfig | None = None,
) -> ConversationRuntime:
    steps: list[BaseException | list[AssistantEvent | BaseException]] = [
        [_fetch(url, n), MessageStop()] for n, url in enumerate(urls)
    ]
    steps.append([TextDelta(text="done"), MessageStop()])
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=policy or default_policy(WW),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(max_iterations=8, web_fetch=web_fetch or WebFetchConfig(enabled=True)),
    )


async def _run(
    rt: ConversationRuntime, prompter: object
) -> tuple[TurnSummary, list[ToolResultBlock]]:
    summary = await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    results = [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]
    return summary, results


YES = PermissionOutcome(decision="allow", reason="y")
NO = PermissionOutcome(decision="deny", reason="n")
A = "https://example.com/a"
B = "https://docs.example.com/b"


async def test_first_fetch_asks_once(tmp_path: Path) -> None:
    prompter = _Prompter(YES)
    summary, results = await _run(_runtime(tmp_path, [A, B]), prompter)
    assert prompter.calls == [(A, "first web_fetch this session: example.com")]
    assert [r.output for r in results] == ["HTTP 200\n\npage /a", "HTTP 200\n\npage /b"]
    assert summary.first_fetch_asks == 1


async def test_declining_asks_again_next_time(tmp_path: Path) -> None:
    prompter = _Prompter(NO, YES)
    _, results = await _run(_runtime(tmp_path, [A, B]), prompter)
    assert prompter.calls == [
        (A, "first web_fetch this session: example.com"),
        (B, "first web_fetch this session: docs.example.com"),
    ]
    assert results[0].is_error and not results[1].is_error


async def test_noninteractive_refuses_and_names_the_setting(tmp_path: Path) -> None:
    summary, [result] = await _run(_runtime(tmp_path, [A]), DenyPrompter())
    assert result.is_error
    assert result.output.startswith("permission required but running non-interactively: ")
    assert "web_fetch.confirm (not a tool argument)" in result.output
    assert summary.first_fetch_asks == 1


async def test_confirm_never_does_not_ask(tmp_path: Path) -> None:
    prompter = _Prompter()
    rt = _runtime(tmp_path, [A], web_fetch=WebFetchConfig(enabled=True, confirm="never"))
    summary, [result] = await _run(rt, prompter)
    assert prompter.calls == [] and not result.is_error and summary.first_fetch_asks == 0


async def test_confirm_always_asks_every_time(tmp_path: Path) -> None:
    prompter = _Prompter(YES)
    await _run(
        _runtime(tmp_path, [A, B], web_fetch=WebFetchConfig(enabled=True, confirm="always")),
        prompter,
    )
    assert prompter.calls == [(A, "web_fetch: example.com"), (B, "web_fetch: docs.example.com")]


async def test_a_rule_naming_the_url_skips_the_card(tmp_path: Path) -> None:
    policy = default_policy(WW)
    policy.allow.append(
        PermissionRule(
            match=RuleMatch(tool="web_fetch", field="url", glob="https://example.com/*"),
            label="example",
        )
    )
    prompter = _Prompter(YES)
    await _run(_runtime(tmp_path, [A, B], policy=policy), prompter)
    assert prompter.calls == [(B, "first web_fetch this session: docs.example.com")]


async def test_an_ask_rule_still_asks_after_approval(tmp_path: Path) -> None:
    policy = default_policy(WW)
    policy.ask.append(PermissionRule(match=RuleMatch(tool="web_fetch"), label="fetch"))
    prompter = _Prompter(YES)
    summary, _ = await _run(_runtime(tmp_path, [A, B], policy=policy), prompter)
    assert prompter.calls == [(A, None), (B, None)]
    assert summary.first_fetch_asks == 0


async def test_remembering_records_the_url_and_clears_the_session(tmp_path: Path) -> None:
    resolver = PermissionResolver(base=default_policy(WW))
    prompter = _Prompter(PermissionOutcome(decision="allow", reason="a", remember=True))
    await _run(_runtime(tmp_path, [A, B], policy=resolver), prompter)
    assert len(prompter.calls) == 1
    assert resolver.authorize("web_fetch", {"url": A}, PermissionMode.READ_ONLY).scoped
    assert not resolver.authorize("web_fetch", {"url": B}, PermissionMode.READ_ONLY).scoped


async def test_approval_is_shared_with_runtimes_on_the_same_resolver(tmp_path: Path) -> None:
    resolver = PermissionResolver(base=default_policy(WW))
    await _run(_runtime(tmp_path, [A], policy=resolver), _Prompter(YES))
    _, [child] = await _run(_runtime(tmp_path, [B], policy=resolver), DenyPrompter())
    assert not child.is_error and child.output.endswith("page /b")


@pytest.mark.parametrize("word", ["none", "None", " NONE "])
def test_none_in_settings_disables_web_search(word: str) -> None:
    assert WebSearchConfig.model_validate({"provider": word}).provider is None


def test_web_tools_are_off_by_default() -> None:
    assert WebSearchConfig().provider is None
    assert WebFetchConfig().enabled is False and WebFetchConfig().confirm == "first-use"


async def test_disabled_web_fetch_shows_no_card(tmp_path: Path) -> None:
    prompter = _Prompter(YES)
    rt = _runtime(tmp_path, [A], web_fetch=WebFetchConfig())
    summary, [result] = await _run(rt, prompter)
    assert prompter.calls == [] and summary.first_fetch_asks == 0
    assert result.is_error and result.output.startswith("web_fetch is turned off.")
