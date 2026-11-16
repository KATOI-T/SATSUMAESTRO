"""外の内容（web・MCP）を読んだ後の確認（permission.confirm_after_external）。"""

from __future__ import annotations

import ipaddress
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from satsumaestro.config.models import (
    HookConfig,
    PermissionConfig,
    RuntimeConfig,
    SubagentConfig,
    ToolsConfig,
    WebFetchConfig,
    WebSearchConfig,
)
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import RuntimeFactory
from satsumaestro.engine.turn import DenyPrompter
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
from satsumaestro.tools.base import ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.builtins import web_fetch as web_fetch_mod
from tests.fixtures.mock_provider import ScriptedFailureProvider

WW = PermissionMode.WORKSPACE_WRITE
YES = PermissionOutcome(decision="allow", reason="y")
NO = PermissionOutcome(decision="deny", reason="n")
NOTE = "web or MCP content was read earlier in this turn"


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


class _McpInput(BaseModel):
    model_config = ConfigDict(extra="allow")


class _Echo:
    """MCP のツールの代わり（``mcp__demo__echo``）。"""

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="mcp__demo__echo",
        description="Echo.",
        input_schema={"type": "object"},
        required_mode=PermissionMode.PROMPT,
    )
    Input: ClassVar[type[BaseModel]] = _McpInput

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        extra = inp.model_extra or {}
        return ToolResult(output=f"echo {extra.get('text', '')}", is_error=bool(extra.get("fail")))


class _Prompter:
    def __init__(self, *decisions: PermissionOutcome) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._decisions = list(decisions) or [YES]

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.calls.append((call.name, warning))
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


def _call(name: str, n: int = 0, **inp: Any) -> ToolUse:
    return ToolUse(id=f"{name}{n}", name=name, input=inp)


def _fetch(url: str = "https://example.com/a", n: int = 0) -> ToolUse:
    return _call("web_fetch", n, url=url, mode="raw")


def _write(path: str = "a.txt", n: int = 0) -> ToolUse:
    return _call("write_file", n, path=path, content="x\n")


def _steps(*calls: ToolUse) -> list[list[AssistantEvent]]:
    steps: list[list[AssistantEvent]] = [[c, MessageStop()] for c in calls]
    return [*steps, [TextDelta(text="done"), MessageStop()]]


def _config(
    *,
    on: bool = True,
    allowed_hosts: list[str] | None = None,
    web_search: WebSearchConfig | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        max_iterations=12,
        permission=PermissionConfig(confirm_after_external=on),
        web_fetch=WebFetchConfig(enabled=True, confirm="never", allowed_hosts=allowed_hosts or []),
        web_search=web_search or WebSearchConfig(),
        subagent=SubagentConfig(default_model="mock-model"),
        tools=ToolsConfig(deferred_tool_loading=False),
    )


def _runtime(
    tmp_path: Path,
    steps: list[list[AssistantEvent]],
    *,
    config: RuntimeConfig | None = None,
    policy: PermissionPolicy | PermissionResolver | None = None,
    factory: bool = False,
) -> ConversationRuntime:
    cfg = config or _config()
    tools = default_registry()
    tools.register(_Echo)  # type: ignore[arg-type]
    pol = policy or default_policy(WW)
    hooks = HookRunner(HookConfig(), tmp_path)
    provider = ScriptedFailureProvider(steps)  # type: ignore[arg-type]
    made = None
    if factory:
        made = RuntimeFactory(
            provider=provider,  # type: ignore[arg-type]
            model="mock-model",
            config=cfg,
            tools=tools,
            policy=pol,
            hooks=hooks,
            cwd=tmp_path,
            provider_builder=lambda model: provider,  # type: ignore[arg-type,return-value]
        )
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=tools,
        policy=pol,
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=hooks,
        config=cfg,
        runtime_factory=made,
    )


async def _run(rt: ConversationRuntime, prompter: object) -> list[ToolResultBlock]:
    await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    return [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


def _allowing(*tools: str, mode: PermissionMode = WW) -> PermissionPolicy:
    policy = default_policy(mode)
    for tool in tools:
        policy.allow.append(PermissionRule(match=RuleMatch(tool=tool), label=tool))
    return policy


async def test_off_by_default_nothing_changes(tmp_path: Path) -> None:
    assert RuntimeConfig().permission.confirm_after_external is False
    prompter = _Prompter()
    rt = _runtime(tmp_path, _steps(_fetch(), _write()), config=_config(on=False))
    await _run(rt, prompter)
    assert prompter.calls == [] and (tmp_path / "a.txt").exists()


async def test_writes_before_any_external_content_do_not_ask(tmp_path: Path) -> None:
    prompter = _Prompter()
    await _run(_runtime(tmp_path, _steps(_write(), _fetch())), prompter)
    assert prompter.calls == [] and (tmp_path / "a.txt").exists()


async def test_a_write_after_web_fetch_asks_and_a_no_stops_it(tmp_path: Path) -> None:
    prompter = _Prompter(NO)
    results = await _run(_runtime(tmp_path, _steps(_fetch(), _write())), prompter)
    assert prompter.calls == [("write_file", f"{NOTE} (web_fetch: example.com)")]
    assert results[-1].is_error and not (tmp_path / "a.txt").exists()


@pytest.mark.parametrize(
    ("call", "policy"),
    [
        (_write(), None),
        (_call("edit_file", path="e.txt", old="old", new="new"), None),
        (_call("append_file", path="e.txt", content="more\n"), None),
        (_call("bash", command="touch b.txt"), default_policy(PermissionMode.DANGER_FULL_ACCESS)),
        (_call("mcp__demo__echo", text="hi"), _allowing("mcp__demo__echo")),
        (_call("agent", description="look", prompt="look around"), None),
    ],
    ids=["write_file", "edit_file", "append_file", "bash", "mcp", "agent"],
)
async def test_calls_that_change_or_send_ask_after_web_fetch(
    tmp_path: Path, call: ToolUse, policy: PermissionPolicy | None
) -> None:
    (tmp_path / "e.txt").write_text("old\n")
    prompter = _Prompter(NO)
    await _run(_runtime(tmp_path, _steps(_fetch(), call), policy=policy), prompter)
    assert prompter.calls == [(call.name, f"{NOTE} (web_fetch: example.com)")]


@pytest.mark.parametrize("fail", [False, True], ids=["result", "error"])
async def test_an_mcp_result_counts_even_when_it_is_an_error(tmp_path: Path, fail: bool) -> None:
    prompter = _Prompter(YES, NO)
    echo = _call("mcp__demo__echo", text="hi", fail=fail)
    rt = _runtime(tmp_path, _steps(echo, _write()), policy=_allowing("mcp__demo__echo"))
    await _run(rt, prompter)
    assert prompter.calls == [("write_file", f"{NOTE} (mcp__demo__echo)")]


async def test_a_remembered_allow_still_asks(tmp_path: Path) -> None:
    resolver = PermissionResolver(base=default_policy(WW))
    resolver.session.add(RuleMatch(tool="write_file", field="path", regex=r"\Aa\.txt\Z"))
    prompter = _Prompter(NO)
    await _run(_runtime(tmp_path, _steps(_fetch(), _write()), policy=resolver), prompter)
    assert [name for name, _ in prompter.calls] == ["write_file"]


@pytest.mark.parametrize("mode", [PermissionMode.DANGER_FULL_ACCESS, PermissionMode.ALLOW])
async def test_wide_modes_still_ask(tmp_path: Path, mode: PermissionMode) -> None:
    prompter = _Prompter(NO)
    steps = _steps(_fetch(), _call("bash", command="touch b.txt"))
    await _run(_runtime(tmp_path, steps, policy=default_policy(mode)), prompter)
    assert [name for name, _ in prompter.calls] == ["bash"]
    assert not (tmp_path / "b.txt").exists()


async def test_read_only_bash_is_still_allowed(tmp_path: Path) -> None:
    prompter = _Prompter(NO)
    steps = _steps(_fetch(), _call("bash", command="ls"))
    results = await _run(_runtime(tmp_path, steps), prompter)
    assert prompter.calls == [] and not results[-1].is_error


async def test_more_web_requests_ask(tmp_path: Path) -> None:
    search = WebSearchConfig(provider="searxng", base_url="http://search.example")
    prompter = _Prompter(NO)
    steps = _steps(
        _fetch(),
        _call("web_search", query="next"),
        _fetch("https://other.example.org/b", 1),
    )
    await _run(_runtime(tmp_path, steps, config=_config(web_search=search)), prompter)
    assert [name for name, _ in prompter.calls] == ["web_search", "web_fetch"]


async def test_a_fetch_to_an_allowed_host_does_not_ask(tmp_path: Path) -> None:
    prompter = _Prompter(NO)
    steps = _steps(_fetch(), _fetch("https://docs.example.com/b", 1))
    cfg = _config(allowed_hosts=["example.com"])
    results = await _run(_runtime(tmp_path, steps, config=cfg), prompter)
    assert prompter.calls == [] and not results[-1].is_error


async def test_a_refused_fetch_does_not_count(tmp_path: Path) -> None:
    prompter = _Prompter(NO)
    steps = _steps(_fetch("https://other.example.org/"), _write())
    cfg = _config(allowed_hosts=["example.com"])
    results = await _run(_runtime(tmp_path, steps, config=cfg), prompter)
    assert results[0].is_error and prompter.calls == [] and (tmp_path / "a.txt").exists()


async def test_the_next_turn_starts_clean(tmp_path: Path) -> None:
    steps = [
        [_fetch(), MessageStop()],
        [TextDelta(text="read"), MessageStop()],
        [_write(), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    prompter = _Prompter(NO)
    rt = _runtime(tmp_path, steps)
    await rt.run_turn("read it", prompter=prompter)  # type: ignore[arg-type]
    await rt.run_turn("now write", prompter=prompter)  # type: ignore[arg-type]
    assert prompter.calls == [] and (tmp_path / "a.txt").exists()


async def test_a_subagent_reading_the_web_counts_for_the_parent(tmp_path: Path) -> None:
    steps = [
        [_call("agent", description="look", prompt="read the page"), MessageStop()],
        [_fetch(), MessageStop()],
        [TextDelta(text="the page says hi"), MessageStop()],
        [_write(), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    prompter = _Prompter(NO)
    await _run(_runtime(tmp_path, steps, factory=True), prompter)
    assert prompter.calls == [("write_file", f"{NOTE} (web_fetch: example.com)")]


def _agent_writes_after_fetch() -> list[list[AssistantEvent]]:
    return [
        [_fetch(), MessageStop()],
        [_call("agent", description="write", prompt="write a.txt"), MessageStop()],
        [_write(), MessageStop()],
        [TextDelta(text="child done"), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]


async def test_a_subagent_started_after_web_content_cannot_write(tmp_path: Path) -> None:
    prompter = _Prompter(YES)
    await _run(_runtime(tmp_path, _agent_writes_after_fetch(), factory=True), prompter)
    assert prompter.calls == [("agent", f"{NOTE} (web_fetch: example.com)")]
    assert not (tmp_path / "a.txt").exists()


async def test_a_subagent_writes_when_the_setting_is_off(tmp_path: Path) -> None:
    prompter = _Prompter(YES)
    steps = _agent_writes_after_fetch()
    await _run(_runtime(tmp_path, steps, config=_config(on=False), factory=True), prompter)
    assert prompter.calls == [] and (tmp_path / "a.txt").exists()


async def test_a_subagent_in_the_next_turn_writes(tmp_path: Path) -> None:
    steps = [
        [_fetch(), MessageStop()],
        [TextDelta(text="read"), MessageStop()],
        [_call("agent", description="write", prompt="write a.txt"), MessageStop()],
        [_write(), MessageStop()],
        [TextDelta(text="child done"), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    prompter = _Prompter(NO)
    rt = _runtime(tmp_path, steps, factory=True)
    await rt.run_turn("read it", prompter=prompter)  # type: ignore[arg-type]
    await rt.run_turn("now write", prompter=prompter)  # type: ignore[arg-type]
    assert prompter.calls == [] and (tmp_path / "a.txt").exists()


def test_children_share_the_count_with_the_parent(tmp_path: Path) -> None:
    from satsumaestro.engine.subagent import resolve_subagent_type

    rt = _runtime(tmp_path, _steps(), factory=True)
    factory = rt._runtime_factory
    assert factory is not None
    child = factory.build_child(preset=resolve_subagent_type("general"), depth=1)
    assert child._external_reads is rt._external_reads is factory.external_reads


async def test_without_a_prompter_it_is_refused_with_the_setting_named(tmp_path: Path) -> None:
    results = await _run(_runtime(tmp_path, _steps(_fetch(), _write())), DenyPrompter())
    assert results[-1].is_error
    assert results[-1].output.startswith("permission required but running non-interactively: ")
    assert "permission.confirm_after_external (not a tool argument)" in results[-1].output
    assert not (tmp_path / "a.txt").exists()


async def test_the_note_joins_a_warning_already_on_the_card(tmp_path: Path) -> None:
    prompter = _Prompter(NO)
    steps = _steps(_fetch(), _write(".satsuma.json"))
    await _run(_runtime(tmp_path, steps), prompter)
    [(name, warning)] = prompter.calls
    assert name == "write_file" and warning is not None
    assert warning.startswith("agent settings file: .satsuma.json; ")
    assert warning.endswith(f"{NOTE} (web_fetch: example.com)")
