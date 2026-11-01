"""MCP 統合の結線テスト。"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, McpServerConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.mcp.manager import _MAX_OUTPUT_CHARS, CALL_TIMEOUT_S, MCPManager
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import (
    PermissionPolicy,
    PermissionRule,
    RuleMatch,
    default_policy,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.mcp_bridge import (
    mcp_tool_name,
    normalize_server,
    raw_tool_name,
    register_mcp_tools,
)
from satsumaestro.tools.registry import ToolRegistry
from tests.fixtures.mock_provider import SequencedMockProvider

_MOCK_SCRIPT = str(Path(__file__).resolve().parent / "fixtures" / "mock_mcp.py")


def _mock_servers(name: str = "mock", *, broken: bool = False) -> dict[str, McpServerConfig]:
    """モック stdio サーバ設定（``broken=True`` で存在しない command → degrade 検証用）。"""
    if broken:
        return {name: McpServerConfig(transport="stdio", command="___no_such_binary___")}
    return {name: McpServerConfig(transport="stdio", command=sys.executable, args=[_MOCK_SCRIPT])}


async def _start(name: str = "mock", *, broken: bool = False) -> MCPManager:
    return await MCPManager.start(_mock_servers(name, broken=broken))


def test_names_normalization() -> None:
    assert normalize_server("github.com") == "github_com"
    assert mcp_tool_name("github.com", "search") == "mcp__github_com__search"
    assert raw_tool_name("github.com", "mcp__github_com__search") == "search"


async def test_stdio_connect_and_list() -> None:
    mgr = await _start()
    try:
        handles = mgr.handles()
        assert len(handles) == 1 and handles[0].connected
        reg = default_registry()
        n = register_mcp_tools(reg, mgr)
        assert n >= 2
        names = {t["name"] for t in reg.json_schemas()}
        assert "mcp__mock__echo" in names
        assert "mcp__mock__big" in names
        spec = reg.get("mcp__mock__echo")
        assert spec is not None and spec.spec.required_mode is PermissionMode.PROMPT
        assert spec.spec.parallel_safe is False
    finally:
        await mgr.aclose()


async def test_bridge_call_roundtrip() -> None:
    mgr = await _start()
    try:
        res = await mgr.call("mock", "echo", {"text": "ping"})
        assert res.is_error is False
        assert "ping" in res.output
    finally:
        await mgr.aclose()


async def test_large_output_truncated() -> None:
    mgr = await _start()
    try:
        res = await mgr.call("mock", "big", {})
        assert len(res.output) <= _MAX_OUTPUT_CHARS + 64
        assert "truncate" in res.output
    finally:
        await mgr.aclose()


async def test_call_unknown_server_is_error() -> None:
    mgr = await _start()
    try:
        res = await mgr.call("nope", "echo", {"text": "x"})
        assert res.is_error is True
    finally:
        await mgr.aclose()


def _runtime_with_mcp(
    tmp_path: Path,
    provider: object,
    reg: ToolRegistry,
    mgr: MCPManager,
    *,
    policy: object | None = None,
    hooks: HookRunner | None = None,
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=reg,
        policy=policy or default_policy(PermissionMode.WORKSPACE_WRITE),  # type: ignore[arg-type]
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=hooks or HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
        mcp=mgr,
    )


async def test_permission_default_prompt_denied_noninteractive(tmp_path: Path) -> None:
    mgr = await _start()
    try:
        reg = default_registry()
        register_mcp_tools(reg, mgr)
        provider = SequencedMockProvider(
            [
                [ToolUse(id="m1", name="mcp__mock__echo", input={"text": "hi"}), MessageStop()],
                [TextDelta(text="after deny"), MessageStop()],
            ]
        )
        rt = _runtime_with_mcp(tmp_path, provider, reg, mgr)
        summary = await rt.run_turn("call echo")
        assert summary.final_text == "after deny"
        tool_msg = rt.session.messages[2]
        assert tool_msg.role == "tool" and tool_msg.blocks[0].is_error is True
    finally:
        await mgr.aclose()


async def test_permission_session_allow_executes(tmp_path: Path) -> None:
    mgr = await _start()
    try:
        reg = default_registry()
        register_mcp_tools(reg, mgr)
        resolver = PermissionResolver(default_policy(PermissionMode.WORKSPACE_WRITE))
        resolver.session.add(RuleMatch(tool="mcp__mock__echo"))
        provider = SequencedMockProvider(
            [
                [ToolUse(id="m1", name="mcp__mock__echo", input={"text": "pong"}), MessageStop()],
                [TextDelta(text="done"), MessageStop()],
            ]
        )
        rt = _runtime_with_mcp(tmp_path, provider, reg, mgr, policy=resolver)
        summary = await rt.run_turn("call echo")
        assert summary.final_text == "done"
        tool_msg = rt.session.messages[2]
        assert tool_msg.role == "tool" and tool_msg.blocks[0].is_error is False
        assert "pong" in tool_msg.blocks[0].output
    finally:
        await mgr.aclose()


async def test_policy_allow_rule_executes_noninteractive(tmp_path: Path) -> None:
    mgr = await _start()
    try:
        reg = default_registry()
        register_mcp_tools(reg, mgr)
        policy = PermissionPolicy(
            mode=PermissionMode.WORKSPACE_WRITE,
            allow=[PermissionRule(match=RuleMatch(tool="mcp__mock__echo"), label="allow mcp echo")],
        )
        provider = SequencedMockProvider(
            [
                [ToolUse(id="m1", name="mcp__mock__echo", input={"text": "pong"}), MessageStop()],
                [TextDelta(text="done"), MessageStop()],
            ]
        )
        rt = _runtime_with_mcp(tmp_path, provider, reg, mgr, policy=policy)
        summary = await rt.run_turn("call echo")
        assert summary.final_text == "done"
        tool_msg = rt.session.messages[2]
        assert tool_msg.role == "tool" and tool_msg.blocks[0].is_error is False
        assert "pong" in tool_msg.blocks[0].output
    finally:
        await mgr.aclose()


async def test_hook_path_is_shared(tmp_path: Path) -> None:
    hook = tmp_path / "deny_hook.py"
    hook.write_text(
        "import json,sys\njson.load(sys.stdin)\n"
        "print(json.dumps({'deny': True, 'reason': 'blocked by hook'}))\n",
        encoding="utf-8",
    )
    mgr = await _start()
    try:
        reg = default_registry()
        register_mcp_tools(reg, mgr)
        hooks = HookRunner(HookConfig(pre_tool_use=[f"{sys.executable} {hook}"]), tmp_path)
        provider = SequencedMockProvider(
            [
                [ToolUse(id="m1", name="mcp__mock__echo", input={"text": "x"}), MessageStop()],
                [TextDelta(text="ok"), MessageStop()],
            ]
        )
        rt = _runtime_with_mcp(
            tmp_path,
            provider,
            reg,
            mgr,
            policy=PermissionPolicy(mode=PermissionMode.ALLOW),
            hooks=hooks,
        )
        await rt.run_turn("call echo")
        tool_msg = rt.session.messages[2]
        assert tool_msg.blocks[0].is_error is True
        assert "blocked by hook" in tool_msg.blocks[0].output
    finally:
        await mgr.aclose()


async def test_degrade_on_connect_failure() -> None:
    servers = {**_mock_servers("mock"), **_mock_servers("bad", broken=True)}
    mgr = await MCPManager.start(servers)
    try:
        by_name = {h.name: h for h in mgr.handles()}
        assert by_name["mock"].connected is True
        assert by_name["bad"].connected is False
        assert by_name["bad"].error
        reg = default_registry()
        assert register_mcp_tools(reg, mgr) >= 2
    finally:
        await mgr.aclose()


def _mock_pids() -> set[str]:
    r = subprocess.run(["pgrep", "-f", _MOCK_SCRIPT], capture_output=True, text=True, check=False)
    return {p for p in r.stdout.split() if p}


async def test_aclose_no_leak() -> None:
    baseline = _mock_pids()
    mgr = await _start()
    spawned = _mock_pids() - baseline
    assert spawned, "モックサーバの子プロセスが起動しているはず"
    await mgr.aclose()
    for _ in range(20):
        if not (_mock_pids() & spawned):
            break
        await asyncio.sleep(0.1)
    assert not (_mock_pids() & spawned), "aclose 後に子プロセスが残っている"


async def test_build_runtime_closes_mcp_when_construction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from satsumaestro.cli import app as cli_app

    state = tmp_path / ".satsuma"
    state.mkdir()
    servers = {n: c.model_dump() for n, c in _mock_servers().items()}
    (state / "settings.json").write_text(json.dumps({"mcp_servers": servers}), encoding="utf-8")
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli_app, "build_provider", lambda *a, **kw: object())

    def _boom(*a: object, **kw: object) -> None:
        raise RuntimeError("policy load failed")

    monkeypatch.setattr(cli_app, "load_policies", _boom)

    baseline = _mock_pids()
    with pytest.raises(RuntimeError):
        await cli_app._assemble_runtime("mock-model", tmp_path, interactive=False, trusted=True)
    for _ in range(20):
        if not (_mock_pids() - baseline):
            break
        await asyncio.sleep(0.1)
    assert not (_mock_pids() - baseline), "構築失敗時に MCP 子プロセスが残っている"


def test_call_timeout_constant_is_positive() -> None:
    assert CALL_TIMEOUT_S > 0
