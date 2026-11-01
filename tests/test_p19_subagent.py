"""サブエージェント（agent ツール・RuntimeFactory）のテスト。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig, SubagentConfig, ToolsConfig
from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    Usage,
)
from satsumaestro.engine.recovery import SessionAbortError
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import (
    EXPLORE_TOOLS,
    PLAN_TOOLS,
    RuntimeFactory,
    resolve_subagent_type,
)
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.registry import expand_model_name
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import PermissionPolicy, default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.agent import Agent
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import (
    MockProvider,
    ScriptedFailureProvider,
    SequencedMockProvider,
)


class _RecordingBuilder:
    """呼び出しモデルを記録し、与えた子プロバイダを順に返す provider_builder。"""

    def __init__(self, children: list[object]) -> None:
        self.calls: list[str] = []
        self._children = children

    def __call__(self, model: str) -> object:
        self.calls.append(model)
        idx = min(len(self.calls) - 1, len(self._children) - 1)
        return self._children[idx]


def _factory_and_runtime(
    parent_provider: object,
    tmp_path: Path,
    *,
    children: list[object],
    config: RuntimeConfig | None = None,
    policy: PermissionPolicy | None = None,
) -> tuple[ConversationRuntime, RuntimeFactory, _RecordingBuilder]:
    """親 runtime + RuntimeFactory + 記録付きビルダを組む（test_run_turn の流儀）。"""
    cfg = config or RuntimeConfig(
        subagent=SubagentConfig(default_model="child-mock"),
        tools=ToolsConfig(deferred_tool_loading=False),
    )
    pol = policy or default_policy(PermissionMode.WORKSPACE_WRITE)
    tools = default_registry()
    hooks = HookRunner(HookConfig(), tmp_path)
    builder = _RecordingBuilder(children)
    factory = RuntimeFactory(
        provider=parent_provider,  # type: ignore[arg-type]
        model="mock-model",
        config=cfg,
        tools=tools,
        policy=pol,
        hooks=hooks,
        cwd=tmp_path,
        provider_builder=builder,  # type: ignore[arg-type]
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=parent_provider,  # type: ignore[arg-type]
        tools=tools,
        policy=pol,
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=hooks,
        config=cfg,
        runtime_factory=factory,
    )
    return rt, factory, builder


def _agent_call(call_id: str = "a1", **overrides: str) -> ToolUse:
    inp: dict[str, str] = {"description": "delegate task", "prompt": "do the sub-task"}
    inp.update(overrides)
    return ToolUse(id=call_id, name="agent", input=inp)


async def test_agent_empty_final_is_error_with_diagnostic(tmp_path: Path) -> None:
    child = MockProvider([MessageStop()])
    parent = SequencedMockProvider(
        [
            [_agent_call(), MessageStop()],
            [TextDelta(text="parent done"), MessageStop()],
        ]
    )
    rt, _factory, _ = _factory_and_runtime(parent, tmp_path, children=[child])
    await rt.run_turn("go")
    block = rt.session.messages[2].blocks[0]
    assert block.is_error is True  # type: ignore[union-attr]
    assert "empty final answer" in block.output  # type: ignore[union-attr]


async def test_agent_empty_final_falls_back_to_last_text(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("x", encoding="utf-8")
    child = SequencedMockProvider(
        [
            [
                TextDelta(text="partial answer"),
                ToolUse(id="c1", name="read_file", input={"path": "hello.txt"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [MessageStop()],
        ]
    )
    parent = SequencedMockProvider(
        [
            [_agent_call(), MessageStop()],
            [TextDelta(text="parent done"), MessageStop()],
        ]
    )
    rt, _factory, _ = _factory_and_runtime(parent, tmp_path, children=[child])
    await rt.run_turn("go")
    block = rt.session.messages[2].blocks[0]
    assert block.is_error is False  # type: ignore[union-attr]
    assert block.output == "partial answer"  # type: ignore[union-attr]


async def test_agent_empty_final_falls_back_to_reasoning(tmp_path: Path) -> None:
    child = MockProvider([Thinking(text="reasoned answer", signature=None), MessageStop()])
    parent = SequencedMockProvider(
        [
            [_agent_call(), MessageStop()],
            [TextDelta(text="parent done"), MessageStop()],
        ]
    )
    rt, _factory, _ = _factory_and_runtime(parent, tmp_path, children=[child])
    await rt.run_turn("go")
    block = rt.session.messages[2].blocks[0]
    assert block.is_error is False  # type: ignore[union-attr]
    assert block.output.startswith("(answer found only in reasoning)")  # type: ignore[union-attr]
    assert "reasoned answer" in block.output  # type: ignore[union-attr]


async def test_agent_returns_only_summary(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("SECRET-FILE-BODY", encoding="utf-8")
    child = SequencedMockProvider(
        [
            [ToolUse(id="c1", name="read_file", input={"path": "hello.txt"}), MessageStop()],
            [TextDelta(text="child summary"), MessageStop()],
        ]
    )
    parent = SequencedMockProvider(
        [
            [_agent_call(), MessageStop()],
            [TextDelta(text="parent done"), MessageStop()],
        ]
    )
    rt, _factory, builder = _factory_and_runtime(parent, tmp_path, children=[child])
    summary = await rt.run_turn("go")

    assert summary.final_text == "parent done"
    assert [m.role for m in rt.session.messages] == ["user", "assistant", "tool", "assistant"]
    block = rt.session.messages[2].blocks[0]
    assert block.is_error is False  # type: ignore[union-attr]
    assert block.output == "child summary"  # type: ignore[union-attr]
    dump = "".join(m.model_dump_json() for m in rt.session.messages)
    assert "SECRET-FILE-BODY" not in dump
    assert builder.calls == ["child-mock"]
    session_files = list((tmp_path / ".satsuma" / "sessions").rglob("*.jsonl"))
    assert len(session_files) == 1


async def test_depth_limit_blocks_recursion(tmp_path: Path) -> None:
    cfg = RuntimeConfig(subagent=SubagentConfig(default_model="child-mock"))
    _rt, factory, builder = _factory_and_runtime(
        MockProvider([]), tmp_path, children=[MockProvider([])], config=cfg
    )
    jail = WorkspaceJail(tmp_path)
    ctx = ToolContext(
        workspace_root=jail.root,
        jail=jail,
        config=cfg,
        runtime_factory=factory,
        depth=cfg.subagent.max_depth,
    )
    res = await Agent().run(Agent.Input(description="d", prompt="p"), ctx)
    assert res.is_error
    assert "depth limit" in res.output
    assert builder.calls == []

    bare = ToolContext(workspace_root=jail.root, jail=jail, config=cfg)
    res = await Agent().run(Agent.Input(description="d", prompt="p"), bare)
    assert res.is_error
    assert "not available" in res.output


async def test_empty_fields_are_reported_together(tmp_path: Path) -> None:
    cfg = RuntimeConfig(subagent=SubagentConfig(default_model="child-mock"))
    _rt, factory, builder = _factory_and_runtime(
        MockProvider([]), tmp_path, children=[MockProvider([])], config=cfg
    )
    jail = WorkspaceJail(tmp_path)
    ctx = ToolContext(workspace_root=jail.root, jail=jail, config=cfg, runtime_factory=factory)
    both = await Agent().run(Agent.Input(description=" ", prompt=""), ctx)
    assert both.is_error
    assert both.output.startswith("agent was not started: description and prompt are empty.")
    one = await Agent().run(Agent.Input(description="look up", prompt="  "), ctx)
    assert one.output.startswith("agent was not started: prompt is empty.")
    assert builder.calls == []


async def test_agent_excluded_from_child_tools(tmp_path: Path) -> None:
    _rt, factory, _builder = _factory_and_runtime(
        MockProvider([]), tmp_path, children=[MockProvider([])]
    )
    child = factory.build_child(preset=resolve_subagent_type("general"), depth=1)
    assert child.tools.get("agent") is None
    assert factory.tools.get("agent") is not None
    assert child.tools.get("write_file") is not None


async def test_model_override(tmp_path: Path) -> None:
    child_provider = MockProvider([])
    _rt, factory, builder = _factory_and_runtime(
        MockProvider([]), tmp_path, children=[child_provider]
    )
    preset = resolve_subagent_type("general")

    child = factory.build_child(preset=preset, depth=1, model="opus")
    assert child.session.model == expand_model_name("opus")
    assert builder.calls == [expand_model_name("opus")]
    assert child.provider is child_provider

    same = factory.build_child(preset=preset, depth=1, model="mock-model")
    assert same.provider is factory.provider
    assert builder.calls == [expand_model_name("opus")]


async def test_explore_preset_tool_subset(tmp_path: Path) -> None:
    _rt, factory, _builder = _factory_and_runtime(
        MockProvider([]), tmp_path, children=[MockProvider([])]
    )
    child = factory.build_child(preset=resolve_subagent_type("Explore"), depth=1)
    names = {s["name"] for s in child.tools.json_schemas()}
    assert names == set(EXPLORE_TOOLS)


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("Explore", "Explore"),
        ("explore", "Explore"),
        ("Explorer", "Explore"),
        ("exploration", "Explore"),
        ("explore_agent", "Explore"),
        (" Plan ", "Plan"),
        ("planner", "Plan"),
        ("Plan-Agent", "Plan"),
        ("general", "general-purpose"),
        ("general-purpose", "general-purpose"),
        ("GeneralPurposeAgent", "general-purpose"),
        ("", "general-purpose"),
        ("unknown-kind", "general-purpose"),
        ("explain", "general-purpose"),
    ],
)
def test_subagent_type_is_read_by_its_stem(raw: str, key: str) -> None:
    assert resolve_subagent_type(raw).key == key


def test_presets_pair_the_role_text_with_the_tools() -> None:
    explore, plan, general = (resolve_subagent_type(k) for k in ("Explore", "Plan", "general"))
    assert explore.allowed_tools == EXPLORE_TOOLS and plan.allowed_tools == PLAN_TOOLS
    assert general.allowed_tools is None
    for preset in (explore, plan, general):
        assert preset.system_suffix.startswith("# Delegated task\n")
        assert f"as its `{preset.key}` helper" in preset.system_suffix
    assert "do not change anything" in explore.system_suffix
    assert "numbered plan" in plan.system_suffix
    assert "do not change anything" not in general.system_suffix


@pytest.mark.parametrize(
    "case",
    ["provider_error", "session_abort", "max_iterations"],
)
async def test_child_failure_is_error_not_crash(tmp_path: Path, case: str) -> None:
    (tmp_path / "loop.txt").write_text("x", encoding="utf-8")
    child: object
    cfg = RuntimeConfig(
        subagent=SubagentConfig(default_model="child-mock", max_iterations=1),
        tools=ToolsConfig(deferred_tool_loading=False),
    )
    if case == "provider_error":
        child = ScriptedFailureProvider([ProviderError("boom")])
    elif case == "session_abort":
        child = ScriptedFailureProvider([SessionAbortError("child session unhealthy")])
    else:
        child = MockProvider(
            [ToolUse(id="c1", name="read_file", input={"path": "loop.txt"}), MessageStop()]
        )
    parent = SequencedMockProvider(
        [
            [_agent_call(), MessageStop()],
            [TextDelta(text="parent done"), MessageStop()],
        ]
    )
    rt, _factory, _builder = _factory_and_runtime(parent, tmp_path, children=[child], config=cfg)
    summary = await rt.run_turn("go")

    assert summary.final_text == "parent done"
    block = rt.session.messages[2].blocks[0]
    assert block.is_error is True  # type: ignore[union-attr]
    assert "subagent failed" in block.output  # type: ignore[union-attr]


class _BarrierProvider:
    """barrier を通過してから応答する子プロバイダ（並行実行の証明用）。"""

    def __init__(self, barrier: asyncio.Barrier, text: str) -> None:
        self.kind = ProviderKind.ANTHROPIC
        self._barrier = barrier
        self._text = text

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        await asyncio.wait_for(self._barrier.wait(), timeout=2.0)
        yield TextDelta(text=self._text)
        yield MessageStop()

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


async def test_parallel_agents_gathered(tmp_path: Path) -> None:
    assert Agent.spec.parallel_safe is True
    barrier = asyncio.Barrier(2)
    parent = SequencedMockProvider(
        [
            [_agent_call("a1"), _agent_call("a2"), MessageStop()],
            [TextDelta(text="parent done"), MessageStop()],
        ]
    )
    rt, _factory, _builder = _factory_and_runtime(
        parent,
        tmp_path,
        children=[_BarrierProvider(barrier, "c1"), _BarrierProvider(barrier, "c2")],
    )
    summary = await rt.run_turn("go")
    assert summary.final_text == "parent done"
    blocks = rt.session.messages[2].blocks
    assert [b.is_error for b in blocks] == [False, False]  # type: ignore[union-attr]
    assert [b.output for b in blocks] == ["c1", "c2"]  # type: ignore[union-attr]


async def test_child_inherits_jail_and_policy_deny(tmp_path: Path) -> None:
    policy = default_policy(PermissionMode.WORKSPACE_WRITE).model_copy(
        update={"blocked_tools": ["write_file"]}
    )
    child_provider = SequencedMockProvider(
        [
            [
                ToolUse(id="w1", name="write_file", input={"path": "o.txt", "content": "x"}),
                MessageStop(),
            ],
            [TextDelta(text="child done"), MessageStop()],
        ]
    )
    _rt, factory, _builder = _factory_and_runtime(
        MockProvider([]), tmp_path, children=[child_provider], policy=policy
    )
    child = factory.build_child(preset=resolve_subagent_type("general"), depth=1)
    assert child.policy is factory.policy
    assert child.tool_context.jail.root == WorkspaceJail(tmp_path).root

    summary = await child.run_turn("write it")
    assert summary.final_text == "child done"
    block = child.session.messages[2].blocks[0]
    assert block.is_error is True  # type: ignore[union-attr]
    assert "listed under Denied tools" in block.output  # type: ignore[union-attr]
    assert not (tmp_path / "o.txt").exists()

    ro_policy = default_policy(PermissionMode.READ_ONLY)
    ask_child_provider = SequencedMockProvider(
        [
            [
                ToolUse(id="w2", name="write_file", input={"path": "p.txt", "content": "x"}),
                MessageStop(),
            ],
            [TextDelta(text="child done"), MessageStop()],
        ]
    )
    _rt2, ro_factory, _b2 = _factory_and_runtime(
        MockProvider([]), tmp_path, children=[ask_child_provider], policy=ro_policy
    )
    ro_child = ro_factory.build_child(preset=resolve_subagent_type("general"), depth=1)
    summary = await ro_child.run_turn("write it")
    assert summary.final_text == "child done"
    block = ro_child.session.messages[2].blocks[0]
    assert block.is_error is True  # type: ignore[union-attr]
    assert not (tmp_path / "p.txt").exists()


async def test_child_usage_aggregated(tmp_path: Path) -> None:
    child = MockProvider(
        [TextDelta(text="child summary"), Usage(input_tokens=100, output_tokens=7), MessageStop()]
    )
    parent = SequencedMockProvider(
        [
            [_agent_call(), Usage(input_tokens=10, output_tokens=1), MessageStop()],
            [TextDelta(text="parent done"), Usage(input_tokens=20, output_tokens=2), MessageStop()],
        ]
    )
    rt, _factory, _builder = _factory_and_runtime(parent, tmp_path, children=[child])
    summary = await rt.run_turn("go")
    assert (summary.usage.input_tokens, summary.usage.output_tokens) == (30, 3)
    assert (summary.delegated_usage.input_tokens, summary.delegated_usage.output_tokens) == (100, 7)
