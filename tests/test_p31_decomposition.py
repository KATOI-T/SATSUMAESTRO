"""自動タスク分解と並列振り分けのテスト。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from satsumaestro.config.models import (
    DecompositionConfig,
    HookConfig,
    ModelRoles,
    OrchestrationConfig,
    ProviderConfig,
    RuntimeConfig,
    SubagentConfig,
)
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import NullSessionStore, RuntimeFactory
from satsumaestro.engine.tool_call_repair import ToolCallRepairPolicy
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.orchestration.decompose import Decomposer, static_gate
from satsumaestro.orchestration.dispatch import Dispatcher
from satsumaestro.orchestration.executor import PlanExecutor, extract_evidence
from satsumaestro.orchestration.hook import DecompositionCoordinator, _CircuitBreaker
from satsumaestro.orchestration.plan import (
    DispatchMode,
    SubtaskKind,
    SubtaskSpec,
    TaskPlan,
    ValidatedPlan,
    paths_collide,
    validate_plan,
)
from satsumaestro.orchestration.runner import Orchestrator
from satsumaestro.orchestration.task_ledger import TaskStatus
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import OrchestrationError
from tests.fixtures.mock_provider import MockProvider, SequencedMockProvider


def _spec(
    id: str,
    *,
    kind: SubtaskKind = SubtaskKind.LOOKUP,
    depends_on: list[str] | None = None,
    writes: list[str] | None = None,
    est_calls: int = 2,
) -> SubtaskSpec:
    return SubtaskSpec(
        id=id,
        objective=f"objective for {id}",
        kind=kind,
        depends_on=depends_on or [],
        writes=writes or [],
        est_calls=est_calls,
    )


def _validate(plan: TaskPlan, *, max_subtasks: int = 6, max_depth: int = 3) -> ValidatedPlan:
    return validate_plan(plan, max_subtasks=max_subtasks, max_depth=max_depth)


def test_static_gate_rejects_short_single_and_write_intent() -> None:
    cfg = DecompositionConfig(enabled=True)
    long_pad = "背景として、既存の設計と実装の対応関係を確認したい状況にある。"

    multi = "認証の実装を調べて。課金の実装も調べて。両者を比較して報告して。"

    assert static_gate(multi, DecompositionConfig()).reason == "static_gate:disabled"
    off = static_gate(multi, cfg, orchestration_enabled=False)
    assert off.single and off.reason == "static_gate:disabled"
    assert static_gate(multi, cfg, breaker_open=True).reason == "breaker_open"
    assert static_gate(multi, cfg, plan_mode=True).reason == "static_gate:plan_mode"

    cases: list[tuple[str, str]] = [
        ("この関数は何?", "static_gate:short_prompt"),
        (
            "このリポジトリの認証まわりの実装がどうなっているのか全体像をできるだけ詳しく教えてほしい",
            "static_gate:single_sentence",
        ),
        (long_pad + "ログイン画面のバグをできるだけ早く直してください", "static_gate:write_intent"),
        (long_pad + "リトライ回数の設定を追加して、テストも実装して", "static_gate:write_intent"),
    ]
    for text, expected in cases:
        result = static_gate(text, cfg)
        assert result.single, text
        assert result.reason == expected, text

    passing = [
        "認証まわりの実装がどこにあるか調べて。課金まわりの実装も調べて。両者の共通点を教えて。",
        long_pad + "エラー処理・リトライ・タイムアウトの方針をそれぞれ整理して報告してほしい",
    ]
    for text in passing:
        result = static_gate(text, cfg)
        assert not result.single, text
        assert result.reason == "static_gate:pass"


def test_validate_plan_rejects_cycle_unknown_dep_and_overlap() -> None:
    cycle = TaskPlan(
        subtasks=[
            _spec("s1", depends_on=["s2"]),
            _spec("s2", depends_on=["s1"]),
        ]
    )
    with pytest.raises(OrchestrationError, match="cycle"):
        _validate(cycle)

    unknown = TaskPlan(subtasks=[_spec("s1", depends_on=["s9"])])
    with pytest.raises(OrchestrationError, match="unknown"):
        _validate(unknown)

    dup = TaskPlan(subtasks=[_spec("s1"), _spec("s1")])
    with pytest.raises(OrchestrationError, match="duplicate"):
        _validate(dup)

    empty = TaskPlan(subtasks=[SubtaskSpec(id="s1", objective="   ", kind=SubtaskKind.LOOKUP)])
    with pytest.raises(OrchestrationError, match="objective"):
        _validate(empty)

    conflict = TaskPlan(
        subtasks=[
            _spec("s1", kind=SubtaskKind.EDIT, writes=["app/models"]),
            _spec("s2", kind=SubtaskKind.EDIT, writes=["app/models/user.py"]),
        ]
    )
    with pytest.raises(OrchestrationError, match="conflict"):
        _validate(conflict)

    serialized = TaskPlan(
        subtasks=[
            _spec("s1", kind=SubtaskKind.EDIT, writes=["app/models"]),
            _spec("s2", kind=SubtaskKind.EDIT, writes=["app/models"], depends_on=["s1"]),
        ]
    )
    assert _validate(serialized).levels == [["s1"], ["s2"]]

    too_many = TaskPlan(subtasks=[_spec(f"s{i}") for i in range(7)])
    with pytest.raises(OrchestrationError, match="too many"):
        _validate(too_many)
    chain = TaskPlan(
        subtasks=[
            _spec("s1"),
            _spec("s2", depends_on=["s1"]),
            _spec("s3", depends_on=["s2"]),
            _spec("s4", depends_on=["s3"]),
        ]
    )
    with pytest.raises(OrchestrationError, match="deep"):
        _validate(chain, max_depth=3)


def test_paths_collide_by_whole_components() -> None:
    assert paths_collide("src", "src")
    assert paths_collide("src", "src/api")
    assert paths_collide("src/api/routes.py", "src")
    assert paths_collide("docs/", "docs/guide.md")
    assert not paths_collide("src", "src_old")
    assert not paths_collide("src/api", "src/web")


def test_levels_and_background_inline_split() -> None:
    plan = TaskPlan(
        subtasks=[
            _spec("s1"),
            _spec("s2", kind=SubtaskKind.ANALYSIS),
            _spec("s3", kind=SubtaskKind.EDIT, writes=["a.py"]),
            _spec("s4", kind=SubtaskKind.VERIFY),
            _spec("s5", depends_on=["s3"]),
            _spec("s6", kind=SubtaskKind.ANALYSIS, depends_on=["s5"]),
        ]
    )
    validated = _validate(plan)
    assert validated.levels == [["s1", "s2", "s3", "s4"], ["s5"], ["s6"]]
    assert validated.background == {"s1", "s2"}
    assert validated.inline == {"s3", "s4", "s5", "s6"}

    sneaky = _validate(TaskPlan(subtasks=[_spec("s1", writes=["b.py"]), _spec("s2")]))
    assert sneaky.background == {"s2"}
    assert sneaky.inline == {"s1"}

    layered = _validate(
        TaskPlan(subtasks=[_spec("s1"), _spec("s2"), _spec("s3", depends_on=["s1"])])
    )
    assert layered.background == {"s1", "s2", "s3"}


def test_dispatch_width_and_gain_gate() -> None:
    cfg = DecompositionConfig(enabled=True)

    three = _validate(TaskPlan(subtasks=[_spec("s1"), _spec("s2"), _spec("s3")]))
    decision = Dispatcher(cfg, capacity=4).decide(three)
    assert decision.mode is DispatchMode.PARALLEL
    assert decision.reason == "parallel(serial=6,parallel=3,width=3)"
    assert (decision.serial_cost, decision.parallel_cost, decision.width) == (6, 3, 3)

    two = _validate(TaskPlan(subtasks=[_spec("s1"), _spec("s2")]))
    decision = Dispatcher(cfg, capacity=4).decide(two)
    assert decision.mode is DispatchMode.PARALLEL
    assert decision.reason == "parallel(serial=4,parallel=3,width=2)"

    decision = Dispatcher(DecompositionConfig(overhead_calls=2), capacity=4).decide(two)
    assert decision.mode is DispatchMode.SINGLE
    assert decision.reason == "no_gain(serial=4,parallel=4)"

    chain = _validate(TaskPlan(subtasks=[_spec("s1"), _spec("s2", depends_on=["s1"])]))
    decision = Dispatcher(cfg, capacity=4).decide(chain)
    assert decision.mode is DispatchMode.SINGLE
    assert decision.reason == "width=1"

    decision = Dispatcher(cfg, capacity=1).decide(three)
    assert decision.mode is DispatchMode.SINGLE
    assert decision.reason == "capacity=1"

    decision = Dispatcher(cfg, capacity=2).decide(three)
    assert decision.mode is DispatchMode.SINGLE
    assert decision.reason == "no_gain(serial=6,parallel=5)"

    heavy = _validate(TaskPlan(subtasks=[_spec(f"s{i}", est_calls=3) for i in range(1, 4)]))
    assert (
        Dispatcher(DecompositionConfig(gain_margin=0.0), capacity=4).decide(heavy).mode
        is DispatchMode.PARALLEL
    )
    assert (
        Dispatcher(DecompositionConfig(gain_margin=1.5), capacity=4).decide(heavy).mode
        is DispatchMode.SINGLE
    )

    mixed = _validate(
        TaskPlan(
            subtasks=[
                _spec("s1"),
                _spec("s2", kind=SubtaskKind.EDIT, writes=["a.py"]),
            ]
        )
    )
    decision = Dispatcher(cfg, capacity=4).decide(mixed)
    assert decision.mode is DispatchMode.SINGLE
    assert decision.reason == "width=1"


_GOOD_PLAN_INPUT = {
    "single": False,
    "subtasks": [
        {"id": "s1", "objective": "調査 A", "kind": "lookup"},
        {"id": "s2", "objective": "調査 B", "kind": "analysis", "depends_on": ["s1"]},
    ],
    "reduce_hint": "両者を比較する",
}


class _BoomProvider:
    kind = ProviderKind.OPENAI

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        raise RuntimeError("provider down")
        yield TextDelta(text="")  # pragma: no cover

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


async def test_decomposer_degrades_on_bad_json() -> None:
    policy = ToolCallRepairPolicy()

    ok = MockProvider(
        [ToolUse(id="d1", name="decompose_plan", input=_GOOD_PLAN_INPUT), MessageStop()]
    )
    plan = await Decomposer(provider=ok, model="m", policy=policy).decompose(
        "調査 A と調査 B をそれぞれ", max_subtasks=6
    )
    assert plan is not None and [s.id for s in plan.subtasks] == ["s1", "s2"]
    assert len(ok.requests) == 1
    req = ok.requests[0]
    assert req.thinking is False and req.tools[0]["name"] == "decompose_plan"

    repaired = SequencedMockProvider(
        [
            [
                ToolUse(
                    id="d1",
                    name="decompose_plan",
                    input={"single": False, "subtasks": [{"id": "s1"}]},
                ),
                MessageStop(),
            ],
            [ToolUse(id="d2", name="decompose_plan", input=_GOOD_PLAN_INPUT), MessageStop()],
        ]
    )
    plan = await Decomposer(provider=repaired, model="m", policy=policy).decompose(
        "q", max_subtasks=6
    )
    assert plan is not None and len(repaired.requests) == 2
    hint = repaired.requests[1].messages[-1]
    assert hint.role == "user"
    hint_text = hint.blocks[0].text  # type: ignore[union-attr]
    assert "decompose_plan" in hint_text and "invalid" in hint_text

    broken = SequencedMockProvider(
        [
            [ToolUse(id="d1", name="decompose_plan", input={"subtasks": "oops"}), MessageStop()],
            [TextDelta(text="ツールを呼ばずに文章で答える"), MessageStop()],
        ]
    )
    result = await Decomposer(provider=broken, model="m", policy=policy).decompose(
        "q", max_subtasks=6
    )
    assert result is None and len(broken.requests) == 2

    off = SequencedMockProvider(
        [[ToolUse(id="d1", name="decompose_plan", input={"subtasks": "oops"}), MessageStop()]]
    )
    result = await Decomposer(
        provider=off, model="m", policy=ToolCallRepairPolicy(enabled=False)
    ).decompose("q", max_subtasks=6)
    assert result is None and len(off.requests) == 1

    result = await Decomposer(provider=_BoomProvider(), model="m", policy=policy).decompose(
        "q", max_subtasks=6
    )
    assert result is None


async def test_decomposer_normalizes_alias_keys_and_missing_ids() -> None:
    aliased = {
        "single": False,
        "subtasks": [
            {"kind": "lookup", "task": "権限モードを調べる"},
            {"id": "s1", "kind": "analysis", "description": "結果を比較する"},
            {"kind": "lookup", "task": "上限件数を調べる"},
        ],
    }
    provider = MockProvider([ToolUse(id="d1", name="decompose_plan", input=aliased), MessageStop()])
    plan = await Decomposer(provider=provider, model="m", policy=ToolCallRepairPolicy()).decompose(
        "q", max_subtasks=6
    )
    assert plan is not None and len(provider.requests) == 1
    assert [s.id for s in plan.subtasks] == ["s2", "s1", "s3"]
    assert [s.objective for s in plan.subtasks] == [
        "権限モードを調べる",
        "結果を比較する",
        "上限件数を調べる",
    ]
    hopeless = MockProvider(
        [
            ToolUse(
                id="d1",
                name="decompose_plan",
                input={"single": False, "subtasks": [{"kind": "lookup"}]},
            ),
            MessageStop(),
        ]
    )
    assert (
        await Decomposer(provider=hopeless, model="m", policy=ToolCallRepairPolicy()).decompose(
            "q", max_subtasks=6
        )
        is None
    )


class _GateProvider:
    """``gate`` が開くまで応答を返さない子（timeout 経路の検証用）。"""

    def __init__(self, gate: asyncio.Event) -> None:
        self.kind = ProviderKind.ANTHROPIC
        self.gate = gate

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        await self.gate.wait()
        yield TextDelta(text="late")
        yield MessageStop()

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


class _RoundRobinBuilder:
    """``build_child`` ごとに次の子プロバイダを返す（_RecordingBuilder と同型）。"""

    def __init__(self, children: list[object]) -> None:
        self.calls = 0
        self._children = children

    def __call__(self, model: str) -> object:
        idx = min(self.calls, len(self._children) - 1)
        self.calls += 1
        return self._children[idx]


def _orchestrator(tmp_path: Path, children: list[object]) -> Orchestrator:
    cfg = RuntimeConfig(
        subagent=SubagentConfig(default_model="child-mock", max_iterations=8),
        orchestration=OrchestrationConfig(enabled=True, max_concurrency=4),
    )
    factory = RuntimeFactory(
        provider=MockProvider([]),  # type: ignore[arg-type]
        model="mock-model",
        config=cfg,
        tools=default_registry(orchestration=True),
        policy=default_policy(PermissionMode.DANGER_FULL_ACCESS),
        hooks=HookRunner(HookConfig(), tmp_path),
        cwd=tmp_path,
        provider_builder=_RoundRobinBuilder(children),  # type: ignore[arg-type]
    )
    return Orchestrator(factory=factory, config=cfg.orchestration, cwd=tmp_path)


async def test_executor_embeds_parent_output_into_child_prompt(tmp_path: Path) -> None:
    chained = TaskPlan(
        subtasks=[
            _spec("s1"),
            _spec("s2"),
            _spec("s3", kind=SubtaskKind.ANALYSIS, depends_on=["s1"]),
        ]
    )
    validated = _validate(chained)
    children = [
        MockProvider([TextDelta(text="A は satsumaestro/a.py の定数"), MessageStop()]),
        MockProvider([TextDelta(text="B は satsumaestro/b.py を参照"), MessageStop()]),
        MockProvider([TextDelta(text="A+B の結論。根拠 satsumaestro/a.py"), MessageStop()]),
    ]
    orch = _orchestrator(tmp_path, children)
    try:
        executor = PlanExecutor(orch, DecompositionConfig(enabled=True))
        result = await executor.run(validated)
        assert [r.status for r in result.results] == ["ok", "ok", "ok"]
        assert result.fallback_ids == []
        assert result.results[0].evidence == ["satsumaestro/a.py"]
        tasks = orch.list_tasks()
        assert len(tasks) == 3 and all(t.preset == "Explore" for t in tasks)
        s3_task = tasks[2]
        assert "# Background" in s3_task.prompt and "satsumaestro/a.py の定数" in s3_task.prompt
        assert "do not fill the gap with a guess" in s3_task.prompt
        assert "# Background" not in tasks[1].prompt
        plans = orch.snapshot()["plans"]
        assert len(plans) == 1
        (record,) = plans.values()
        assert record["result"]["fallback_ids"] == []
        assert {t.plan_id for t in tasks} == set(plans.keys())
        context = executor.build_context("元の問い", validated, result)
        assert "A は satsumaestro/a.py の定数" in context and "統合" not in context.split("\n")[0]
    finally:
        await orch.aclose()


async def test_executor_timeout_and_empty_fallback(tmp_path: Path) -> None:
    gate = asyncio.Event()
    orch = _orchestrator(tmp_path, [_GateProvider(gate)])
    try:
        executor = PlanExecutor(orch, DecompositionConfig(enabled=True, per_call_budget_s=0.05))
        validated = _validate(TaskPlan(subtasks=[_spec("s1", est_calls=1), _spec("s2")]))
        result = await executor.run(validated)
        assert {r.status for r in result.results} == {"timeout"}
        assert sorted(result.fallback_ids) == ["s1", "s2"]
        stopped = orch.list_tasks()
        assert all(t.status is TaskStatus.STOPPED for t in stopped)
        context = executor.build_context("q", validated, result)
        assert "Not finished (timeout): look into this yourself" in context
    finally:
        gate.set()
        await orch.aclose()

    orch2 = _orchestrator(tmp_path, [MockProvider([MessageStop()])])
    try:
        executor2 = PlanExecutor(orch2, DecompositionConfig(enabled=True))
        validated2 = _validate(TaskPlan(subtasks=[_spec("s1"), _spec("s2")]))
        result2 = await executor2.run(validated2)
        assert {r.status for r in result2.results} == {"empty"}
        assert sorted(result2.fallback_ids) == ["s1", "s2"]
    finally:
        await orch2.aclose()


async def test_executor_requires_evidence(tmp_path: Path) -> None:
    orch = _orchestrator(
        tmp_path,
        [
            MockProvider([TextDelta(text="答えは 42 です"), MessageStop()]),
            MockProvider([TextDelta(text="see https://example.com/doc"), MessageStop()]),
        ],
    )
    try:
        executor = PlanExecutor(orch, DecompositionConfig(enabled=True))
        validated = _validate(TaskPlan(subtasks=[_spec("s1"), _spec("s2")]))
        result = await executor.run(validated)
        by_id = {r.id: r for r in result.results}
        assert by_id["s1"].status == "failed" and by_id["s1"].output == "答えは 42 です"
        assert by_id["s2"].status == "ok"
        assert by_id["s2"].evidence == ["https://example.com/doc"]
        assert result.fallback_ids == ["s1"]
    finally:
        await orch.aclose()


async def test_executor_skips_inline_subtasks(tmp_path: Path) -> None:
    orch = _orchestrator(
        tmp_path, [MockProvider([TextDelta(text="found satsumaestro/x.py"), MessageStop()])]
    )
    try:
        executor = PlanExecutor(orch, DecompositionConfig(enabled=True))
        validated = _validate(
            TaskPlan(
                subtasks=[
                    _spec("s1"),
                    _spec("s2", kind=SubtaskKind.EDIT, writes=["satsumaestro/x.py"]),
                ]
            )
        )
        result = await executor.run(validated)
        by_id = {r.id: r for r in result.results}
        assert by_id["s1"].status == "ok" and by_id["s2"].status == "inline"
        assert result.fallback_ids == ["s2"]
        assert len(orch.list_tasks()) == 1
    finally:
        await orch.aclose()


def test_extract_evidence_rules() -> None:
    text = (
        "根拠は satsumaestro/config/models.py と models.py、"
        "https://example.com/a?b=1 を参照。satsumaestro/config/models.py 再掲。"
    )
    found = extract_evidence(text)
    assert "https://example.com/a?b=1" in found
    assert "satsumaestro/config/models.py" in found
    assert found.count("satsumaestro/config/models.py") == 1
    assert extract_evidence("パスも URL も無い文") == []


def _hooked_runtime(
    tmp_path: Path,
    *,
    parent: MockProvider,
    with_hook: bool,
    decomposer: MockProvider | None = None,
    children: list[object] | None = None,
) -> tuple[ConversationRuntime, Orchestrator, DecompositionCoordinator | None]:
    cfg = RuntimeConfig(
        provider=ProviderConfig(roles=ModelRoles(decompose="decompose-mock")),
        subagent=SubagentConfig(default_model="child-mock", max_iterations=8),
        orchestration=OrchestrationConfig(enabled=True, max_concurrency=4),
        decomposition=DecompositionConfig(enabled=True, provider_parallelism=4),
    )
    rr = _RoundRobinBuilder(children or [])

    def builder(model: str) -> object:
        if model == "decompose-mock":
            assert decomposer is not None
            return decomposer
        return rr(model)

    tools = default_registry(orchestration=True)
    policy = default_policy(PermissionMode.DANGER_FULL_ACCESS)
    hooks = HookRunner(HookConfig(), tmp_path)
    factory = RuntimeFactory(
        provider=parent,  # type: ignore[arg-type]
        model="mock-model",
        config=cfg,
        tools=tools,
        policy=policy,
        hooks=hooks,
        cwd=tmp_path,
        provider_builder=builder,  # type: ignore[arg-type]
    )
    orch = Orchestrator(factory=factory, config=cfg.orchestration, cwd=tmp_path)
    session = new_session(tmp_path, "mock-model")
    coordinator = (
        DecompositionCoordinator(
            orchestrator=orch,
            config=cfg,
            provider=parent,  # type: ignore[arg-type]
            model="mock-model",
            session=session,
            cwd=tmp_path,
            env={},
            provider_builder=builder,  # type: ignore[arg-type]
        )
        if with_hook
        else None
    )
    rt = ConversationRuntime(
        session=session,
        provider=parent,  # type: ignore[arg-type]
        tools=tools,
        policy=policy,
        prompt_builder=SystemPromptBuilder(),
        store=NullSessionStore(),
        hooks=hooks,
        config=cfg,
        runtime_factory=factory,
        orchestrator=orch,
        decomposition=coordinator,
    )
    return rt, orch, coordinator


async def test_runtime_single_path_is_byte_identical(tmp_path: Path) -> None:
    question = "この関数は何?"
    parent_a = MockProvider([TextDelta(text="answer"), MessageStop()])
    parent_b = MockProvider([TextDelta(text="answer"), MessageStop()])
    rt_a, orch_a, _ = _hooked_runtime(tmp_path, parent=parent_a, with_hook=False)
    rt_b, orch_b, _ = _hooked_runtime(tmp_path, parent=parent_b, with_hook=True)
    try:
        summary_a = await rt_a.run_turn(question)
        summary_b = await rt_b.run_turn(question)
    finally:
        await orch_a.aclose()
        await orch_b.aclose()
    dump_a = [m.model_dump(mode="json") for m in rt_a.session.messages]
    dump_b = [m.model_dump(mode="json") for m in rt_b.session.messages]
    assert dump_a == dump_b
    assert rt_a.session.prompt_history == rt_b.session.prompt_history == [question]
    assert parent_a.requests[0].model_dump(mode="json") == parent_b.requests[0].model_dump(
        mode="json"
    )
    assert rt_b.session.todos == []
    assert summary_a.decomposition is None
    assert summary_b.decomposition is not None
    assert summary_b.decomposition.decision.reason == "static_gate:short_prompt"
    assert summary_b.final_text == summary_a.final_text == "answer"


async def test_runtime_parallel_adds_one_context_message(tmp_path: Path) -> None:
    question = "このリポジトリの認証の実装を調べて。課金周りの実装も調べて。設定の既定値も調べて。"
    plan_input = {
        "single": False,
        "subtasks": [
            {"id": "s1", "objective": "認証の実装を調べる", "kind": "lookup"},
            {"id": "s2", "objective": "課金の実装を調べる", "kind": "lookup"},
            {"id": "s3", "objective": "設定の既定値を調べる", "kind": "lookup"},
        ],
        "reduce_hint": "3 点をまとめる",
    }
    decomposer = MockProvider(
        [ToolUse(id="d1", name="decompose_plan", input=plan_input), MessageStop()]
    )
    children: list[object] = [
        MockProvider(
            [
                TextDelta(text=f"結果 {i}: satsumaestro/m{i}.py を参照"),
                Usage(input_tokens=5, output_tokens=3),
                MessageStop(),
            ]
        )
        for i in range(1, 4)
    ]
    parent = MockProvider([TextDelta(text="統合回答"), MessageStop()])
    rt, orch, _ = _hooked_runtime(
        tmp_path, parent=parent, with_hook=True, decomposer=decomposer, children=children
    )
    try:
        summary = await rt.run_turn(question)
    finally:
        await orch.aclose()

    roles = [m.role for m in rt.session.messages]
    assert roles == ["user", "user", "assistant"]
    context_text = rt.session.messages[1].blocks[0].text  # type: ignore[union-attr]
    assert context_text.startswith("[background research results]")
    assert "satsumaestro/m1.py" in context_text
    assert rt.session.prompt_history == [question]
    assert len(parent.requests[0].messages) == 2
    record = summary.decomposition
    assert record is not None and record.decision.mode is DispatchMode.PARALLEL
    assert record.result is not None
    assert [r.status for r in record.result.results] == ["ok", "ok", "ok"]
    assert summary.delegated_usage.input_tokens == 15
    assert summary.delegated_usage.output_tokens == 9
    assert summary.usage.input_tokens == 0
    assert [t.status for t in rt.session.todos] == ["completed"] * 3
    plans = orch.snapshot()["plans"]
    (payload,) = plans.values()
    assert set(payload) == {"decision", "plan", "result"}
    assert payload["decision"]["mode"] == "parallel"
    assert summary.final_text == "統合回答"


async def test_circuit_breaker_opens_after_losses(tmp_path: Path) -> None:
    breaker = _CircuitBreaker(2)
    breaker.note(wall_ms=100, serial_estimate_ms=50)
    assert not breaker.open
    breaker.note(wall_ms=100, serial_estimate_ms=50)
    assert breaker.open

    reset = _CircuitBreaker(2)
    reset.note(wall_ms=100, serial_estimate_ms=50)
    reset.note(wall_ms=10, serial_estimate_ms=50)
    reset.note(wall_ms=100, serial_estimate_ms=50)
    assert not reset.open

    parent = MockProvider([TextDelta(text="a"), MessageStop()])
    rt, orch, coordinator = _hooked_runtime(tmp_path, parent=parent, with_hook=True)
    assert coordinator is not None
    coordinator._breaker.note(wall_ms=100, serial_estimate_ms=50)
    coordinator._breaker.note(wall_ms=100, serial_estimate_ms=50)
    try:
        context = await coordinator.before_turn(
            "このリポジトリの認証の実装を調べて。課金周りの実装も調べて。設定の既定値も調べて。"
        )
    finally:
        await orch.aclose()
    record = coordinator.take_record()
    assert context is None
    assert record is not None and record.decision.reason == "breaker_open"


async def test_eval_spec_parses_decompose_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from satsumaestro.cli import app as cli_app
    from satsumaestro.evals.agents import SatsumaAgent

    monkeypatch.setattr(cli_app, "build_provider", lambda cfg, m, env: MockProvider([]))

    async def _no_fit(cfg: object, model: object, env: object) -> None:
        return None

    monkeypatch.setattr(cli_app, "apply_context_fit", _no_fit)
    agent = await cli_app._build_eval_agent(
        "satsuma:gpt-oss:120b+decompose", None, 600.0, tmp_path, trusted=True
    )
    assert isinstance(agent, SatsumaAgent)
    assert agent.name == "satsuma+decompose" and agent.model == "gpt-oss:120b"
    agent = await cli_app._build_eval_agent(
        "satsuma:gpt-oss:120b", None, 600.0, tmp_path, trusted=True
    )
    assert isinstance(agent, SatsumaAgent)
    assert agent.name == "satsuma" and agent.model == "gpt-oss:120b"
    agent = await cli_app._build_eval_agent(
        "satsuma+decompose", "gpt-oss:120b", 600.0, tmp_path, trusted=True
    )
    assert isinstance(agent, SatsumaAgent)
    assert agent.name == "satsuma+decompose" and agent.model == "gpt-oss:120b"


async def test_satsuma_agent_decompose_wires_and_reports(tmp_path: Path) -> None:
    from satsumaestro.evals.agents import SatsumaAgent

    (tmp_path / "a.py").write_text("X = 1\n", encoding="utf-8")
    plan_input = {
        "single": False,
        "subtasks": [
            {"id": "s1", "objective": "A を調べる", "kind": "lookup"},
            {"id": "s2", "objective": "B を調べる", "kind": "lookup"},
            {"id": "s3", "objective": "C を調べる", "kind": "lookup"},
        ],
    }
    decomposer = MockProvider(
        [ToolUse(id="d1", name="decompose_plan", input=plan_input), MessageStop()]
    )
    children = [
        MockProvider(
            [
                TextDelta(text=f"結果 {i}: a.py を参照"),
                Usage(input_tokens=5, output_tokens=3),
                MessageStop(),
            ]
        )
        for i in range(3)
    ]
    rr = _RoundRobinBuilder(children)

    def builder(model: str) -> object:
        return decomposer if model == "decompose-mock" else rr(model)

    parent = MockProvider([TextDelta(text="統合回答"), MessageStop()])
    cfg = RuntimeConfig(
        provider=ProviderConfig(roles=ModelRoles(decompose="decompose-mock")),
        subagent=SubagentConfig(default_model="child-mock", max_iterations=8),
        decomposition=DecompositionConfig(provider_parallelism=4),
    )
    agent = SatsumaAgent(
        parent,  # type: ignore[arg-type]
        cfg,
        "mock-model",
        decompose=True,
        provider_builder=builder,  # type: ignore[arg-type]
        env={},
    )
    answer = await agent.run(
        "このリポジトリの認証の実装を調べて。課金周りの実装も調べて。設定の既定値も調べて。",
        tmp_path,
    )
    assert answer.final_text == "統合回答"
    assert answer.decomposition_reason.startswith("parallel(")
    assert answer.delegated_usage.input_tokens == 15
    assert answer.usage.input_tokens == 0

    plain = SatsumaAgent(
        MockProvider([TextDelta(text="ans"), MessageStop()]),  # type: ignore[arg-type]
        cfg,
        "mock-model",
    )
    answer = await plain.run("この関数は何?", tmp_path)
    assert answer.final_text == "ans"
    assert answer.decomposition_reason == ""
    assert answer.delegated_usage.input_tokens == 0


async def test_parallel_guidance_prompt_toggle(tmp_path: Path) -> None:
    marker = "in a single response"
    on = (
        await SystemPromptBuilder().build(
            tmp_path, "mock-model", RuntimeConfig(parallel_guidance=True)
        )
    ).joined()
    off = (await SystemPromptBuilder().build(tmp_path, "mock-model", RuntimeConfig())).joined()
    assert marker in on and marker not in off
    assert (
        on.replace(
            "\n- When several independent read-only lookups are needed (searching or reading "
            "unrelated files), issue those tool calls together in a single response so they "
            "run in parallel; keep a call that needs another call's result in a later response.",
            "",
        )
        == off
    )
