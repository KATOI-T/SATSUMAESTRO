"""並列オーケストレーションのテスト。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from satsumaestro.config.models import (
    HookConfig,
    OrchestrationConfig,
    RuntimeConfig,
    SubagentConfig,
    ToolsConfig,
)
from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    Usage,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import (
    CHILD_EXCLUDED_TOOLS,
    EXPLORE_TOOLS,
    RuntimeFactory,
    resolve_subagent_type,
)
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.orchestration import (
    CronLedger,
    FailureCause,
    Freshness,
    Heartbeat,
    RunEvent,
    RunEventKind,
    TaskLedger,
    TaskStatus,
    TeamLedger,
    TeamStatus,
)
from satsumaestro.orchestration.runner import Orchestrator
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionPolicy, default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import bash as bash_mod
from satsumaestro.tools.builtins.bash import Bash
from satsumaestro.tools.builtins.orchestration_tools import (
    MUTATING_TOOL_NAMES,
    ORCHESTRATION_TOOL_NAMES,
    ORCHESTRATION_TOOLS,
    AnnotateTask,
    CreateTeam,
    DeleteCron,
    DeleteTeam,
    GetTask,
    ReadTaskOutput,
    StopTask,
    _failure,
    known_ids_hint,
)
from satsumaestro.tools.registry import ToolRegistry
from satsumaestro.util.errors import OrchestrationError, ProviderError
from tests.fixtures.mock_provider import (
    MockProvider,
    ScriptedFailureProvider,
    SequencedMockProvider,
)

_KINDS = {"started", "ready", "prompt_misdelivered", "blocked", "finished", "failed", "closed"}
_CAUSES = {
    "prompt_not_delivered",
    "trust_pending",
    "branch_behind",
    "mcp_init",
    "tool_failure",
    "outside_workspace",
    "infrastructure",
}


def test_run_event_values_and_shape() -> None:
    assert {k.value for k in RunEventKind} == _KINDS
    assert {c.value for c in FailureCause} == _CAUSES
    ev = RunEvent.failed(3, cause=FailureCause.TOOL_FAILURE, detail="2 failed", data={"attempt": 1})
    dumped = ev.model_dump(mode="json")
    assert isinstance(dumped["at"], int)
    assert dumped == {
        "kind": "failed",
        "seq": 3,
        "at": dumped["at"],
        "cause": "tool_failure",
        "detail": "2 failed",
        "data": {"attempt": 1},
    }


def test_run_event_factories() -> None:
    assert [k for k in RunEventKind if k.final] == [
        RunEventKind.FINISHED,
        RunEventKind.FAILED,
        RunEventKind.CLOSED,
    ]
    assert RunEvent.started(1).final is False and RunEvent.ready(2).final is False
    assert RunEvent.closed(3, detail="bye").final is True
    misdelivered = RunEvent.prompt_misdelivered(4, detail="wrong cwd")
    assert misdelivered.cause is FailureCause.PROMPT_NOT_DELIVERED and misdelivered.final is False
    blocked = RunEvent.blocked(5, cause=FailureCause.TRUST_PENDING)
    assert blocked.kind is RunEventKind.BLOCKED and blocked.detail is None and blocked.data is None
    finished = RunEvent.finished(6, data={"iterations": 2})
    assert finished.cause is None and finished.data == {"iterations": 2}
    assert [RunEvent.started(s).seq for s in (0, 7)] == [0, 7]


def test_task_lifecycle() -> None:
    reg = TaskLedger()
    assert len(reg) == 0
    t = reg.create("do it", "desc")
    assert t.task_id.startswith("task_") and t.status is TaskStatus.CREATED
    assert t.notes == [] and t.output == "" and t.team_id is None and t.heartbeat is None
    reg.move_to(t.task_id, TaskStatus.RUNNING)
    reg.add_output(t.task_id, "hello ")
    reg.add_output(t.task_id, "world")
    assert reg.output(t.task_id) == "hello world"
    upd = reg.note(t.task_id, "note")
    assert [n.author for n in upd.notes] == ["user"] and upd.notes[0].text == "note"
    upd.output = "tampered"
    assert reg.output(t.task_id) == "hello world"
    reg.move_to(t.task_id, TaskStatus.COMPLETED)
    with pytest.raises(OrchestrationError, match=r"already finished \(completed\)"):
        reg.stop(t.task_id)
    t2 = reg.create("second")
    assert reg.stop(t2.task_id).status is TaskStatus.STOPPED
    with pytest.raises(OrchestrationError, match="stopped"):
        reg.stop(t2.task_id)
    t3 = reg.create("third")
    reg.move_to(t3.task_id, TaskStatus.FAILED)
    with pytest.raises(OrchestrationError, match="failed"):
        reg.stop(t3.task_id)
    assert [x.task_id for x in reg.list(TaskStatus.STOPPED)] == [t2.task_id]
    assert len(reg.list()) == 3
    for op in (reg.stop, reg.output):
        with pytest.raises(OrchestrationError, match="unknown task id: missing"):
            op("missing")
    assert reg.remove("missing") is None
    assert reg.remove(t3.task_id) is not None and len(reg) == 2


def test_board_groups_and_freshness() -> None:
    reg = TaskLedger()
    running = reg.create("running")
    reg.move_to(running.task_id, TaskStatus.RUNNING)
    reg.pulse(running.task_id, Heartbeat(at=100_000, note="running"))
    dead = reg.create("dead transport")
    reg.pulse(dead.task_id, Heartbeat(at=105_000, connected=False))
    blocked = reg.create("blocked")
    reg.move_to(blocked.task_id, TaskStatus.BLOCKED)
    reg.pulse(blocked.task_id, Heartbeat(at=10_000))
    done = reg.create("done")
    reg.move_to(done.task_id, TaskStatus.COMPLETED)
    done_dead = reg.create("done dead")
    reg.pulse(done_dead.task_id, Heartbeat(at=100, connected=False))
    reg.move_to(done_dead.task_id, TaskStatus.COMPLETED)
    edge = reg.create("edge")
    reg.pulse(edge.task_id, Heartbeat(at=80_000))

    board = reg.board_at(110_000, 30_000)
    assert board.as_of == 110_000
    by_id = {e.task_id: e for group in (board.active, board.blocked, board.finished) for e in group}
    assert {e.task_id for e in board.active} == {running.task_id, dead.task_id, edge.task_id}
    assert [e.task_id for e in board.blocked] == [blocked.task_id]
    assert [e.task_id for e in board.finished] == [done.task_id, done_dead.task_id]
    assert by_id[running.task_id].freshness is Freshness.HEALTHY
    assert by_id[dead.task_id].freshness is Freshness.DISCONNECTED
    assert by_id[blocked.task_id].freshness is Freshness.STALLED
    assert by_id[done.task_id].freshness is Freshness.FINISHED
    assert by_id[done_dead.task_id].freshness is Freshness.FINISHED
    assert by_id[edge.task_id].freshness is Freshness.HEALTHY
    assert board.model_dump(mode="json")["blocked"][0]["freshness"] == "stalled"


def test_team_soft_delete() -> None:
    reg = TeamLedger()
    team = reg.create("alpha", ["task_1", "task_2"])
    assert team.team_id.startswith("team_") and team.status is TeamStatus.CREATED
    deleted = reg.delete(team.team_id)
    assert deleted.status is TeamStatus.DELETED
    still = reg.get(team.team_id)
    assert still is not None and still.status is TeamStatus.DELETED and len(reg) == 1
    with pytest.raises(OrchestrationError, match="unknown team id: nope"):
        reg.delete("nope")
    assert reg.remove(team.team_id) is not None and len(reg) == 0
    assert reg.remove(team.team_id) is None


def test_cron_enabled_filter() -> None:
    reg = CronLedger()
    a = reg.create("0 * * * *", "hourly")
    b = reg.create("0 0 * * *", "daily", "nightly job")
    assert a.cron_id.startswith("cron_") and a.enabled and a.runs == 0
    assert a.last_ran_at is None and a.description is None and b.description == "nightly job"
    reg.disable(a.cron_id)
    assert [e.cron_id for e in reg.list(only_enabled=True)] == [b.cron_id]
    assert len(reg.list()) == 2
    reg.mark_ran(b.cron_id)
    reg.mark_ran(b.cron_id)
    got = reg.get(b.cron_id)
    assert got is not None and got.runs == 2 and got.last_ran_at is not None
    removed = reg.delete(b.cron_id)
    assert removed.cron_id == b.cron_id and reg.get(b.cron_id) is None and len(reg) == 1
    for op in (reg.delete, reg.disable, reg.mark_ran):
        with pytest.raises(OrchestrationError, match="unknown cron id: x"):
            op("x")


def test_orchestration_config_defaults() -> None:
    cfg = RuntimeConfig()
    assert cfg.orchestration.enabled is False
    assert cfg.orchestration.max_concurrency == 4 and cfg.orchestration.drain_on_exit is True
    with pytest.raises(ValidationError):
        OrchestrationConfig(bogus=True)  # type: ignore[call-arg]


class _RecordingBuilder:
    def __init__(self, children: list[object]) -> None:
        self.calls: list[str] = []
        self._children = children

    def __call__(self, model: str) -> object:
        self.calls.append(model)
        idx = min(len(self.calls) - 1, len(self._children) - 1)
        return self._children[idx]


class _GateProvider:
    """``gate`` が開くまで応答を返さない子プロバイダ（停止・並列度の検証用）。"""

    def __init__(self, gate: asyncio.Event, text: str = "done") -> None:
        self.kind = ProviderKind.ANTHROPIC
        self.gate = gate
        self.text = text
        self.started = asyncio.Event()
        self.calls = 0

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.calls += 1
        self.started.set()
        await self.gate.wait()
        yield TextDelta(text=self.text)
        yield MessageStop()

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


def _harness(
    parent: object,
    tmp_path: Path,
    *,
    children: list[object],
    policy: PermissionPolicy | None = None,
    max_concurrency: int = 4,
) -> tuple[ConversationRuntime, Orchestrator, _RecordingBuilder]:
    cfg = RuntimeConfig(
        subagent=SubagentConfig(default_model="child-mock"),
        orchestration=OrchestrationConfig(enabled=True, max_concurrency=max_concurrency),
        tools=ToolsConfig(deferred_tool_loading=False),
    )
    pol = policy or default_policy(PermissionMode.DANGER_FULL_ACCESS)
    tools = default_registry(orchestration=True)
    hooks = HookRunner(HookConfig(), tmp_path)
    builder = _RecordingBuilder(children)
    factory = RuntimeFactory(
        provider=parent,  # type: ignore[arg-type]
        model="mock-model",
        config=cfg,
        tools=tools,
        policy=pol,
        hooks=hooks,
        cwd=tmp_path,
        provider_builder=builder,  # type: ignore[arg-type]
    )
    orch = Orchestrator(
        factory=factory,
        config=cfg.orchestration,
        cwd=tmp_path,
        state_path=tmp_path / ".satsuma" / "orchestration-state.json",
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=parent,  # type: ignore[arg-type]
        tools=tools,
        policy=pol,
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=hooks,
        config=cfg,
        runtime_factory=factory,
        orchestrator=orch,
    )
    return rt, orch, builder


async def test_background_task_runs_to_completion(tmp_path: Path) -> None:
    child = MockProvider([TextDelta(text="bg result"), MessageStop()])
    _rt, orch, builder = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("summarize", "bg job")
    assert task.status is TaskStatus.CREATED and orch.running_count() == 1
    await orch.drain()
    done = orch.get_task(task.task_id)
    assert done.status is TaskStatus.COMPLETED and done.description == "bg job"
    assert orch.task_output(task.task_id) == "bg result"
    assert builder.calls == ["child-mock"] and orch.running_count() == 0
    events = orch.events(task.task_id)
    assert [e.kind for e in events] == [RunEventKind.STARTED, RunEventKind.FINISHED]
    assert events[-1].final
    assert events[-1].data == {"iterations": 1, "tool_calls": 0}
    assert [e.seq for e in events] == sorted(e.seq for e in events)
    assert done.heartbeat is not None and done.heartbeat.connected is False
    board = orch.board()
    assert [e.task_id for e in board.finished] == [task.task_id] and board.active == []


async def test_empty_final_task_fails_with_diagnostic(tmp_path: Path) -> None:
    child = MockProvider([MessageStop()])
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("empty job")
    await orch.drain()
    done = orch.get_task(task.task_id)
    assert done.status is TaskStatus.FAILED
    out = orch.task_output(task.task_id)
    assert out and "empty final answer" in out
    failed = [e for e in orch.events(task.task_id) if e.kind is RunEventKind.FAILED]
    assert len(failed) == 1
    assert failed[0].cause is FailureCause.TOOL_FAILURE


async def test_empty_final_falls_back_to_last_text(tmp_path: Path) -> None:
    child = SequencedMockProvider(
        [
            [
                TextDelta(text="found: test_a, test_b"),
                ToolUse(id="c1", name="read_file", input={"path": "x"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [MessageStop()],
        ]
    )
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("listing job")
    await orch.drain()
    done = orch.get_task(task.task_id)
    assert done.status is TaskStatus.COMPLETED
    assert orch.task_output(task.task_id) == "found: test_a, test_b"


async def test_empty_final_falls_back_to_reasoning(tmp_path: Path) -> None:
    child = MockProvider([Thinking(text="the answer is 42", signature=None), MessageStop()])
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("reasoning job")
    await orch.drain()
    done = orch.get_task(task.task_id)
    assert done.status is TaskStatus.COMPLETED
    out = orch.task_output(task.task_id)
    assert out.startswith("(answer found only in reasoning)") and "42" in out


async def test_stop_cancels_running_task(tmp_path: Path) -> None:
    gate = asyncio.Event()
    child = _GateProvider(gate)
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("long job")
    await asyncio.wait_for(child.started.wait(), timeout=2.0)
    assert orch.get_task(task.task_id).status is TaskStatus.RUNNING
    stopped = orch.stop_task(task.task_id)
    assert stopped.status is TaskStatus.STOPPED
    await orch.drain()
    after = orch.get_task(task.task_id)
    assert after.status is TaskStatus.STOPPED and orch.running_count() == 0
    assert after.heartbeat is not None and after.heartbeat.connected is False
    names = [e.kind for e in orch.events(task.task_id)]
    assert RunEventKind.CLOSED in names and RunEventKind.FINISHED not in names
    with pytest.raises(OrchestrationError, match=r"already finished \(stopped\)"):
        orch.stop_task(task.task_id)


async def test_stop_queued_task_never_runs(tmp_path: Path) -> None:
    gate = asyncio.Event()
    first = _GateProvider(gate, "first")
    second = MockProvider([TextDelta(text="second"), MessageStop()])
    _rt, orch, builder = _harness(
        MockProvider([]), tmp_path, children=[first, second], max_concurrency=1
    )
    t1 = await orch.create_task("one")
    t2 = await orch.create_task("two")
    await asyncio.wait_for(first.started.wait(), timeout=2.0)
    assert orch.get_task(t2.task_id).status is TaskStatus.CREATED
    orch.stop_task(t2.task_id)
    gate.set()
    await orch.drain()
    assert orch.get_task(t1.task_id).status is TaskStatus.COMPLETED
    assert orch.get_task(t2.task_id).status is TaskStatus.STOPPED
    assert builder.calls == ["child-mock"]
    assert len(second.requests) == 0


async def test_concurrency_semaphore(tmp_path: Path) -> None:
    g1, g2 = asyncio.Event(), asyncio.Event()
    c1, c2 = _GateProvider(g1, "a"), _GateProvider(g2, "b")
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[c1, c2], max_concurrency=1)
    t1 = await orch.create_task("one")
    t2 = await orch.create_task("two")
    await asyncio.wait_for(c1.started.wait(), timeout=2.0)
    await asyncio.sleep(0)
    assert c2.calls == 0 and orch.get_task(t2.task_id).status is TaskStatus.CREATED
    assert len(orch.board().active) == 2
    g1.set()
    await asyncio.wait_for(c2.started.wait(), timeout=2.0)
    g2.set()
    await orch.drain()
    assert {orch.task_output(t1.task_id), orch.task_output(t2.task_id)} == {"a", "b"}


async def test_child_failure_marks_failed(tmp_path: Path) -> None:
    child = ScriptedFailureProvider([ProviderError("boom")])
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("will fail")
    await orch.drain()
    failed = orch.get_task(task.task_id)
    assert failed.status is TaskStatus.FAILED and "ProviderError: boom" in failed.output
    events = orch.events(task.task_id)
    terminal = [e for e in events if e.kind is RunEventKind.FAILED]
    assert len(terminal) == 1 and terminal[0].cause is FailureCause.INFRASTRUCTURE
    assert terminal[0].final


async def test_aclose_cancels_running(tmp_path: Path) -> None:
    gate = asyncio.Event()
    child = _GateProvider(gate)
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("never finishes")
    await asyncio.wait_for(child.started.wait(), timeout=2.0)
    await orch.aclose()
    assert orch.get_task(task.task_id).status is TaskStatus.STOPPED
    assert orch.running_count() == 0
    last = orch.events(task.task_id)[-1]
    assert last.kind is RunEventKind.CLOSED and last.final
    assert last.detail == "stopped by orchestrator shutdown"
    snap = json.loads((tmp_path / ".satsuma" / "orchestration-state.json").read_text("utf-8"))
    assert snap["tasks"][0]["status"] == "stopped"
    assert snap["events"][task.task_id][-1]["kind"] == "closed"


async def test_aclose_after_stop_task_does_not_duplicate_closed(tmp_path: Path) -> None:
    gate = asyncio.Event()
    child = _GateProvider(gate)
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("stopped before shutdown")
    await asyncio.wait_for(child.started.wait(), timeout=2.0)
    orch.stop_task(task.task_id)
    await orch.aclose()
    closed = [e for e in orch.events(task.task_id) if e.kind is RunEventKind.CLOSED]
    assert len(closed) == 1 and closed[0].detail == "stopped by task_stop"
    assert orch.running_count() == 0


async def test_team_create_validates_ids_and_delete_stops_members(tmp_path: Path) -> None:
    gate = asyncio.Event()
    c1, c2 = _GateProvider(gate), _GateProvider(gate)
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[c1, c2])
    t1 = await orch.create_task("one")
    t2 = await orch.create_task("two")
    with pytest.raises(OrchestrationError, match="unknown task id: ghost"):
        orch.create_team("team", [t1.task_id, "ghost"])
    assert orch.teams.list() == []
    team = orch.create_team("team", [t1.task_id, t2.task_id])
    assert orch.get_task(t1.task_id).team_id == team.team_id
    assert orch.board().active[0].team_id == team.team_id
    deleted = orch.delete_team(team.team_id)
    assert deleted.status is TeamStatus.DELETED
    await orch.drain()
    assert orch.get_task(t1.task_id).status is TaskStatus.STOPPED
    assert orch.get_task(t2.task_id).status is TaskStatus.STOPPED
    team2 = orch.create_team("again", [t1.task_id])
    assert orch.delete_team(team2.team_id).status is TeamStatus.DELETED


async def test_snapshot_written_atomically(tmp_path: Path) -> None:
    child = MockProvider([TextDelta(text="x"), MessageStop()])
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    path = tmp_path / ".satsuma" / "orchestration-state.json"
    task = await orch.create_task("snap")
    snap = json.loads(path.read_text("utf-8"))
    assert snap["version"] == 1 and snap["pid"] == os.getpid()
    assert snap["tasks"][0]["task_id"] == task.task_id
    assert not path.with_suffix(".json.tmp").exists()
    await orch.drain()
    snap = json.loads(path.read_text("utf-8"))
    assert snap["board"]["finished"][0]["freshness"] == "finished"
    assert snap["events"][task.task_id][-1]["kind"] == "finished"
    assert "at" in snap["events"][task.task_id][-1]
    orch.create_cron("* * * * *", "tick")
    assert json.loads(path.read_text("utf-8"))["crons"][0]["schedule"] == "* * * * *"


async def test_tools_via_run_turn(tmp_path: Path) -> None:
    child = MockProvider([TextDelta(text="bg done"), MessageStop()])
    parent = SequencedMockProvider(
        [
            [
                ToolUse(
                    id="t1", name="task_create", input={"prompt": "bg work", "description": "bg"}
                ),
                MessageStop(),
            ],
            [ToolUse(id="t2", name="task_list", input={}), MessageStop()],
            [TextDelta(text="parent done"), MessageStop()],
        ]
    )
    rt, orch, _ = _harness(parent, tmp_path, children=[child])
    summary = await rt.run_turn("go")
    assert summary.final_text == "parent done"
    created = rt.session.messages[2].blocks[0]
    assert created.is_error is False  # type: ignore[union-attr]
    task_id = json.loads(created.output)["task_id"]  # type: ignore[union-attr]
    listed = json.loads(rt.session.messages[4].blocks[0].output)  # type: ignore[union-attr]
    assert listed["count"] == 1 and listed["tasks"][0]["task_id"] == task_id
    assert "board" in listed
    await orch.drain()
    assert orch.task_output(task_id) == "bg done"

    parent2 = SequencedMockProvider(
        [
            [ToolUse(id="t1", name="task_create", input={"prompt": "bg"}), MessageStop()],
            [TextDelta(text="parent done"), MessageStop()],
        ]
    )
    rt2, orch2, _ = _harness(
        parent2, tmp_path, children=[child], policy=default_policy(PermissionMode.WORKSPACE_WRITE)
    )
    await rt2.run_turn("go")
    denied = rt2.session.messages[2].blocks[0]
    assert denied.is_error is True  # type: ignore[union-attr]
    assert orch2.list_tasks() == []


async def test_not_found_errors_list_known_ids(tmp_path: Path) -> None:
    child = MockProvider([TextDelta(text="x"), MessageStop()])
    rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child, child])
    t1 = await orch.create_task("job 1")
    t2 = await orch.create_task("job 2")
    await orch.drain()
    ctx = rt.tool_context
    for tool, inp in (
        (GetTask, {"task_id": "task_2"}),
        (ReadTaskOutput, {"task_id": "task_2"}),
        (StopTask, {"task_id": "task_2"}),
        (AnnotateTask, {"task_id": "task_2", "message": "m"}),
    ):
        res = await tool().run(tool.Input(**inp), ctx)  # type: ignore[arg-type]
        assert res.is_error, tool.spec.name
        assert "unknown task id: task_2" in res.output, tool.spec.name
        assert t1.task_id in res.output and t2.task_id in res.output, tool.spec.name
    res = await DeleteTeam().run(DeleteTeam.Input(team_id="team_9"), ctx)
    assert "unknown team id: team_9" in res.output and "no known team_ids" in res.output
    res = await DeleteCron().run(DeleteCron.Input(cron_id="cron_9"), ctx)
    assert "unknown cron id: cron_9" in res.output and "no known cron_ids" in res.output


async def test_known_ids_listing_truncates(tmp_path: Path) -> None:
    rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[MockProvider([])])
    ids = [orch.tasks.create(f"p{i}", None).task_id for i in range(12)]
    out = known_ids_hint("task", orch)
    assert out.startswith("(known task_ids: ") and "… and 2 more" in out
    assert re.search(rf"{re.escape(ids[0])}\b", out) is None
    assert re.search(rf"{re.escape(ids[1])}\b", out) is None
    assert all(re.search(rf"{re.escape(i)}\b", out) for i in ids[2:])
    assert _failure(OrchestrationError("prompt is required"), orch).output == "prompt is required"
    assert known_ids_hint("team", orch) == "(no known team_ids)"


async def test_tools_degrade_without_orchestrator(tool_ctx: ToolContext) -> None:
    inputs: dict[str, dict[str, object]] = {
        "task_create": {"prompt": "p"},
        "task_get": {"task_id": "x"},
        "task_list": {},
        "task_stop": {"task_id": "x"},
        "task_update": {"task_id": "x", "message": "m"},
        "task_output": {"task_id": "x"},
        "team_create": {"name": "n", "task_ids": ["x"]},
        "team_delete": {"team_id": "x"},
        "cron_create": {"schedule": "* * * * *", "prompt": "p"},
        "cron_delete": {"cron_id": "x"},
        "cron_list": {},
    }
    assert set(inputs) == ORCHESTRATION_TOOL_NAMES
    for tool in ORCHESTRATION_TOOLS:
        res = await tool().run(tool.Input(**inputs[tool.spec.name]), tool_ctx)
        assert res.is_error and "not available" in res.output, tool.spec.name


def test_schema_lint_and_modes() -> None:
    registry = ToolRegistry(strict_schema=True)
    for tool in ORCHESTRATION_TOOLS:
        registry.register(tool)
    read_only = {"task_get", "task_list", "task_output", "cron_list"}
    for tool in ORCHESTRATION_TOOLS:
        if tool.spec.name in read_only:
            assert tool.spec.required_mode is PermissionMode.READ_ONLY and tool.spec.parallel_safe
        else:
            assert tool.spec.required_mode is PermissionMode.DANGER_FULL_ACCESS
            assert not tool.spec.parallel_safe
        assert len(tool.spec.input_schema["properties"]) <= 8
    assert MUTATING_TOOL_NAMES == ORCHESTRATION_TOOL_NAMES - read_only
    ids = CreateTeam.spec.input_schema["properties"]["task_ids"]
    assert ids["type"] == "array" and ids["items"]["type"] == "string"


def test_default_registry_orchestration_flag() -> None:
    base = [s["name"] for s in default_registry().json_schemas()]
    assert not (set(base) & ORCHESTRATION_TOOL_NAMES)
    full = [s["name"] for s in default_registry(orchestration=True).json_schemas()]
    assert full[: len(base)] == base
    assert full[len(base) :] == [t.spec.name for t in ORCHESTRATION_TOOLS]


async def test_child_registry_excludes_orchestration_tools(tmp_path: Path) -> None:
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[MockProvider([])])
    assert ORCHESTRATION_TOOL_NAMES <= CHILD_EXCLUDED_TOOLS and "agent" in CHILD_EXCLUDED_TOOLS
    child = orch._factory.build_child(preset=resolve_subagent_type("general"), depth=1)
    names = {s["name"] for s in child.tools.json_schemas()}
    assert not (names & ORCHESTRATION_TOOL_NAMES) and "agent" not in names
    assert "write_file" in names
    assert child.tool_context.orchestrator is None


async def test_bash_cancel_kills_subprocess(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[asyncio.subprocess.Process] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args: object, **kwargs: object) -> asyncio.subprocess.Process:
        proc = await real(*args, **kwargs)  # type: ignore[arg-type]
        created.append(proc)
        return proc

    monkeypatch.setattr(bash_mod.asyncio, "create_subprocess_exec", spy)
    run = asyncio.create_task(Bash().run(Bash.Input(command="sleep 30"), tool_ctx))
    while not created:
        await asyncio.sleep(0.01)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert created[0].returncode is not None


async def test_bash_cancel_after_process_exit_keeps_cancelled() -> None:
    class _ExitedProc:
        returncode = 0
        waited = False

        def kill(self) -> None:
            raise ProcessLookupError

        async def wait(self) -> int:
            self.waited = True
            return 0

    proc = _ExitedProc()

    async def doomed() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await bash_mod._kill_and_reap(cast(asyncio.subprocess.Process, proc))
            raise

    run = asyncio.create_task(doomed())
    await asyncio.sleep(0)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert proc.waited


async def test_wait_returns_terminal_statuses(tmp_path: Path) -> None:
    child = MockProvider([TextDelta(text="done"), MessageStop()])
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("quick job")
    await orch.drain()
    statuses = await orch.wait([task.task_id], timeout_s=0.01)
    assert statuses == {task.task_id: TaskStatus.COMPLETED}
    with pytest.raises(OrchestrationError, match="unknown task id"):
        await orch.wait([task.task_id, "task-999"])


async def test_wait_blocks_until_completion_and_times_out(tmp_path: Path) -> None:
    gate = asyncio.Event()
    child = _GateProvider(gate)
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("slow job")
    await child.started.wait()
    statuses = await orch.wait([task.task_id], timeout_s=0.05)
    assert statuses[task.task_id] is TaskStatus.RUNNING

    async def _open() -> None:
        await asyncio.sleep(0.01)
        gate.set()

    opener = asyncio.create_task(_open())
    statuses = await orch.wait([task.task_id])
    assert statuses[task.task_id] is TaskStatus.COMPLETED
    await opener


async def test_children_are_explore_and_cannot_fan_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = MockProvider([TextDelta(text="found"), MessageStop()])
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    factory = orch._factory
    seen: dict[str, object] = {}
    original = factory.build_child

    def spy(*, preset: object, depth: int, model: str = "") -> object:
        runtime = original(preset=preset, depth=depth, model=model)  # type: ignore[arg-type]
        seen["preset"] = preset.key  # type: ignore[attr-defined]
        seen["tools"] = {s["name"] for s in runtime.tools.json_schemas()}
        return runtime

    monkeypatch.setattr(factory, "build_child", spy)
    task = await orch.create_task("explore job", preset="Explore", plan_id="plan-1")
    assert task.preset == "Explore" and task.plan_id == "plan-1"
    await orch.drain()
    assert seen["preset"] == "Explore"
    assert seen["tools"] == set(EXPLORE_TOOLS)
    assert not (set(EXPLORE_TOOLS) & CHILD_EXCLUDED_TOOLS)
    assert orch.get_task(task.task_id).status is TaskStatus.COMPLETED


async def test_task_records_usage_and_interval(tmp_path: Path) -> None:
    child = MockProvider(
        [TextDelta(text="bg"), Usage(input_tokens=11, output_tokens=7), MessageStop()]
    )
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("usage job")
    assert task.preset == "general-purpose" and task.plan_id is None
    assert task.started_at is None and task.finished_at is None
    await orch.drain()
    done = orch.get_task(task.task_id)
    assert done.usage.input_tokens == 11 and done.usage.output_tokens == 7
    assert done.started_at is not None and done.finished_at is not None
    assert done.started_at <= done.finished_at


async def test_stop_stamps_finished_at(tmp_path: Path) -> None:
    gate = asyncio.Event()
    child = _GateProvider(gate)
    _rt, orch, _ = _harness(MockProvider([]), tmp_path, children=[child])
    task = await orch.create_task("to stop")
    await child.started.wait()
    orch.stop_task(task.task_id)
    stopped = orch.get_task(task.task_id)
    assert stopped.status is TaskStatus.STOPPED and stopped.finished_at is not None
    await orch.aclose()
