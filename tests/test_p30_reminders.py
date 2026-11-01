"""状態トリガ式 system-reminder のテスト。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.reminders import (
    PLAN_MODE_DENY_PREFIX,
    ReminderState,
    format_reminder,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import (
    ModeState,
    Session,
    TodoItem,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider


def _session(tmp_path: Path) -> Session:
    return new_session(tmp_path, "mock-model")


def _call(name: str, cid: str = "c1") -> ToolUseBlock:
    return ToolUseBlock(id=cid, name=name, input={})


def _result(
    name: str, cid: str = "c1", *, error: bool = False, output: str = "x"
) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=cid, tool_name=name, output=output, is_error=error)


def test_post_compaction_fires_once(tmp_path: Path) -> None:
    state = ReminderState()
    session = _session(tmp_path)
    state.note_compaction()
    notes = state.collect([_call("read_file")], [_result("read_file")], session)
    assert len(notes) == 1 and "compacted" in notes[0]
    assert state.collect([_call("read_file")], [_result("read_file")], session) == []
    state.note_compaction()
    assert len(state.collect([_call("read_file")], [_result("read_file")], session)) == 1


def test_failure_streak_fires_and_clears(tmp_path: Path) -> None:
    state = ReminderState()
    session = _session(tmp_path)
    fail = [_result("grep_search", error=True)]
    assert state.collect([_call("grep_search")], fail, session) == []
    notes = state.collect([_call("grep_search")], fail, session)
    assert len(notes) == 1 and "grep_search" in notes[0]
    assert state.collect([_call("grep_search")], fail, session) == []
    assert state.collect([_call("grep_search")], [_result("grep_search")], session) == []
    state.collect([_call("grep_search")], fail, session)
    assert len(state.collect([_call("grep_search")], fail, session)) == 1


def test_plan_mode_trigger_requires_denial(tmp_path: Path) -> None:
    state = ReminderState()
    session = _session(tmp_path)
    session.mode = ModeState(main="plan")
    deny = [_result("write_file", error=True, output=f"{PLAN_MODE_DENY_PREFIX} denied")]
    ok = [_result("read_file")]
    assert state.collect([_call("read_file")], ok, session) == []
    assert len(state.collect([_call("write_file")], deny, session)) == 1
    assert state.collect([_call("write_file")], deny, session) == []
    session.mode = ModeState(main="code")
    assert state.collect([_call("read_file")], ok, session) == []
    session.mode = ModeState(main="plan")
    assert len(state.collect([_call("write_file")], deny, session)) == 1


def test_todo_stale_counts_iterations(tmp_path: Path) -> None:
    state = ReminderState(todo_stale_after=2)
    session = _session(tmp_path)
    session.todos = [TodoItem(content="task", active_form="doing", status="in_progress")]
    ok = [_result("read_file")]
    assert state.collect([_call("read_file")], ok, session) == []
    notes = state.collect([_call("read_file")], ok, session)
    assert len(notes) == 1 and "todo_write" in notes[0]
    assert state.collect([_call("read_file")], ok, session) == []
    assert state.collect([_call("todo_write")], [_result("todo_write")], session) == []
    assert state.collect([_call("read_file")], ok, session) == []
    assert len(state.collect([_call("read_file")], ok, session)) == 1
    done = ReminderState(todo_stale_after=1)
    session.todos = [TodoItem(content="task", active_form="doing", status="completed")]
    assert done.collect([_call("read_file")], ok, session) == []


def test_format_reminder_shape() -> None:
    text = format_reminder(["a", "b"])
    assert text.startswith("<system-reminder>\n") and text.endswith("\n</system-reminder>")
    assert "a\nb" in text


def _runtime(provider: object, tmp_path: Path, cfg: RuntimeConfig) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg,
    )


def _streak_provider() -> SequencedMockProvider:
    return SequencedMockProvider(
        [
            [
                ToolUse(id="c1", name="read_file", input={}),
                ToolUse(id="c2", name="read_file", input={}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )


async def test_runtime_injects_into_last_tool_result(tmp_path: Path) -> None:
    rt = _runtime(_streak_provider(), tmp_path, RuntimeConfig())
    summary = await rt.run_turn("hi")
    assert summary.final_text == "done"
    assert summary.reminders == 1
    tool_msgs = [m for m in rt.session.messages if m.role == "tool"]
    blocks = tool_msgs[0].blocks
    assert isinstance(blocks[0], ToolResultBlock) and isinstance(blocks[1], ToolResultBlock)
    assert "<system-reminder>" not in blocks[0].output
    assert "</system-reminder>" in blocks[1].output and "read_file" in blocks[1].output
    assert blocks[1].tool_use_id == "c2"


async def test_runtime_reminders_disabled(tmp_path: Path) -> None:
    cfg = RuntimeConfig(guidance=GuidanceConfig(reminders=False))
    rt = _runtime(_streak_provider(), tmp_path, cfg)
    summary = await rt.run_turn("hi")
    assert summary.reminders == 0
    tool_msgs = [m for m in rt.session.messages if m.role == "tool"]
    assert all("<system-reminder>" not in b.output for m in tool_msgs for b in m.blocks)  # type: ignore[union-attr]


async def test_injected_session_roundtrips(tmp_path: Path) -> None:
    rt = _runtime(_streak_provider(), tmp_path, RuntimeConfig())
    await rt.run_turn("hi")
    restored = Session.model_validate_json(rt.session.model_dump_json())
    assert restored.messages == rt.session.messages


async def test_prompt_mentions_reminder_tag_only_when_enabled(tmp_path: Path) -> None:
    on = (await SystemPromptBuilder().build(tmp_path, "m", RuntimeConfig(), None)).joined()
    assert "<system-reminder>" in on
    off_cfg = RuntimeConfig(guidance=GuidanceConfig(reminders=False))
    off = (await SystemPromptBuilder().build(tmp_path, "m", off_cfg, None)).joined()
    assert "<system-reminder>" not in off


async def test_empty_final_retries_once_and_succeeds(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [MessageStop()],
            [TextDelta(text="the answer is 42"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("q")
    assert summary.final_text == "the answer is 42"
    assert summary.iterations == 2
    assert summary.reminders == 1
    injected = [
        m
        for m in rt.session.messages
        if m.role == "user" and "<system-reminder>" in (m.blocks[0].text or "")  # type: ignore[union-attr]
    ]
    assert len(injected) == 1
    assert "final answer" in injected[0].blocks[0].text  # type: ignore[union-attr]


async def test_empty_final_gives_up_after_one_retry(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[MessageStop()], [MessageStop()]])
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("q")
    assert summary.final_text == ""
    assert summary.iterations == 2
    assert summary.reminders == 1


async def test_empty_final_retry_disabled(tmp_path: Path) -> None:
    cfg = RuntimeConfig(guidance=GuidanceConfig(empty_final_retry=False))
    provider = SequencedMockProvider([[MessageStop()]])
    rt = _runtime(provider, tmp_path, cfg)
    summary = await rt.run_turn("q")
    assert summary.final_text == ""
    assert summary.iterations == 1
    assert summary.reminders == 0


async def test_empty_final_respects_max_iterations(tmp_path: Path) -> None:
    cfg = RuntimeConfig(max_iterations=1)
    provider = SequencedMockProvider([[MessageStop()]])
    rt = _runtime(provider, tmp_path, cfg)
    summary = await rt.run_turn("q")
    assert summary.final_text == ""
    assert summary.iterations == 1


async def test_empty_final_does_not_fire_on_normal_answer(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[TextDelta(text="done"), MessageStop()]])
    rt = _runtime(provider, tmp_path, RuntimeConfig())
    summary = await rt.run_turn("q")
    assert summary.final_text == "done"
    assert summary.iterations == 1
    assert summary.reminders == 0


def test_failure_streak_latches_per_tool(tmp_path: Path) -> None:
    state = ReminderState()
    session = _session(tmp_path)
    fail_a = [_result("grep_search", error=True)]
    fail_ab = [_result("grep_search", error=True), _result("read_file", cid="c2", error=True)]
    state.collect([_call("grep_search")], fail_a, session)
    notes = state.collect([_call("grep_search")], fail_a, session)
    assert len(notes) == 1 and "grep_search" in notes[0]
    state.collect([_call("grep_search"), _call("read_file", "c2")], fail_ab, session)
    notes = state.collect([_call("grep_search"), _call("read_file", "c2")], fail_ab, session)
    assert len(notes) == 1 and "read_file" in notes[0]
