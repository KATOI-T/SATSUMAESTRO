"""plan mode と TODO を system に置かない: system が動かないこと・会話の中での伝え方。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.plan_request import (
    PLAN_MODE_OFF_NOTE,
    PLAN_MODE_SUFFIX,
    PLAN_REQUEST_SUFFIX,
    plan_left_open,
)
from satsumaestro.engine.reminders import ReminderState, timeout_note
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import (
    ConversationMessage,
    TextBlock,
    TodoItem,
    ToolResultBlock,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider

_OFF_HEAD = PLAN_MODE_OFF_NOTE.split(":", 1)[0]
_PLAN_HEAD = PLAN_REQUEST_SUFFIX.split(":", 1)[0].strip()
_RULES_HEAD = PLAN_MODE_SUFFIX.split(":", 1)[0].strip()


def _runtime(tmp_path: Path, replies: int) -> tuple[ConversationRuntime, SequencedMockProvider]:
    provider = SequencedMockProvider(
        [[TextDelta(text="ok"), MessageStop()] for _ in range(replies)]
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,
        tools=default_registry(),
        policy=PermissionResolver(default_policy(PermissionMode.WORKSPACE_WRITE), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(guidance=GuidanceConfig(step_budget=False)),
    )
    return rt, provider


def _user_texts(rt: ConversationRuntime) -> list[str]:
    return [
        b.text
        for m in rt.session.messages
        if m.role == "user"
        for b in m.blocks
        if isinstance(b, TextBlock)
    ]


def _user(text: str) -> ConversationMessage:
    return ConversationMessage(role="user", blocks=[TextBlock(text=text)])


def _result(tool: str, output: str) -> ConversationMessage:
    return ConversationMessage(
        role="tool",
        blocks=[ToolResultBlock(tool_use_id="x", tool_name=tool, output=output)],
    )


def test_plan_left_open_reads_the_last_plan_signal() -> None:
    plan_turn = _user("作って" + PLAN_REQUEST_SUFFIX.format(plan_file="p.md", state="", steps=""))
    assert plan_left_open([]) is False
    assert plan_left_open([_user("こんにちは")]) is False
    assert plan_left_open([plan_turn]) is True
    assert plan_left_open([plan_turn, _result("exit_plan_mode", "Exited plan mode. ok")]) is False
    assert plan_left_open([plan_turn, _result("exit_plan_mode", "not approved")]) is True
    off = _user("消して\n\n<system-reminder>\n" + PLAN_MODE_OFF_NOTE.format(sub="x"))
    assert plan_left_open([plan_turn, off]) is False
    entered = _result("enter_plan_mode", "Entered plan mode. File writes ...")
    assert plan_left_open([_user("hi"), entered]) is True


async def test_plan_turn_carries_the_rules_and_the_system_stays_the_same(tmp_path: Path) -> None:
    rt, provider = _runtime(tmp_path, 3)
    await rt.run_turn("こんにちは")
    rt.set_mode(main="plan")
    await rt.run_turn("ダッシュボードを作って")
    await rt.run_turn("計画を立てて")
    texts = _user_texts(rt)
    assert _PLAN_HEAD not in texts[0]
    assert _PLAN_HEAD in texts[1] and rt.plan_file_relpath() in texts[1]
    assert _RULES_HEAD in texts[2] and "treat this as a request to plan first" not in texts[2]
    assert len({(r.system, r.system_dynamic) for r in provider.requests}) == 1
    assert "# Plan mode" not in provider.requests[1].system_dynamic


async def test_switching_back_to_code_says_plan_mode_ended_once(tmp_path: Path) -> None:
    rt, provider = _runtime(tmp_path, 3)
    rt.set_mode(main="plan")
    await rt.run_turn("ダッシュボードを作って")
    rt.set_mode(main="code")
    await rt.run_turn("やっぱりファイルを 1 つ消して")
    await rt.run_turn("ありがとう")
    texts = _user_texts(rt)
    assert _OFF_HEAD in texts[1]
    assert PermissionMode.WORKSPACE_WRITE.value in texts[1]
    assert _OFF_HEAD not in texts[2]
    assert len({(r.system, r.system_dynamic) for r in provider.requests}) == 1


async def test_prompt_history_keeps_the_users_words(tmp_path: Path) -> None:
    rt, _ = _runtime(tmp_path, 2)
    rt.set_mode(main="plan")
    await rt.run_turn("ダッシュボードを作って")
    rt.set_mode(main="code")
    await rt.run_turn("消して")
    assert rt.session.prompt_history == ["ダッシュボードを作って", "消して"]


def test_post_compaction_note_carries_the_current_todos(tmp_path: Path) -> None:
    session = new_session(tmp_path, "m")
    session.todos = [
        TodoItem(content="A を作る", activeForm="A を作成中", status="in_progress"),
        TodoItem(content="B を直す", activeForm="B を修正中", status="pending"),
    ]
    state = ReminderState()
    state.note_compaction()
    notes = state.collect([], [], session)
    assert len(notes) == 1
    assert "Current TODO list:" in notes[0]
    assert "A を作る" in notes[0] and "B を直す" in notes[0]


def test_post_compaction_note_without_todos_has_no_list(tmp_path: Path) -> None:
    state = ReminderState()
    state.note_compaction()
    notes = state.collect([], [], new_session(tmp_path, "m"))
    assert len(notes) == 1
    assert "Current TODO list" not in notes[0]


def test_timeout_note_does_not_blame_the_tool_call_alone() -> None:
    """読み直しだけで 120 秒を使い切ることもあるので、原因を 1 つに決めつけない。"""
    note = timeout_note(120.0)
    assert note.startswith("Your previous response was cancelled: nothing arrived for 120 seconds")
    assert "too much text" in note
    assert "re-reading a long conversation" in note
    assert "almost always" not in note
    assert "append_file" in note and "simply continue" in note
