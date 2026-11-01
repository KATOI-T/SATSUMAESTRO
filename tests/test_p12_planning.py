"""プランニング・ステートフル TODO / plan mode のテスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter, PlanExitVerdict
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import (
    PermissionPolicy,
    PermissionRule,
    RuleMatch,
    default_policy,
)
from satsumaestro.session.models import ModeState, Session, TodoItem
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.plan_mode import EnterPlanMode, ExitPlanMode
from satsumaestro.tools.builtins.todo_write import TodoWrite
from tests.fixtures.mock_provider import SequencedMockProvider


def _todo(
    content: str,
    status: Literal["pending", "in_progress", "completed"] = "pending",
    active_form: str | None = None,
) -> TodoItem:
    return TodoItem(content=content, active_form=active_form or f"Doing {content}", status=status)


def _session_ctx(tmp_path: Path) -> ToolContext:
    jail = WorkspaceJail(tmp_path)
    return ToolContext(
        workspace_root=jail.root,
        jail=jail,
        config=RuntimeConfig(),
        session=new_session(tmp_path, "mock-model"),
    )


def _make_runtime(
    provider: object,
    tmp_path: Path,
    *,
    policy: PermissionPolicy | None = None,
    session: Session | None = None,
    hooks: HookRunner | None = None,
) -> ConversationRuntime:
    return ConversationRuntime(
        session=session or new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(planning_guidance=True),
        policy=policy or default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=hooks or HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


async def test_todo_write_persists_to_session(tmp_path: Path) -> None:
    ctx = _session_ctx(tmp_path)
    todos = [_todo("A", "completed"), _todo("B", "in_progress"), _todo("C")]
    result = await TodoWrite().run(TodoWrite.Input(todos=todos), ctx)
    assert result.is_error is False
    assert result.output.startswith("[x] A\n[~] B\n[ ] C")
    assert ctx.session is not None
    assert [t.content for t in ctx.session.todos] == ["A", "B", "C"]
    assert [t.status for t in ctx.session.todos] == ["completed", "in_progress", "pending"]


async def test_todo_write_clears_when_all_completed(tmp_path: Path) -> None:
    ctx = _session_ctx(tmp_path)
    assert ctx.session is not None
    ctx.session.todos = [_todo("A", "in_progress")]
    result = await TodoWrite().run(
        TodoWrite.Input(todos=[_todo("A", "completed"), _todo("B", "completed")]), ctx
    )
    assert result.is_error is False
    assert ctx.session.todos == []


async def test_todo_write_validation(tmp_path: Path) -> None:
    ctx = _session_ctx(tmp_path)
    empty = await TodoWrite().run(TodoWrite.Input(todos=[]), ctx)
    assert empty.is_error is True
    blank_content = await TodoWrite().run(
        TodoWrite.Input(todos=[TodoItem(content="  ", active_form="Doing")]), ctx
    )
    assert blank_content.is_error is True
    blank_form = await TodoWrite().run(
        TodoWrite.Input(todos=[TodoItem(content="A", active_form=" ")]), ctx
    )
    assert blank_form.is_error is False
    missing_form = await TodoWrite().run(
        TodoWrite.Input(todos=[TodoItem.model_validate({"content": "A"})]), ctx
    )
    assert missing_form.is_error is False and "[ ] A" in missing_form.output
    multi = await TodoWrite().run(
        TodoWrite.Input(todos=[_todo("A", "in_progress"), _todo("B", "in_progress")]), ctx
    )
    assert multi.is_error is False


async def test_todo_check_note(tmp_path: Path) -> None:
    ctx = _session_ctx(tmp_path)
    done2 = [_todo("Write the parser", "completed"), _todo("Wire the CLI", "completed")]
    done3 = [*done2, _todo("Add a help text", "completed")]
    noted = await TodoWrite().run(TodoWrite.Input(todos=done3), ctx)
    assert noted.output.splitlines()[-1].startswith("Every item is completed, but none of them")
    assert "with bash" in noted.output
    for check in (
        "Verify with tests",
        "Run make check",
        "Add tests",
        "動作確認",
        "ブラウザでテスト",
    ):
        plain = await TodoWrite().run(
            TodoWrite.Input(todos=[*done3, _todo(check, "completed")]), ctx
        )
        assert plain.output.count("\n") == 3, check
    latest = await TodoWrite().run(
        TodoWrite.Input(todos=[*done2, _todo("Use the latest API", "completed")]), ctx
    )
    assert "none of them confirms" in latest.output
    two = await TodoWrite().run(TodoWrite.Input(todos=done2), ctx)
    assert two.output == "[x] Write the parser\n[x] Wire the CLI"
    open_ = await TodoWrite().run(
        TodoWrite.Input(todos=[_todo("A", "completed"), _todo("B", "in_progress")]), ctx
    )
    assert open_.output == "[x] A\n[~] B"


def test_todo_roundtrip_store(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    session = new_session(tmp_path, "mock-model")
    session.todos = [_todo("A", "in_progress"), _todo("B")]
    store.save(session)
    loaded = store.load(store.path_for(tmp_path, session.session_id))
    assert loaded.todos == session.todos


async def test_enter_plan_mode_sets_state(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="p1", name="enter_plan_mode", input={}), MessageStop()],
            [TextDelta(text="planned"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path)
    await rt.run_turn("plan it")
    state = rt.session.mode
    assert state.main == "plan"
    assert state.sub == PermissionMode.WORKSPACE_WRITE


class _ApprovePrompter(DenyPrompter):
    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        return PlanExitVerdict(approved=True)


async def test_plan_mode_denies_write_and_bash(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("body", encoding="utf-8")
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="plan", sub=PermissionMode.WORKSPACE_WRITE)
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="w1", name="write_file", input={"path": "out.txt", "content": "x"}),
                ToolUse(
                    id="e1", name="edit_file", input={"path": "src.txt", "old": "b", "new": "c"}
                ),
                ToolUse(id="b1", name="bash", input={"command": "echo hi"}),
                ToolUse(id="r1", name="read_file", input={"path": "src.txt"}),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path, session=session)
    await rt.run_turn("try tools")
    blocks = rt.session.messages[2].blocks
    by_id = {b.tool_use_id: b for b in blocks}  # type: ignore[union-attr]
    for tid in ("w1", "e1", "b1"):
        assert by_id[tid].is_error is True
        assert "plan mode" in by_id[tid].output
    assert by_id["r1"].is_error is False
    assert by_id["r1"].output.startswith("body")
    assert "<system-reminder>" in by_id["r1"].output
    assert not (tmp_path / "out.txt").exists()
    assert (tmp_path / "src.txt").read_text(encoding="utf-8") == "body"


async def test_exit_plan_mode_restores_mode(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="p1", name="enter_plan_mode", input={}), MessageStop()],
            [ToolUse(id="p2", name="exit_plan_mode", input={}), MessageStop()],
            [TextDelta(text="executing"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path)
    plan_path = tmp_path / rt.plan_file_relpath()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("# Plan\n\n## Steps\n1. a (~0 lines)\n", encoding="utf-8")
    await rt.run_turn("plan then execute", prompter=_ApprovePrompter())
    assert rt.session.mode.main == "code"
    assert rt._base_mode() == PermissionMode.WORKSPACE_WRITE


async def test_plan_mode_rolls_back_when_post_hook_errors(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="p1", name="enter_plan_mode", input={}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    hooks = HookRunner(HookConfig(post_tool_use=["echo '{\"is_error\": true}'"]), tmp_path)
    rt = _make_runtime(provider, tmp_path, hooks=hooks)
    await rt.run_turn("plan it")
    block = rt.session.messages[2].blocks[0]
    assert block.is_error is True  # type: ignore[union-attr]
    assert rt.session.mode.main == "code"


def test_plan_mode_roundtrip_store(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="plan", sub=PermissionMode.WORKSPACE_WRITE)
    store.save(session)
    loaded = store.load(store.path_for(tmp_path, session.session_id))
    assert loaded.mode == session.mode


async def test_hard_deny_absolute_under_plan_mode(tmp_path: Path) -> None:
    policy = default_policy(PermissionMode.WORKSPACE_WRITE)
    policy.deny.append(PermissionRule(match=RuleMatch(tool="read_file"), label="no reads"))
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="plan", sub=PermissionMode.WORKSPACE_WRITE)
    provider = SequencedMockProvider(
        [
            [ToolUse(id="r1", name="read_file", input={"path": "x.txt"}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path, policy=policy, session=session)
    await rt.run_turn("read it")
    block = rt.session.messages[2].blocks[0]
    assert block.is_error is True  # type: ignore[union-attr]
    assert "deny rule: no reads" in block.output  # type: ignore[union-attr]


async def test_plan_mode_tools_idempotent(tmp_path: Path) -> None:
    ctx = _session_ctx(tmp_path)
    assert ctx.session is not None
    first = await EnterPlanMode().run(EnterPlanMode.Input(), ctx)
    assert first.is_error is False
    assert ctx.session.mode.main == "plan"
    assert ".satsuma/plans/" in first.output
    again = await EnterPlanMode().run(EnterPlanMode.Input(), ctx)
    assert "already active" in again.output
    left = await ExitPlanMode().run(ExitPlanMode.Input(), ctx)
    assert left.is_error is False
    assert ctx.session.mode.main == "code"
    noop = await ExitPlanMode().run(ExitPlanMode.Input(), ctx)
    assert "not active" in noop.output


async def test_todos_and_plan_mode_stay_out_of_the_system_prompt(tmp_path: Path) -> None:
    """TODO と plan mode は system に置かない（変わるたびに会話全体が読み直しになるため）。"""
    session = new_session(tmp_path, "mock-model")
    session.todos = [_todo("A", "in_progress"), _todo("B")]
    session.mode = ModeState(main="plan")
    joined = (
        await SystemPromptBuilder().build(tmp_path, "claude-opus-4-8", RuntimeConfig())
    ).joined()
    assert "# Current TODOs" not in joined
    assert "# Plan mode" not in joined
    assert "[~] A" not in joined
