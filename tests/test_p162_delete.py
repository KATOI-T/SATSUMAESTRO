"""削除の定型化のテスト。"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli.render import RichRenderer, action_preview, change_preview
from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, ToolUse
from satsumaestro.engine.reminders import ReminderState
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.deletion_guard import PROHIBITED_DELETION_PREFIX, check_bash, check_path
from satsumaestro.security.enforcer import PermissionMode, classify
from satsumaestro.security.policy import PermissionOutcome, PermissionPolicy, default_policy
from satsumaestro.session.models import ToolResultBlock, ToolUseBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.bash import benign_exit
from satsumaestro.tools.builtins.delete_path import DeletePath
from satsumaestro.util.errors import MaxIterationsError, PathEscapeError
from tests.fixtures.mock_provider import MockProvider

WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "build" / "sub").mkdir(parents=True)
    (root / "build" / "a.o").write_text("x", encoding="utf-8")
    (root / "build" / "sub" / "b.o").write_text("y", encoding="utf-8")
    (root / "empty").mkdir()
    (root / "note.txt").write_text("n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf ..",
        "rm -rf .",
        "rm -rf ~",
        "rm -rf $HOME",
        "cd .. && rm -rf project",
        "find .. -delete",
        "rm -rf /",
        "rm -rf /usr",
        "rmdir /tmp",
        "find / -name x -delete",
        "sudo rm -rf build",
        "sudo -u root rm note.txt",
        "doas rm note.txt",
        "rm -rf build ; rm -rf ..",
        "rm -rf .. > /dev/null 2>&1",
    ],
)
def test_prohibited_bash_deletions(ws: Path, command: str) -> None:
    reason = check_bash(command, ws)
    assert reason is not None and reason.startswith(PROHIBITED_DELETION_PREFIX)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        f"rm -rf {'{ws}'}/build",
        "rm note.txt",
        "find . -name '*.o' -delete",
        "ls -la | grep build",
        "echo rm -rf /",
        'rm -rf "$TARGET"',
    ],
)
def test_allowed_bash_commands(ws: Path, command: str) -> None:
    assert check_bash(command.replace("{ws}", str(ws)), ws) is None


def test_prohibited_paths_for_delete_tool(ws: Path) -> None:
    assert check_path("build", ws) is None
    for path in (".", "..", "/", "/usr", str(ws.parent)):
        reason = check_path(path, ws)
        assert reason is not None and reason.startswith(PROHIBITED_DELETION_PREFIX), path


@pytest.mark.skipif(os.geteuid() == 0, reason="root はどこでも消せる")
def test_needs_privilege(ws: Path) -> None:
    locked = ws / "locked"
    locked.mkdir()
    (locked / "f").write_text("x", encoding="utf-8")
    locked.chmod(0o555)
    try:
        reason = check_path("locked/f", ws)
        assert reason is not None and "sudo" in reason
    finally:
        locked.chmod(0o755)


class _CapturePrompter:
    def __init__(self, decision: str = "allow") -> None:
        self.calls: list[str] = []
        self._decision = decision

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.calls.append(call.name)
        return PermissionOutcome(decision=self._decision, reason="captured")  # type: ignore[arg-type]


def _runtime(ws: Path, call: ToolUse, policy: PermissionPolicy) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(ws, "mock-model"),
        provider=MockProvider([call, MessageStop()]),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=policy,
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), ws),
        config=RuntimeConfig(max_iterations=1),
    )


async def _run(rt: ConversationRuntime, prompter: object) -> ToolResultBlock:
    with pytest.raises(MaxIterationsError):
        await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    block = rt.session.messages[2].blocks[0]
    assert isinstance(block, ToolResultBlock)
    return block


@pytest.mark.parametrize(
    "call",
    [
        ToolUse(id="t1", name="bash", input={"command": "rm -rf .."}),
        ToolUse(id="t1", name="delete_path", input={"path": "..", "recursive": True}),
    ],
)
async def test_prohibited_is_refused_without_prompt_even_in_danger_mode(
    ws: Path, call: ToolUse
) -> None:
    prompter = _CapturePrompter(decision="allow")
    block = await _run(_runtime(ws, call, default_policy(DFA)), prompter)
    assert block.is_error is True
    assert block.output.startswith(PROHIBITED_DELETION_PREFIX)
    assert "Do not retry" in block.output
    assert prompter.calls == []
    assert ws.exists() and (ws / "note.txt").exists()


def _ctx(ws: Path) -> ToolContext:
    from satsumaestro.security.jail import WorkspaceJail

    jail = WorkspaceJail(ws)
    return ToolContext(workspace_root=jail.root, jail=jail, config=RuntimeConfig())


async def test_delete_file_and_directories(ws: Path) -> None:
    tool = DeletePath()
    ctx = _ctx(ws)
    res = await tool.run(DeletePath.Input(path="note.txt"), ctx)
    assert not res.is_error and not (ws / "note.txt").exists()
    res = await tool.run(DeletePath.Input(path="empty"), ctx)
    assert not res.is_error and not (ws / "empty").exists()
    res = await tool.run(DeletePath.Input(path="build"), ctx)
    assert res.is_error and "recursive=true" in res.output and "3 entries" in res.output
    assert (ws / "build").exists()
    res = await tool.run(DeletePath.Input(path="build", recursive=True), ctx)
    assert not res.is_error and not (ws / "build").exists()
    res = await tool.run(DeletePath.Input(path="missing"), ctx)
    assert res.is_error and "not found" in res.output


async def test_delete_symlink_removes_only_the_link(ws: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("k", encoding="utf-8")
    (ws / "link").symlink_to(outside)
    res = await DeletePath().run(DeletePath.Input(path="link", recursive=True), _ctx(ws))
    assert not res.is_error and not (ws / "link").exists()
    assert (outside / "keep").exists()


async def test_delete_outside_workspace_is_rejected(ws: Path, tmp_path: Path) -> None:
    (tmp_path / "other").mkdir()
    with pytest.raises(PathEscapeError):
        await DeletePath().run(DeletePath.Input(path="../other"), _ctx(ws))
    assert (tmp_path / "other").exists()


def test_delete_always_asks() -> None:
    assert classify("delete_path", DeletePath.Input(path="x"), DeletePath.spec.required_mode) is DFA
    assert default_policy(WW).authorize("delete_path", {"path": "x"}, DFA).decision == "ask"


async def test_runtime_prompts_then_deletes(ws: Path) -> None:
    call = ToolUse(id="t1", name="delete_path", input={"path": "note.txt"})
    prompter = _CapturePrompter(decision="allow")
    block = await _run(_runtime(ws, call, default_policy(WW)), prompter)
    assert prompter.calls == ["delete_path"]
    assert block.is_error is False and not (ws / "note.txt").exists()


def test_always_answer_is_not_remembered_for_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    from satsumaestro.cli.repl import InteractivePrompter

    p = InteractivePrompter(Console(file=io.StringIO(), force_terminal=False))
    answers = iter(["a", "y"])
    monkeypatch.setattr(p.console, "input", lambda *a, **k: next(answers))
    out = p.decide(ToolUse(id="1", name="delete_path", input={"path": "x"}), DFA)
    assert out.decision == "allow" and out.remember is False
    assert isinstance(p.console.file, io.StringIO)
    assert "この操作は永続許可できません" in p.console.file.getvalue()


def _plain(renderable: object) -> str:
    c = Console(file=io.StringIO(), force_terminal=False, width=100)
    c.print(renderable)
    assert isinstance(c.file, io.StringIO)
    return c.file.getvalue()


def test_delete_display_lists_contents(ws: Path) -> None:
    call = ToolUse(id="t1", name="delete_path", input={"path": "build", "recursive": True})
    preview = _plain(change_preview(call, ws))
    assert (
        "中身 3 件" in preview and "a.o" in preview and "sub/" in preview and "sub/b.o" in preview
    )
    card = _plain(action_preview(call, DFA.value))
    assert "build" in card and "中身ごと削除" in card and "元に戻せません" in card

    buf = io.StringIO()
    r = RichRenderer(
        console=Console(file=buf, force_terminal=True, width=100, color_system=None), cwd=ws
    )
    r.start_turn()
    r.on_tool_start(call)
    r.end_turn()
    out = buf.getvalue()
    assert "・削除" in out and "- build" in out and "delete_path" not in out


def _observe(state: ReminderState, output: str, tool: str = "bash") -> list[str]:
    call = ToolUseBlock(id="c", name=tool, input={"command": "rm -rf /x"})
    result = ToolResultBlock(tool_use_id="c", tool_name=tool, output=output, is_error=True)
    return state.collect([call], [result], new_session("/tmp", "m"))


def test_policy_deny_tells_model_not_to_retry() -> None:
    state = ReminderState()
    first = _observe(state, "deny rule: bash when command re: ^rm")
    assert len(first) == 1 and "do not retry" in first[0] and "delete_path" in first[0]
    second = _observe(state, "deny rule: bash when command re: ^rm")
    assert len(second) == 1 and "blocked again" in second[0]
    assert _observe(state, "deny rule: bash when command re: ^rm") == []


def test_user_deny_and_prohibited() -> None:
    state = ReminderState()
    notes = _observe(state, "denied by user")
    assert len(notes) == 1 and "user declined" in notes[0]
    state = ReminderState()
    assert _observe(state, f"{PROHIBITED_DELETION_PREFIX} x\nDo not retry") == []
    assert "blocked again" in _observe(state, f"{PROHIBITED_DELETION_PREFIX} x\nDo not retry")[0]


def test_denials_do_not_count_as_failure_streak() -> None:
    state = ReminderState()
    _observe(state, "deny rule: r")
    notes = _observe(state, "deny rule: r")
    assert not any("failed" in n for n in notes)


@pytest.mark.parametrize(
    ("command", "code", "benign"),
    [
        ("ls -la | grep aws", 1, True),
        ("grep x f", 1, True),
        ("rg x", 1, True),
        ("diff a b", 1, True),
        ("[ -f x ]", 1, True),
        ("grep x f 2>&1", 1, True),
        ("LC_ALL=C grep x f", 1, True),
        ("grep x f", 2, False),
        ("false && grep x f", 1, False),
        ("grep x f; echo", 1, False),
        ("make test", 1, False),
        ("grep x f | wc -l", 1, False),
    ],
)
def test_benign_exit(command: str, code: int, benign: bool) -> None:
    assert benign_exit(command, code) is benign
