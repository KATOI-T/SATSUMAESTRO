"""エージェントの状態ディレクトリ（.satsuma）への書き込みの絶対禁止。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.turn import PermissionOutcome
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.state_write_guard import (
    PROHIBITED_STATE_WRITE_PREFIX,
    check_bash,
    check_tool,
)
from satsumaestro.session.models import ToolResultBlock, plan_file_relpath
from tests.test_p145_plan_exit_nudge import _runtime


class _Capture:
    """確認カードに回った呼び出しを記録して、決めた答えを返すユーザー。"""

    def __init__(self, decision: str = "allow") -> None:
        self.asked: list[str] = []
        self._decision = decision

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.asked.append(call.name)
        return PermissionOutcome(decision=self._decision, reason="denied by user")  # type: ignore[arg-type]


def _outputs(rt: object) -> dict[str, ToolResultBlock]:
    return {
        b.tool_use_id: b
        for m in rt.session.messages  # type: ignore[attr-defined]
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("SATSUMA_CONFIG_HOME", raising=False)
    (tmp_path / ".satsuma" / "plans").mkdir(parents=True)
    (tmp_path / ".satsuma" / "policy.md").write_text("mode: workspace-write\n")
    return tmp_path


@pytest.mark.parametrize(
    "command",
    [
        "echo 'mode: allow' > .satsuma/policy.md",
        "echo x >> .satsuma/memory.md",
        "sed -i '' 's/workspace-write/allow/' .satsuma/policy.md",
        "cp /tmp/x .satsuma/settings.local.json",
        "mv notes.md .satsuma/memory.md",
        "tee -a .satsuma/memory.md",
        "dd if=/dev/zero of=.satsuma/rag.db",
        "sort -o .satsuma/policy.md .satsuma/policy.md",
        "cd .satsuma && echo x > policy.md",
        "touch .satsuma/sessions/a.jsonl",
        "echo '{}' > ~/.satsuma/settings.json",
        "mkdir -p .satsuma/new-dir",
    ],
)
def test_bash_writes_into_state_are_prohibited(ws: Path, command: str) -> None:
    reason = check_bash(command, ws)
    assert reason is not None and reason.startswith(PROHIBITED_STATE_WRITE_PREFIX), command


@pytest.mark.parametrize(
    "command",
    [
        "cat .satsuma/policy.md",
        "ls -la .satsuma",
        "grep -rn mode .satsuma",
        "find .satsuma -name '*.md'",
        "cat ~/.satsuma/logs/satsumaestro.log",
        "echo x > out.txt",
        "mv a.txt b.txt",
        "git status",
        "mkdir -p .satsuma/plans",
    ],
)
def test_reads_and_other_writes_are_left_alone(ws: Path, command: str) -> None:
    assert check_bash(command, ws) is None, command


def test_the_current_plan_file_is_allowed(ws: Path) -> None:
    plan = ws / ".satsuma" / "plans" / "abc.md"
    assert check_bash("echo x > .satsuma/plans/abc.md", ws, visible=[plan]) is None
    assert check_tool("write_file", ".satsuma/plans/abc.md", ws, visible=[plan]) is None
    assert check_tool("write_file", ".satsuma/plans/other.md", ws, visible=[plan]) is not None


def test_write_tools(ws: Path) -> None:
    for name in ("write_file", "edit_file", "append_file", "notebook_edit"):
        assert check_tool(name, ".satsuma/policy.md", ws) is not None, name
    assert check_tool("write_file", "src/app.py", ws) is None
    assert check_tool("read_file", ".satsuma/policy.md", ws) is None


async def test_bash_rewrite_of_the_policy_is_refused_even_in_full_access(tmp_path: Path) -> None:
    """カードが出ない danger-full-access でも、カードの前で拒否してファイルは変わらない。"""
    ws = tmp_path
    rt, script = _runtime(ws)
    (ws / ".satsuma" / "policy.md").write_text("mode: workspace-write\n")
    rt.set_mode(main="code", sub="danger-full-access")
    script[:] = [
        [
            ToolUse(
                id="b1", name="bash", input={"command": "echo 'mode: allow' > .satsuma/policy.md"}
            ),
            MessageStop(),
        ],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture()
    summary = await rt.run_turn("設定を変えて", user)  # type: ignore[arg-type]
    assert user.asked == []
    assert _outputs(rt)["b1"].output.startswith(PROHIBITED_STATE_WRITE_PREFIX)
    assert summary.state_write_blocks == 1
    assert (ws / ".satsuma" / "policy.md").read_text() == "mode: workspace-write\n"


async def test_write_tool_is_refused_before_the_card(tmp_path: Path) -> None:
    """read-only で書き込みがカードに回る設定でも、カードを出さずに拒否する。"""
    ws = tmp_path
    rt, script = _runtime(ws)
    (ws / ".satsuma" / "policy.md").write_text("mode: workspace-write\n")
    rt.set_mode(main="code", sub="read-only")
    script[:] = [
        [
            ToolUse(
                id="w1",
                name="write_file",
                input={"path": ".satsuma/settings.local.json", "content": "{}"},
            ),
            MessageStop(),
        ],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture()
    await rt.run_turn("書いて", user)  # type: ignore[arg-type]
    assert user.asked == []
    assert _outputs(rt)["w1"].output.startswith(PROHIBITED_STATE_WRITE_PREFIX)
    assert not (ws / ".satsuma" / "settings.local.json").exists()


async def test_plan_file_is_still_writable_in_plan_mode(tmp_path: Path) -> None:
    ws = tmp_path
    rt, script = _runtime(ws)
    (ws / ".satsuma" / "policy.md").write_text("mode: workspace-write\n")
    plan = plan_file_relpath(rt.session.session_id)
    script[:] = [
        [
            ToolUse(id="p1", name="write_file", input={"path": plan, "content": "# Plan\n"}),
            MessageStop(),
        ],
        [TextDelta(text="ok"), MessageStop()],
    ]
    summary = await rt.run_turn("計画して", _Capture())  # type: ignore[arg-type]
    assert _outputs(rt)["p1"].is_error is False
    assert summary.state_write_blocks == 0
    assert (ws / plan).read_text() == "# Plan\n"
