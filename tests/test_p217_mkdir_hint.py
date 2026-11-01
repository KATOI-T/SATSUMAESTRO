"""書く前の mkdir を、確認カードに回す前に「要らない」と返す。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.turn import PermissionOutcome
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.tools.mkdir_hint import MKDIR_NOT_NEEDED, pure_mkdir_targets
from tests.test_p145_plan_exit_nudge import _runtime


class _Capture:
    """確認カードに回った bash を記録して、決めた答えを返すユーザー。"""

    def __init__(self, decision: str = "deny") -> None:
        self.asked: list[str] = []
        self._decision = decision

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.asked.append(str(call.input.get("command")))
        return PermissionOutcome(decision=self._decision, reason="denied by user")  # type: ignore[arg-type]


def _bash(call_id: str, command: str) -> ToolUse:
    return ToolUse(id=call_id, name="bash", input={"command": command})


def _outputs(rt: object) -> dict[str, str]:
    return {
        b.tool_use_id: b.output
        for m in rt.session.messages  # type: ignore[attr-defined]
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }


def test_pure_mkdir_inside_the_workspace(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    assert pure_mkdir_targets("mkdir -p dashboards", tmp_path) == [root / "dashboards"]
    assert pure_mkdir_targets("mkdir a b/c", tmp_path) == [root / "a", root / "b" / "c"]
    assert pure_mkdir_targets(f"mkdir -p {tmp_path}/web", tmp_path) == [root / "web"]
    assert pure_mkdir_targets(f"cd {tmp_path} && mkdir -p src/x", tmp_path) == [root / "src" / "x"]


def test_anything_else_is_left_to_the_normal_path(tmp_path: Path) -> None:
    (tmp_path / "out").symlink_to("/tmp")
    for command in (
        "mkdir -p dashboards && touch dashboards/a",
        "mkdir -m 700 secret",
        "mkdir -p ../outside",
        "mkdir -p /tmp/elsewhere",
        "mkdir -p out/x",
        "mkdir -p $HOME/x",
        "mkdir -p .",
        "mkdir",
        "ls dashboards",
    ):
        assert pure_mkdir_targets(command, tmp_path) is None, command


async def test_mkdir_before_writing_is_answered_without_a_card(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [_bash("m1", "mkdir -p dashboards"), MessageStop()],
        [
            ToolUse(
                id="w1",
                name="write_file",
                input={"path": "dashboards/index.html", "content": "<html></html>\n"},
            ),
            MessageStop(),
        ],
        [TextDelta(text="作成しました。"), MessageStop()],
    ]
    user = _Capture()
    summary = await rt.run_turn("作って", user)  # type: ignore[arg-type]
    assert user.asked == []
    assert _outputs(rt)["m1"] == MKDIR_NOT_NEEDED.format(example="dashboards/<file>")
    assert summary.mkdir_skips == 1
    assert (tmp_path / "dashboards" / "index.html").exists()


async def test_second_mkdir_of_the_same_place_goes_to_the_card(tmp_path: Path) -> None:
    """本当に空のディレクトリが要るなら、2 回目は従来どおりユーザーに確認する。"""
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [_bash("m1", "mkdir -p build"), MessageStop()],
        [_bash("m2", "mkdir -p build"), MessageStop()],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture(decision="allow")
    summary = await rt.run_turn("build を作って", user)  # type: ignore[arg-type]
    assert user.asked == ["mkdir -p build"]
    assert summary.mkdir_skips == 1
    assert (tmp_path / "build").is_dir()


async def test_other_commands_still_ask(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [_bash("m1", "mkdir -p out && touch out/a"), MessageStop()],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture()
    summary = await rt.run_turn("作って", user)  # type: ignore[arg-type]
    assert user.asked == ["mkdir -p out && touch out/a"]
    assert summary.mkdir_skips == 0


async def test_flag_off_keeps_the_card(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    rt.config.tools.mkdir_hint = False
    script[:] = [
        [_bash("m1", "mkdir -p dashboards"), MessageStop()],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture()
    await rt.run_turn("作って", user)  # type: ignore[arg-type]
    assert user.asked == ["mkdir -p dashboards"]
