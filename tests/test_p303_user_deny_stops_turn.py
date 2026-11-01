"""権限カードでユーザーが拒否したら、そのターンを止めて入力待ちに戻す。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.turn import USER_DENY_ORIGIN, DenyPrompter, PermissionOutcome
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import ToolResultBlock, ToolUseBlock
from satsumaestro.telemetry.log import TraceLog
from satsumaestro.telemetry.recorder import TraceRecorder
from tests.test_p145_plan_exit_nudge import _runtime
from tests.test_p165_plan_steps import _Approve, _exit_then, _write_plan
from tests.test_p165_plan_steps import _runtime as _step_runtime
from tests.test_p217_mkdir_hint import _bash, _outputs


class _User:
    """確認カードに回った bash を記録し、決めた答えを返すユーザー（REPL と同じ印を付ける）。"""

    def __init__(self, decision: str = "deny", *, origin: str | None = USER_DENY_ORIGIN) -> None:
        self.asked: list[str] = []
        self._decision = decision
        self._origin = origin

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.asked.append(str(call.input.get("command")))
        return PermissionOutcome(
            decision=self._decision,  # type: ignore[arg-type]
            reason="denied by user",
            origin=self._origin,
        )


class _ApproveThenDeny(_Approve):
    """計画は承認し、段の中の確認カードは拒否するユーザー。"""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.asked.append(str(call.input.get("command")))
        return PermissionOutcome(decision="deny", reason="denied by user", origin=USER_DENY_ORIGIN)


def _retrying(command: str, times: int) -> list[list[AssistantEvent]]:
    """拒否されても同じ呼び出しを ``times`` 回出し直し、最後に返答するモデル。"""
    calls = [[_bash(f"b{i}", command), MessageStop()] for i in range(times)]
    return [*calls, [TextDelta(text="完了しました"), MessageStop()]]


def _pairs(rt: object) -> tuple[list[str], list[str]]:
    uses = [
        b.id
        for m in rt.session.messages  # type: ignore[attr-defined]
        for b in m.blocks
        if isinstance(b, ToolUseBlock)
    ]
    results = [
        b.tool_use_id
        for m in rt.session.messages  # type: ignore[attr-defined]
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    ]
    return uses, results


async def test_user_deny_stops_the_turn_without_asking_the_model_again(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = _retrying("touch x", 3)
    user = _User()
    summary = await rt.run_turn("x を作って", user)  # type: ignore[arg-type]
    assert user.asked == ["touch x"]
    assert summary.stop_reason == "user_denied" and summary.final_text == ""
    assert summary.iterations == 1
    assert len(rt.provider.requests) == 1  # type: ignore[attr-defined]
    output = _outputs(rt)["b0"]
    assert output.splitlines()[0] == "denied by user"
    assert "the turn was stopped there" in output
    assert not (tmp_path / "x").exists()


async def test_the_rest_of_the_response_is_not_run(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [_bash("b1", "touch a"), _bash("b2", "touch b"), MessageStop()],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _User()
    summary = await rt.run_turn("a と b を作って", user)  # type: ignore[arg-type]
    assert user.asked == ["touch a"]
    assert summary.stop_reason == "user_denied"
    outputs = _outputs(rt)
    assert outputs["b1"].splitlines()[0] == "denied by user"
    assert outputs["b2"].startswith("not run: the user declined `bash`")
    uses, results = _pairs(rt)
    assert uses == results == ["b1", "b2"]
    assert not (tmp_path / "b").exists()


async def test_a_deny_without_the_user_mark_keeps_going(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = _retrying("touch x", 2)
    summary = await rt.run_turn("x を作って", DenyPrompter())
    assert summary.stop_reason == "end_turn" and summary.final_text == "完了しました"
    rt2, script2 = _runtime(tmp_path / "other")
    rt2.set_mode(main="code")
    script2[:] = _retrying("touch y", 2)
    user = _User(origin=None)
    summary2 = await rt2.run_turn("y を作って", user)  # type: ignore[arg-type]
    assert summary2.stop_reason == "end_turn" and user.asked == ["touch y", "touch y"]


async def test_an_allowed_call_does_not_stop_the_turn(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = _retrying("touch x", 1)
    summary = await rt.run_turn("x を作って", _User("allow"))  # type: ignore[arg-type]
    assert summary.stop_reason == "end_turn" and (tmp_path / "x").exists()


async def test_the_next_turn_runs_normally(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = _retrying("touch x", 1)
    await rt.run_turn("x を作って", _User())  # type: ignore[arg-type]
    script[:] = [[TextDelta(text="了解です"), MessageStop()]]
    summary = await rt.run_turn("やっぱりやめて", _User())  # type: ignore[arg-type]
    assert summary.stop_reason == "end_turn" and summary.final_text == "了解です"
    last = rt.provider.requests[-1].messages  # type: ignore[attr-defined]
    roles = [m.role for m in last]
    assert roles[-3:] == ["assistant", "tool", "user"]


async def test_a_deny_inside_a_plan_step_keeps_the_steps(tmp_path: Path) -> None:
    provider = _exit_then([_bash("s1", "touch out.txt"), MessageStop()])
    rt = _step_runtime(provider, tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    user = _ApproveThenDeny()
    summary = await rt.run_turn("作って", prompter=user)  # type: ignore[arg-type]
    assert summary.stop_reason == "user_denied" and user.asked == ["touch out.txt"]
    assert rt.session.plan_step_index == 1 and len(rt.session.plan_steps) == 3


async def test_the_telemetry_records_the_stop(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = _retrying("touch x", 2)
    log = TraceLog()
    recorder = TraceRecorder(rt.session.session_id, log)
    await rt.run_turn("x を作って", _User(), observer=recorder)  # type: ignore[arg-type]
    done = [e for e in log.entries if e.event == "turn.end"]
    assert [e.data["stop_reason"] for e in done] == ["user_denied"]
