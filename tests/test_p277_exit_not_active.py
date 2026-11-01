"""plan でないときの exit_plan_mode をエラーで返し、何をすべきかを伝える。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.reminders import empty_final_approved_note, empty_final_note
from satsumaestro.engine.runtime import PLAN_EXIT_LOOP_LIMIT
from satsumaestro.tools.builtins.plan_mode import (
    FLAG_EXIT_NOT_ACTIVE,
    FLAG_EXIT_STEP_DONE,
    PLAN_EXIT_NOT_ACTIVE,
    ExitPlanMode,
)
from satsumaestro.util.errors import PlanExitLoopError
from tests.test_p12_planning import _session_ctx
from tests.test_p36_mode_state import _blocks
from tests.test_p145_plan_exit_nudge import _Approve, _reminders, _runtime


def _exit(call_id: str) -> ToolUse:
    return ToolUse(id=call_id, name="exit_plan_mode", input={})


async def test_exit_outside_plan_is_an_error_with_guidance(tmp_path: Path) -> None:
    ctx = _session_ctx(tmp_path)
    out = await ExitPlanMode().run(ExitPlanMode.Input(), ctx)
    assert out.is_error is True
    assert out.output == PLAN_EXIT_NOT_ACTIVE
    assert out.flags == [FLAG_EXIT_NOT_ACTIVE]


async def test_exit_during_steps_is_recorded_as_step_done(tmp_path: Path) -> None:
    """段階実行の途中の exit は、エラーではなく「段の完了」として返す。"""
    ctx = _session_ctx(tmp_path)
    assert ctx.session is not None
    ctx.session.plan_steps = ["write a.html (~40 lines)", "write a.css (~40 lines)", "check"]
    ctx.session.plan_step_index = 2
    out = await ExitPlanMode().run(ExitPlanMode.Input(), ctx)
    assert out.is_error is False
    assert out.output.startswith("Plan step recorded: step 2 of 3 is done.")
    assert FLAG_EXIT_STEP_DONE in out.flags


async def test_repeated_exit_outside_plan_trips_the_failure_streak(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [_exit("x1"), MessageStop()],
        [_exit("x2"), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    summary = await rt.run_turn("続けて", _Approve())
    assert summary.plan_exit_not_active == 2
    assert "has failed 2 times in a row" in _blocks(rt)["x2"].output
    assert summary.final_text == "done"


async def test_exit_outside_plan_does_not_count_as_approval(tmp_path: Path) -> None:
    """成功扱いだったころは「承認された」と数え、空の返答に承認後の文を出していた。"""
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [_exit("x1"), MessageStop()],
        [MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    await rt.run_turn("hi", _Approve())
    notes = _reminders(rt)
    assert any(empty_final_note() in n for n in notes)
    assert not any(empty_final_approved_note() in n for n in notes)


async def test_three_noop_exits_in_a_row_end_the_turn(tmp_path: Path) -> None:
    """案内を受けても続くループは、反復の上限まで空回りさせずに 3 回で打ち切る。"""
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [TextDelta(text="段 4 完了"), _exit("x1"), MessageStop()],
        [_exit("x2"), MessageStop()],
        [_exit("x3"), MessageStop()],
        [_exit("x4"), MessageStop()],
    ]
    with pytest.raises(PlanExitLoopError) as info:
        await rt.run_turn("続けて", _Approve())
    assert info.value.calls == PLAN_EXIT_LOOP_LIMIT == 3
    assert info.value.last_text == "段 4 完了"
    assert "x4" not in _blocks(rt)
    assert "x3" in _blocks(rt)


async def test_other_tool_calls_reset_the_loop_count(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [_exit("x1"), MessageStop()],
        [_exit("x2"), MessageStop()],
        [ToolUse(id="r1", name="glob_search", input={"pattern": "*"}), MessageStop()],
        [_exit("x3"), MessageStop()],
        [_exit("x4"), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    summary = await rt.run_turn("続けて", _Approve())
    assert summary.plan_exit_not_active == 4
    assert summary.final_text == "done"
