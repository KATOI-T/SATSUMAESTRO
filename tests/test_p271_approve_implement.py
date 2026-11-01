"""段に分けない計画の承認後にも「承認された。ツールで実行せよ」を伝える。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.session.models import plan_file_relpath
from satsumaestro.tools.builtins.plan_mode import (
    PLAN_EXIT_APPROVED_IMPLEMENT,
    PLAN_EXIT_APPROVED_WITH_NOTE,
    PLAN_EXIT_NOT_ACTIVE,
    PLAN_EXIT_NOT_APPROVED,
)
from tests.fixtures.mock_provider import SequencedMockProvider
from tests.test_p36_mode_state import (
    _Approve,
    _ApproveWithNote,
    _blocks,
    _plan_then_exit,
    _runtime,
    _with_plan,
)

STEPPED_PLAN = "# Plan\n\n## Steps\n1. write a.html (~40 lines)\n2. write a.css (~40 lines)\n"


def _implement(rt: object) -> str:
    return PLAN_EXIT_APPROVED_IMPLEMENT.format(
        plan_file=plan_file_relpath(rt.session.session_id)  # type: ignore[attr-defined]
    )


async def test_unstepped_approval_tells_the_model_to_implement_with_tools(
    tmp_path: Path,
) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path)
    await rt.run_turn("plan", prompter=_Approve())
    out = _blocks(rt)["x1"].output
    assert out.startswith("Exited plan mode")
    assert out.endswith(_implement(rt))
    assert "write_file" in out and "do not paste file contents" in out


async def test_note_approval_keeps_its_own_wording(tmp_path: Path) -> None:
    """一言つき承認は「Implement the plan … note」だけ（実装の指示を 2 回言わない）。"""
    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path)
    await rt.run_turn("plan", prompter=_ApproveWithNote())
    out = _blocks(rt)["x1"].output
    assert out.endswith(
        PLAN_EXIT_APPROVED_WITH_NOTE.format(note="also add tests; name it greet_user")
    )
    assert _implement(rt) not in out


async def test_stepped_approval_hands_step_one_instead(tmp_path: Path) -> None:
    """段に分けた承認は段 1 の指示（Do ONLY step 1）を渡す。"""
    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path, STEPPED_PLAN)
    await rt.run_turn("plan", prompter=_Approve())
    out = _blocks(rt)["x1"].output
    assert "Do ONLY step 1 of 2" in out
    assert _implement(rt) not in out


async def test_not_approved_gets_no_implement_instruction(tmp_path: Path) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path)
    await rt.run_turn("plan")
    assert _blocks(rt)["x1"].output == PLAN_EXIT_NOT_APPROVED


async def test_exit_outside_plan_mode_gets_no_implement_instruction(tmp_path: Path) -> None:
    """plan でないときの exit_plan_mode（``plan mode is not active``）に承認の文を付けない。"""
    provider = SequencedMockProvider(
        [
            [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
            [TextDelta(text="ok"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    await rt.run_turn("hi", prompter=_Approve())
    out = _blocks(rt)["x1"].output
    assert out == PLAN_EXIT_NOT_ACTIVE
