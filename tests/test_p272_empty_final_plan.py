"""plan 中・承認直後の空の返答に「ツールを呼ばずに答えろ」と言わない。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.reminders import (
    empty_final_approved_note,
    empty_final_note,
    empty_final_plan_note,
)
from tests.test_p145_plan_exit_nudge import _Approve, _reminders, _runtime, _write_plan


def _exit() -> ToolUse:
    return ToolUse(id="x1", name="exit_plan_mode", input={})


async def test_empty_reply_after_approval_asks_to_carry_out_the_plan(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [_write_plan(rt), MessageStop()],
        [_exit(), MessageStop()],
        [MessageStop()],
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
    summary = await rt.run_turn("計画を立てて", _Approve())
    notes = _reminders(rt)
    assert any(empty_final_approved_note() in n for n in notes)
    assert not any(empty_final_note() in n for n in notes)
    assert "without calling any tools" not in "".join(notes)
    assert (tmp_path / "dashboards" / "index.html").exists()
    assert summary.final_text == "作成しました。"


async def test_empty_reply_in_plan_mode_asks_for_the_plan_and_exit(tmp_path: Path) -> None:
    """計画も書かずに空で終えた（承認への差し戻しの条件に当たらない）ときも、ツールで続けさせる。"""
    rt, script = _runtime(tmp_path)
    script[:] = [
        [MessageStop()],
        [TextDelta(text="計画はまだです。"), MessageStop()],
    ]
    await rt.run_turn("計画を立てて", _Approve())
    notes = _reminders(rt)
    assert len(notes) == 1
    assert empty_final_plan_note(rt.plan_file_relpath()) in notes[0]
    assert "exit_plan_mode" in notes[0] and "without calling any tools" not in notes[0]


async def test_plain_code_mode_keeps_the_original_empty_final(tmp_path: Path) -> None:
    """plan と関係ない会話（本文空の再試行の本来の形）は、従来どおり本文で答えさせる。"""
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        [MessageStop()],
        [TextDelta(text="答えです。"), MessageStop()],
    ]
    summary = await rt.run_turn("質問です")
    notes = _reminders(rt)
    assert len(notes) == 1 and empty_final_note() in notes[0]
    assert summary.final_text == "答えです。"
