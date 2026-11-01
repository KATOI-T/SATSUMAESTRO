"""段階実行の exit_plan_mode と、段の節の選び方。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.plan_steps import parse_plan_steps
from satsumaestro.tools.builtins.plan_mode import PLAN_EXIT_STEP_DONE_PREFIX
from tests.fixtures.mock_provider import SequencedMockProvider
from tests.test_p165_plan_steps import (
    _Approve,
    _blocks,
    _done,
    _injected,
    _read,
    _runtime,
    _user_texts,
    _write_plan,
)


def _summary_then_exit(call_id: str, summary: str) -> list[AssistantEvent]:
    """実機の再現: 段の要約を本文に書き、続けて exit_plan_mode を呼ぶ。"""
    return [
        TextDelta(text=summary),
        ToolUse(id=call_id, name="exit_plan_mode", input={}),
        MessageStop(),
    ]


def _provider(*replies: list[AssistantEvent]) -> SequencedMockProvider:
    """承認の exit_plan_mode（x1）の後、与えた応答を順に返す。"""
    return SequencedMockProvider(
        [[ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()], *replies]
    )


async def test_summary_plus_exit_advances_to_the_next_step(tmp_path: Path) -> None:
    rt = _runtime(
        _provider(
            _read("a"),
            _summary_then_exit("e1", "ステップ 1 完了：雛形を作成しました。"),
            _read("b"),
            _done("step 2 done"),
            _read("c"),
            _done("final"),
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())

    out = _blocks(rt)["e1"]
    assert not out.is_error
    assert out.output.startswith(f"{PLAN_EXIT_STEP_DONE_PREFIX} step 1 of 3 is done.")
    notes = _injected(rt)
    assert "Now do ONLY step 2 of 3" in notes[0]
    assert summary.plan_exit_step_done == 1
    assert summary.plan_exit_not_active == 0
    assert summary.plan_steps_run == 3
    assert summary.final_text == "final"
    assert rt.session.plan_steps == []


async def test_exit_on_the_last_step_finishes_and_the_summary_ends_the_turn(
    tmp_path: Path,
) -> None:
    rt = _runtime(
        _provider(
            _read("a"),
            _done("1"),
            _read("b"),
            _done("2"),
            _read("c"),
            _summary_then_exit("e3", "ステップ 3 完了"),
            _done("すべて作成しました。"),
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())

    assert _blocks(rt)["e3"].output.startswith(f"{PLAN_EXIT_STEP_DONE_PREFIX} all 3 steps")
    assert rt.session.plan_steps == [] and rt.session.plan_step_index == 0
    assert summary.final_text == "すべて作成しました。"
    assert summary.plan_exit_step_done == 1


async def test_exit_without_any_work_is_sent_back_once(tmp_path: Path) -> None:
    """宣言だけで exit しても、返答を終えたときと同じく 1 回だけ段をやり直させる。"""
    rt = _runtime(
        _provider(
            _summary_then_exit("e1", "ステップ 1 完了"),
            _read("a"),
            _summary_then_exit("e2", "ステップ 1 完了"),
            _read("b"),
            _done("2"),
            _read("c"),
            _done("final"),
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())

    notes = _injected(rt)
    assert "step 1 of 3" in notes[0] and "Now do ONLY step 2" not in notes[0]
    assert "Now do ONLY step 2 of 3" in notes[1]
    assert summary.plan_exit_step_done == 2
    assert summary.final_text == "final"


async def test_repeated_exits_no_longer_abort_the_turn(tmp_path: Path) -> None:
    """実機（19 回）の再現: 段ごとに要約＋exit を続けても、ループ止めで中断しない。"""
    rt = _runtime(
        _provider(
            _read("a"),
            _summary_then_exit("e1", "1"),
            _read("b"),
            _summary_then_exit("e2", "2"),
            _read("c"),
            _summary_then_exit("e3", "3"),
            _done("final"),
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())
    assert summary.plan_exit_step_done == 3
    assert summary.plan_steps_run == 3
    assert summary.final_text == "final"


async def test_exit_in_a_resumed_step_advances(tmp_path: Path) -> None:
    """「続けて」で再開した段でも、要約＋exit で次の段へ進む。"""
    rt = _runtime(
        _provider(
            _done("BLOCKED: 色は？"),
            _read("a"),
            _summary_then_exit("e1", "ステップ 1 完了"),
            _read("b"),
            _done("2"),
            _read("c"),
            _done("final"),
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    second = await rt.run_turn("続けて")
    assert "Resuming step 1 of 3" in "".join(_user_texts(rt))
    assert second.plan_exit_step_done == 1
    assert second.final_text == "final" and rt.session.plan_steps == []


_TWO_LISTS = """# ダッシュボードデザイン作成の計画

## 背景と目的
モダンでレスポンシブな UI を提供します。

## 作業手順
### ステップ概要
1. `index.html` の雛形を作成し、ダッシュボードの主要セクションを配置。
2. `style.css` を作成し、レイアウト、配色、ダーク/ライトテーマ切替を実装。

## Steps
1. `index.html` を作成する (~60 lines)
2. `style.css` を作成する (~100 lines)
"""


def test_requested_steps_heading_wins_over_an_earlier_japanese_one() -> None:
    steps = parse_plan_steps(_TWO_LISTS)
    assert [s.estimated_lines for s in steps] == [60, 100]
    assert steps[0].text.startswith("`index.html` を作成する")


def test_other_exact_titles_still_work_without_steps() -> None:
    plan = "## 作業手順\n1. a (~10 lines)\n2. b (~20 lines)\n"
    assert [s.estimated_lines for s in parse_plan_steps(plan)] == [10, 20]
