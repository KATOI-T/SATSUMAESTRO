"""段階実行の反復予算を「段ごと」にする。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.config.models import GuidanceConfig, RuntimeConfig
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.util.errors import MaxIterationsError
from tests.fixtures.mock_provider import SequencedMockProvider
from tests.test_p165_plan_steps import (
    _Approve,
    _done,
    _exit_then,
    _read,
    _runtime,
    _step_work,
    _write_plan,
)


def _cfg(max_iterations: int) -> RuntimeConfig:
    return RuntimeConfig(max_iterations=max_iterations, guidance=GuidanceConfig(step_budget=False))


def _reads(rt: object) -> list[ToolResultBlock]:
    return [
        b
        for m in rt.session.messages  # type: ignore[attr-defined]
        for b in m.blocks
        if isinstance(b, ToolResultBlock) and b.tool_name == "read_file"
    ]


async def test_each_step_gets_its_own_budget(tmp_path: Path) -> None:
    provider = _exit_then(*_step_work("a"), *_step_work("b"), *_step_work("c", "final"))
    rt = _runtime(provider, tmp_path, cfg=_cfg(2))
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())
    assert summary.final_text == "final" and summary.plan_steps_run == 3
    assert summary.iterations == 7
    assert rt.session.plan_steps == [] and rt.session.plan_step_index == 0


async def test_a_runaway_step_exhausts_its_own_budget(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(*_step_work("a"), _read("b")), tmp_path, cfg=_cfg(3))
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    with pytest.raises(MaxIterationsError) as info:
        await rt.run_turn("作って", prompter=_Approve())
    exc = info.value
    assert exc.limit == 3 and exc.step == (2, 3)
    assert str(exc).startswith("reached max_iterations=3 without end_turn")
    assert "in plan step 2 of 3" in str(exc)
    assert rt.session.plan_step_index == 2 and len(rt.session.plan_steps) == 3
    assert len(_reads(rt)) == 1 + 3


async def test_resending_an_idle_step_does_not_reopen_the_budget(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(_done("分けて書きます"), _read("a")), tmp_path, cfg=_cfg(2))
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    with pytest.raises(MaxIterationsError) as info:
        await rt.run_turn("作って", prompter=_Approve())
    assert info.value.step == (1, 3)
    assert len(_reads(rt)) == 1
    assert rt.session.plan_step_index == 1 and len(rt.session.plan_steps) == 3


async def test_the_next_step_is_handed_even_when_the_window_is_used_up(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(*_step_work("a"), *_step_work("b", "final")), tmp_path, cfg=_cfg(2))
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path, "# plan\n\n## Steps\n1. one\n2. two\n")
    summary = await rt.run_turn("作って", prompter=_Approve())
    assert summary.final_text == "final" and summary.plan_steps_run == 2
    assert summary.iterations == 5 and rt.session.plan_steps == []


async def test_turns_without_steps_keep_the_turn_budget(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([_read("x")]), tmp_path, cfg=_cfg(1))
    with pytest.raises(MaxIterationsError) as info:
        await rt.run_turn("読んで")
    assert info.value.step is None
    assert str(info.value) == "reached max_iterations=1 without end_turn"
    assert len(_reads(rt)) == 1
