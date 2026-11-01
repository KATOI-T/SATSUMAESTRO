"""段の実行中、1 回の書き込みが段の上限を超えたら書かずに差し戻す。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from satsumaestro.engine import runtime as runtime_module
from satsumaestro.engine.events import AssistantEvent, MessageStop, ToolUse, Usage
from satsumaestro.engine.plan_steps import STEP_WRITE_REJECT_LIMIT, step_write_reject_message
from satsumaestro.engine.turn import TurnSummary
from satsumaestro.evals.agents import _answer_from
from satsumaestro.evals.models import TaskResult
from satsumaestro.session.models import TokenUsage
from tests.fixtures.mock_provider import SequencedMockProvider
from tests.test_p237_step_budget import _Approve, _blocks, _exit, _runtime, _say, _write_plan

_PLAN = "## Steps\n1. out.html の頭 (~50 lines)\n2. out.html の残り (~50 lines)\n"
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """時計を 1 呼び出し 10 秒進める（応答 1 回 = 10 秒）。"""
    ticks = iter(range(0, 1_000_000, 10))
    monkeypatch.setattr(
        runtime_module, "time", SimpleNamespace(monotonic=lambda: float(next(ticks)))
    )


def _lines(n: int) -> str:
    return "".join(f"<p>{i}</p>\n" for i in range(n))


def _write(id: str, tool: str, n: int) -> list[AssistantEvent]:
    call = ToolUse(id=id, name=tool, input={"path": "out.html", "content": _lines(n)})
    return [call, Usage(output_tokens=30), MessageStop()]


async def _run(tmp_path: Path, steps: list[list[AssistantEvent]]) -> tuple[object, TurnSummary]:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [_write_plan(rt, _PLAN), _exit("x1"), *steps]  # type: ignore[attr-defined]
    summary = await rt.run_turn("作って", _Approve())
    return rt, summary


def _out(tmp_path: Path) -> int:
    path = tmp_path / "out.html"
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


async def test_a_write_over_the_step_limit_is_sent_back(tmp_path: Path, clock: None) -> None:
    rt, summary = await _run(
        tmp_path,
        [
            _write("w1", "write_file", 400),
            _write("w2", "write_file", 200),
            _say("段 1 done"),
            _write("a1", "append_file", 50),
            _say("done"),
        ],
    )

    w1 = _blocks(rt)["w1"]  # type: ignore[arg-type]
    assert w1.is_error
    assert w1.output.splitlines()[0] == (
        "write_file rejected: 400 lines is over this step's limit of 300 lines per write"
    )
    assert _out(tmp_path) == 250
    assert (summary.step_write_rejects, summary.step_write_overrides) == (1, 0)
    first, second = summary.plan_step_records
    assert (first.actual_lines, first.max_write_lines, first.writes) == (200, 400, 2)
    assert second.actual_lines == 50


async def test_the_third_over_limit_write_in_a_step_is_written(tmp_path: Path, clock: None) -> None:
    assert STEP_WRITE_REJECT_LIMIT == 2
    rt, summary = await _run(
        tmp_path,
        [
            _write("w1", "write_file", 400),
            _write("w2", "write_file", 400),
            _write("w3", "write_file", 400),
            _say("段 1 done"),
            _say("done"),
        ],
    )

    blocks = _blocks(rt)  # type: ignore[arg-type]
    assert [blocks[i].is_error for i in ("w1", "w2", "w3")] == [True, True, False]
    assert _out(tmp_path) == 400
    assert (summary.step_write_rejects, summary.step_write_overrides) == (2, 1)


async def test_the_count_starts_again_in_the_next_step(tmp_path: Path, clock: None) -> None:
    rt, summary = await _run(
        tmp_path,
        [
            _write("w1", "write_file", 400),
            _write("w2", "write_file", 400),
            _write("w3", "write_file", 400),
            _say("段 1 done"),
            _write("a1", "append_file", 400),
            _write("a2", "append_file", 100),
            _say("done"),
        ],
    )

    a1 = _blocks(rt)["a1"]  # type: ignore[arg-type]
    assert a1.is_error and a1.output.startswith("append_file rejected: 400 lines")
    assert "- Append the content in parts of at most 300 lines, one append_file each." in a1.output
    assert _out(tmp_path) == 500
    assert (summary.step_write_rejects, summary.step_write_overrides) == (3, 1)


async def test_writes_outside_the_steps_are_not_limited(tmp_path: Path, clock: None) -> None:
    rt = _runtime(SequencedMockProvider([_write("w1", "write_file", 400), _say("done")]), tmp_path)
    rt.set_mode(main="code")

    summary = await rt.run_turn("作って")

    assert not _blocks(rt)["w1"].is_error
    assert _out(tmp_path) == 400 and summary.step_write_rejects == 0


def test_reject_message() -> None:
    assert step_write_reject_message("write_file", "out.html", 688, 290) == (
        "write_file rejected: 688 lines is over this step's limit of 290 lines per write\n"
        "out.html was not changed. One response has to finish within 120 seconds; the limit "
        "keeps one write well inside that for this model.\n"
        "- Write only what this step of the plan needs; leave later steps for later.\n"
        "- To write a long file, write_file the first part (at most 290 lines), then "
        "append_file each following part of the same size."
    )


def test_counts_reach_the_answer_and_old_reports_load() -> None:
    summary = TurnSummary(
        final_text="done",
        stop_reason="end_turn",
        iterations=3,
        usage=TokenUsage(),
        step_write_rejects=2,
        step_write_overrides=1,
    )
    answer = _answer_from(summary, plan_exits=1, iteration_limit=50)
    assert (answer.step_write_rejects, answer.step_write_overrides) == (2, 1)
    result = TaskResult.model_validate({"task_id": "t", "passed": True, **answer.model_dump()})
    assert (result.step_write_rejects, result.step_write_overrides) == (2, 1)
    old = TaskResult.model_validate({"task_id": "t", "passed": True})
    assert (old.step_write_rejects, old.step_write_overrides) == (0, 0)


def test_ab_summary_folds_the_counts(tmp_path: Path) -> None:
    before = tmp_path / "b.json"
    before.write_text(
        json.dumps({"results": [{"task_id": "t0", "passed": True}]}), encoding="utf-8"
    )
    after = tmp_path / "a.json"
    rows = [
        {"task_id": "t0", "passed": True, "step_write_rejects": 2, "step_write_overrides": 1},
        {"task_id": "t1", "passed": True, "step_write_rejects": 1},
    ]
    after.write_text(json.dumps({"agent": "satsuma+plan", "results": rows}), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "step", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 段の上限を超えた書き込みの差し戻し | 0.0 | 3.0 | +3.0 |" in out
    assert "| 段の上限の差し戻しの後で書いた | 0.0 | 1.0 | +1.0 |" in out
