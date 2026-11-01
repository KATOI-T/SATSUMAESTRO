"""段の記録に、edit_file で直した行数を別の欄で残す。"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from satsumaestro.cli.render import RichRenderer
from satsumaestro.engine import runtime as runtime_module
from satsumaestro.engine.events import AssistantEvent, MessageStop, ToolUse, Usage
from satsumaestro.session.models import PlanStepRecord
from tests.fixtures.mock_provider import SequencedMockProvider
from tests.test_p237_step_budget import _Approve, _exit, _runtime, _say, _write_plan

_PLAN = "## Steps\n1. out.html を作る (~50 lines)\n2. out.html を直す (~10 lines)\n"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """時計を 1 呼び出し 10 秒進める（応答 1 回 = 10 秒）。"""
    ticks = iter(range(0, 1_000_000, 10))
    monkeypatch.setattr(
        runtime_module, "time", SimpleNamespace(monotonic=lambda: float(next(ticks)))
    )


def _tool(id: str, name: str, **inp: object) -> list[AssistantEvent]:
    return [ToolUse(id=id, name=name, input=inp), Usage(output_tokens=30), MessageStop()]


async def test_an_edit_only_step_records_its_edited_lines(tmp_path: Path, clock: None) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, _PLAN),
        _exit("x1"),
        _tool("o1", "write_file", path="out.html", content="a\nb\nc\n"),
        _say("段 1 done"),
        _tool("e1", "edit_file", path="out.html", old="b", new="B\nB2"),
        _tool("e2", "edit_file", path="out.html", old="no such text", new="x\ny\nz"),
        _tool("e3", "edit_file", path="out.html", old="c", new="C"),
        _say("done"),
    ]

    summary = await rt.run_turn("作って", _Approve())

    first, second = summary.plan_step_records
    assert (first.actual_lines, first.edits, first.edited_lines) == (3, 0, 0)
    assert (second.actual_lines, second.edits, second.edited_lines) == (0, 2, 3)
    assert (tmp_path / "out.html").read_text(encoding="utf-8") == "a\nB\nB2\nC\n"


def _done_line(record: PlanStepRecord) -> str:
    buf = io.StringIO()
    r = RichRenderer(Console(file=buf, force_terminal=True, width=200, color_system=None))
    r.on_plan_step_done(record)
    r.end_turn()
    return next(line for line in buf.getvalue().splitlines() if "完了" in line)


def test_done_line_adds_the_edited_lines_only_when_there_were_edits() -> None:
    edited = PlanStepRecord(
        index=2, total=3, estimated_lines=10, budget_lines=280, edits=2, edited_lines=18
    )
    plain = PlanStepRecord(
        index=1, total=3, estimated_lines=50, budget_lines=280, actual_lines=48, max_write_lines=48
    )

    assert _done_line(edited).endswith(
        "— 段 2/3 完了: 見積 10 行 / 実績 0 行 / 最大 1 回 0 行 / 最長応答 0 秒 / 編集 18 行"
    )
    assert _done_line(plain).endswith(
        "— 段 1/3 完了: 見積 50 行 / 実績 48 行 / 最大 1 回 48 行 / 最長応答 0 秒"
    )


def test_old_records_load_without_the_new_fields() -> None:
    old = PlanStepRecord.model_validate({"index": 1, "actual_lines": 5})
    assert (old.edits, old.edited_lines) == (0, 0)
