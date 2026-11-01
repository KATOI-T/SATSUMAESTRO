"""段の実行中の read タイムアウトは段ごとに 2 回まで復帰し、2 回目は段の再開の文を積む。"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest
from rich.console import Console

from satsumaestro.cli.render import RichRenderer
from satsumaestro.config.models import GuidanceConfig, RuntimeConfig
from satsumaestro.engine import runtime as runtime_module
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.plan_steps import (
    first_step_note,
    next_step_note,
    resume_step_note,
    timeout_resume_step_note,
)
from satsumaestro.engine.reminders import timeout_note
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.trace import TurnAborted
from satsumaestro.engine.turn import NullRenderer, TurnSummary
from satsumaestro.evals.agents import SatsumaAgent
from satsumaestro.evals.models import EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.session.models import TextBlock, TokenUsage
from satsumaestro.telemetry.recorder import describe, summary_fields
from satsumaestro.util import chunking
from satsumaestro.util.errors import MaxIterationsError, ProviderError
from tests.fixtures.mock_provider import ScriptedFailureProvider, SequencedMockProvider
from tests.test_p237_step_budget import _Approve, _exit, _runtime, _say, _write_plan

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"
PLAN = "## Steps\n1. out.html (~50 lines)\n2. verify (~0 lines)\n"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """時計を 1 呼び出し 10 秒進める（``tests/test_p237_step_budget.py`` と同じ）。"""
    ticks = iter(range(0, 1_000_000, 10))
    monkeypatch.setattr(
        runtime_module, "time", SimpleNamespace(monotonic=lambda: float(next(ticks)))
    )


def _timeout(message: str = "timed out") -> ProviderError:
    """プロバイダ境界が作るのと同じ形（``__cause__`` に httpx 例外）を再現する。"""
    error = ProviderError(f"OpenAI 互換 API 呼び出しに失敗: {message}", retryable=True)
    error.__cause__ = httpx.ReadTimeout(
        message, request=httpx.Request("POST", "http://localhost:11434")
    )
    return error


def _write_out(id: str = "o1", text: str = "a\n") -> list[AssistantEvent]:
    return [
        ToolUse(id=id, name="write_file", input={"path": "out.html", "content": text}),
        Usage(output_tokens=10),
        MessageStop(),
    ]


def _cancelled_notes(rt: ConversationRuntime) -> list[str]:
    """打ち切りの後に積んだ合成 user メッセージ（段を渡す文は含めない）。"""
    return [
        b.text
        for m in rt.session.messages
        if m.role == "user"
        for b in m.blocks
        if isinstance(b, TextBlock) and "<system-reminder>" in b.text and "cancelled" in b.text
    ]


class _Recording(NullRenderer):
    """打ち切りの通知（1 回目）と段の再開の通知（2 回目）を別々に記録する。"""

    aborted: ClassVar[list[float | None]] = []
    resumed: ClassVar[list[tuple[float, int, int, int, int]]] = []

    def __init__(self) -> None:
        type(self).aborted = []
        type(self).resumed = []

    def on_stream_aborted(self, timeout_recovery_s: float | None = None) -> None:
        type(self).aborted.append(timeout_recovery_s)

    def on_step_timeout_resume(
        self, timeout_s: float, index: int, total: int, attempt: int, limit: int
    ) -> None:
        type(self).resumed.append((timeout_s, index, total, attempt, limit))


def _scripted(
    tmp_path: Path, steps: list[Any], cfg: RuntimeConfig | None = None
) -> ConversationRuntime:
    rt = _runtime(ScriptedFailureProvider([[MessageStop()]]), tmp_path, cfg=cfg)
    rt.provider._steps[:] = steps  # type: ignore[attr-defined]
    return rt


async def test_second_timeout_in_a_step_resumes_instead_of_failing(
    tmp_path: Path, clock: None
) -> None:
    rt = _scripted(tmp_path, [])
    rt.provider._steps[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN),
        _exit(),
        _timeout(),
        _timeout(),
        _write_out(),
        _say("段 1 done"),
        _say("done"),
    ]
    render = _Recording()
    summary = await rt.run_turn("作って", _Approve(), render=render)

    assert summary.final_text == "done"
    assert (summary.timeout_recoveries, summary.timeout_resumes) == (2, 1)
    first = summary.plan_step_records[0]
    assert (first.timeouts, first.writes, first.actual_lines) == (2, 1, 1)
    assert summary.plan_step_records[1].timeouts == 0

    notes = _cancelled_notes(rt)
    assert len(notes) == 2
    assert "Your previous response was cancelled" in notes[0]
    assert "at most ~300 lines" in notes[0] and "~150 lines" not in notes[0]
    assert "Resuming" not in notes[0]
    second = notes[1]
    assert "cancelled again" in second
    assert "2 cancelled responses in a row in step 1 of 2" in second
    assert "Resuming step 1 of 2 of the approved plan: out.html (~50 lines)" in second
    assert "Start with ONE call of at most ~300 lines" in second
    assert "Keep each write_file/append_file call under ~300 lines" in second
    assert "This step adds about ~50 lines in total" in second
    assert "~60 tokens/s measured in this conversation" in second
    assert "at least ~7,200 tokens" in second and "roughly 600 lines" in second
    assert "about 2× the limit of ~300 lines per call" in second
    assert "files are exactly as they were before" in second
    assert "The user's messages since you stopped" not in second
    assert _Recording.aborted == [120.0]
    assert _Recording.resumed == [(120.0, 1, 2, 2, 2)]


async def test_third_timeout_in_a_step_is_raised_with_the_step_on_the_error(
    tmp_path: Path, clock: None
) -> None:
    rt = _scripted(tmp_path, [])
    rt.provider._steps[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN),
        _exit(),
        _timeout(),
        _timeout(),
        _timeout(),
    ]
    render = _Recording()
    with pytest.raises(ProviderError) as excinfo:
        await rt.run_turn("作って", _Approve(), render=render)
    assert excinfo.value.step == (1, 2)
    assert excinfo.value.timeout_recoveries == 2
    assert len(_cancelled_notes(rt)) == 2
    assert _Recording.aborted == [120.0, None]
    assert _Recording.resumed == [(120.0, 1, 2, 2, 2)]
    assert rt.session.plan_step_index == 1


async def test_allowance_is_counted_per_step(tmp_path: Path, clock: None) -> None:
    """段 1 で 1 回、段 2 で 1 回。"""
    rt = _scripted(tmp_path, [])
    rt.provider._steps[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN),
        _exit(),
        _timeout(),
        _write_out(),
        _say("段 1 done"),
        _timeout(),
        _write_out("o2", "b\n"),
        _say("done"),
    ]
    summary = await rt.run_turn("作って", _Approve())
    assert summary.final_text == "done"
    assert (summary.timeout_recoveries, summary.timeout_resumes) == (2, 0)
    assert [r.timeouts for r in summary.plan_step_records] == [1, 1]
    notes = _cancelled_notes(rt)
    assert len(notes) == 2 and all("cancelled again" not in n for n in notes)


async def test_timeout_before_approval_does_not_use_the_step_allowance(
    tmp_path: Path, clock: None
) -> None:
    """計画を書く途中の打ち切りは段の外の 1 回。"""
    rt = _scripted(tmp_path, [])
    rt.provider._steps[:] = [  # type: ignore[attr-defined]
        _timeout(),
        _write_plan(rt, PLAN),
        _exit(),
        _timeout(),
        _timeout(),
        _write_out(),
        _say("段 1 done"),
        _say("done"),
    ]
    summary = await rt.run_turn("作って", _Approve())
    assert summary.final_text == "done"
    assert (summary.timeout_recoveries, summary.timeout_resumes) == (3, 1)
    assert summary.plan_step_records[0].timeouts == 2


async def test_allowance_of_one_keeps_the_previous_behaviour(tmp_path: Path, clock: None) -> None:
    """``step_timeout_recoveries=1`` ＝ 修正前（同一バイナリ A/B の before 条件）。"""
    cfg = RuntimeConfig(guidance=GuidanceConfig(step_timeout_recoveries=1))
    rt = _scripted(tmp_path, [], cfg=cfg)
    rt.provider._steps[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN),
        _exit(),
        _timeout(),
        _timeout(),
        _write_out(),
    ]
    with pytest.raises(ProviderError) as excinfo:
        await rt.run_turn("作って", _Approve())
    assert excinfo.value.step == (1, 2) and excinfo.value.timeout_recoveries == 1
    assert len(_cancelled_notes(rt)) == 1


async def test_plain_turn_still_recovers_once(tmp_path: Path) -> None:
    """段の実行でないターンは従来どおり 1 回。"""
    rt = _scripted(tmp_path, [_timeout(), _timeout("again"), _say("x")])
    rt.set_mode(main="code")
    with pytest.raises(ProviderError, match="again") as excinfo:
        await rt.run_turn("300 行の HTML を作って")
    assert excinfo.value.step is None and excinfo.value.timeout_recoveries == 1
    rt2 = _scripted(tmp_path / "b", [_timeout(), _say("x")])
    rt2.set_mode(main="code")
    summary = await rt2.run_turn("300 行の HTML を作って")
    assert (summary.timeout_recoveries, summary.timeout_resumes) == (1, 0)
    assert timeout_note(120.0) in _cancelled_notes(rt2)[0]


def test_step_allowance_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        GuidanceConfig(step_timeout_recoveries=0)
    assert GuidanceConfig().step_timeout_recoveries == 2


def test_first_reminder_takes_the_step_budget_as_its_limit() -> None:
    assert "(at most ~220 lines)" in timeout_note(120.0, lines=220)
    assert f"(at most {chunking.LINES})" in timeout_note(120.0)
    assert timeout_note(120.0, lines=None) == timeout_note(120.0)
    assert timeout_note(120.0, lines=0) == timeout_note(120.0)


def test_resume_note_quotes_the_measured_size_as_a_lower_bound() -> None:
    note = timeout_resume_step_note(
        1, 9, "骨格と CSS (~220 lines)", seconds=120.0, attempt=2, rate_tok_s=58.0, budget_lines=220
    )
    assert "2 cancelled responses in a row in step 1 of 9" in note
    assert "~58 tokens/s" in note and "at least ~6,960 tokens" in note
    assert "roughly 580 lines" in note and "about 3× the limit of ~220 lines per call" in note
    assert "Resuming step 1 of 9 of the approved plan: 骨格と CSS (~220 lines)" in note
    assert "Start with ONE call of at most ~220 lines" in note
    assert "write_file if the file does not exist yet, append_file if it does" in note
    assert "do not start the next step" in note and "final answer" not in note
    last = timeout_resume_step_note(
        9, 9, "確認", seconds=120.0, attempt=2, rate_tok_s=58.0, budget_lines=220
    )
    assert "final answer" in last and "do not start the next step" not in last


def test_resume_note_without_a_measurement_has_no_numbers_but_the_limit() -> None:
    note = timeout_resume_step_note(
        2, 3, "x", seconds=120.0, attempt=2, rate_tok_s=None, budget_lines=220
    )
    assert "tokens/s" not in note and "at least" not in note
    assert (
        "A tool call that takes 120 seconds to produce is far above the limit of ~220 lines" in note
    )
    small = timeout_resume_step_note(
        2, 3, "x", seconds=120.0, attempt=2, rate_tok_s=20.0, budget_lines=300
    )
    assert "roughly 200 lines inside one tool call — the limit is ~300 lines per call" in small
    assert "×" not in small
    none = timeout_resume_step_note(2, 3, "x", seconds=120.0, attempt=2, rate_tok_s=None)
    assert f"far above the limit of {chunking.LINES} per call" in none
    assert "Keep each" not in none


def test_step_notes_carry_the_step_estimate_only_with_a_budget() -> None:
    first = first_step_note(3, "a", "p.md", budget_lines=300, estimated_lines=40)
    assert "under ~300 lines" in first
    assert "This step adds about ~40 lines in total across all its calls; later steps add" in first
    assert "in total" not in first_step_note(3, "a", "p.md", budget_lines=300)
    assert "in total" not in first_step_note(3, "a", "p.md", budget_lines=300, estimated_lines=0)
    assert "in total" not in first_step_note(3, "a", "p.md", estimated_lines=40)
    assert "~40 lines in total" in next_step_note(2, 3, "b", budget_lines=300, estimated_lines=40)
    assert "~40 lines in total" in resume_step_note(2, 3, "b", budget_lines=300, estimated_lines=40)
    assert next_step_note(2, 3, "b", budget_lines=300) == next_step_note(
        2, 3, "b", budget_lines=300, estimated_lines=None
    )


def _resumed(attempt: int, limit: int) -> tuple[RichRenderer, str]:
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=True, width=120, color_system=None))
    r.start_turn()
    r.on_step_timeout_resume(120.0, 1, 9, attempt, limit)
    return r, buf.getvalue()


def test_resume_notice_is_one_line_and_says_when_the_next_abort_stops() -> None:
    r, printed = _resumed(2, 2)
    out = "".join(printed.split())
    assert (
        "—120秒応答がなかったので中断し、段1/9を小さく分けて再開するよう依頼しました"
        "（この段で2回目・もう1度切れたら中止）"
    ) in out
    assert "分けて出し直すよう依頼しました" not in out
    status = io.StringIO()
    Console(file=status, force_terminal=False, width=100).print(r._live_renderable())
    assert "出し直しを待っています…" in status.getvalue()
    _, more = _resumed(2, 3)
    assert "（この段で2回目）" in "".join(more.split()) and "中止" not in more


def test_aborted_turn_record_carries_the_step_and_the_recoveries() -> None:
    error = ProviderError("OpenAI 互換 API 呼び出しに失敗: ReadTimeout", retryable=True)
    error.step = (1, 9)
    error.timeout_recoveries = 2
    assert describe(TurnAborted(7, error)) == (
        "turn.abort",
        {"iteration": 7, "error": "ProviderError", "step": [1, 9], "timeout_recoveries": 2},
    )
    rows = [
        describe(TurnAborted(8, MaxIterationsError(50, step=(3, 9)))),
        describe(TurnAborted(9, RuntimeError("x"))),
    ]
    assert rows[0] is not None and rows[1] is not None
    assert rows[0][1]["step"] == [3, 9] and rows[0][1]["timeout_recoveries"] == 0
    assert rows[1][1]["step"] is None and rows[1][1]["timeout_recoveries"] == 0
    fresh = ProviderError("x")
    assert fresh.step is None and fresh.timeout_recoveries == 0


def test_summary_metric_reaches_the_telemetry_record() -> None:
    summary = TurnSummary(
        final_text="x",
        stop_reason="end_turn",
        iterations=3,
        usage=TokenUsage(),
        timeout_recoveries=2,
        timeout_resumes=1,
    )
    data = summary_fields(summary)
    assert data["timeout_recoveries"] == 2 and data["timeout_resumes"] == 1


async def test_metric_reaches_the_report_json(tmp_path: Path) -> None:
    """評価の経路（SatsumaAgent → EvalRunner → TaskResult）。"""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    provider = SequencedMockProvider([[TextDelta(text="できました"), MessageStop()]])
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")
    task = EvalTask.model_validate(
        {
            "id": "t",
            "prompt": "x",
            "fixture": str(fixture),
            "permission_mode": "workspace-write",
            "checks": {"contains_all": ["できました"]},
        }
    )
    report = await EvalRunner(agent).run([task])
    dumped = json.loads(report.model_dump_json())
    assert dumped["results"][0]["timeout_resumes"] == 0
    assert EvalReport.model_validate(dumped).results[0].timeout_resumes == 0
    assert TaskResult.model_validate({"task_id": "t", "passed": True}).timeout_resumes == 0


def test_ab_summary_folds_the_resumes(tmp_path: Path) -> None:
    before = tmp_path / "b.json"
    before.write_text(
        json.dumps({"results": [{"task_id": "t0", "passed": True}]}), encoding="utf-8"
    )
    after = tmp_path / "a.json"
    after.write_text(
        json.dumps(
            {
                "agent": "satsuma",
                "results": [
                    {
                        "task_id": "t0",
                        "passed": True,
                        "timeout_recoveries": 2,
                        "timeout_resumes": 1,
                    },
                    {"task_id": "t1", "passed": True, "timeout_recoveries": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "to", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| タイムアウト復帰 | 0.0 | 3.0 | +3.0 |" in out
    assert "| 打ち切り 2 回目の段の再開 | 0.0 | 1.0 | +1.0 |" in out
