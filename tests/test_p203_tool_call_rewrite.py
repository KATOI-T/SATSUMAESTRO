"""tool_call のパース失敗が誘導付きの再送の後も尽きたら、ターンを落とさず書き直しを依頼する。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from satsumaestro.config.models import GuidanceConfig, RuntimeConfig
from satsumaestro.engine import runtime as runtime_module
from satsumaestro.engine.events import AssistantEvent, MessageStop, ToolUse, Usage
from satsumaestro.engine.reminders import tool_call_rewrite_note
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.tool_call_repair import ToolCallRepairState
from satsumaestro.engine.turn import NullRenderer, TurnSummary
from satsumaestro.evals.agents import _answer_from
from satsumaestro.evals.models import TaskResult
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.notices import ProviderNotice, describe
from satsumaestro.session.models import TextBlock, TokenUsage
from satsumaestro.util.errors import ProviderError, RetriesExhaustedError
from tests.fixtures.mock_provider import ScriptedFailureProvider
from tests.test_p237_step_budget import _Approve, _exit, _runtime, _say, _write_plan

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"
PLAN = "## Steps\n1. a.txt (~5 lines)\n2. b.txt (~5 lines)\n"


def _broken() -> RetriesExhaustedError:
    """専用の再試行の後に RetryingProvider が投げる形（解析の 500 が尽きた）。"""
    return RetriesExhaustedError(
        2,
        3.5,
        last_error='error parsing tool call: raw=\'{"content":"a\nb"}\', '
        "err=invalid character '\\n' in string literal",
    )


def _write(path: str, id: str) -> list[AssistantEvent]:
    return [
        ToolUse(id=id, name="write_file", input={"path": path, "content": "x\n"}),
        Usage(output_tokens=10),
        MessageStop(),
    ]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter(range(0, 1_000_000, 10))
    monkeypatch.setattr(
        runtime_module, "time", SimpleNamespace(monotonic=lambda: float(next(ticks)))
    )


def _scripted(
    tmp_path: Path, steps: list[Any], cfg: RuntimeConfig | None = None, *, code: bool = True
) -> ConversationRuntime:
    """台本どおりに失敗するランタイム。"""
    rt = _runtime(ScriptedFailureProvider([[MessageStop()]]), tmp_path, cfg=cfg)
    rt.provider._steps[:] = steps  # type: ignore[attr-defined]
    if code:
        rt.set_mode(main="code")
    return rt


def _rewrite_notes(rt: ConversationRuntime) -> list[str]:
    return [
        b.text
        for m in rt.session.messages
        if m.role == "user"
        for b in m.blocks
        if isinstance(b, TextBlock) and "not valid JSON" in b.text
    ]


class _Notices(NullRenderer):
    seen: ClassVar[list[ProviderNotice]] = []

    def __init__(self) -> None:
        type(self).seen = []

    def on_provider_notice(self, notice: ProviderNotice) -> None:
        type(self).seen.append(notice)


async def test_a_call_still_broken_after_the_hint_is_rewritten(tmp_path: Path) -> None:
    rt = _scripted(tmp_path, [_broken(), _broken(), _write("a.txt", "w1"), _say("done")])
    render = _Notices()
    summary = await rt.run_turn("a.txt を作って", render=render)

    assert summary.final_text == "done"
    assert (summary.tool_call_repairs, summary.tool_call_rewrites) == (1, 1)
    assert (tmp_path / "a.txt").exists()
    (note,) = _rewrite_notes(rt)
    assert tool_call_rewrite_note() in note
    assert [n.kind for n in _Notices.seen] == ["retry", "rewrite"]


async def test_the_second_time_in_a_turn_still_fails(tmp_path: Path) -> None:
    rt = _scripted(tmp_path, [_broken()] * 4 + [_say("done")])
    with pytest.raises(RetriesExhaustedError):
        await rt.run_turn("a.txt を作って")
    assert len(_rewrite_notes(rt)) == 1
    assert rt.aborted_summary is not None
    assert rt.aborted_summary.tool_call_rewrites == 1


async def test_setting_off_keeps_the_old_failure(tmp_path: Path) -> None:
    cfg = RuntimeConfig(guidance=GuidanceConfig(tool_call_rewrite=False))
    rt = _scripted(tmp_path, [_broken(), _broken(), _say("done")], cfg)
    with pytest.raises(RetriesExhaustedError):
        await rt.run_turn("a.txt を作って")
    assert _rewrite_notes(rt) == []


async def test_other_failures_are_not_rewritten(tmp_path: Path) -> None:
    other = RetriesExhaustedError(9, 60.0, last_error="503 Service Unavailable")
    rt = _scripted(tmp_path, [other, _say("done")])
    with pytest.raises(ProviderError):
        await rt.run_turn("a.txt を作って")
    assert _rewrite_notes(rt) == []


async def test_no_rewrite_on_the_last_iteration(tmp_path: Path) -> None:
    rt = _scripted(tmp_path, [_broken(), _broken(), _say("done")], RuntimeConfig(max_iterations=1))
    with pytest.raises(RetriesExhaustedError):
        await rt.run_turn("a.txt を作って")
    assert _rewrite_notes(rt) == []


async def test_each_step_gets_its_own_hint_and_rewrite(tmp_path: Path, clock: None) -> None:
    rt = _scripted(tmp_path, [], code=False)
    rt.provider._steps[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN),
        _exit(),
        _broken(),
        _broken(),
        _write("a.txt", "a1"),
        _say("段 1 done"),
        _broken(),
        _broken(),
        _write("b.txt", "b1"),
        _say("done"),
    ]
    summary = await rt.run_turn("作って", _Approve())

    assert summary.final_text == "done"
    assert (summary.tool_call_repairs, summary.tool_call_rewrites) == (2, 2)
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()
    assert len(_rewrite_notes(rt)) == 2


def test_the_hint_window_restarts_but_the_total_does_not() -> None:
    state = ToolCallRepairState()
    exc = _broken()
    assert state.should_repair(exc)
    state.repaired_request(ApiRequest(model="m", system="s", messages=[], tools=[]))
    assert not state.should_repair(exc)
    state.new_section()
    assert state.should_repair(exc)
    assert (state.repairs, state.window) == (1, 0)


def test_the_note_names_the_usual_breakages() -> None:
    note = tool_call_rewrite_note()
    assert note.startswith("Your previous tool call was rejected by the server again")
    for words in ("not valid JSON", "raw line break", 'unescaped `"`', "backslash escape"):
        assert words in note
    assert 'write line breaks as `\\n` and double quotes as `\\"`' in note
    assert "Do not resend the same call unchanged." in note


def test_the_screen_line() -> None:
    assert describe(ProviderNotice(kind="rewrite")) == (
        "モデルの出力が繰り返し壊れたので、書き直すよう依頼しました"
    )


def test_counts_reach_the_answer_and_old_reports_load() -> None:
    summary = TurnSummary(
        final_text="done",
        stop_reason="end_turn",
        iterations=3,
        usage=TokenUsage(),
        tool_call_rewrites=2,
    )
    answer = _answer_from(summary, plan_exits=0, iteration_limit=50)
    assert answer.tool_call_rewrites == 2
    result = TaskResult.model_validate({"task_id": "t", "passed": True, **answer.model_dump()})
    assert result.tool_call_rewrites == 2
    assert TaskResult.model_validate({"task_id": "t", "passed": True}).tool_call_rewrites == 0


def test_ab_summary_folds_the_count(tmp_path: Path) -> None:
    before = tmp_path / "b.json"
    before.write_text(json.dumps({"results": [{"task_id": "t0", "passed": True}]}), "utf-8")
    after = tmp_path / "a.json"
    rows = [{"task_id": "t0", "passed": True, "tool_call_rewrites": 1}]
    after.write_text(json.dumps({"agent": "satsuma", "results": rows}), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "rw", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 壊れた呼び出しの書き直し | 0.0 | 1.0 | +1.0 |" in out
