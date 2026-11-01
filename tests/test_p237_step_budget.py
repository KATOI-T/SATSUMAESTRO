"""段の予算: ハーネスが測り、検算し、数字で差し戻す。"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from rich.console import Console

import satsumaestro.engine.runtime as runtime_module
from satsumaestro.cli.render import RichRenderer
from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.plan_budget import (
    DEFAULT_RATE_TOK_S,
    MAX_SENDBACKS,
    BudgetVerdict,
    PlanBudget,
    check_plan_budget,
    compute_plan_budget,
    lines_for,
    measured_rate,
)
from satsumaestro.engine.plan_request import ensure_plan_request
from satsumaestro.engine.plan_steps import (
    PLAN_STEPS_REQUEST,
    PLAN_STEPS_REQUEST_LEGACY,
    PlanStep,
    first_step_note,
    next_step_note,
    parse_plan_steps,
    parse_step_estimate,
    parse_steps,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter, PlanBudgetAware, PlanExitVerdict
from satsumaestro.evals.agents import AgentAnswer, EvalAgent
from satsumaestro.evals.defects import detect_defects
from satsumaestro.evals.models import EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ModeState, PlanStepRecord, TextBlock, ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util import chunking
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import ScriptedFailureProvider, SequencedMockProvider

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"
WW = PermissionMode.WORKSPACE_WRITE

PLAN_BUNDLED = """# ダッシュボード

## Steps
1. dashboard/index.html にヘッダー・サイドバー・KPI カード・グラフ・テーブルを作る (~400 lines)
2. ブラウザで表示を確認する (~0 lines)
"""
PLAN_SPLIT = """# ダッシュボード

## Steps
1. dashboard/index.html にヘッダーとサイドバー (~100 lines)
2. append_file で KPI カードとグラフ (~100 lines)
3. append_file でテーブルとスクリプト (~100 lines)
4. ブラウザで表示を確認する (~0 lines)
"""
PLAN_NO_ESTIMATES = """# 調査

## Steps
1. cart.py を読む
2. 呼び出し元を答える
"""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ヘッダーを書く (~150 lines)", 150),
        ("ヘッダーを書く（約150行）", 150),
        ("ヘッダーを書く (150 lines)", 150),
        ("ヘッダーを書く ~150 lines", 150),
        ("ヘッダーを書く 約150行", 150),
        ("Step 2: header (~40 lines of HTML)", 40),
        ("テストを実行する (~0 lines)", 0),
    ],
)
def test_estimate_forms_are_accepted(raw: str, expected: int) -> None:
    assert parse_step_estimate(raw) == expected


def test_bare_line_counts_are_not_estimates() -> None:
    assert parse_step_estimate("20 行のデータ表を作る") is None
    assert parse_step_estimate("a 20-line table") is None
    assert parse_step_estimate("(20 行のデータ表)") is None


def test_several_estimates_in_one_step_take_the_max() -> None:
    assert (
        parse_step_estimate("index.html (~120 lines) - header (~40 lines) - body (~80 lines)")
        == 120
    )


def test_estimate_survives_truncation() -> None:
    long = "x" * 400 + " (~120 lines)"
    steps = parse_plan_steps(f"## Steps\n1. {long}\n2. verify (~0 lines)\n")
    assert [s.estimated_lines for s in steps] == [120, 0]
    assert steps[0].text.endswith("…") and len(steps[0].text) == 300


def test_parse_steps_is_unchanged_and_single_step_plans_are_only_in_parse_plan_steps() -> None:
    assert parse_steps(PLAN_SPLIT) == [s.text for s in parse_plan_steps(PLAN_SPLIT)]
    assert [s.estimated_lines for s in parse_plan_steps(PLAN_SPLIT)] == [100, 100, 100, 0]
    one = "## Steps\n1. 全部を 1 回で書く (~600 lines)\n"
    assert parse_steps(one) == []
    assert parse_plan_steps(one) == [PlanStep("全部を 1 回で書く (~600 lines)", 600)]
    assert parse_plan_steps("# nothing\n\n- a\n") == []


def _sample(rate: float, tokens: int = 1000) -> tuple[int, float]:
    return tokens, tokens / rate


def test_budget_math_with_the_measured_models() -> None:
    assert compute_plan_budget(120.0, [_sample(66.9)]).lines == 330
    assert compute_plan_budget(120.0, [_sample(57.5)]).lines == 280
    assert compute_plan_budget(120.0, [_sample(18.1)]).lines == 90
    budget = compute_plan_budget(120.0, [_sample(57.5)])
    assert (budget.source, budget.samples, budget.tokens) == ("measured", 1, 3450)
    assert "58 tokens/s measured in this conversation" in budget.explain_en()
    assert "3,450 tokens ≈ 280 lines" in budget.explain_en()
    assert budget.explain_card() == (
        "1 段の上限 280 行 ＝ この会話で計測した 58 tok/s × 120 秒 × 安全率 0.5（≈ 3,450 トークン）"
    )


def test_rate_is_the_median_over_responses_of_at_least_300_tokens() -> None:
    samples = [(100, 1.0), (300, 10.0), (600, 10.0), (900, 10.0)]
    assert measured_rate(samples) == (60.0, 3)
    assert measured_rate([(100, 1.0)]) is None
    assert measured_rate([(600, 0.0)]) is None


def test_no_samples_falls_back_to_the_default_and_says_so() -> None:
    budget = compute_plan_budget(120.0, [])
    assert (budget.source, budget.rate_tok_s, budget.lines) == ("default", DEFAULT_RATE_TOK_S, 100)
    assert "default; not yet measured" in budget.explain_en()
    assert "まだ未計測" in budget.explain_card()


def test_pin_overrides_measurement() -> None:
    budget = compute_plan_budget(120.0, [_sample(57.5)], pin=150)
    assert (budget.source, budget.lines) == ("pinned", 150)
    assert "pinned by settings" in budget.explain_en() and "設定で固定" in budget.explain_card()


def test_lines_are_rounded_down_and_clamped() -> None:
    assert lines_for(3450) == 280
    assert lines_for(100) == chunking.STEP_BUDGET_MIN_LINES
    assert lines_for(10_000) == chunking.STEP_BUDGET_MAX_LINES


_BUDGET = PlanBudget(120.0, 57.5, "measured", 1, 3450, 280)


def test_verdict_over_budget_names_each_step_with_numbers() -> None:
    steps = parse_plan_steps(PLAN_BUNDLED)
    verdict = check_plan_budget(steps, _BUDGET)
    assert verdict is not None and verdict.kind == "over" and verdict.over == ((1, 400),)
    note = verdict.note(".satsuma/plans/x.md")
    assert "Step 1 is estimated at 400 lines; the limit per step is 280 lines" in note
    assert "58 tokens/s measured in this conversation" in note
    assert "write_file for the first part, append_file for the rest" in note
    assert "`.satsuma/plans/x.md`" in note and note.endswith("call exit_plan_mode again.")
    assert verdict.summary() == "段 1 が予算超過（400 行 > 280 行）"

    two = check_plan_budget([PlanStep("a", 400), PlanStep("b", 100), PlanStep("c", 350)], _BUDGET)
    assert two is not None and two.over == ((1, 400), (3, 350))
    assert "Steps 1 (400 lines) and 3 (350 lines) exceed the limit per step" in two.note("p.md")
    assert two.summary() == "段 1・3 が予算超過（400 行・350 行 > 280 行）"


def test_missing_estimates_count_as_zero_and_all_missing_is_sent_back_once() -> None:
    assert check_plan_budget(parse_plan_steps(PLAN_SPLIT), _BUDGET) is None
    assert check_plan_budget([PlanStep("a", 100), PlanStep("b", None)], _BUDGET) is None
    verdict = check_plan_budget(parse_plan_steps(PLAN_NO_ESTIMATES), _BUDGET)
    assert verdict is not None and verdict.kind == "no_estimates"
    assert verdict.sends_back and verdict.sendback_limit == 1
    assert "(~0 lines) for steps that only read, run or verify" in verdict.note("p.md")
    assert verdict.summary() == "段に見積もり (~N lines) が無い（上限 280 行）"
    assert verdict.card().endswith("検算はできないので承認を求めます")


def test_no_steps_section_is_recorded_not_sent_back() -> None:
    verdict = check_plan_budget([], _BUDGET)
    assert verdict is not None and verdict.kind == "no_steps" and not verdict.sends_back
    assert verdict.sendback_limit == 0
    assert 'Add a "## Steps" section' in verdict.note("p.md")
    assert verdict.summary() == "## Steps 節が無い（上限 280 行）"
    assert "段ごとには渡しません" in verdict.card()
    over = check_plan_budget([PlanStep("a", 400)], _BUDGET)
    assert over is not None and over.sends_back and over.sendback_limit == 2
    assert over.card().endswith("承認を求めます")


def test_request_texts_are_selected_by_the_flags() -> None:
    on = ensure_plan_request("作って", "p.md", steps=True, budget=True)
    off = ensure_plan_request("作って", "p.md", steps=True, budget=False)
    assert PLAN_STEPS_REQUEST in on and PLAN_STEPS_REQUEST_LEGACY not in on
    assert PLAN_STEPS_REQUEST_LEGACY in off and PLAN_STEPS_REQUEST not in off
    assert "tokens/s" not in PLAN_STEPS_REQUEST


def test_handoff_notes_carry_the_budget_only_when_given() -> None:
    with_budget = first_step_note(3, "a", "p.md", budget_lines=280)
    assert "Keep each write_file/append_file call under ~280 lines" in with_budget
    assert with_budget.index("Do ONLY step 1 of 3") < with_budget.index("under ~280 lines")
    assert "Keep each" not in first_step_note(3, "a", "p.md")
    assert "under ~280 lines" in next_step_note(2, 3, "b", budget_lines=280)
    assert next_step_note(2, 3, "b", budget_lines=None) == next_step_note(2, 3, "b")


class _Approve(DenyPrompter):
    approvals = 0

    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        self.approvals += 1
        return PlanExitVerdict(approved=True)


class _ApproveAware(_Approve):
    """承認カードに予算を出せるプロンプタ（REPL の InteractivePrompter と同じ境界）。"""

    def __init__(self) -> None:
        self.seen: list[tuple[PlanBudget, int, BudgetVerdict | None]] = []

    def set_plan_budget(
        self, budget: PlanBudget, steps: Any, verdict: BudgetVerdict | None
    ) -> None:
        self.seen.append((budget, len(steps), verdict))


def _runtime(
    provider: object, tmp_path: Path, *, cfg: RuntimeConfig | None = None
) -> ConversationRuntime:
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=PermissionResolver(default_policy(WW), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg or RuntimeConfig(),
    )
    rt.set_mode(main="plan")
    (tmp_path / ".satsuma" / "plans").mkdir(parents=True, exist_ok=True)
    return rt


def _write_plan(rt: ConversationRuntime, text: str, *, id: str = "w1") -> list[AssistantEvent]:
    """計画ファイルを書く応答。"""
    return [
        ToolUse(id=id, name="write_file", input={"path": rt.plan_file_relpath(), "content": text}),
        Usage(output_tokens=600),
        MessageStop(),
    ]


def _exit(id: str = "x1") -> list[AssistantEvent]:
    return [ToolUse(id=id, name="exit_plan_mode", input={}), MessageStop()]


def _say(text: str) -> list[AssistantEvent]:
    return [TextDelta(text=text), MessageStop()]


def _blocks(rt: ConversationRuntime) -> dict[str, ToolResultBlock]:
    return {
        b.tool_use_id: b
        for m in rt.session.messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }


def _reminders(rt: ConversationRuntime) -> list[str]:
    return [
        b.text
        for m in rt.session.messages[1:]
        if m.role == "user"
        for b in m.blocks
        if isinstance(b, TextBlock) and "<system-reminder>" in b.text
    ]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """時計を 1 呼び出し 10 秒進める（応答 1 回 = 開始と終端の 2 呼び出し = 10 秒）。"""
    ticks = iter(range(0, 1_000_000, 10))
    monkeypatch.setattr(
        runtime_module, "time", SimpleNamespace(monotonic=lambda: float(next(ticks)))
    )


async def test_over_budget_plan_is_sent_back_before_the_card(tmp_path: Path, clock: None) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN_BUNDLED),
        _exit("x1"),
        _write_plan(rt, PLAN_SPLIT, id="w2"),
        _exit("x2"),
        _say("段 1 done"),
    ]
    prompter = _ApproveAware()
    summary = await rt.run_turn("最先端のダッシュボードを作って", prompter)

    budget = rt.last_plan_budget
    assert budget is not None and budget.source == "measured" and budget.samples == 2
    assert budget.rate_tok_s == 60.0 and budget.lines == 300
    sent_back = _blocks(rt)["x1"].output
    assert "Step 1 is estimated at 400 lines; the limit per step is 300 lines" in sent_back
    assert "60 tokens/s measured in this conversation" in sent_back
    assert f"`{rt.plan_file_relpath()}`" in sent_back
    assert prompter.approvals == 1 and summary.plan_budget_sendbacks == 1
    assert prompter.seen == [(budget, 4, None)]
    handed = _blocks(rt)["x2"].output
    assert "Do ONLY step 1 of 4" in handed and "under ~300 lines" in handed
    assert rt.session.mode.main == "code"
    assert summary.plan_budget_lines == 300
    assert summary.plan_steps_run == 4 and len(summary.plan_step_records) == 4
    assert [r.estimated_lines for r in summary.plan_step_records] == [100, 100, 100, 0]
    assert summary.plan_budget_verdicts == ["over"] and not summary.plan_budget_forced
    assert all(r.budget_lines == 300 and r.total == 4 for r in summary.plan_step_records)


async def test_after_two_sendbacks_the_card_is_shown_anyway(tmp_path: Path, clock: None) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN_BUNDLED),
        _exit("x1"),
        _exit("x2"),
        _exit("x3"),
        _say("done"),
    ]
    prompter = _ApproveAware()
    summary = await rt.run_turn("作って", prompter)
    assert summary.plan_budget_sendbacks == MAX_SENDBACKS == 2
    assert prompter.approvals == 1
    assert summary.plan_budget_forced and summary.plan_budget_verdicts == ["over"] * 3
    (budget, count, verdict) = prompter.seen[0]
    assert verdict is not None and verdict.kind == "over" and count == 2
    assert "Exited plan mode" in _blocks(rt)["x3"].output


async def test_sendback_does_not_confuse_the_plan_exit_nudge(tmp_path: Path, clock: None) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN_BUNDLED),
        _exit("x1"),
        _write_plan(rt, PLAN_SPLIT, id="w2"),
        _say("分けました。承認してください。"),
        _exit("x2"),
        _say("done"),
    ]
    prompter = _Approve()
    summary = await rt.run_turn("作って", prompter)
    assert summary.plan_budget_sendbacks == 1 and summary.plan_exit_nudges == 1
    assert prompter.approvals == 1 and rt.session.mode.main == "code"


async def test_sendback_alone_does_not_trigger_the_nudge(tmp_path: Path, clock: None) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN_BUNDLED),
        _exit("x1"),
        _say("どう分けるか相談させてください。"),
    ]
    summary = await rt.run_turn("作って", _Approve())
    assert summary.plan_budget_sendbacks == 1 and summary.plan_exit_nudges == 0
    assert rt.session.mode.main == "plan" and summary.final_text.startswith("どう分ける")


async def test_sendback_respects_the_iteration_budget(tmp_path: Path, clock: None) -> None:
    cfg = RuntimeConfig(max_iterations=1)
    rt = _runtime(SequencedMockProvider([_exit("x1"), _say("done")]), tmp_path, cfg=cfg)
    (tmp_path / rt.plan_file_relpath()).write_text(PLAN_BUNDLED, encoding="utf-8")
    prompter = _ApproveAware()
    summary = await rt.run_turn("作って", prompter)
    assert summary.final_text == "done" and summary.plan_steps_run == 2
    assert rt.session.plan_steps == []
    assert rt._plan_budget_sendbacks == 0 and prompter.approvals == 1
    assert prompter.seen[0][2] is not None and prompter.seen[0][2].kind == "over"


async def test_pinned_budget_is_used_instead_of_the_measurement(
    tmp_path: Path, clock: None
) -> None:
    cfg = RuntimeConfig(guidance=GuidanceConfig(step_budget_lines=500))
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path, cfg=cfg)
    rt.provider._scripts[:] = [_write_plan(rt, PLAN_BUNDLED), _exit("x1"), _say("done")]  # type: ignore[attr-defined]
    prompter = _Approve()
    summary = await rt.run_turn("作って", prompter)
    assert summary.plan_budget_sendbacks == 0 and prompter.approvals == 1
    assert rt.last_plan_budget is not None and rt.last_plan_budget.source == "pinned"
    assert "under ~500 lines" in _blocks(rt)["x1"].output


async def test_step_records_group_write_chunks_by_step(tmp_path: Path, clock: None) -> None:
    plan = "## Steps\n1. out.html の頭 (~50 lines)\n2. out.html の残り (~50 lines)\n"
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, plan),
        _exit("x1"),
        [
            ToolUse(id="o1", name="write_file", input={"path": "out.html", "content": "a\nb\nc\n"}),
            Usage(output_tokens=30),
            MessageStop(),
        ],
        _say("段 1 done"),
        [
            ToolUse(
                id="o2",
                name="append_file",
                input={"path": "out.html", "content": "d\ne\nf\ng\nh\n"},
            ),
            Usage(output_tokens=50),
            MessageStop(),
        ],
        _say("done"),
    ]
    summary = await rt.run_turn("作って", _Approve())
    first, second = summary.plan_step_records
    assert (first.index, first.total, first.estimated_lines, first.budget_lines) == (1, 2, 50, 300)
    assert (first.writes, first.actual_lines, first.max_write_lines) == (1, 3, 3)
    assert (second.index, second.writes, second.actual_lines, second.max_write_lines) == (
        2,
        1,
        5,
        5,
    )
    assert first.max_response_s == 10.0 and first.timeouts == 0 and first.iterations == 2
    assert not first.over_budget
    assert rt.session.plan_steps == [] and rt.session.plan_step_estimates == []
    assert rt.session.plan_budget_lines == 0


def _timeout() -> ProviderError:
    error = ProviderError("OpenAI 互換 API 呼び出しに失敗: timed out", retryable=True)
    error.__cause__ = httpx.ReadTimeout(
        "timed out", request=httpx.Request("POST", "http://localhost:11434")
    )
    return error


async def test_timeout_inside_a_step_is_counted_in_its_record(tmp_path: Path, clock: None) -> None:
    plan = "## Steps\n1. out.html (~50 lines)\n2. verify (~0 lines)\n"
    rt = _runtime(ScriptedFailureProvider([[MessageStop()]]), tmp_path)
    rt.provider._steps[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, plan),
        _exit("x1"),
        _timeout(),
        [
            ToolUse(id="o1", name="write_file", input={"path": "out.html", "content": "a\n"}),
            Usage(output_tokens=10),
            MessageStop(),
        ],
        _say("段 1 done"),
        _say("done"),
    ]
    summary = await rt.run_turn("作って", _Approve())
    assert summary.timeout_recoveries == 1
    first = summary.plan_step_records[0]
    assert (first.timeouts, first.writes, first.actual_lines) == (1, 1, 1)
    assert summary.plan_step_records[1].timeouts == 0


async def test_blocked_step_is_not_recorded(tmp_path: Path, clock: None) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN_SPLIT),
        _exit("x1"),
        _say("BLOCKED: 配色を決めてください"),
    ]
    summary = await rt.run_turn("作って", _Approve())
    assert summary.plan_steps_run == 1 and summary.plan_step_records == []
    assert rt.session.plan_step_index == 1 and rt.session.plan_budget_lines == 300


async def test_plan_without_estimates_is_sent_back_once_then_shown(
    tmp_path: Path, clock: None
) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN_NO_ESTIMATES),
        _exit("x1"),
        _exit("x2"),
        _say("done"),
    ]
    prompter = _ApproveAware()
    summary = await rt.run_turn("調べて", prompter)
    assert "no size estimates" in _blocks(rt)["x1"].output
    assert summary.plan_budget_sendbacks == 1 and summary.plan_budget_forced
    assert summary.plan_budget_verdicts == ["no_estimates", "no_estimates"]
    assert prompter.approvals == 1 and prompter.seen[0][2] is not None
    assert prompter.seen[0][2].kind == "no_estimates"
    assert "Do ONLY step 1 of 2" in _blocks(rt)["x2"].output
    assert [r.estimated_lines for r in summary.plan_step_records] == [None, None]


async def test_plan_without_steps_is_handed_whole_as_before(tmp_path: Path, clock: None) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, "# The plan\n- just do it\n"),
        _exit("x1"),
        _say("done"),
    ]
    prompter = _Approve()
    summary = await rt.run_turn("作って", prompter)
    assert summary.plan_budget_sendbacks == 0 and summary.plan_budget_verdicts == ["no_steps"]
    assert prompter.approvals == 1
    out = _blocks(rt)["x1"].output
    assert "Exited plan mode" in out and "Do ONLY step" not in out
    assert rt.session.plan_steps == [] and summary.plan_step_records == []


async def test_flag_off_keeps_the_previous_behaviour(tmp_path: Path, clock: None) -> None:
    cfg = RuntimeConfig(guidance=GuidanceConfig(step_budget=False))
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path, cfg=cfg)
    rt.provider._scripts[:] = [  # type: ignore[attr-defined]
        _write_plan(rt, PLAN_NO_ESTIMATES),
        _exit("x1"),
        _say("done"),
    ]
    prompter = _ApproveAware()
    summary = await rt.run_turn("調べて", prompter)
    assert summary.plan_budget_sendbacks == 0 and summary.plan_budget_lines == 0
    assert summary.plan_budget_verdicts == [] and not summary.plan_budget_forced
    assert prompter.approvals == 1 and prompter.seen == []
    handed = _blocks(rt)["x1"].output
    assert "Do ONLY step 1 of 2" in handed and "Keep each write_file" not in handed
    assert summary.plan_step_records == [] and rt.last_plan_budget is None
    text = rt.session.messages[0].blocks[0]
    assert isinstance(text, TextBlock) and PLAN_STEPS_REQUEST_LEGACY in text.text


def test_prompter_protocol_is_recognised_by_isinstance() -> None:
    assert isinstance(_ApproveAware(), PlanBudgetAware)
    assert not isinstance(_Approve(), PlanBudgetAware)
    assert isinstance(InteractivePrompter(Console(file=io.StringIO())), PlanBudgetAware)


def test_estimates_round_trip_through_jsonl(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    s = new_session(tmp_path, "m")
    s.mode = ModeState(main="code", sub=WW)
    s.plan_steps, s.plan_step_index = ["a (~100 lines)", "b (~0 lines)"], 1
    s.plan_step_estimates, s.plan_budget_lines = [100, 0], 280
    store.save(s)
    loaded = store.load(store.path_for(tmp_path, s.session_id))
    assert (loaded.plan_step_estimates, loaded.plan_budget_lines) == ([100, 0], 280)
    path = store.path_for(tmp_path, s.session_id)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("type") == "plan_steps":
            row.pop("estimates"), row.pop("budget_lines")
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    old = store.load(path)
    assert old.plan_steps == s.plan_steps and old.plan_step_estimates == []
    assert old.plan_budget_lines == 0


def test_rich_renderer_prints_sendback_and_step_done_lines() -> None:
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=True, width=100, color_system=None))
    r.start_turn()
    r.on_plan_sendback("段 2 が予算超過（400 行 > 280 行）")
    r.on_plan_step_done(
        PlanStepRecord(
            index=2,
            total=7,
            estimated_lines=120,
            budget_lines=280,
            actual_lines=143,
            max_write_lines=143,
            writes=1,
            max_response_s=38.2,
        )
    )
    r.on_plan_step_done(
        PlanStepRecord(index=3, total=7, budget_lines=280, actual_lines=400, max_write_lines=400)
    )
    r.end_turn()
    out = buf.getvalue()
    assert "計画を差し戻し: 段 2 が予算超過（400 行 > 280 行）" in out
    assert "段 2/7 完了: 見積 120 行 / 実績 143 行 / 最大 1 回 143 行 / 最長応答 38 秒" in out
    assert "段 3/7 完了: 見積 なし / 実績 400 行 / 最大 1 回 400 行（予算超過）" in out


def test_approval_card_explains_the_budget() -> None:
    def card(verdict: BudgetVerdict | None) -> str:
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=100, color_system=None)
        console.input = lambda *a, **k: "y"  # type: ignore[method-assign]
        prompter = InteractivePrompter(console, plan_steps=True)
        prompter.set_plan_budget(_BUDGET, parse_plan_steps(PLAN_SPLIT), verdict)
        assert prompter.confirm_plan_exit(PLAN_SPLIT).approved
        return buf.getvalue()

    out = card(None)
    assert "4 段に分けて" in out
    assert "1 段の上限 280 行 ＝ この会話で計測した 58 tok/s × 120 秒 × 安全率 0.5" in out
    assert "最大の段は 1（100 行）" in out
    assert "差し戻し上限" not in out
    forced = card(BudgetVerdict("over", _BUDGET, ((1, 400),)))
    assert (
        "段 1 が予算超過（400 行 > 280 行）のまま。差し戻し上限に達したため承認を求めます" in forced
    )
    assert "段に見積もり (~N lines) が無いまま" in card(BudgetVerdict("no_estimates", _BUDGET))


class _Stepped(EvalAgent):
    name = "satsuma+plan"
    model = "mock"

    async def run(
        self,
        prompt: str,
        workspace: Path,
        max_iterations: int | None = None,
        *,
        permission_mode: str = "read-only",
    ) -> AgentAnswer:
        return AgentAnswer(
            text="ok",
            plan_exits=1,
            plan_step_records=[
                PlanStepRecord(
                    index=1,
                    total=2,
                    estimated_lines=100,
                    budget_lines=280,
                    actual_lines=90,
                    max_write_lines=90,
                    writes=1,
                ),
                PlanStepRecord(
                    index=2,
                    total=2,
                    estimated_lines=100,
                    budget_lines=280,
                    actual_lines=420,
                    max_write_lines=420,
                    writes=1,
                    timeouts=1,
                ),
            ],
            plan_budget_sendbacks=1,
            plan_budget_lines=280,
        )


async def test_step_records_reach_the_task_result_and_old_reports_load(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    task = EvalTask.model_validate({"id": "t", "prompt": "hi", "fixture": str(fixture)})
    report = await EvalRunner(_Stepped()).run([task])
    result = report.results[0]
    assert (result.plan_budget_sendbacks, result.plan_budget_lines) == (1, 280)
    assert [r.over_budget for r in result.plan_step_records] == [False, True]
    dumped = json.loads(report.model_dump_json())
    assert dumped["results"][0]["plan_step_records"][1]["max_write_lines"] == 420
    assert EvalReport.model_validate(dumped).results[0] == result
    old = TaskResult.model_validate({"task_id": "t", "passed": True})
    assert (old.plan_step_records, old.plan_budget_sendbacks, old.plan_budget_lines) == ([], 0, 0)
    kinds = [d.kind for d in detect_defects(result, "satsuma+plan")]
    assert "step over budget" in kinds and "plan send-back limit" not in kinds
    result.plan_budget_forced = True
    assert "plan send-back limit" in [d.kind for d in detect_defects(result, "satsuma+plan")]


def test_ab_summary_folds_step_records(tmp_path: Path) -> None:
    def report(path: Path, results: list[dict[str, object]]) -> Path:
        path.write_text(json.dumps({"agent": "satsuma+plan", "results": results}), encoding="utf-8")
        return path

    before = report(
        tmp_path / "b.json",
        [
            {
                "task_id": "t0",
                "passed": True,
                "plan_budget_sendbacks": 1,
                "plan_step_records": [
                    {"index": 1, "budget_lines": 280, "max_write_lines": 90},
                    {"index": 2, "budget_lines": 280, "max_write_lines": 420},
                ],
            }
        ],
    )
    after = report(tmp_path / "a.json", [{"task_id": "t0", "passed": True}])
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "gen", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 段の予算超過 (段数) | 1.0 | 0.0 | -1.0 |" in out
    assert "| 計画の差し戻し (予算超過) | 1.0 | 0.0 | -1.0 |" in out
