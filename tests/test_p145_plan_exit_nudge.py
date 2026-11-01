"""plan モードで ``exit_plan_mode`` を呼ばずに終えたターンを承認へ繋ぐ。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.reminders import (
    ReminderTrigger,
    empty_final_note,
    empty_final_plan_note,
    plan_exit_nudge_note,
    plan_missing_plan_note,
)
from satsumaestro.engine.runtime import ConversationRuntime, _promises_a_plan
from satsumaestro.engine.turn import DenyPrompter, PlanExitVerdict
from satsumaestro.evals.models import TaskResult
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"

WW = PermissionMode.WORKSPACE_WRITE


class _Approve(DenyPrompter):
    """承認カードで「はい」を押すユーザー（評価の AutoApprovePlanPrompter と同じ振る舞い）。"""

    approvals = 0

    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        self.approvals += 1
        return PlanExitVerdict(approved=True)


class _Revise(DenyPrompter):
    """承認カードで修正を指示するユーザー（＝この計画では進めない）。"""

    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        return PlanExitVerdict(approved=False, feedback="段 2 を分けて")


def _runtime(
    tmp_path: Path, *, nudge: bool = True
) -> tuple[ConversationRuntime, list[list[AssistantEvent]]]:
    """plan モードの runtime と、**後から差し替えられる**台本のリストを返す。"""
    config = RuntimeConfig()
    config.guidance.plan_exit_nudge = nudge
    config.guidance.step_budget = False
    script: list[list[AssistantEvent]] = [[MessageStop()]]
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(script),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=PermissionResolver(default_policy(WW), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config,
    )
    rt.set_mode(main="plan")
    (tmp_path / ".satsuma" / "plans").mkdir(parents=True)
    return rt, script


def _write_plan(rt: ConversationRuntime, *, id: str = "p1") -> ToolUse:
    return ToolUse(
        id=id,
        name="write_file",
        input={"path": rt.plan_file_relpath(), "content": "# 調査計画\n\n1. 読む\n2. 答える\n"},
    )


def _reminders(rt: ConversationRuntime) -> list[str]:
    """モデルへ差し込んだ ``<system-reminder>`` の本文（ターン内の注入だけ）。"""
    return [
        b.text
        for m in rt.session.messages[1:]
        if m.role == "user"
        for b in m.blocks
        if isinstance(b, TextBlock) and "<system-reminder>" in b.text
    ]


def _nudged(rt: ConversationRuntime) -> bool:
    note = plan_exit_nudge_note(rt.plan_file_relpath())
    return any(note in text for text in _reminders(rt))


async def test_plan_written_then_prose_approval_is_handed_back_to_the_card(
    tmp_path: Path,
) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [ToolUse(id="g1", name="glob_search", input={"pattern": "**/*.py"}), MessageStop()],
        [_write_plan(rt), MessageStop()],
        [
            TextDelta(text="以下のプランを作成しました。ご確認のうえ、実行を許可してください。"),
            MessageStop(),
        ],
        [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
        [TextDelta(text="計画どおり実施しました。"), MessageStop()],
    ]
    prompter = _Approve()
    summary = await rt.run_turn("この関数の呼び出し元を調べて", prompter)

    assert summary.plan_exit_nudges == 1
    assert _nudged(rt) is True
    assert prompter.approvals == 1
    assert rt.session.mode.main == "code"
    assert summary.final_text == "計画どおり実施しました。"


async def test_plan_written_then_direct_answer_is_also_handed_back(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [_write_plan(rt), MessageStop()],
        [TextDelta(text="`Cart.total` は `apply_discount` を使っています。"), MessageStop()],
        [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
        [TextDelta(text="上記が回答です。"), MessageStop()],
    ]
    summary = await rt.run_turn("呼び出し関係を教えて", _Approve())

    assert summary.plan_exit_nudges == 1
    assert _nudged(rt) is True


async def test_read_only_question_answered_without_a_plan_is_left_alone(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    (tmp_path / "cart.py").write_text("def total(): ...\n", encoding="utf-8")
    script[:] = [
        [ToolUse(id="g1", name="grep_search", input={"pattern": "with_tax"}), MessageStop()],
        [ToolUse(id="r1", name="read_file", input={"path": "cart.py"}), MessageStop()],
        [TextDelta(text="`with_tax` を呼んでいるのは `Cart.total` です。"), MessageStop()],
    ]
    summary = await rt.run_turn("with_tax の呼び出し元は？", _Approve())

    assert summary.plan_exit_nudges == 0
    assert _reminders(rt) == []
    assert rt.session.mode.main == "plan"


async def test_denied_write_without_a_plan_file_asks_for_the_plan_first(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [
            ToolUse(
                id="w1",
                name="write_file",
                input={"path": "dashboards/index.html", "content": "<html>"},
            ),
            MessageStop(),
        ],
        [TextDelta(text="プランモードのため直接書けません。ご承認いただけますか？"), MessageStop()],
        [_write_plan(rt), MessageStop()],
        [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
        [TextDelta(text="作成しました。"), MessageStop()],
    ]
    summary = await rt.run_turn("ダッシュボードを作って", _Approve())

    assert summary.plan_exit_nudges == 1
    assert any(plan_missing_plan_note(rt.plan_file_relpath()) in t for t in _reminders(rt))
    assert _nudged(rt) is False
    assert rt.session.mode.main == "code"


async def test_exit_plan_mode_written_as_the_last_line_is_handed_back(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    (tmp_path / rt.plan_file_relpath()).write_text("# plan", encoding="utf-8")
    script[:] = [
        [
            TextDelta(
                text="Please approve the plan to proceed with the implementation.\nexit_plan_mode"
            ),
            MessageStop(),
        ],
        [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    summary = await rt.run_turn("進めて", _Approve())

    assert summary.plan_exit_nudges == 1
    assert _nudged(rt) is True


async def test_declaring_the_call_without_making_it_is_handed_back(tmp_path: Path) -> None:
    """何も書かないまま「これから呼びます」と宣言して終える形（A/B で観測）。"""
    rt, script = _runtime(tmp_path)
    script[:] = [
        [ToolUse(id="g1", name="glob_search", input={"pattern": "**/*"}), MessageStop()],
        [
            TextDelta(
                text="まず Plan を作成します。次に Plan を書き込み、`exit_plan_mode` を"
                "呼び出します。以上の手順で進めます。"
            ),
            MessageStop(),
        ],
        [_write_plan(rt), MessageStop()],
        [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
        [TextDelta(text="作成しました。"), MessageStop()],
    ]
    summary = await rt.run_turn("ダッシュボードを作って", _Approve())

    assert summary.plan_exit_nudges == 1
    assert any(plan_missing_plan_note(rt.plan_file_relpath()) in t for t in _reminders(rt))
    assert rt.session.mode.main == "code"


async def test_no_nudge_after_the_user_declined_the_card(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [_write_plan(rt), MessageStop()],
        [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
        [TextDelta(text="どの段を分けますか？"), MessageStop()],
    ]
    summary = await rt.run_turn("計画を立てて", _Revise())

    assert summary.plan_exit_nudges == 0
    assert rt.session.mode.main == "plan"


async def test_nudge_returns_after_the_plan_is_revised(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [_write_plan(rt), MessageStop()],
        [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
        [_write_plan(rt, id="p2"), MessageStop()],
        [TextDelta(text="直しました。ご確認ください。"), MessageStop()],
        [TextDelta(text="お返事をお待ちします。"), MessageStop()],
    ]
    summary = await rt.run_turn("計画を立てて", _Revise())

    assert summary.plan_exit_nudges == 1
    assert _nudged(rt) is True


async def test_nudge_fires_at_most_once_per_turn(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [_write_plan(rt), MessageStop()],
        [TextDelta(text="ご承認ください。"), MessageStop()],
        [TextDelta(text="やはりご承認ください。"), MessageStop()],
    ]
    summary = await rt.run_turn("計画を立てて", _Approve())

    assert summary.plan_exit_nudges == 1
    assert summary.final_text == "やはりご承認ください。"
    assert len(_reminders(rt)) == 1


async def test_nudge_does_not_consume_the_empty_final_retry(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [_write_plan(rt), MessageStop()],
        [MessageStop()],
        [MessageStop()],
        [TextDelta(text="最終回答です。"), MessageStop()],
    ]
    summary = await rt.run_turn("計画を立てて", _Approve())

    notes = _reminders(rt)
    assert summary.plan_exit_nudges == 1
    assert plan_exit_nudge_note(rt.plan_file_relpath()) in notes[0]
    assert empty_final_plan_note(rt.plan_file_relpath()) in notes[1]
    assert empty_final_note() not in notes[1]
    assert summary.final_text == "最終回答です。"


async def test_disabled_keeps_the_previous_behaviour(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path, nudge=False)
    script[:] = [
        [_write_plan(rt), MessageStop()],
        [TextDelta(text="ご承認ください。"), MessageStop()],
    ]
    summary = await rt.run_turn("計画を立てて", _Approve())

    assert summary.plan_exit_nudges == 0
    assert _reminders(rt) == []
    assert rt.session.mode.main == "plan"


def test_the_two_templates_differ_and_name_the_plan_file() -> None:
    nudge = plan_exit_nudge_note(".satsuma/plans/s.md")
    missing = plan_missing_plan_note(".satsuma/plans/s.md")

    assert ".satsuma/plans/s.md" in nudge and ".satsuma/plans/s.md" in missing
    assert nudge != missing
    assert "does not start the approval" in nudge
    assert "does not ask for approval" in missing
    assert {ReminderTrigger.PLAN_EXIT_NUDGE, ReminderTrigger.PLAN_EXIT_NO_PLAN} <= set(
        ReminderTrigger
    )


@pytest.mark.parametrize(
    "text",
    [
        "承認をお願いします。\n\nexit_plan_mode\n",
        "承認をお願いします。\n\n`exit_plan_mode`\n",
        "その後、exit_plan_mode を呼び出して承認をお願いします。",
        "計画を `.satsuma/plans/s.md` に書き込みます。",
    ],
)
def test_text_forms_that_promise_a_plan(text: str) -> None:
    assert _promises_a_plan(text, ".satsuma/plans/s.md") is True


@pytest.mark.parametrize(
    "text",
    ["", "`Cart.total` は `apply_discount` を使っています。", "プランの承認をお願い致します。"],
)
def test_text_forms_that_do_not_promise_a_plan(text: str) -> None:
    assert _promises_a_plan(text, ".satsuma/plans/s.md") is False


def test_old_report_without_the_key_still_loads() -> None:
    assert TaskResult.model_validate({"task_id": "t", "passed": True}).plan_exit_nudges == 0


def _report(path: Path, rows: list[tuple[int, int]]) -> Path:
    results = [
        {"task_id": f"t{i}", "passed": True, "plan_exits": exits, "plan_exit_nudges": nudges}
        for i, (exits, nudges) in enumerate(rows)
    ]
    path.write_text(json.dumps({"agent": "satsuma+plan", "results": results}), encoding="utf-8")
    return path


def test_ab_summary_shows_nudges_next_to_plan_exits(tmp_path: Path) -> None:
    before = _report(tmp_path / "b.json", [(0, 0), (0, 0)])
    after = _report(tmp_path / "a.json", [(1, 1), (1, 1)])
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "plan", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "| 計画→承認の回数 (plan_exits) | 0.0 | 2.0 | +2.0 |" in out
    assert "| 承認へ差し戻した回数 | 0.0 | 2.0 | +2.0 |" in out
