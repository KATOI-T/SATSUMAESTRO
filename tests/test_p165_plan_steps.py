"""承認済み計画の段階実行。"""

from __future__ import annotations

import io
import json
from pathlib import Path

from rich.console import Console

from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.cli.render import RichRenderer
from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.plan_request import ensure_plan_request
from satsumaestro.engine.plan_steps import (
    PLAN_STEPS_REQUEST,
    PLAN_STEPS_REQUEST_LEGACY,
    first_step_note,
    is_blocked_reply,
    next_step_note,
    parse_steps,
    plan_steps_request,
    wants_resume,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter, NullRenderer, PlanExitVerdict
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ModeState, Session, TextBlock, ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util import chunking
from tests.fixtures.mock_provider import SequencedMockProvider

WW = PermissionMode.WORKSPACE_WRITE

PLAN = """# The plan

Background text that is not a step.

## Steps

1. Create dashboards/index.html with the page skeleton
2. Append the CSS block
3. Append the JavaScript block
"""


def _runtime(
    provider: object, tmp_path: Path, *, cfg: RuntimeConfig | None = None
) -> ConversationRuntime:
    if cfg is None:
        cfg = RuntimeConfig(guidance=GuidanceConfig(step_budget=False))
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(planning_guidance=True),
        policy=PermissionResolver(default_policy(WW), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg,
    )


def _write_plan(rt: ConversationRuntime, tmp_path: Path, text: str = PLAN) -> Path:
    path = tmp_path / rt.plan_file_relpath()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _blocks(rt: ConversationRuntime) -> dict[str, ToolResultBlock]:
    return {
        b.tool_use_id: b
        for m in rt.session.messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }


def _injected(rt: ConversationRuntime) -> list[str]:
    """合成 user メッセージとして積まれた system-reminder の本文。"""
    return [
        block.text
        for message in rt.session.messages
        if message.role == "user"
        for block in message.blocks
        if isinstance(block, TextBlock) and "<system-reminder>" in block.text
    ]


def _user_texts(rt: ConversationRuntime) -> list[str]:
    return [
        block.text
        for message in rt.session.messages
        if message.role == "user"
        for block in message.blocks
        if isinstance(block, TextBlock)
    ]


class _Approve(DenyPrompter):
    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        return PlanExitVerdict(approved=True)


class _ApproveWithNote(DenyPrompter):
    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        return PlanExitVerdict(approved=True, note="use tabs, not spaces")


def _exit_then(*replies: list[AssistantEvent]) -> SequencedMockProvider:
    """``exit_plan_mode`` を 1 回呼び、以後は与えた応答を順に返す。"""
    return SequencedMockProvider(
        [[ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()], *replies]
    )


def _done(text: str = "done") -> list[AssistantEvent]:
    return [TextDelta(text=text), MessageStop()]


def _read(path: str) -> list[AssistantEvent]:
    """段の中で 1 回ツールを呼ぶ応答（空振り差し戻しを起こさないため）。"""
    return [ToolUse(id=f"r{path}", name="read_file", input={"path": path}), MessageStop()]


def _step_work(tag: str, summary: str = "done") -> list[list[AssistantEvent]]:
    """1 段ぶんの振る舞い: ツールを 1 回呼んでから 1 行で報告して止まる。"""
    return [_read(tag), _done(summary)]


def test_parses_numbered_bullet_and_checkbox_items() -> None:
    for body in ("1. one\n2. two\n3. three", "- one\n- two\n- three", "* one\n* two\n* three"):
        assert parse_steps(f"## Steps\n{body}\n") == ["one", "two", "three"]
    assert parse_steps("## Steps\n- [ ] one\n- [x] two\n") == ["one", "two"]
    assert parse_steps("## Steps\n1) one\n2) two\n") == ["one", "two"]


def test_sub_items_are_ignored_and_wrapped_lines_are_joined() -> None:
    plan = "## Steps\n1. write the file\n   with the header block\n   - sub detail\n2. second\n"
    assert parse_steps(plan) == ["write the file with the header block", "second"]


def test_section_ends_at_the_next_heading_of_same_or_shallower_depth() -> None:
    plan = "## Steps\n1. one\n2. two\n\n## Risks\n- not a step\n"
    assert parse_steps(plan) == ["one", "two"]


def test_heading_variants_are_accepted() -> None:
    assert parse_steps("### step\n- a\n- b\n") == ["a", "b"]
    assert parse_steps("## 手順\n- a\n- b\n") == ["a", "b"]
    assert parse_steps("## 3. Implementation steps\n- a\n- b\n") == ["a", "b"]


def test_falls_back_to_sub_headings_when_there_are_no_bullets() -> None:
    plan = "## Steps\n\n### Step 1: add the tool\nIt goes in tools/.\n\n### Step 2: wire it up\n"
    assert parse_steps(plan) == ["Step 1: add the tool It goes in tools/.", "Step 2: wire it up"]


def test_sub_headings_win_over_the_bullets_beneath_them() -> None:
    plan = (
        "## Steps\n\n### Step 1: skeleton\n- create index.html\n- link the css\n\n"
        "### Step 2: styles\n- variables\n#### detail\n- grid\n"
    )
    assert parse_steps(plan) == [
        "Step 1: skeleton - create index.html - link the css",
        "Step 2: styles - variables #### detail - grid",
    ]


def test_real_plans_split_at_the_headings_the_model_wrote() -> None:
    plans = Path(__file__).parent / "fixtures" / "plans"
    seven = parse_steps((plans / "issue210_steps_heading_7.md").read_text(encoding="utf-8"))
    assert len(seven) == 7
    assert seven[0].startswith("Step 1: プロジェクト構造を作成 - `dashboard/`")
    assert seven[-1].startswith("Step 7: テストと最適化") and "実行しますか" not in seven[-1]
    five = parse_steps((plans / "issue210_jissou_step_5.md").read_text(encoding="utf-8"))
    assert [s.split(":")[0] for s in five] == [f"ステップ {n}" for n in range(1, 6)]


def test_a_rule_between_sub_headings_is_skipped_not_a_terminator() -> None:
    plan = "## Steps\n### A\nfirst\n\n---\n\n### B\nsecond\n\n---\n\nShall I proceed?\n"
    assert parse_steps(plan) == ["A first", "B second"]


def test_bullets_stay_the_steps_when_headings_cannot_be() -> None:
    assert parse_steps("## Steps\n### Order\n- a\n- b\n") == ["a", "b"]
    plan = "## Steps\n1. a\n2. b\n### Notes\n- n1\n### Risks\n- r1\n"
    assert parse_steps(plan) == ["a", "b"]


def test_long_steps_are_cut() -> None:
    [only] = parse_steps(f"## Steps\n- {'x' * 500}\n- second\n")[:1]
    assert len(only) == 300 and only.endswith("…")


def test_missing_heading_or_single_step_means_no_stepping() -> None:
    assert parse_steps("# The plan\n- do everything\n- and more\n") == []
    assert parse_steps("## Steps\n1. only one\n") == []
    assert parse_steps(None) == [] and parse_steps("") == []


def test_an_empty_heading_does_not_shadow_the_real_steps() -> None:
    """空の見出しで確定すると段階実行が黙って無効になる（実機 3 回中 2 回）。"""
    plan = "## ステップ\n\n## Steps\n1. one\n2. two\n"
    assert parse_steps(plan) == ["one", "two"]
    assert parse_steps("## Steps\n1. one\n2. two\n\n## ステップ\n") == ["one", "two"]


def test_a_single_step_section_does_not_shadow_a_real_one() -> None:
    plan = "## 手順\n1. only one\n\n## Steps\n1. one\n2. two\n3. three\n"
    assert parse_steps(plan) == ["one", "two", "three"]


def test_exact_titles_are_still_tried_before_loose_ones() -> None:
    plan = "## Next steps\n- loose a\n- loose b\n\n## Steps\n- exact a\n- exact b\n"
    assert parse_steps(plan) == ["exact a", "exact b"]
    empty_exact = "## Steps\n\n## Next steps\n- loose a\n- loose b\n"
    assert parse_steps(empty_exact) == ["loose a", "loose b"]


def test_no_section_has_steps_still_means_no_stepping() -> None:
    assert parse_steps("## ステップ\n\n## Steps\n\n## Risks\n- r\n") == []


def test_the_plan_that_broke_stepping_in_the_repl() -> None:
    """実機（gpt-oss:120b）の計画ファイルそのもの。"""
    plans = Path(__file__).parent / "fixtures" / "plans"
    text = (plans / "issue228_empty_heading_before_steps.md").read_text(encoding="utf-8")
    steps = parse_steps(text)
    assert len(steps) == 6
    assert steps[0].startswith("**プロジェクト構造の作成**")
    assert steps[-1].startswith("**最終調整と文書化**")


def test_later_steps_say_not_to_redo_what_an_earlier_step_did() -> None:
    hint = "If an earlier step already did this, do not redo or polish it"
    assert hint in next_step_note(2, 3, "styles")
    last = next_step_note(3, 3, "verify")
    assert hint in last and "final answer to the original request" in last
    assert hint not in first_step_note(3, "skeleton", ".satsuma/plans/x.md")


def test_resume_and_blocked_vocabulary() -> None:
    assert wants_resume("続けて") and wants_resume("再開して") and wants_resume("continue")
    assert wants_resume("進めてください")
    assert wants_resume("許可します") and wants_resume("いいよ")
    assert not wants_resume("続けるべきですか？")
    assert not wants_resume("ログイン画面のバグを直して")
    assert not wants_resume("作成したファイルを削除してください")
    assert not wants_resume("続けないで") and not wants_resume("まだ待って")
    assert not wants_resume("不許可です")
    assert is_blocked_reply("BLOCKED: which database?")
    assert is_blocked_reply("**BLOCKED:** which database?")
    assert not is_blocked_reply("I am blocked on nothing")


async def test_approved_plan_is_handed_one_step_at_a_time(tmp_path: Path) -> None:
    rt = _runtime(
        _exit_then(*_step_work("a"), *_step_work("b"), *_step_work("c", "final")), tmp_path
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("ダッシュボードを作って", prompter=_Approve())

    first = _blocks(rt)["x1"].output
    assert "Do ONLY step 1 of 3" in first and "page skeleton" in first
    assert "Implement the plan" not in first
    notes = _injected(rt)
    assert "Now do ONLY step 2 of 3" in notes[0] and "CSS block" in notes[0]
    assert "Now do the last step (3 of 3)" in notes[1] and "JavaScript block" in notes[1]
    assert "write the final answer to the original request" in notes[1]
    assert "Reply in Japanese." in notes[0]
    assert summary.final_text == "final"
    assert summary.plan_steps_run == 3
    assert rt.session.plan_steps == [] and rt.session.plan_step_index == 0


async def test_note_on_approval_is_kept_but_does_not_say_implement_the_plan(
    tmp_path: Path,
) -> None:
    rt = _runtime(_exit_then(_done("stop")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_ApproveWithNote())
    out = _blocks(rt)["x1"].output
    assert "User note for implementation" in out and "use tabs" in out
    assert "Do ONLY step 1 of 3" in out
    assert "Implement the plan" not in out


async def test_blocked_reply_stops_and_keeps_the_step(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(_done("BLOCKED: which database?")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())
    assert summary.final_text == "BLOCKED: which database?"
    assert rt.session.plan_steps and rt.session.plan_step_index == 1
    assert _injected(rt) == []


async def test_unrelated_request_after_blocked_does_not_resume(tmp_path: Path) -> None:
    rt = _runtime(
        _exit_then(
            _done("BLOCKED: dashboard/ を作る必要があります。許可してください。"),
            _done("削除するファイルはありません"),
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    second = await rt.run_turn("作成したファイルを削除してください")
    assert "Resuming step" not in "".join(_user_texts(rt))
    assert second.final_text == "削除するファイルはありません"
    assert rt.session.plan_steps and rt.session.plan_step_index == 1


async def test_answer_only_after_blocked_does_not_resume_until_asked(tmp_path: Path) -> None:
    rt = _runtime(
        _exit_then(
            _done("BLOCKED: which database?"),
            _done("了解しました"),
            _read("a"),
            _read("b"),
            _done("final"),
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    await rt.run_turn("postgres で")
    assert "Resuming step" not in "".join(_user_texts(rt))
    assert rt.session.plan_step_index == 1
    third = await rt.run_turn("続けて")
    resumed = [t for t in _user_texts(rt) if "Resuming step 1 of 3" in t]
    assert len(resumed) == 1 and "messages since you stopped" in resumed[0]
    assert third.final_text == "final" and rt.session.plan_steps == []


async def test_permission_word_after_blocked_resumes(tmp_path: Path) -> None:
    rt = _runtime(
        _exit_then(
            _done("BLOCKED: dashboard/ を作ります。許可してください。"),
            _read("a"),
            _read("b"),
            _done("final"),
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    second = await rt.run_turn("許可します")
    assert "Resuming step 1 of 3" in "".join(_user_texts(rt))
    assert second.final_text == "final" and rt.session.plan_steps == []


async def test_unrelated_message_does_not_resume_pending_steps(tmp_path: Path) -> None:
    rt = _runtime(
        _exit_then(_done("BLOCKED: which db?"), _done("その関数は 3 を返します")), tmp_path
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    await rt.run_turn("ところで foo() は何を返しますか")
    assert "Resuming step" not in "".join(_user_texts(rt))
    assert rt.session.plan_step_index == 1


async def test_explicit_continue_resumes(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(_done("BLOCKED: ?"), _read("a"), _read("b"), _done("final")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    await rt.run_turn("続けて")
    assert "Resuming step 1 of 3" in "".join(_user_texts(rt))


async def test_idle_step_is_resent_once_then_advances(tmp_path: Path) -> None:
    rt = _runtime(
        _exit_then(_done("これから書きます"), _done("やはり書きます"), _read("b"), _done("final")),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())
    notes = _injected(rt)
    assert "without calling any tool" in notes[0] and "step 1 of 3" in notes[0]
    assert "Now do ONLY step 2 of 3" in notes[1]
    assert sum("without calling any tool" in n for n in notes) == 1
    assert summary.plan_steps_run == 3


async def test_idle_step_is_not_resent_once_work_has_started(tmp_path: Path) -> None:
    """段 1 で計画全体を書き切ったモデルは、残りの段で「もう済んでいる」と答える。"""
    rt = _runtime(
        _exit_then(*_step_work("a"), _done("段 2 は段 1 で済んでいます"), _done("final")), tmp_path
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())
    assert not any("without calling any tool" in n for n in _injected(rt))
    assert summary.plan_steps_run == 3 and rt.session.plan_steps == []


async def test_last_step_without_tools_is_not_resent(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(*_step_work("a"), *_step_work("b"), _done("答えです")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())
    assert summary.final_text == "答えです"
    assert not any("without calling any tool" in n for n in _injected(rt))


async def test_mid_step_empty_body_does_not_consume_the_empty_final_latch(
    tmp_path: Path,
) -> None:
    rt = _runtime(_exit_then([MessageStop()], _read("b"), _done("final")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    assert not any("empty body" in n for n in _injected(rt))


async def test_iteration_budget_is_per_step(tmp_path: Path) -> None:
    cfg = RuntimeConfig(max_iterations=2, guidance=GuidanceConfig(step_budget=False))
    rt = _runtime(
        _exit_then(*_step_work("a"), *_step_work("b"), *_step_work("c", "final")), tmp_path, cfg=cfg
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())
    assert summary.final_text == "final" and summary.plan_steps_run == 3
    assert summary.iterations == 7
    assert "Now do ONLY step 2 of 3" in _injected(rt)[0]


async def test_plan_without_steps_section_keeps_the_legacy_wording(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(_done("after")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path, "# The plan\n- just do it\n")
    await rt.run_turn("作って", prompter=_Approve())
    out = _blocks(rt)["x1"].output
    assert "Exited plan mode" in out and "Do ONLY step" not in out
    assert rt.session.plan_steps == []


async def test_flag_off_disables_stepping_and_the_prompt_sentence(tmp_path: Path) -> None:
    cfg = RuntimeConfig(guidance=GuidanceConfig(plan_steps=False))
    rt = _runtime(_exit_then(_done("after")), tmp_path, cfg=cfg)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    assert "Do ONLY step" not in _blocks(rt)["x1"].output
    assert rt.session.plan_steps == []


def test_plan_mode_notice_asks_for_steps_only_when_enabled() -> None:
    on = ensure_plan_request("作って", ".satsuma/plans/x.md", steps=True, budget=True)
    off = ensure_plan_request("作って", ".satsuma/plans/x.md", steps=False, budget=True)
    assert PLAN_STEPS_REQUEST in on and PLAN_STEPS_REQUEST not in off
    asked = ensure_plan_request("計画を立てて", ".satsuma/plans/x.md", steps=True, budget=True)
    assert PLAN_STEPS_REQUEST in asked


def test_the_steps_request_rules_out_the_conversation_itself() -> None:
    """段は「承認後に自分が行う作業」。"""
    assert "after the user approves" in PLAN_STEPS_REQUEST
    assert "you will do with tools" in PLAN_STEPS_REQUEST
    assert "Planning, approval and anything the user does are not steps" in PLAN_STEPS_REQUEST
    assert "(~N lines)" in PLAN_STEPS_REQUEST and "(~0 lines)" in PLAN_STEPS_REQUEST
    assert "checked when you call exit_plan_mode" in PLAN_STEPS_REQUEST
    assert "fewest steps that respect the limit" in PLAN_STEPS_REQUEST
    assert f"more than {chunking.LINES} of new content" in PLAN_STEPS_REQUEST
    assert "tokens/s" not in PLAN_STEPS_REQUEST


def test_the_legacy_request_is_byte_stable() -> None:
    """``guidance.step_budget=false`` で使う予算の門より前の文面（同一バイナリ A/B の before）。"""
    assert "you make in the workspace with tools" in PLAN_STEPS_REQUEST_LEGACY
    assert "Do not list steps for planning, for approval, or for anything the user does" in (
        PLAN_STEPS_REQUEST_LEGACY
    )
    assert "Split the work that produces the output" in PLAN_STEPS_REQUEST_LEGACY
    assert "one large file is not one step" in PLAN_STEPS_REQUEST_LEGACY
    assert "each of those sub-items is a step" in PLAN_STEPS_REQUEST_LEGACY
    assert "append the rest, one step each" in PLAN_STEPS_REQUEST_LEGACY
    assert f"more than {chunking.LINES} of new content" in PLAN_STEPS_REQUEST_LEGACY
    assert plan_steps_request(True, False) == f" {PLAN_STEPS_REQUEST_LEGACY}"
    assert plan_steps_request(True, True) == f" {PLAN_STEPS_REQUEST}"
    assert plan_steps_request(False, True) == ""


async def test_entering_plan_mode_again_drops_pending_steps(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(_done("BLOCKED: ?")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    assert rt.session.plan_steps
    rt.set_mode(main="plan")
    assert rt.session.plan_steps == [] and rt.session.plan_step_index == 0


async def test_entering_plan_mode_during_a_step_does_not_crash(tmp_path: Path) -> None:
    """段の途中で ``enter_plan_mode`` が呼ばれて段の状態が消えても落ちない（実機）。"""
    rt = _runtime(
        SequencedMockProvider(
            [
                [ToolUse(id="p1", name="enter_plan_mode", input={}), MessageStop()],
                _done("新しい計画を立てます"),
            ]
        ),
        tmp_path,
    )
    rt.session.plan_steps = ["a", "b", "c"]
    rt.session.plan_step_index = 2
    summary = await rt.run_turn("続けて")
    assert summary.final_text == "新しい計画を立てます"
    assert rt.session.mode.main == "plan"
    assert rt.session.plan_steps == [] and rt.session.plan_step_index == 0
    assert summary.plan_step_records == []


async def test_clear_command_drops_pending_steps(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(_done("BLOCKED: ?")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って", prompter=_Approve())
    console = Console(record=True, width=120)
    await SlashDispatcher(rt, console).dispatch(parse_slash("/clear"))  # type: ignore[arg-type]
    assert rt.session.plan_steps == [] and rt.session.plan_step_index == 0


async def test_approve_plan_from_text_hands_the_first_step(tmp_path: Path) -> None:
    rt = _runtime(
        SequencedMockProvider([*_step_work("a"), *_step_work("b"), *_step_work("c", "final")]),
        tmp_path,
    )
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    assert rt.approve_plan_from_text("進めてください") is True
    assert rt.session.plan_steps and rt.session.plan_step_index == 0
    summary = await rt.run_turn("進めてください")
    first_user = _user_texts(rt)[0]
    assert "Do ONLY step 1 of 3" in first_user
    assert first_user.index("Do ONLY step 1 of 3") < first_user.index("Reply in Japanese")
    assert summary.plan_steps_run == 3 and rt.session.plan_steps == []


async def test_step_is_handed_even_when_batched_with_another_tool(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="w1", name="write_file", input={"path": "x.md", "content": "hi"}),
                ToolUse(id="x1", name="exit_plan_mode", input={}),
                MessageStop(),
            ],
            *_step_work("a"),
            *_step_work("b"),
            *_step_work("c", "final"),
        ]
    )
    rt = _runtime(provider, tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    summary = await rt.run_turn("作って", prompter=_Approve())
    assert "Do ONLY step 1 of 3" in _blocks(rt)["x1"].output
    assert summary.plan_steps_run == 3


async def test_not_approved_does_not_stage_any_step(tmp_path: Path) -> None:
    rt = _runtime(_exit_then(_done("still planning")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    await rt.run_turn("作って")
    assert rt.session.plan_steps == [] and rt.session.mode.main == "plan"
    assert "Do ONLY step" not in _blocks(rt)["x1"].output


async def test_decomposition_is_skipped_while_stepping(tmp_path: Path) -> None:
    class _Decomposition:
        def __init__(self) -> None:
            self.calls = 0

        async def before_turn(self, user_text: str) -> str | None:
            self.calls += 1
            return None

        def take_record(self) -> None:
            return None

    rt = _runtime(
        SequencedMockProvider([*_step_work("a"), *_step_work("b"), *_step_work("c", "final")]),
        tmp_path,
    )
    rt.decomposition = _Decomposition()  # type: ignore[assignment]
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    assert rt.approve_plan_from_text("進めてください") is True
    await rt.run_turn("進めてください")
    assert rt.decomposition.calls == 0  # type: ignore[union-attr]


def test_steps_round_trip_through_jsonl(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    s = new_session(tmp_path, "m")
    s.plan_steps = ["one", "two"]
    s.plan_step_index = 1
    store.save(s)
    restored = store.load(store.path_for(tmp_path, s.session_id))
    assert restored.plan_steps == ["one", "two"] and restored.plan_step_index == 1


def test_no_row_when_not_stepping_and_old_sessions_still_load(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    s = new_session(tmp_path, "m")
    store.save(s)
    path = store.path_for(tmp_path, s.session_id)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert not any(r.get("type") == "plan_steps" for r in rows)
    restored = store.load(path)
    assert restored.plan_steps == [] and restored.plan_step_index == 0


async def test_renderer_is_told_each_step(tmp_path: Path) -> None:
    class _Recording(NullRenderer):
        def __init__(self) -> None:
            self.seen: list[tuple[int, int, str]] = []

        def on_plan_step(self, index: int, total: int, text: str) -> None:
            self.seen.append((index, total, text))

    rt = _runtime(_exit_then(_read("a"), _read("b"), _read("c"), _done("final")), tmp_path)
    rt.set_mode(main="plan")
    _write_plan(rt, tmp_path)
    render = _Recording()
    await rt.run_turn("作って", prompter=_Approve(), render=render)
    assert [(i, n) for i, n, _ in render.seen] == [(1, 3), (2, 3), (3, 3)]


def test_rich_renderer_prints_the_step_line() -> None:
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=True, width=80, color_system=None))
    r.start_turn()
    r.on_plan_step(2, 3, "Append the CSS block")
    r.end_turn()
    assert "段 2/3: Append the CSS block" in buf.getvalue()


def test_approval_card_announces_the_number_of_steps() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=80, color_system=None)
    console.input = lambda *a, **k: "y"  # type: ignore[method-assign]
    verdict = InteractivePrompter(console, plan_steps=True).confirm_plan_exit(PLAN)
    assert verdict.approved and "3 段に分けて" in buf.getvalue()


def test_approval_card_stays_quiet_when_the_feature_is_off() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=80, color_system=None)
    console.input = lambda *a, **k: "y"  # type: ignore[method-assign]
    InteractivePrompter(console, plan_steps=False).confirm_plan_exit(PLAN)
    assert "段に分けて" not in buf.getvalue()


def test_enter_plan_mode_output_asks_for_a_steps_section() -> None:
    from satsumaestro.tools.builtins.plan_mode import PLAN_EXIT_APPROVED_NOTE_WHILE_STEPPING

    assert "Implement the plan" not in PLAN_EXIT_APPROVED_NOTE_WHILE_STEPPING
    assert "User note for implementation" in PLAN_EXIT_APPROVED_NOTE_WHILE_STEPPING


def test_session_defaults_are_backward_compatible() -> None:
    s = Session(session_id="x", created_at_ms=0, updated_at_ms=0, workspace_root="/tmp", model="m")
    assert s.plan_steps == [] and s.plan_step_index == 0
    assert s.mode == ModeState()
