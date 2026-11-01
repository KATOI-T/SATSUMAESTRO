"""plan / code メインモードと権限サブモード。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli.commands import SlashDispatcher, cycle_main_mode, parse_slash
from satsumaestro.cli.modes import mode_markup, mode_prompt_fragments
from satsumaestro.cli.repl import InteractivePrompter, _permission_prompt, _plan_prompt
from satsumaestro.config.models import GuidanceConfig, HookConfig, PermissionConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.plan_approval import is_plan_approval_text
from satsumaestro.engine.plan_request import (
    PLAN_FILE_EXISTS,
    PLAN_FILE_NEW,
    PLAN_FILE_READ_MISSING,
    PLAN_MODE_SUFFIX,
    PLAN_REQUEST_SUFFIX,
    ensure_plan_request,
    wants_plan,
)
from satsumaestro.engine.reminders import PLAN_MODE_DENY_PREFIX
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import (
    DETAIL_REQUEST_ORIGIN,
    DenyPrompter,
    PermissionDetail,
    PlanExitVerdict,
)
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import (
    ModeState,
    Session,
    ToolResultBlock,
    plan_file_relpath,
    resolve_initial_mode,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.builtins.plan_mode import (
    ENTER_PLAN_DESC_GUIDED,
    ENTER_PLAN_DESC_ON_REQUEST,
    PLAN_EXIT_APPROVED_WITH_NOTE,
    PLAN_EXIT_NO_PLAN,
    PLAN_EXIT_NOT_APPROVED,
)
from tests.fixtures.mock_provider import SequencedMockProvider

WW = PermissionMode.WORKSPACE_WRITE
RO = PermissionMode.READ_ONLY


def _runtime(
    provider: object, tmp_path: Path, *, session: Session | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=session or new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(planning_guidance=True),
        policy=PermissionResolver(default_policy(WW), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(guidance=GuidanceConfig(step_budget=False)),
    )


def _blocks(rt: ConversationRuntime) -> dict[str, ToolResultBlock]:
    out: dict[str, ToolResultBlock] = {}
    for m in rt.session.messages:
        for b in m.blocks:
            if isinstance(b, ToolResultBlock):
                out[b.tool_use_id] = b
    return out


class _Approve(DenyPrompter):
    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        self.seen = plan_text
        return PlanExitVerdict(approved=True)


class _Revise(DenyPrompter):
    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        return PlanExitVerdict(approved=False, feedback="add tests for X")


class _ApproveWithNote(DenyPrompter):
    def confirm_plan_exit(self, plan_text: str | None) -> PlanExitVerdict:
        return PlanExitVerdict(approved=True, note="also add tests; name it greet_user")


def test_mode_state_default_sub_matches_permission_config_default() -> None:
    assert ModeState().sub.value == PermissionConfig().default_mode
    assert ModeState().main == "code"


@pytest.mark.parametrize(
    ("spec", "main", "sub"),
    [
        ("plan", "plan", WW),
        ("code", "code", WW),
        ("code:read-only", "code", RO),
        ("CODE:Danger-Full-Access", "code", PermissionMode.DANGER_FULL_ACCESS),
        ("plan:allow", "plan", PermissionMode.ALLOW),
    ],
)
def test_mode_state_parse(spec: str, main: str, sub: PermissionMode) -> None:
    state = ModeState.parse(spec, WW)
    assert state.main == main and state.sub == sub


@pytest.mark.parametrize("spec", ["foo", "code:bogus", "", "plan:code"])
def test_mode_state_parse_rejects(spec: str) -> None:
    with pytest.raises(ValueError):
        ModeState.parse(spec, WW)


def test_mode_state_parse_empty_sub_uses_default() -> None:
    assert ModeState.parse("code:", RO) == ModeState(main="code", sub=RO)


def test_mode_label() -> None:
    assert ModeState(main="plan", sub=RO).label() == "plan"
    assert ModeState(main="code", sub=RO).label() == "code:read-only"
    assert ModeState().label() == "code:workspace-write"


def test_mode_short_label_omits_default_sub() -> None:
    """接頭辞用: 既定 workspace-write は省略、それ以外の権限は残す。"""
    assert ModeState().short_label() == "code"
    assert ModeState(main="plan", sub=WW).short_label() == "plan"
    assert ModeState(main="plan", sub=RO).short_label() == "plan"
    assert ModeState(main="code", sub=RO).short_label() == "code:read-only"
    assert (
        ModeState(main="code", sub=PermissionMode.DANGER_FULL_ACCESS).short_label()
        == "code:danger-full-access"
    )


def test_mode_prompt_fragments_colors_main_mode() -> None:
    """plan=緑 / code=青。"""
    plan = list(mode_prompt_fragments(ModeState(main="plan")))
    assert plan == [("", "["), ("fg:ansigreen bold", "plan"), ("", "] satsuma> ")]
    code = list(mode_prompt_fragments(ModeState()))
    assert code == [("", "["), ("fg:ansiblue bold", "code"), ("", "] satsuma> ")]
    ro = list(mode_prompt_fragments(ModeState(main="code", sub=RO)))
    assert ro == [
        ("", "["),
        ("fg:ansiblue bold", "code"),
        ("", ":read-only"),
        ("", "] satsuma> "),
    ]


def test_mode_markup_uses_same_colors() -> None:
    assert mode_markup(ModeState(main="plan")) == "[green]plan[/green]"
    assert mode_markup(ModeState()) == "[blue]code[/blue]"
    assert mode_markup(ModeState(), full=True) == "[blue]code[/blue]:workspace-write"
    assert mode_markup(ModeState(main="code", sub=RO)) == "[blue]code[/blue]:read-only"


def test_resolve_initial_mode_precedence() -> None:
    saved = ModeState(main="plan", sub=RO)
    assert resolve_initial_mode(saved, "code", WW) == ModeState(main="code", sub=WW)
    assert resolve_initial_mode(saved, None, WW) == saved
    assert resolve_initial_mode(None, None, RO) == ModeState(main="code", sub=RO)


async def test_set_mode_sub_updates_policy_and_ctx(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="w1", name="write_file", input={"path": "o.txt", "content": "x"})],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    assert rt.session.mode == ModeState(main="code", sub=WW)
    rt.set_mode(sub=RO)
    assert isinstance(rt.policy, PermissionResolver)
    assert rt.policy.base.mode is RO
    assert rt._ctx.base_mode is RO
    await rt.run_turn("write")
    block = _blocks(rt)["w1"]
    assert block.is_error is True and not (tmp_path / "o.txt").exists()


def test_set_mode_plan_keeps_sub_and_code_restores(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    before = rt.session.mode
    rt.set_mode(main="plan")
    assert rt.session.mode.main == "plan" and rt.session.mode.sub == WW
    assert rt.session.mode is not before
    rt.set_mode(main="code")
    assert rt.session.mode == ModeState(main="code", sub=WW)


async def test_plan_file_write_allowed_but_other_writes_denied(tmp_path: Path) -> None:
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="plan", sub=WW)
    plan_rel = plan_file_relpath(session.session_id)
    (tmp_path / ".satsuma" / "plans").mkdir(parents=True)
    (tmp_path / "src.txt").write_text("body", encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="p1", name="write_file", input={"path": plan_rel, "content": "# Plan"}),
                ToolUse(
                    id="p2",
                    name="edit_file",
                    input={"path": f"./{plan_rel}", "old": "Plan", "new": "Plan v2"},
                ),
                ToolUse(
                    id="o1",
                    name="write_file",
                    input={"path": ".satsuma/plans/other.md", "content": "x"},
                ),
                ToolUse(id="o2", name="write_file", input={"path": "src.txt", "content": "x"}),
                ToolUse(id="b1", name="bash", input={"command": "echo hi"}),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, session=session)
    await rt.run_turn("plan")
    by_id = _blocks(rt)
    assert by_id["p1"].is_error is False and by_id["p2"].is_error is False
    assert (tmp_path / plan_rel).read_text(encoding="utf-8") == "# Plan v2"
    for tid in ("o2", "b1"):
        assert by_id[tid].is_error is True and "plan mode is active:" in by_id[tid].output
    assert by_id["o1"].is_error is True and by_id["o1"].output.startswith("write prohibited:")
    assert not (tmp_path / ".satsuma" / "plans" / "other.md").exists()
    assert (tmp_path / "src.txt").read_text(encoding="utf-8") == "body"
    assert rt.session.mode.main == "plan"


async def test_plan_file_exception_only_under_plan(tmp_path: Path) -> None:
    session = new_session(tmp_path, "mock-model")
    plan_rel = plan_file_relpath(session.session_id)
    provider = SequencedMockProvider(
        [
            [ToolUse(id="p1", name="write_file", input={"path": plan_rel, "content": "x"})],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, session=session)
    rt.set_mode(sub=RO)
    await rt.run_turn("write")
    assert _blocks(rt)["p1"].is_error is True


def _with_plan(rt: ConversationRuntime, tmp_path: Path, text: str = "# The plan\n- step") -> None:
    """計画ファイルを置く（承認カードの経路を見るテスト用。無いとカードの前で差し戻す）。"""
    plan_path = tmp_path / plan_file_relpath(rt.session.session_id)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(text, encoding="utf-8")


def _plan_then_exit() -> SequencedMockProvider:
    return SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
            [TextDelta(text="after"), MessageStop()],
        ]
    )


async def test_exit_plan_mode_not_approved_stays_in_plan(tmp_path: Path) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path)
    await rt.run_turn("plan")
    block = _blocks(rt)["x1"]
    assert block.is_error is False
    assert block.output == PLAN_EXIT_NOT_APPROVED
    assert rt.session.mode.main == "plan"
    assert rt._base_mode() is WW


async def test_exit_plan_mode_revise_returns_feedback(tmp_path: Path) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path)
    await rt.run_turn("plan", prompter=_Revise())
    block = _blocks(rt)["x1"]
    assert block.is_error is False
    assert "User feedback:\nadd tests for X" in block.output
    assert plan_file_relpath(rt.session.session_id) in block.output
    assert rt.session.mode.main == "plan"


async def test_exit_plan_mode_approved_restores_code_and_passes_plan_text(
    tmp_path: Path,
) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    plan_path = tmp_path / plan_file_relpath(rt.session.session_id)
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("# The plan\n- step", encoding="utf-8")
    prompter = _Approve()
    await rt.run_turn("plan", prompter=prompter)
    assert prompter.seen == "# The plan\n- step"
    block = _blocks(rt)["x1"]
    assert block.is_error is False and "Exited plan mode" in block.output
    assert "User note for implementation" not in block.output
    assert rt.session.mode == ModeState(main="code", sub=WW)


async def test_exit_plan_mode_approved_with_note_appends_note_to_result(
    tmp_path: Path,
) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path)
    await rt.run_turn("plan", prompter=_ApproveWithNote())
    block = _blocks(rt)["x1"]
    assert block.is_error is False
    assert block.output.startswith("Exited plan mode")
    assert block.output.endswith(
        PLAN_EXIT_APPROVED_WITH_NOTE.format(note="also add tests; name it greet_user")
    )
    assert "User note for implementation:\nalso add tests; name it greet_user" in block.output
    assert rt.session.mode == ModeState(main="code", sub=WW)


async def test_exit_plan_mode_note_dropped_when_tool_fails(tmp_path: Path) -> None:
    from satsumaestro.tools.base import ToolResult

    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path)

    async def _fail(tool_use: object, res: ToolResult) -> ToolResult:
        if getattr(tool_use, "name", "") == "exit_plan_mode":
            return ToolResult(output="hook failed", is_error=True)
        return res

    rt.hooks.post_tool_use = _fail  # type: ignore[method-assign]
    await rt.run_turn("plan", prompter=_ApproveWithNote())
    block = _blocks(rt)["x1"]
    assert block.is_error is True and "User note" not in block.output
    assert rt.session.mode.main == "plan"


PROSE_PLAN = "Plan:\n- add the tool\n- wire it up\n"


async def test_exit_plan_mode_without_plan_file_falls_back_to_last_text(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [TextDelta(text=PROSE_PLAN), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    await rt.run_turn("plan")
    assert rt._read_plan_file() == PROSE_PLAN.strip()


@pytest.mark.parametrize("text", ["...", "計画を作成します。", "Let me write the plan now."])
async def test_filler_text_is_not_taken_as_the_plan(tmp_path: Path, text: str) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [TextDelta(text=text), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    await rt.run_turn("plan")
    assert rt._read_plan_file() is None


async def test_blank_plan_file_counts_as_missing(tmp_path: Path) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    _with_plan(rt, tmp_path, "  \n")
    assert rt._read_plan_file() is None


async def test_exit_without_plan_is_sent_back_once_without_a_card(tmp_path: Path) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    prompter = _Approve()
    await rt.run_turn("plan", prompter=prompter)
    assert not hasattr(prompter, "seen")
    block = _blocks(rt)["x1"]
    assert block.is_error is False
    assert block.output == PLAN_EXIT_NO_PLAN.format(
        plan_file=plan_file_relpath(rt.session.session_id)
    )
    assert "write_file" in block.output and "already exists" in block.output
    assert rt.session.mode.main == "plan"


async def test_exit_after_writing_the_plan_shows_the_card(tmp_path: Path) -> None:
    rt = _runtime(_plan_then_exit(), tmp_path)
    plan_file = plan_file_relpath(rt.session.session_id)
    rt.provider = SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
            [
                ToolUse(
                    id="w1", name="write_file", input={"path": plan_file, "content": "# P\n- a"}
                ),
                MessageStop(),
            ],
            [ToolUse(id="x2", name="exit_plan_mode", input={}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    prompter = _Approve()
    await rt.run_turn("plan", prompter=prompter)
    blocks = _blocks(rt)
    assert blocks["x1"].output.startswith("No plan file yet")
    assert blocks["w1"].is_error is False
    assert prompter.seen == "# P\n- a"
    assert rt.session.mode.main == "code"


async def test_second_exit_without_plan_leaves_it_to_the_user(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
            [ToolUse(id="x2", name="exit_plan_mode", input={}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    prompter = _Approve()
    await rt.run_turn("plan", prompter=prompter)
    assert _blocks(rt)["x1"].output.startswith("No plan file yet")
    assert prompter.seen is None


async def test_enter_plan_mode_says_whether_the_plan_file_exists(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [ToolUse(id="e2", name="enter_plan_mode", input={}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    await rt.run_turn("plan")
    blocks = _blocks(rt)
    assert PLAN_FILE_NEW in blocks["e1"].output and PLAN_FILE_NEW in blocks["e2"].output
    _with_plan(rt, tmp_path)
    rt.provider = SequencedMockProvider(
        [
            [ToolUse(id="e3", name="enter_plan_mode", input={}), MessageStop()],
            [TextDelta(text="ok"), MessageStop()],
        ]
    )
    await rt.run_turn("again")
    assert PLAN_FILE_EXISTS in _blocks(rt)["e3"].output


async def test_plan_mode_deny_says_the_directory_exists(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [
                ToolUse(id="b1", name="bash", input={"command": "mkdir -p .satsuma/plans"}),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    await rt.run_turn("plan")
    out = _blocks(rt)["b1"].output
    assert out.startswith("plan mode is active:") and "directory already exists" in out


async def test_exit_plan_mode_when_not_in_plan_is_noop_without_card(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    prompter = _Approve()
    await rt.run_turn("exit", prompter=prompter)
    assert not hasattr(prompter, "seen")
    assert "not active" in _blocks(rt)["x1"].output


@pytest.mark.parametrize(
    "text",
    [
        "実装に移ってください",
        "実装して",
        "この計画で進めてください",
        "進めて",
        "承認します",
        "OK",
        "はい",
        "それでお願いします",
        "go ahead",
        "Approved, implement it",
        "LGTM",
        "proceed with the plan",
    ],
)
def test_is_plan_approval_text_accepts(text: str) -> None:
    assert is_plan_approval_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "実装するには何が必要ですか？",
        "まだ実装しないで",
        "計画を修正してから進めて",
        "テストの方針を先に見直して",
        "don't implement yet",
        "wait, change step 2 first",
        "認証周りの改修計画を立てて",
        "Tailwind ではなく plain CSS にして",
        "",
        "実装して " + "x" * 100,
    ],
)
def test_is_plan_approval_text_rejects(text: str) -> None:
    assert not is_plan_approval_text(text)


def test_approve_plan_from_text_requires_plan_file(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.set_mode(main="plan")
    assert rt.approve_plan_from_text("実装に移って") is False
    assert rt.session.mode.main == "plan"
    plan_path = tmp_path / plan_file_relpath(rt.session.session_id)
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("# plan", encoding="utf-8")
    assert rt.approve_plan_from_text("ステップ 2 を見直して") is False
    assert rt.session.mode.main == "plan"
    assert rt.approve_plan_from_text("実装に移って") is True
    assert rt.session.mode == ModeState(main="code", sub=WW)
    assert rt._base_mode() is WW
    assert rt.approve_plan_from_text("実装に移って") is False


async def test_approve_plan_from_text_then_writes_succeed(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="w1", name="write_file", input={"path": "o.txt", "content": "x"})],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    rt.set_mode(main="plan")
    plan_path = tmp_path / plan_file_relpath(rt.session.session_id)
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("# plan", encoding="utf-8")
    assert rt.approve_plan_from_text("進めてください") is True
    await rt.run_turn("進めてください")
    assert _blocks(rt)["w1"].is_error is False
    assert (tmp_path / "o.txt").read_text(encoding="utf-8") == "x"


@pytest.mark.parametrize(
    "text",
    [
        "認証周りの実装計画を立てて",
        "プランを出して",
        "段取りを考えて",
        "Make a plan for the migration",
        "Planning: how to split this module",
    ],
)
def test_wants_plan_accepts(text: str) -> None:
    assert wants_plan(text) is True
    out = ensure_plan_request(text, ".satsuma/plans/x.md")
    assert out == text + PLAN_MODE_SUFFIX.format(
        plan_file=".satsuma/plans/x.md", state=PLAN_FILE_NEW, steps=""
    )
    assert "treat this as a request to plan first" not in out


@pytest.mark.parametrize(
    "text", ["モダンな解析ダッシュボードを作成してください", "Create hello.py", "AWSのコスト監視"]
)
def test_ensure_plan_request_appends_suffix(text: str) -> None:
    out = ensure_plan_request(text, ".satsuma/plans/x.md")
    assert out.startswith(text)
    assert out.endswith(
        PLAN_REQUEST_SUFFIX.format(plan_file=".satsuma/plans/x.md", state=PLAN_FILE_NEW, steps="")
    )
    assert "exit_plan_mode" in out and ".satsuma/plans/x.md" in out


@pytest.mark.parametrize(("exists", "expected"), [(False, PLAN_FILE_NEW), (True, PLAN_FILE_EXISTS)])
def test_ensure_plan_request_says_whether_the_plan_file_exists(exists: bool, expected: str) -> None:
    out = ensure_plan_request("ダッシュボードを作って", ".satsuma/plans/x.md", exists=exists)
    other = PLAN_FILE_EXISTS if not exists else PLAN_FILE_NEW
    assert expected in out and other not in out


async def test_plan_request_suffix_tracks_the_plan_file_state(tmp_path: Path) -> None:
    reply = [TextDelta(text="ok"), MessageStop()]
    rt = _runtime(SequencedMockProvider([reply, reply]), tmp_path)

    def last_user_text() -> str:
        users = [m for m in rt.session.messages if m.role == "user"]
        return getattr(users[-1].blocks[0], "text", "")

    rt.set_mode(main="plan")
    await rt.run_turn("ダッシュボードを作って")
    assert PLAN_FILE_NEW in last_user_text()
    _with_plan(rt, tmp_path)
    await rt.run_turn("ダッシュボードを作って")
    assert PLAN_FILE_EXISTS in last_user_text()


async def test_reading_the_unwritten_plan_file_returns_guidance_not_an_error(
    tmp_path: Path,
) -> None:
    """未作成の計画ファイルの read_file は失敗にせず「まず書け」と返す。"""
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="plan", sub=WW)
    plan_rel = plan_file_relpath(session.session_id)
    (tmp_path / ".satsuma" / "plans").mkdir(parents=True)
    (tmp_path / ".satsuma" / "plans" / "other.md").write_text("# 別セッション", encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [ToolUse(id="r1", name="read_file", input={"path": plan_rel}), MessageStop()],
            [
                ToolUse(id="w1", name="write_file", input={"path": plan_rel, "content": "# Plan"}),
                MessageStop(),
            ],
            [ToolUse(id="r2", name="read_file", input={"path": plan_rel}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, session=session)
    await rt.run_turn("plan")
    by_id = _blocks(rt)
    assert by_id["r1"].is_error is False
    assert by_id["r1"].output == PLAN_FILE_READ_MISSING.format(plan_file=plan_rel)
    assert "FileNotFoundError" not in by_id["r1"].output
    assert by_id["r2"].is_error is False and by_id["r2"].output.startswith("# Plan")


async def test_reading_the_plan_file_outside_plan_mode_is_not_intercepted(tmp_path: Path) -> None:
    session = new_session(tmp_path, "mock-model")
    plan_rel = plan_file_relpath(session.session_id)
    provider = SequencedMockProvider(
        [
            [ToolUse(id="r1", name="read_file", input={"path": plan_rel}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, session=session)
    await rt.run_turn("read it")
    block = _blocks(rt)["r1"]
    assert block.is_error is True
    assert PLAN_FILE_READ_MISSING.format(plan_file=plan_rel) not in block.output


def test_eval_protocol_gets_only_the_plan_mode_rules() -> None:
    from satsumaestro.evals.agents import PLAN_EVAL_PROTOCOL

    prompt = "What does foo() return?" + PLAN_EVAL_PROTOCOL
    out = ensure_plan_request(prompt, ".satsuma/plans/x.md")
    assert out.startswith(prompt)
    assert "treat this as a request to plan first" not in out
    assert ".satsuma/plans/x.md" in out


async def test_run_turn_in_plan_augments_model_text_but_keeps_history(tmp_path: Path) -> None:
    reply = [TextDelta(text="ok"), MessageStop()]
    rt = _runtime(SequencedMockProvider([reply, reply]), tmp_path)

    def last_user_text() -> str:
        users = [m for m in rt.session.messages if m.role == "user"]
        return getattr(users[-1].blocks[0], "text", "")

    rt.set_mode(main="plan")
    await rt.run_turn("hello.py を作って")
    assert rt.session.prompt_history[-1] == "hello.py を作って"
    sent = last_user_text()
    assert sent.startswith("hello.py を作って")
    assert rt.plan_file_relpath() in sent
    assert sent.index(rt.plan_file_relpath()) < sent.index("(Reply in Japanese.")
    rt.set_mode(main="code")
    await rt.run_turn("hello.py を作って")
    assert last_user_text().startswith("hello.py を作って")
    assert rt.plan_file_relpath() not in last_user_text()


class _RecordingRenderer:
    def __init__(self) -> None:
        self.starts: list[str] = []
        self.results: list[tuple[str, bool]] = []

    def on_assistant_event(self, ev: object) -> None:
        pass

    def on_tool_start(self, call: ToolUse) -> None:
        self.starts.append(call.name)

    def on_tool_result(self, block: ToolResultBlock) -> None:
        self.results.append((block.tool_name, block.is_error))


async def test_plan_clamp_denies_are_not_rendered(tmp_path: Path) -> None:
    rt = _runtime(
        SequencedMockProvider(
            [
                [
                    ToolUse(id="w1", name="write_file", input={"path": "o.txt", "content": "x"}),
                    ToolUse(id="b1", name="bash", input={"command": "mkdir out"}),
                    ToolUse(id="r1", name="read_file", input={"path": "missing.txt"}),
                    MessageStop(),
                ],
                [MessageStop()],
            ]
        ),
        tmp_path,
    )
    rt.set_mode(main="plan")
    render = _RecordingRenderer()
    await rt.run_turn("計画を立てて", render=render)
    blocks = _blocks(rt)
    assert blocks["w1"].is_error and blocks["w1"].output.startswith(PLAN_MODE_DENY_PREFIX)
    assert blocks["b1"].is_error and blocks["b1"].output.startswith(PLAN_MODE_DENY_PREFIX)
    assert render.starts == ["read_file"]
    assert [name for name, _ in render.results] == ["read_file"]
    assert not (tmp_path / "o.txt").exists()


async def test_plan_file_write_is_still_rendered_in_plan(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.set_mode(main="plan")
    plan = rt.plan_file_relpath()
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="p1", name="write_file", input={"path": plan, "content": "# plan"}),
                MessageStop(),
            ],
            [MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, session=rt.session)
    render = _RecordingRenderer()
    await rt.run_turn("計画を立てて", render=render)
    assert _blocks(rt)["p1"].is_error is False
    assert render.starts == ["write_file"] and render.results == [("write_file", False)]


def test_plan_clamp_predicted(tmp_path: Path) -> None:
    from satsumaestro.session.models import ToolUseBlock

    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)

    def tub(name: str, inp: dict[str, object]) -> ToolUseBlock:
        return ToolUseBlock(id="t", name=name, input=inp)

    write = tub("write_file", {"path": "o.txt", "content": "x"})
    assert rt._plan_clamp_predicted(write) is False
    rt.set_mode(main="plan")
    assert rt._plan_clamp_predicted(write) is True
    assert rt._plan_clamp_predicted(tub("bash", {"command": "ls"})) is True
    assert rt._plan_clamp_predicted(tub("read_file", {"path": "o.txt"})) is False
    plan = tub("write_file", {"path": rt.plan_file_relpath(), "content": "# plan"})
    assert rt._plan_clamp_predicted(plan) is False
    assert rt._plan_clamp_predicted(tub("write_file", {"nope": 1})) is False
    assert rt._plan_clamp_predicted(tub("no_such_tool", {})) is False


def test_store_roundtrip_mode_row(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="code", sub=RO)
    store.save(session)
    path = store.path_for(tmp_path, session.session_id)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {"type": "mode", "main": "code", "sub": "read-only"} in rows
    assert not any(r["type"] == "plan_mode" for r in rows)
    assert store.load(path).mode == session.mode


def test_store_legacy_plan_mode_row_maps_to_mode_state(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    session = new_session(tmp_path, "mock-model")
    path = store.path_for(tmp_path, session.session_id)
    path.parent.mkdir(parents=True)
    meta = {
        "type": "session_meta",
        "version": 1,
        "session_id": session.session_id,
        "created_at_ms": 1,
        "updated_at_ms": 1,
        "workspace_root": str(tmp_path),
        "model": "m",
    }
    legacy = {"type": "plan_mode", "active": True, "previous_mode": "read-only"}
    path.write_text(json.dumps(meta) + "\n" + json.dumps(legacy) + "\n", encoding="utf-8")
    assert store.load(path).mode == ModeState(main="plan", sub=RO)
    legacy2 = {"type": "plan_mode", "active": False, "previous_mode": None}
    path.write_text(json.dumps(meta) + "\n" + json.dumps(legacy2) + "\n", encoding="utf-8")
    assert store.load(path).mode == ModeState()
    both = json.dumps(meta) + "\n" + json.dumps(legacy) + "\n"
    both += json.dumps({"type": "mode", "main": "code", "sub": "allow"}) + "\n"
    path.write_text(both, encoding="utf-8")
    assert store.load(path).mode == ModeState(main="code", sub=PermissionMode.ALLOW)


async def test_mode_command_lists_and_switches(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=160)
    d = SlashDispatcher(rt, console)

    async def run(line: str) -> str:
        await d.dispatch(parse_slash(line))  # type: ignore[arg-type]
        return console.export_text(clear=True)

    assert "mode: code:workspace-write" in await run("/mode")
    await run("/mode code read-only")
    assert rt.session.mode == ModeState(main="code", sub=RO)
    assert rt._base_mode() is RO
    await run("/mode plan")
    assert rt.session.mode == ModeState(main="plan", sub=RO)
    assert "指定" in await run("/mode plan foo")
    assert "指定" in await run("/mode bogus")
    assert "指定" in await run("/mode code bogus")
    assert rt.session.mode == ModeState(main="plan", sub=RO)
    await run("/mode code")
    assert rt.session.mode == ModeState(main="code", sub=RO)
    loaded = rt.store.load(rt.store.path_for(tmp_path, rt.session.session_id))
    assert loaded.mode == rt.session.mode
    await run("/plan on")
    assert rt.session.mode.main == "plan"
    await run("/plan")
    assert rt.session.mode.main == "code"


async def test_sub_mode_command_shows_and_switches(tmp_path: Path) -> None:
    """``/sub-mode``: 接頭辞で省略した権限サブモードの確認と変更。"""
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=160)
    d = SlashDispatcher(rt, console)

    async def run(line: str) -> str:
        await d.dispatch(parse_slash(line))  # type: ignore[arg-type]
        return console.export_text(clear=True)

    out = await run("/sub-mode")
    assert "sub-mode: workspace-write" in out
    assert "未知のコマンド" not in out
    await run("/sub-mode read-only")
    assert rt.session.mode == ModeState(main="code", sub=RO)
    assert rt._base_mode() is RO
    assert "sub-mode: read-only" in await run("/submode")
    assert "指定" in await run("/sub-mode bogus")
    assert rt.session.mode == ModeState(main="code", sub=RO)
    await run("/mode plan")
    out = await run("/sub-mode")
    assert "sub-mode: read-only" in out and "plan 中" in out
    assert "変更できません" in await run("/sub-mode allow")
    assert rt.session.mode == ModeState(main="plan", sub=RO)


async def test_help_lists_sub_mode(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=200)
    await SlashDispatcher(rt, console).dispatch(parse_slash("/help"))  # type: ignore[arg-type]
    assert "/sub-mode" in console.export_text()


def test_cycle_main_mode_toggles_without_losing_sub(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    rt.set_mode(sub=RO)
    assert cycle_main_mode(rt) == ModeState(main="plan", sub=RO)
    assert cycle_main_mode(rt) == ModeState(main="code", sub=RO)


def test_interactive_prompter_confirm_plan_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    p = InteractivePrompter(console)
    monkeypatch.setattr(p.console, "input", lambda *a, **k: "y")
    assert p.confirm_plan_exit("# plan").approved is True
    answers = iter(["n", "tighten step 2"])
    monkeypatch.setattr(p.console, "input", lambda *a, **k: next(answers))
    verdict = p.confirm_plan_exit("# plan")
    assert verdict == PlanExitVerdict(approved=False, feedback="tighten step 2")
    answers2 = iter(["", ""])
    monkeypatch.setattr(p.console, "input", lambda *a, **k: next(answers2))
    assert p.confirm_plan_exit(None) == PlanExitVerdict(approved=False, feedback=None)

    def _eof(*a: object, **k: object) -> str:
        raise EOFError

    monkeypatch.setattr(p.console, "input", _eof)
    assert p.confirm_plan_exit("# plan").approved is False
    assert isinstance(console.file, io.StringIO)
    assert "プラン承認" in console.file.getvalue()
    monkeypatch.undo()
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    console2 = Console(file=io.StringIO(), force_terminal=False, width=200)
    assert InteractivePrompter(console2).confirm_plan_exit("# plan").approved is True
    assert isinstance(console2.file, io.StringIO)
    shown = console2.file.getvalue()
    assert "[y=承認 / a=承認して一言添える / n=計画を直させる]" in shown
    assert _plan_prompt().strip() in shown


@pytest.mark.parametrize("answer", ["a", "A", "y+", "YES+"])
def test_interactive_prompter_confirm_plan_exit_with_note(
    monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    p = InteractivePrompter(console)
    answers = iter([answer, "  テストも追加して  "])
    monkeypatch.setattr(p.console, "input", lambda *a, **k: next(answers))
    assert p.confirm_plan_exit("# plan") == PlanExitVerdict(approved=True, note="テストも追加して")
    answers2 = iter([answer, ""])
    monkeypatch.setattr(p.console, "input", lambda *a, **k: next(answers2))
    assert p.confirm_plan_exit("# plan") == PlanExitVerdict(approved=True, note=None)

    calls = iter([answer])

    def _note_eof(*a: object, **k: object) -> str:
        try:
            return next(calls)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(p.console, "input", _note_eof)
    assert p.confirm_plan_exit("# plan") == PlanExitVerdict(approved=True, note=None)


def test_enter_plan_mode_is_deferred_by_default_and_core_when_guided() -> None:
    """既定は遅延ツール（モデルの一覧に出ない）。"""
    off = default_registry()
    assert off.is_deferred("enter_plan_mode") is True
    assert "enter_plan_mode" not in off.core_names()
    assert off.is_deferred("exit_plan_mode") is False

    on = default_registry(planning_guidance=True)
    assert on.is_deferred("enter_plan_mode") is False
    assert "enter_plan_mode" in on.core_names()


def test_enter_plan_mode_description_matches_the_planning_guidance() -> None:
    """説明文が system の運用規約と食い違わない（食い違いが勝手な plan 入りの原因だった）。"""
    off = {e["name"]: e["description"] for e in default_registry().deferred_catalog()}
    assert off["enter_plan_mode"] == ENTER_PLAN_DESC_ON_REQUEST
    assert "ONLY when the user explicitly asks" in off["enter_plan_mode"]
    assert "before large changes" not in off["enter_plan_mode"]

    on = {
        e["name"]: e["description"] for e in default_registry(planning_guidance=True).core_schemas()
    }
    assert on["enter_plan_mode"] == ENTER_PLAN_DESC_GUIDED
    assert "before large changes" in on["enter_plan_mode"]


async def test_code_mode_cannot_enter_plan_mode_when_not_guided(tmp_path: Path) -> None:
    """既定では code モードからの enter_plan_mode を差し戻し、モードを変えない。"""
    provider = SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=PermissionResolver(default_policy(WW), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    await rt.run_turn("ダッシュボードを作って")
    block = _blocks(rt)["e1"]
    assert block.is_error is True and "deferred tool" in block.output
    assert rt.session.mode.main == "code"


async def test_plan_mode_still_accepts_enter_plan_mode_even_when_deferred(tmp_path: Path) -> None:
    """Shift+Tab で plan に入った後の念のための呼び出しは、遅延でも失敗にしない（冪等）。"""
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="plan", sub=WW)
    provider = SequencedMockProvider(
        [
            [ToolUse(id="e1", name="enter_plan_mode", input={}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = ConversationRuntime(
        session=session,
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=PermissionResolver(default_policy(WW), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    await rt.run_turn("計画を立てて")
    block = _blocks(rt)["e1"]
    assert block.is_error is False
    assert "already active" in block.output
    assert rt.session.mode.main == "plan"


def test_permission_prompt_shows_all_four_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    """権限確認の選択肢が Rich のマークアップに食われない。"""
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    console = Console(file=io.StringIO(), force_terminal=False, width=200)
    call = ToolUse(id="t1", name="bash", input={"command": "npm test"})
    outcome = InteractivePrompter(console).decide(call, PermissionMode.DANGER_FULL_ACCESS)
    assert outcome.decision == "deny"
    assert isinstance(console.file, io.StringIO)
    shown = console.file.getvalue()
    assert "[y=許可 / n=拒否 / a=永続許可 / d=詳細]" in shown
    assert _permission_prompt(detail=True, always=True).strip() in shown


def test_permission_prompt_keeps_the_choices_after_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """``d=詳細`` で説明を出した後の再入力でも選択肢が出る。"""
    answers = iter(["d", "y"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    console = Console(file=io.StringIO(), force_terminal=False, width=200)
    call = ToolUse(id="t1", name="bash", input={"command": "ls -la"})
    prompter = InteractivePrompter(console)
    asked = prompter.decide(call, PermissionMode.DANGER_FULL_ACCESS)
    assert asked.decision == "ask" and asked.origin == DETAIL_REQUEST_ORIGIN
    detail = PermissionDetail(explanation="ファイル一覧を見ます。", failure=None)
    outcome = prompter.decide_with_detail(call, PermissionMode.DANGER_FULL_ACCESS, None, detail)
    assert outcome.decision == "allow"
    assert isinstance(console.file, io.StringIO)
    shown = console.file.getvalue()
    assert shown.count("[y=許可 / n=拒否 / a=永続許可 / d=詳細]") == 1
    assert shown.count("[y=許可 / n=拒否 / a=永続許可]") == 1
