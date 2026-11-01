"""REPL の入出力・コマンド・失敗の文の英語の画面（日本語の画面は既存のテストが見ている）。"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.cli.errors_ui import turn_failure
from satsumaestro.cli.repl import InteractivePrompter, _detail_panel, _permission_prompt
from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, ToolUse
from satsumaestro.engine.plan_budget import BudgetVerdict, PlanBudget
from satsumaestro.engine.plan_steps import parse_plan_steps
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import PermissionDetail
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.output_guard import card_note
from satsumaestro.security.policy import default_policy
from satsumaestro.security.remember_guard import blocker_note, remember_blocker
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ContextOverflowError, MaxIterationsError, ProviderError
from tests.fixtures.mock_provider import SequencedMockProvider

_JAPANESE = re.compile(r"[　-ヿ一-鿿＀-￯]")
_BUDGET = PlanBudget(120.0, 57.5, "measured", 1, 3450, 280)


@pytest.fixture(autouse=True)
def _english_screen() -> Iterator[None]:
    with pinned_ui_language("en"):
        yield


def _console() -> Console:
    return Console(record=True, width=200)


def _english(text: str) -> str:
    assert not _JAPANESE.search(text), text
    return text


def test_permission_questions() -> None:
    assert _permission_prompt(detail=True, always=True) == (
        "Allow this? [y=allow / n=deny / a=always allow / d=details]: "
    )
    assert _permission_prompt(detail=True, always=False) == (
        "Allow this? [y=allow / n=deny / d=details]: "
    )
    assert _permission_prompt(detail=False, always=False) == "Allow this? [y=allow / n=deny]: "


def test_a_card_that_cannot_be_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    console = _console()
    answers = iter(["a", "y"])
    monkeypatch.setattr(console, "input", lambda *a, **k: next(answers))
    prompter = InteractivePrompter(console)
    call = ToolUse(id="1", name="bash", input={"command": "rm old.txt"})
    outcome = prompter.decide(call, PermissionMode.WORKSPACE_WRITE)
    out = console.export_text()
    assert outcome.decision == "allow" and not outcome.remember
    assert "(This cannot be remembered: it deletes files)" in out
    assert "(This action cannot be remembered. Press y to allow it)" in out
    _english(out)


def test_blocker_notes() -> None:
    git = remember_blocker("bash", {"command": "git push --force"})
    assert git is not None and git.startswith("git:")
    assert blocker_note(git) == (
        "(This cannot be remembered: it is an irreversible git operation (git push))"
    )
    sudo = remember_blocker("bash", {"command": "sudo ls"})
    assert (
        sudo is not None
        and blocker_note(sudo) == "(This cannot be remembered: it uses sudo / doas)"
    )


def test_the_detail_panel_without_an_explanation() -> None:
    detail = PermissionDetail(
        explanation=None,
        failure="timeout",
        model_text="I will delete the build folder.",
        step=(2, 3, "Clean up"),
        user_request="Tidy the project",
    )
    console = _console()
    console.print(_detail_panel(detail))
    out = _english(console.export_text())
    assert "What it is about to do" in out
    assert "Could not get an explanation (stopped after 30 s)" in out
    assert "What the model said just before:" in out and "Current step 2/3: Clean up" in out
    assert "Request: Tidy the project" in out


def test_the_card_question_inside_an_explanation_is_broken() -> None:
    console = _console()
    console.print(
        _detail_panel(PermissionDetail(explanation="Allow this? It edits a.txt.", failure=None))
    )
    assert "Allow this ?" in console.export_text()


def test_the_delete_card_note(tmp_path: Path) -> None:
    files = [tmp_path / name for name in ("a.txt", "b.txt", "c.txt", "d.txt")]
    for f in files:
        f.write_text("x\n", encoding="utf-8")
    note = card_note(files, tmp_path)
    assert note == (
        "Deletes files created in this session: a.txt (1 line), b.txt (1 line), c.txt (1 line) "
        "and 1 more"
    )


PLAN = "## Steps\n1. Create a.txt (~120 lines)\n2. Create b.txt (~400 lines)\n"


def test_the_approval_card(monkeypatch: pytest.MonkeyPatch) -> None:
    console = _console()
    monkeypatch.setattr(console, "input", lambda *a, **k: "y")
    prompter = InteractivePrompter(console, plan_steps=True)
    steps = parse_plan_steps(PLAN)
    prompter.set_plan_budget(_BUDGET, steps, BudgetVerdict("over", _BUDGET, ((2, 400),)))
    assert prompter.confirm_plan_exit(PLAN).approved
    out = _english(console.export_text())
    assert "Plan approval" in out
    assert "(Runs in 2 steps, one at a time)" in out
    assert (
        "(step limit 280 lines = 58 tok/s measured in this conversation × 120 s × safety factor "
        "0.5 (≈ 3,450 tokens). Largest step: 2 (400 lines))" in out
    )
    assert (
        "(step 2 is over budget (400 lines > 280 lines), and the send-back limit was reached, so "
        "approval is requested)" in out
    )


def test_the_approval_card_without_a_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    console = _console()
    monkeypatch.setattr(console, "input", lambda *a, **k: "")
    verdict = InteractivePrompter(console).confirm_plan_exit(None)
    assert not verdict.approved
    assert "(No plan file)" in _english(console.export_text())


def test_budget_summaries() -> None:
    two = BudgetVerdict("over", _BUDGET, ((1, 400), (3, 350)))
    assert two.summary() == "steps 1, 3 are over budget (400 / 350 lines > 280 lines)"
    assert (
        BudgetVerdict("no_estimates", _BUDGET)
        .card()
        .startswith("The steps still have no estimate (~N lines)")
    )
    assert BudgetVerdict("no_section", _BUDGET).summary() == (
        "there is no ## Steps section (limit 280 lines)"
    )


def test_turn_failures() -> None:
    failure = turn_failure(MaxIterationsError(8, step=(2, 3)))
    assert failure.headline == (
        "⚠ The turn could not be finished: Reached the iteration limit (8) inside step 2/3"
    )
    assert failure.next_step.startswith('Say "continue" to resume')
    overflow = turn_failure(
        ContextOverflowError(estimate=140_000, limit=131_072, dropped_request=False)
    )
    assert "does not fit in its context window (estimated about 140,000 / limit about 131,072" in (
        overflow.headline
    )
    assert overflow.next_step.startswith("Fold the conversation with /compact")
    provider = turn_failure(ProviderError("connection refused"))
    assert provider.reason == "Communication with the model failed (connection refused)"


def _runtime(tmp_path: Path) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider([[MessageStop()]]),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


async def test_slash_commands(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    console = _console()
    d = SlashDispatcher(rt, console)

    async def run(line: str) -> str:
        await d.dispatch(parse_slash(line))  # type: ignore[arg-type]
        return console.export_text(clear=True)

    help_text = await run("/help")
    assert "Command" in help_text and "Description" in help_text
    assert "Show the commands" in help_text and "Exit the REPL" in help_text
    assert "Reply language: undetermined (mode=auto)" in await run("/lang")
    assert "Reply language → fixed to en" in await run("/lang en")
    assert "lang must be one of auto|ja|en" in await run("/lang zh")
    assert "Unknown command: /nope — /help for the list" in await run("/nope")
    assert "Nothing to fold (too few messages)" in await run("/compact")
    assert "Use /plan on|off" in await run("/plan maybe")
    assert "Cleared the conversation, TODOs and plan state" in await run("/clear")
    assert "Telemetry is off" in await run("/insights")
    for line in ("/help", "/lang", "/mode", "/sub-mode", "/effort", "/fast"):
        _english(await run(line))
