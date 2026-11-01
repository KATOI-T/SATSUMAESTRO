"""秘密ファイルを指す read_file / edit_file の確認。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, PermissionConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter, TurnSummary
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import (
    MemoryLayer,
    PermissionResolver,
    SessionAllowlist,
    signature_for,
)
from satsumaestro.security.auto_allow import SecretMatcher, secret_path_hit
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import (
    PermissionOutcome,
    PermissionPolicy,
    PermissionRule,
    RuleMatch,
    default_policy,
)
from satsumaestro.session.models import ModeState, ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import ScriptedFailureProvider

WW = PermissionMode.WORKSPACE_WRITE
RO = PermissionMode.READ_ONLY


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / ".env").write_text("API_TOKEN=abc123\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text("API_TOKEN=\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("plain notes\n", encoding="utf-8")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env.local").write_text("LOCAL=1\n", encoding="utf-8")
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "fixtures" / "test.pem").write_text("pem\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (".env", ".env"),
        ("./.env", ".env"),
        ("config/.env.local", "config/.env.local"),
        (".ENV", ".ENV"),
        ("notes.txt", None),
        ("../outside/.env", None),
    ],
)
def test_secret_path_hit(ws: Path, path: str, expected: str | None) -> None:
    hit = secret_path_hit(path, jail=WorkspaceJail(ws), secrets=_matcher())
    assert hit == expected


def test_secret_path_hit_follows_links_both_ways(ws: Path) -> None:
    os.symlink(ws / ".env", ws / "plain.txt")
    os.symlink(ws / "notes.txt", ws / "id_rsa")
    jail = WorkspaceJail(ws)
    assert secret_path_hit("plain.txt", jail=jail, secrets=_matcher()) == ".env"
    assert secret_path_hit("id_rsa", jail=jail, secrets=_matcher()) == "id_rsa"


def test_patterns_remove_drops_a_pattern_or_excludes_a_name() -> None:
    gone = SecretMatcher.from_config(PermissionConfig(secret_patterns_remove=[".env.*"]))
    assert not gone.matches(".env.local") and gone.matches(".env")
    one = SecretMatcher.from_config(PermissionConfig(secret_patterns_remove=[".env.example"]))
    assert not one.matches(".env.example") and not one.matches("sub/.env.example")
    assert one.matches(".env.local") and one.matches(".env")
    tree = SecretMatcher.from_config(PermissionConfig(secret_patterns_remove=["fixtures/**"]))
    assert not tree.matches("fixtures/test.pem") and tree.matches("keys/test.pem")


def test_env_example_is_secret_by_default() -> None:
    assert _matcher().matches(".env.example")


def test_scoped_marks_rules_that_name_a_path() -> None:
    policy = default_policy(WW)
    policy.allow.append(
        PermissionRule(match=RuleMatch(tool="read_file", field="path", glob=".env"), label="env")
    )
    named = policy.authorize("read_file", {"path": ".env"}, RO)
    assert named.decision == "allow" and named.scoped
    tool_wide = policy.authorize("read_file", {"path": "notes.txt"}, RO)
    assert tool_wide.decision == "allow" and not tool_wide.scoped
    by_mode = default_policy(WW).authorize("edit_file", {"path": ".env"}, WW)
    assert by_mode.reason == "within active mode" and not by_mode.scoped


def test_scoped_marks_remembered_allows() -> None:
    session = SessionAllowlist()
    session.add(signature_for("read_file", {"path": ".env"}))
    session.add(RuleMatch(tool="glob_search"))
    resolver = PermissionResolver(base=default_policy(WW), session=session)
    assert resolver.authorize("read_file", {"path": ".env"}, RO).scoped
    assert not resolver.authorize("glob_search", {"pattern": "*"}, RO).scoped


def test_scoped_marks_memory_rules(tmp_path: Path) -> None:
    memory = tmp_path / "memory.md"
    memory.write_text("## Allow\n- read_file when path glob: .env\n", encoding="utf-8")
    resolver = PermissionResolver(
        base=default_policy(WW), memory_layers=[MemoryLayer(memory, writable=False)]
    )
    outcome = resolver.authorize("read_file", {"path": ".env"}, RO)
    assert outcome.decision == "allow" and outcome.scoped


class _Prompter:
    def __init__(self, *decisions: PermissionOutcome) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self._decisions = list(decisions) or [PermissionOutcome(decision="deny", reason="no")]

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.calls.append((str(call.input.get("path")), warning))
        return self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]


def _matcher(cfg: PermissionConfig | None = None) -> SecretMatcher:
    return SecretMatcher.from_config(cfg or PermissionConfig())


def _runtime(
    ws: Path,
    calls: list[ToolUse],
    *,
    policy: PermissionPolicy | PermissionResolver | None = None,
    permission: PermissionConfig | None = None,
    plan_mode: bool = False,
) -> ConversationRuntime:
    steps: list[BaseException | list[AssistantEvent | BaseException]] = [
        [call, MessageStop()] for call in calls
    ]
    steps.append([TextDelta(text="done"), MessageStop()])
    session = new_session(ws, "mock-model")
    if plan_mode:
        session.mode = ModeState(main="plan", sub=WW)
    return ConversationRuntime(
        session=session,
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=policy or default_policy(WW),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), ws),
        config=RuntimeConfig(max_iterations=8, permission=permission or PermissionConfig()),
    )


def _read(path: str, n: int = 1) -> ToolUse:
    return ToolUse(id=f"r{n}", name="read_file", input={"path": path})


async def _run(
    rt: ConversationRuntime, prompter: object
) -> tuple[TurnSummary, list[ToolResultBlock]]:
    summary = await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    results = [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]
    return summary, results


async def test_read_file_of_a_secret_asks_with_the_path(ws: Path) -> None:
    prompter = _Prompter()
    summary, [result] = await _run(_runtime(ws, [_read(".env")]), prompter)
    assert prompter.calls == [(".env", "secret file: .env")]
    assert result.is_error and "abc123" not in result.output
    assert summary.secret_read_checks == 1


async def test_approving_the_card_reads_the_file(ws: Path) -> None:
    prompter = _Prompter(PermissionOutcome(decision="allow", reason="y"))
    _, [result] = await _run(_runtime(ws, [_read(".env")]), prompter)
    assert not result.is_error and "abc123" in result.output


async def test_plain_files_do_not_ask(ws: Path) -> None:
    prompter = _Prompter()
    summary, [result] = await _run(_runtime(ws, [_read("notes.txt")]), prompter)
    assert prompter.calls == [] and "plain notes" in result.output
    assert summary.secret_read_checks == 0


async def test_secret_access_deny_refuses_without_a_card(ws: Path) -> None:
    prompter = _Prompter()
    deny = PermissionConfig(secret_access="deny")
    summary, [result] = await _run(_runtime(ws, [_read(".env")], permission=deny), prompter)
    assert prompter.calls == []
    assert result.is_error and "secret file access denied" in result.output
    assert "abc123" not in result.output and summary.secret_read_checks == 1


async def test_noninteractive_runs_refuse(ws: Path) -> None:
    _, [result] = await _run(_runtime(ws, [_read(".env")]), DenyPrompter())
    assert result.is_error and "abc123" not in result.output


async def test_a_rule_naming_the_path_reads_without_a_card(ws: Path) -> None:
    policy = default_policy(WW)
    policy.allow.append(
        PermissionRule(match=RuleMatch(tool="read_file", field="path", glob=".env"), label="env")
    )
    prompter = _Prompter()
    _, [result] = await _run(_runtime(ws, [_read(".env")], policy=policy), prompter)
    assert prompter.calls == [] and "abc123" in result.output


async def test_a_tool_wide_allow_still_asks(ws: Path) -> None:
    policy = PermissionPolicy(
        mode=WW, allow=[PermissionRule(match=RuleMatch(tool="read_file"), label="read_file")]
    )
    prompter = _Prompter()
    await _run(_runtime(ws, [_read(".env")], policy=policy), prompter)
    assert prompter.calls == [(".env", "secret file: .env")]


async def test_remembering_the_card_skips_it_next_time(ws: Path) -> None:
    resolver = PermissionResolver(base=default_policy(WW))
    prompter = _Prompter(PermissionOutcome(decision="allow", reason="a", remember=True))
    _, results = await _run(
        _runtime(ws, [_read(".env", 1), _read(".env", 2)], policy=resolver), prompter
    )
    assert prompter.calls == [(".env", "secret file: .env")]
    assert all("abc123" in r.output for r in results) and len(results) == 2


async def test_a_link_to_a_secret_asks(ws: Path) -> None:
    os.symlink(ws / ".env", ws / "plain.txt")
    prompter = _Prompter()
    await _run(_runtime(ws, [_read("plain.txt")]), prompter)
    assert prompter.calls == [("plain.txt", "secret file: .env")]


async def test_edit_file_mismatch_does_not_show_secret_lines(ws: Path) -> None:
    edit = ToolUse(
        id="e1", name="edit_file", input={"path": ".env", "old": "API_TOKEN=x", "new": "A=1"}
    )
    prompter = _Prompter()
    summary, [result] = await _run(_runtime(ws, [edit]), prompter)
    assert prompter.calls == [(".env", "secret file: .env; agent settings file: .env")]
    assert "abc123" not in result.output and summary.secret_read_checks == 1


async def test_plan_mode_asks_too(ws: Path) -> None:
    prompter = _Prompter()
    await _run(_runtime(ws, [_read(".env")], plan_mode=True), prompter)
    assert prompter.calls == [(".env", "secret file: .env")]


async def test_excluded_name_reads_without_a_card(ws: Path) -> None:
    cfg = PermissionConfig(secret_patterns_remove=[".env.example"])
    prompter = _Prompter()
    _, results = await _run(
        _runtime(ws, [_read(".env.example", 1), _read(".env", 2)], permission=cfg), prompter
    )
    assert prompter.calls == [(".env", "secret file: .env")]
    assert "API_TOKEN=" in results[0].output and "abc123" not in results[1].output


async def test_a_policy_deny_keeps_its_reason(ws: Path) -> None:
    policy = default_policy(WW)
    policy.deny.append(
        PermissionRule(match=RuleMatch(tool="read_file", field="path", glob=".env"), label="env")
    )
    prompter = _Prompter()
    summary, [result] = await _run(_runtime(ws, [_read(".env")], policy=policy), prompter)
    assert prompter.calls == [] and "deny rule: env" in result.output
    assert summary.secret_read_checks == 0


async def test_an_unreadable_path_does_not_break_the_turn(ws: Path) -> None:
    prompter = _Prompter()
    summary, [result] = await _run(_runtime(ws, [_read("a\x00.env")]), prompter)
    assert result.is_error and summary.stop_reason == "end_turn"
