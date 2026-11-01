"""エージェントの設定ファイル（.satsuma.json・.env）と秘密ファイルへの書き込みの確認。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import (
    PermissionOutcome,
    PermissionPolicy,
    PermissionRule,
    RuleMatch,
    default_policy,
)
from satsumaestro.security.remember_guard import AGENT_CONFIG, remember_blocker
from satsumaestro.security.state_write_guard import config_file_bash, config_file_tool
from satsumaestro.session.models import ModeState, ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import ScriptedFailureProvider

WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS


class _Prompter:
    def __init__(self) -> None:
        self.warnings: list[str | None] = []

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.warnings.append(warning)
        return PermissionOutcome(decision="deny", reason="no")


def _runtime(
    ws: Path,
    calls: list[ToolUse],
    *,
    mode: PermissionMode = WW,
    policy: PermissionPolicy | None = None,
) -> ConversationRuntime:
    steps: list[BaseException | list[AssistantEvent | BaseException]] = [
        [call, MessageStop()] for call in calls
    ]
    steps.append([TextDelta(text="done"), MessageStop()])
    session = new_session(ws, "mock-model")
    session.mode = ModeState(main="code", sub=mode)
    return ConversationRuntime(
        session=session,
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=policy or default_policy(mode),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), ws),
        config=RuntimeConfig(max_iterations=8),
    )


async def _run(rt: ConversationRuntime, prompter: object) -> list[ToolResultBlock]:
    await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    return [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


def _write(path: str, name: str = "write_file") -> ToolUse:
    return ToolUse(id="w1", name=name, input={"path": path, "content": '{"hooks": {}}\n'})


def _bash(command: str) -> ToolUse:
    return ToolUse(id="b1", name="bash", input={"command": command})


@pytest.mark.parametrize("path", [".satsuma.json", ".env", "sub/.SATSUMA.json"])
@pytest.mark.parametrize("mode", [WW, DFA])
async def test_settings_file_writes_ask_in_every_mode(
    tmp_path: Path, path: str, mode: PermissionMode
) -> None:
    (tmp_path / "sub").mkdir()
    prompter = _Prompter()
    [result] = await _run(_runtime(tmp_path, [_write(path)], mode=mode), prompter)
    assert len(prompter.warnings) == 1
    assert "agent settings file:" in (prompter.warnings[0] or "")
    assert result.is_error and not (tmp_path / path).exists()


async def test_a_path_allow_rule_does_not_skip_the_question(tmp_path: Path) -> None:
    policy = default_policy(WW)
    policy.allow.append(
        PermissionRule(
            match=RuleMatch(tool="write_file", field="path", glob=".satsuma.json"), label="cfg"
        )
    )
    prompter = _Prompter()
    await _run(_runtime(tmp_path, [_write(".satsuma.json")], policy=policy), prompter)
    assert prompter.warnings == ["agent settings file: .satsuma.json"]


async def test_bash_writing_settings_asks_with_full_access(tmp_path: Path) -> None:
    prompter = _Prompter()
    command = "echo '{}' > .satsuma.json"
    [result] = await _run(_runtime(tmp_path, [_bash(command)], mode=DFA), prompter)
    assert prompter.warnings == ["agent settings file: .satsuma.json"]
    assert result.is_error and not (tmp_path / ".satsuma.json").exists()


async def test_bash_reading_settings_runs_with_full_access(tmp_path: Path) -> None:
    (tmp_path / ".satsuma.json").write_text("{}\n", encoding="utf-8")
    prompter = _Prompter()
    [result] = await _run(_runtime(tmp_path, [_bash("cat .satsuma.json")], mode=DFA), prompter)
    assert prompter.warnings == [] and not result.is_error


async def test_noninteractive_refuses_settings_writes(tmp_path: Path) -> None:
    [result] = await _run(_runtime(tmp_path, [_write(".satsuma.json")]), DenyPrompter())
    assert result.is_error and not (tmp_path / ".satsuma.json").exists()


@pytest.mark.parametrize("name", ["write_file", "append_file"])
async def test_secret_file_writes_ask(tmp_path: Path, name: str) -> None:
    prompter = _Prompter()
    [result] = await _run(_runtime(tmp_path, [_write("id_rsa", name)]), prompter)
    assert prompter.warnings == ["secret file: id_rsa"]
    assert result.is_error and not (tmp_path / "id_rsa").exists()


def test_settings_file_targets(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("A=1\n", encoding="utf-8")
    os.symlink(tmp_path / ".env", tmp_path / "plain.txt")
    assert config_file_tool("write_file", "plain.txt", tmp_path) == tmp_path / "plain.txt"
    assert config_file_tool("read_file", ".env", tmp_path) is None
    assert config_file_tool("write_file", "notes.txt", tmp_path) is None
    assert config_file_bash("cp a .env", tmp_path) == tmp_path / ".env"
    assert config_file_bash("cat .env", tmp_path) is None


@pytest.mark.parametrize(
    ("tool", "tool_input", "blocked"),
    [
        ("write_file", {"path": ".satsuma.json"}, True),
        ("edit_file", {"path": "sub/.env"}, True),
        ("notebook_edit", {"notebook_path": ".env"}, True),
        ("bash", {"command": "echo A=1 >> .env"}, True),
        ("write_file", {"path": "notes.txt"}, False),
        ("bash", {"command": "echo hi > notes.txt"}, False),
    ],
)
def test_settings_file_changes_cannot_be_remembered(
    tool: str, tool_input: dict[str, object], blocked: bool
) -> None:
    assert (remember_blocker(tool, tool_input) == AGENT_CONFIG) is blocked
