"""確認なしで自動許可した bash の git に、リポジトリの設定を信じさせない設定を重ねる。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, PermissionConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import PermissionOutcome, default_policy
from satsumaestro.security.sandbox import AUTO_ALLOWED_GIT_CONFIG, with_git_config
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.bash import Bash
from tests.fixtures.mock_provider import ScriptedFailureProvider

READ_BARE = "cd b.git && git rev-parse --is-bare-repository"


@pytest.fixture(autouse=True)
def _no_user_git_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """利用者の git の設定（全体・システム・環境変数）を混ぜない。"""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in ("GIT_CONFIG_COUNT", "GIT_DIR", "GIT_WORK_TREE"):
        monkeypatch.delenv(name, raising=False)


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(
        workspace_root=tmp_path, jail=WorkspaceJail(tmp_path), config=RuntimeConfig()
    )


def test_pairs_are_added_as_git_config_variables() -> None:
    assert with_git_config({"PATH": "/bin"}, AUTO_ALLOWED_GIT_CONFIG) == {
        "PATH": "/bin",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "safe.bareRepository",
        "GIT_CONFIG_VALUE_0": "explicit",
        "GIT_CONFIG_KEY_1": "core.fsmonitor",
        "GIT_CONFIG_VALUE_1": "false",
    }


def test_the_users_pairs_stay_in_front() -> None:
    base = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "color.ui", "GIT_CONFIG_VALUE_0": "never"}
    env = with_git_config(base, [("core.fsmonitor", "false")])
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert (env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_KEY_1"]) == ("color.ui", "core.fsmonitor")
    assert base["GIT_CONFIG_COUNT"] == "1"


@pytest.mark.parametrize("count", ["x", "-3"])
def test_an_unreadable_count_starts_over(count: str) -> None:
    env = with_git_config({"GIT_CONFIG_COUNT": count}, [("core.fsmonitor", "false")])
    assert (env["GIT_CONFIG_COUNT"], env["GIT_CONFIG_KEY_0"]) == ("1", "core.fsmonitor")


async def test_only_the_auto_allowed_call_refuses_a_bare_repository(tmp_path: Path) -> None:
    _git("init", "-q", "--bare", "b.git", cwd=tmp_path)
    ctx = _ctx(tmp_path)
    plain = await Bash().run(Bash.Input(command=READ_BARE), ctx)
    assert not plain.is_error and plain.output.strip() == "true"
    hardened = await Bash().run(
        Bash.Input(command=READ_BARE), ctx.model_copy(update={"auto_allowed": True})
    )
    assert hardened.is_error and "safe.bareRepository" in hardened.output


async def test_a_normal_repository_reads_the_same(tmp_path: Path) -> None:
    """作業ディレクトリの普通のリポジトリ（.git）では、status / diff / log の出力が変わらない。"""
    _git("init", "-q", cwd=tmp_path)
    (tmp_path / "f.txt").write_text("a\n", encoding="utf-8")
    _git("add", "f.txt", cwd=tmp_path)
    _git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "c1", cwd=tmp_path)
    (tmp_path / "f.txt").write_text("a\nb\n", encoding="utf-8")
    command = "git status --short && git diff && git log --format=%s"
    ctx = _ctx(tmp_path)
    plain = await Bash().run(Bash.Input(command=command), ctx)
    hardened = await Bash().run(
        Bash.Input(command=command), ctx.model_copy(update={"auto_allowed": True})
    )
    assert not hardened.is_error and hardened.output == plain.output
    assert "+b" in hardened.output and "c1" in hardened.output


class _Prompter:
    def __init__(self) -> None:
        self.calls = 0

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.calls += 1
        return PermissionOutcome(decision="allow", reason="y")


async def _bash_result(tmp_path: Path, auto_allow: str, prompter: _Prompter) -> ToolResultBlock:
    steps: list[BaseException | list[AssistantEvent | BaseException]] = [
        [ToolUse(id="b1", name="bash", input={"command": READ_BARE}), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(
            max_iterations=4,
            permission=PermissionConfig(bash_auto_allow=auto_allow),  # type: ignore[arg-type]
        ),
    )
    await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    [result] = [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]
    return result


async def test_auto_allowed_bash_does_not_open_a_bare_repository(tmp_path: Path) -> None:
    _git("init", "-q", "--bare", "b.git", cwd=tmp_path)
    prompter = _Prompter()
    result = await _bash_result(tmp_path, "read-only", prompter)
    assert prompter.calls == 0
    assert result.is_error and "safe.bareRepository" in result.output


async def test_confirmed_bash_keeps_the_git_behaviour(tmp_path: Path) -> None:
    _git("init", "-q", "--bare", "b.git", cwd=tmp_path)
    prompter = _Prompter()
    result = await _bash_result(tmp_path, "off", prompter)
    assert prompter.calls == 1
    assert not result.is_error and result.output.strip() == "true"
