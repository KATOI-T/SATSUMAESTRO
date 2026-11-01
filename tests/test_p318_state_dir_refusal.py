"""状態フォルダを断ったときと、作業場所の外を断ったときの文。"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

import satsumaestro.cli.repl as repl_mod
from satsumaestro.cli.tool_display import error_detail, error_reason
from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import (
    OUTSIDE_WORKSPACE,
    PATH_ESCAPE_PREFIX,
    STATE_DIR_HIDDEN,
    STATE_DIR_UNDELETABLE,
    WorkspaceJail,
)
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.delete_path import DeletePath
from satsumaestro.util.errors import PathEscapeError, StateDirError
from satsumaestro.util.paths import AGENT_DIRNAME
from tests.fixtures.mock_provider import SequencedMockProvider
from tests.test_repl_turn_failure import _FakeRender, _FakeRuntime, _FakeSession


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / f"{AGENT_DIRNAME}/sessions").mkdir(parents=True)
    (tmp_path / f"{AGENT_DIRNAME}/sessions/old.jsonl").write_text("{}\n", encoding="utf-8")
    return tmp_path


def test_state_dir_is_refused_with_its_own_type(tmp_path: Path) -> None:
    jail = WorkspaceJail(_workspace(tmp_path))
    with pytest.raises(StateDirError) as caught:
        jail.resolve(f"{AGENT_DIRNAME}/sessions/old.jsonl")
    assert str(caught.value).startswith(STATE_DIR_HIDDEN)
    assert isinstance(caught.value, PathEscapeError)
    assert not jail.is_inside(f"{AGENT_DIRNAME}/sessions/old.jsonl")


def test_outside_the_workspace_tells_the_model_to_report_it(tmp_path: Path) -> None:
    jail = WorkspaceJail(_workspace(tmp_path))
    with pytest.raises(PathEscapeError) as caught:
        jail.resolve("../other-project/README.md")
    assert not isinstance(caught.value, StateDirError)
    text = str(caught.value)
    assert text.startswith(OUTSIDE_WORKSPACE)
    assert str(jail.root) in text
    assert "tell them it is outside the workspace" in text and "--cwd" in text


async def test_delete_path_refuses_the_state_dir_with_its_own_type(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    ctx = ToolContext(workspace_root=root, jail=WorkspaceJail(root), config=RuntimeConfig())
    with pytest.raises(StateDirError) as caught:
        await DeletePath().run(DeletePath.Input(path=f"{AGENT_DIRNAME}/sessions/old.jsonl"), ctx)
    assert str(caught.value).startswith(STATE_DIR_UNDELETABLE)
    assert (root / f"{AGENT_DIRNAME}/sessions/old.jsonl").exists()


async def test_tool_results_prefix_only_escapes(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    script: list[list[AssistantEvent]] = [
        [
            ToolUse(
                id="r1", name="read_file", input={"path": f"{AGENT_DIRNAME}/sessions/old.jsonl"}
            ),
            ToolUse(id="r2", name="read_file", input={"path": "../other-project/README.md"}),
            MessageStop(),
        ],
        [TextDelta(text="読めませんでした"), MessageStop()],
    ]
    rt = ConversationRuntime(
        session=new_session(root, "mock-model"),
        provider=SequencedMockProvider(script),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.READ_ONLY),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), root),
        config=RuntimeConfig(),
    )
    await rt.run_turn("前の記録と、隣のプロジェクトの README を読んで")
    results = {
        b.tool_use_id: b
        for m in rt.session.messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }
    assert results["r1"].is_error and results["r1"].output.startswith(STATE_DIR_HIDDEN)
    assert results["r2"].is_error
    assert results["r2"].output.startswith(f"{PATH_ESCAPE_PREFIX}{OUTSIDE_WORKSPACE}")


def _block(output: str) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id="t", tool_name="read_file", output=output, is_error=True)


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (
            f"{STATE_DIR_HIDDEN} {AGENT_DIRNAME}/sessions/old.jsonl",
            "エージェントの状態フォルダは見えません（計画ファイルだけ可）",
        ),
        (
            f"{STATE_DIR_UNDELETABLE} {AGENT_DIRNAME}/plans",
            "エージェントの状態フォルダの中は消せません",
        ),
        (
            f"{PATH_ESCAPE_PREFIX}{OUTSIDE_WORKSPACE} ../x (the workspace is /w). If the user …",
            "作業場所の外です",
        ),
    ],
)
def test_screen_shows_a_short_reason(output: str, reason: str) -> None:
    assert error_reason(_block(output)) == reason
    assert error_detail(_block(output)) == []


def test_other_failures_keep_their_first_line() -> None:
    block = _block(f"{PATH_ESCAPE_PREFIX}redirect '>' target '~/x' is expanded by the shell")
    assert error_reason(block).startswith("path escape: redirect")


async def test_repl_banner_shows_the_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _FakeSession([]))
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    console = Console(record=True, width=120)
    await repl_mod.run_repl(_FakeRuntime([]), None, _FakeRender(), console)  # type: ignore[arg-type]
    assert "作業場所: /work/project" in console.export_text()
