"""エージェント自身の状態ディレクトリをツールから隠す。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.delete_path import DeletePath
from satsumaestro.tools.builtins.glob_search import GlobSearch
from satsumaestro.tools.builtins.read_file import ReadFile
from satsumaestro.tools.builtins.write_file import WriteFile
from satsumaestro.util.errors import PathEscapeError
from satsumaestro.util.paths import AGENT_DIRNAME, in_state_dir
from tests.fixtures.mock_provider import SequencedMockProvider

_CURRENT = f"{AGENT_DIRNAME}/plans/CURRENT.md"


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / f"{AGENT_DIRNAME}/plans").mkdir(parents=True)
    (tmp_path / f"{AGENT_DIRNAME}/sessions/abc").mkdir(parents=True)
    (tmp_path / _CURRENT).write_text("# 今回の計画\n", encoding="utf-8")
    (tmp_path / f"{AGENT_DIRNAME}/plans/OLD.md").write_text("# 前回の計画\n", encoding="utf-8")
    (tmp_path / f"{AGENT_DIRNAME}/sessions/abc/s.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / f"{AGENT_DIRNAME}/policy.md").write_text("policy\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def _ctx(root: Path) -> ToolContext:
    jail = WorkspaceJail(root, visible_state=[root / _CURRENT])
    return ToolContext(workspace_root=root, jail=jail, config=RuntimeConfig())


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        (f"{AGENT_DIRNAME}/plans/x.md", True),
        (f"{AGENT_DIRNAME}", True),
        (f"a/{AGENT_DIRNAME}/b", True),
        ("src/app.py", False),
        (".satsumadb/x", False),
        (".satsuma.json", False),
    ],
)
def test_in_state_dir(relative: str, expected: bool) -> None:
    assert in_state_dir(Path(relative)) is expected


def test_jail_hides_state_except_the_current_plan(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    jail = WorkspaceJail(root, visible_state=[root / _CURRENT])
    assert jail.hides(f"{AGENT_DIRNAME}/plans/OLD.md") is True
    assert jail.hides(f"{AGENT_DIRNAME}/sessions/abc/s.jsonl") is True
    assert jail.hides(_CURRENT) is False
    assert jail.hides("src/app.py") is False
    assert jail.hides("/etc/passwd") is False


def test_resolve_explains_why_it_refused(tmp_path: Path) -> None:
    jail = WorkspaceJail(_workspace(tmp_path))
    with pytest.raises(PathEscapeError) as caught:
        jail.resolve(f"{AGENT_DIRNAME}/sessions/abc/s.jsonl")
    assert "own state" in str(caught.value) and AGENT_DIRNAME in str(caught.value)
    with pytest.raises(PathEscapeError) as escaped:
        jail.resolve("../outside.txt")
    assert "not under the workspace" in str(escaped.value)


async def test_read_file_blocks_past_state_but_not_the_current_plan(tmp_path: Path) -> None:
    ctx = _ctx(_workspace(tmp_path))
    ok = await ReadFile().run(ReadFile.Input(path=_CURRENT), ctx)
    assert "今回の計画" in ok.output
    assert (await ReadFile().run(ReadFile.Input(path="src/app.py"), ctx)).output

    for blocked in (
        f"{AGENT_DIRNAME}/plans/OLD.md",
        f"{AGENT_DIRNAME}/sessions/abc/s.jsonl",
        f"{AGENT_DIRNAME}/policy.md",
    ):
        with pytest.raises(PathEscapeError):
            await ReadFile().run(ReadFile.Input(path=blocked), ctx)


async def test_glob_never_lists_state_even_with_an_explicit_pattern(tmp_path: Path) -> None:
    ctx = _ctx(_workspace(tmp_path))

    async def hits(pattern: str) -> list[str]:
        out = await GlobSearch().run(GlobSearch.Input(path="", pattern=pattern, limit=50), ctx)
        return sorted(line.strip() for line in out.output.splitlines() if line.strip())

    assert await hits("**/*") == ["src", "src/app.py"]
    assert await hits(f"{AGENT_DIRNAME}/plans/*") == [_CURRENT]
    assert await hits(f"{AGENT_DIRNAME}/**") == [_CURRENT]


async def test_writes_into_state_are_blocked_but_the_plan_file_works(tmp_path: Path) -> None:
    ctx = _ctx(_workspace(tmp_path))
    await WriteFile().run(WriteFile.Input(path=_CURRENT, content="# 更新\n"), ctx)
    assert (tmp_path / _CURRENT).read_text(encoding="utf-8") == "# 更新\n"
    await DeletePath().run(DeletePath.Input(path=_CURRENT, recursive=False), ctx)
    assert not (tmp_path / _CURRENT).exists()

    for blocked in (f"{AGENT_DIRNAME}/policy.md", f"{AGENT_DIRNAME}/plans/OLD.md"):
        with pytest.raises(PathEscapeError):
            await WriteFile().run(WriteFile.Input(path=blocked, content="x"), ctx)


async def test_runtime_exposes_only_this_sessions_plan_file(tmp_path: Path) -> None:
    script: list[list[AssistantEvent]] = [[TextDelta(text="ok"), MessageStop()]]
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(script),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    jail = rt._ctx.jail
    mine = rt.plan_file_relpath()
    assert jail.hides(mine) is False
    other = mine.replace(rt.session.session_id, "0" * len(rt.session.session_id))
    assert other != mine
    assert jail.hides(other) is True
    assert jail.hides(f"{AGENT_DIRNAME}/sessions") is True


async def test_the_model_gets_an_explanation_instead_of_a_crash(tmp_path: Path) -> None:
    """拒否はツール結果として返り、ターンは続く（理由を読んで次の手を選べる）。"""
    script: list[list[AssistantEvent]] = [
        [
            ToolUse(
                id="r1",
                name="read_file",
                input={"path": f"{AGENT_DIRNAME}/sessions/other.jsonl"},
            ),
            MessageStop(),
        ],
        [TextDelta(text="読めませんでした"), MessageStop()],
    ]
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(script),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.READ_ONLY),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    summary = await rt.run_turn("過去の記録を見て")
    assert summary.final_text == "読めませんでした"


def test_asyncio_is_used(tmp_path: Path) -> None:
    assert asyncio.iscoroutinefunction(ReadFile.run)
