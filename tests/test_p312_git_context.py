"""system の git 節に diff の本文を入れない。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder, _diff_line_counts
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider

_MARK = "half_done_feature_marker"
_NOTE = "not the user's request"


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True)


def _repo_with_uncommitted_diff(root: Path) -> Path:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-q", "-m", "app")
    (root / "app.py").write_text(
        f"def main():\n    # TODO: finish {_MARK}\n    return {_MARK}()\n", encoding="utf-8"
    )
    (root / "feature.py").write_text(f"def {_MARK}():\n    ...\n", encoding="utf-8")
    return root


async def _git_section(root: Path) -> tuple[str, str]:
    turn = await SystemPromptBuilder().snapshot(root)
    return turn.git, turn.git_state


async def test_git_section_has_no_diff_body(tmp_path: Path) -> None:
    git, state = await _git_section(_repo_with_uncommitted_diff(tmp_path))

    assert _MARK not in git
    assert "$ git diff\n" not in git
    assert "@@" not in git
    assert git.startswith("# Git\n$ git status --short\n")
    assert " M app.py" in git
    assert "?? feature.py" in git
    assert "app.py +2 -1" in git
    assert state in git


async def test_background_note_only_when_there_are_changes(tmp_path: Path) -> None:
    root = _repo_with_uncommitted_diff(tmp_path)
    git, _ = await _git_section(root)
    assert _NOTE in git
    assert git.index("app.py +2 -1") < git.index(_NOTE)

    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "done")
    clean, _ = await _git_section(root)
    assert clean == "# Git\n$ git status --short\n(clean)"


async def test_japanese_paths_are_readable_with_line_counts(tmp_path: Path) -> None:
    root = _repo_with_uncommitted_diff(tmp_path)
    doc = root / "設計書.md"
    doc.write_text("一\n", encoding="utf-8")
    _git(root, "add", "設計書.md")
    _git(root, "commit", "-q", "-m", "doc")
    doc.write_text("一\n二\n", encoding="utf-8")

    git, _ = await _git_section(root)
    assert " M 設計書.md" in git
    assert "設計書.md +1 -0" in git
    assert "\\3" not in git


def test_quoted_diff_header_path() -> None:
    diff = (
        'diff --git "a/x\\ty.txt" "b/x\\ty.txt"\n'
        '--- "a/x\\ty.txt"\n'
        '+++ "b/x\\ty.txt"\n'
        "@@ -0,0 +1 @@\n"
        "+z\n"
    )
    assert _diff_line_counts(diff) == [("x\\ty.txt", 1, 0)]


async def test_non_git_directory_has_no_git_section(tmp_path: Path) -> None:
    assert await _git_section(tmp_path) == ("", "")


def _runtime(provider: object, root: Path, *, freeze: bool) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(root, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), root),
        config=RuntimeConfig(guidance=GuidanceConfig(freeze_git_context=freeze)),
    )


async def test_system_prompt_sent_to_the_model_has_no_diff_body(tmp_path: Path) -> None:
    root = _repo_with_uncommitted_diff(tmp_path)
    for freeze in (True, False):
        provider = SequencedMockProvider(
            [[TextDelta(text="晴れです"), MessageStop(stop_reason="end_turn")]]
        )
        await _runtime(provider, root, freeze=freeze).run_turn("１週間分の東京の天気を教えて")
        request = provider.requests[0]
        system = f"{request.system}\n\n{request.system_dynamic}"
        assert _MARK not in system
        assert "app.py +2 -1" in system and _NOTE in system
