"""system の git 節の凍結と、作業ツリーの変化の通知。"""

from __future__ import annotations

import subprocess
from pathlib import Path

from satsumaestro.config.models import GuidanceConfig, HookConfig, LanguageConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import (
    SystemPromptBuilder,
    _diff_line_counts,
    _git_status_note,
)
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import Session, SessionCompaction, TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider

_NOTICE = "The working tree has changed since the git snapshot"


def _git_repo(path: Path) -> Path:
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "t"],
        ["commit", "-q", "--allow-empty", "-m", "init"],
    ):
        subprocess.run(["git", *args], cwd=path, check=True)
    return path


def _cfg(*, freeze: bool = True, language: str = "off") -> RuntimeConfig:
    return RuntimeConfig(
        guidance=GuidanceConfig(freeze_git_context=freeze),
        language=LanguageConfig(mode=language),  # type: ignore[arg-type]
    )


def _runtime(
    provider: object, root: Path, cfg: RuntimeConfig, session: Session | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=session if session is not None else new_session(root, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), root),
        config=cfg,
    )


def _write_then_answer(path: str) -> list[list[object]]:
    return [
        [
            ToolUse(id="w", name="write_file", input={"path": path, "content": "hello\n"}),
            MessageStop(stop_reason="tool_use"),
        ],
        [TextDelta(text="作りました"), MessageStop(stop_reason="end_turn")],
    ]


def _answer(text: str = "はい") -> list[object]:
    return [TextDelta(text=text), MessageStop(stop_reason="end_turn")]


def _user_texts(rt: ConversationRuntime) -> list[str]:
    return [
        b.text
        for m in rt.session.messages
        if m.role == "user"
        for b in m.blocks
        if isinstance(b, TextBlock)
    ]


def _systems(provider: SequencedMockProvider) -> list[str]:
    return [f"{r.system}\n\n{r.system_dynamic}" for r in provider.requests]


async def test_system_stays_byte_stable_after_a_turn_that_writes(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    provider = SequencedMockProvider([*_write_then_answer("a.txt"), _answer(), _answer()])  # type: ignore[list-item]
    rt = _runtime(provider, root, _cfg())

    await rt.run_turn("a.txt を作って")
    second = await rt.run_turn("消して")
    third = await rt.run_turn("もう一度確認して")

    systems = _systems(provider)
    assert len(systems) == 4
    assert len(set(systems)) == 1
    assert "(Snapshot taken at the start of this session" in systems[0]
    assert second.turn_prefix_break is False and third.turn_prefix_break is False

    users = _user_texts(rt)
    assert _NOTICE not in users[0]
    assert "<system-reminder>" in users[1] and _NOTICE in users[1] and "?? a.txt" in users[1]
    assert "<system-reminder>" not in users[2]


async def test_edit_to_a_tracked_file_reports_changed_line_counts(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    (root / "app.py").write_text("a = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "app"], cwd=root, check=True)
    provider = SequencedMockProvider([_answer(), _answer()])
    rt = _runtime(provider, root, _cfg())

    await rt.run_turn("見て")
    (root / "app.py").write_text("a = 2\nb = 3\n", encoding="utf-8")
    await rt.run_turn("続けて")

    notice = _user_texts(rt)[1]
    assert " M app.py" in notice
    assert "app.py +2 -1" in notice


async def test_notice_comes_before_the_language_directive(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    provider = SequencedMockProvider([*_write_then_answer("a.txt"), _answer()])  # type: ignore[list-item]
    rt = _runtime(provider, root, _cfg(language="auto"))

    await rt.run_turn("a.txt を作ってください")
    await rt.run_turn("作ったファイルを消してください")

    text = _user_texts(rt)[1]
    assert text.index("</system-reminder>") < text.index("(Reply in Japanese.")
    assert text.rstrip().endswith("stay as-is.)")


async def test_frozen_git_survives_save_and_load(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    store = JSONLSessionStore()
    first = SequencedMockProvider(_write_then_answer("a.txt"))  # type: ignore[arg-type]
    rt = _runtime(first, root, _cfg())
    await rt.run_turn("a.txt を作って")
    frozen = rt.session.git_context
    assert frozen is not None and "a.txt" not in frozen

    loaded = store.load(store.path_for(root, rt.session.session_id))
    assert loaded.git_context == frozen
    assert loaded.git_seen == rt.session.git_seen

    second = SequencedMockProvider([_answer()])
    rt2 = _runtime(second, root, _cfg(), session=loaded)
    summary = await rt2.run_turn("消して")
    assert _systems(second)[0] == _systems(first)[-1]
    assert summary.turn_prefix_break is False
    assert "?? a.txt" in _user_texts(rt2)[-1]


async def test_cross_turn_break_is_counted_across_resume(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    store = JSONLSessionStore()
    first = SequencedMockProvider(_write_then_answer("a.txt"))  # type: ignore[arg-type]
    rt = _runtime(first, root, _cfg(freeze=False))
    await rt.run_turn("a.txt を作って")
    assert rt.session.last_system_digest

    loaded = store.load(store.path_for(root, rt.session.session_id))
    assert loaded.last_system_digest == rt.session.last_system_digest
    rt2 = _runtime(SequencedMockProvider([_answer()]), root, _cfg(freeze=False), session=loaded)
    summary = await rt2.run_turn("消して")
    assert summary.turn_prefix_break is True


async def test_git_context_is_refreshed_after_compaction(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    provider = SequencedMockProvider([*_write_then_answer("a.txt"), _answer()])  # type: ignore[list-item]
    rt = _runtime(provider, root, _cfg())
    await rt.run_turn("a.txt を作って")

    rt.session.compaction = SessionCompaction(
        compacted_at_ms=1_234, summary="(summary)", dropped_messages=2
    )
    await rt.run_turn("消して")

    systems = _systems(provider)
    assert "?? a.txt" not in systems[0]
    assert "?? a.txt" in systems[-1]
    assert _NOTICE not in _user_texts(rt)[-1]
    assert rt.session.git_context_compaction == 1_234


async def test_legacy_mode_refreshes_git_each_turn(tmp_path: Path) -> None:
    root = _git_repo(tmp_path)
    provider = SequencedMockProvider([*_write_then_answer("a.txt"), _answer()])  # type: ignore[list-item]
    rt = _runtime(provider, root, _cfg(freeze=False))

    await rt.run_turn("a.txt を作って")
    second = await rt.run_turn("消して")

    systems = _systems(provider)
    assert systems[0] == systems[1]
    assert systems[2] != systems[1]
    assert "?? a.txt" in systems[2]
    assert "(Snapshot taken" not in systems[0]
    assert second.turn_prefix_break is True
    assert all("<system-reminder>" not in t for t in _user_texts(rt))
    assert rt.session.git_context is None


async def test_non_git_directory_never_adds_a_notice(tmp_path: Path) -> None:
    provider = SequencedMockProvider([*_write_then_answer("a.txt"), _answer()])  # type: ignore[list-item]
    rt = _runtime(provider, tmp_path, _cfg())
    await rt.run_turn("a.txt を作って")
    await rt.run_turn("消して")
    assert len(set(_systems(provider))) == 1
    assert all("<system-reminder>" not in t for t in _user_texts(rt))


def test_diff_line_counts_per_file() -> None:
    diff = (
        "diff --git a/app.py b/app.py\n"
        "index 1..2 100644\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1,2 @@\n"
        "-a = 1\n"
        "+a = 2\n"
        "+b = 3\n"
        "diff --git a/notes.md b/notes.md\n"
        "--- a/notes.md\n"
        "+++ b/notes.md\n"
        "@@ -1,2 +1 @@\n"
        "--- removed rule line\n"
        " keep\n"
    )
    assert _diff_line_counts(diff) == [("app.py", 2, 1), ("notes.md", 0, 1)]


def test_git_state_is_clean_and_bounded() -> None:
    assert _git_status_note("", "") == "$ git status --short\n(clean)"
    huge = "\n".join(f"?? file_{i}.txt" for i in range(1_000))
    state = _git_status_note(huge, "")
    assert len(state) < 2_100 and state.endswith("… (truncated)")
