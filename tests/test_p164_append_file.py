"""append_file ツールのテスト。"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli.render import RichRenderer, action_preview, change_preview
from satsumaestro.config.models import CondenseConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.condense import condense
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.reminders import timeout_note
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode, classify
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import (
    ConversationMessage,
    ModeState,
    ToolResultBlock,
    ToolUseBlock,
    plan_file_relpath,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.append_file import AppendFile
from satsumaestro.tools.builtins.write_file import WriteFile
from satsumaestro.util.errors import PathEscapeError
from tests.fixtures.mock_provider import SequencedMockProvider

WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS


async def test_append_creates_missing_file_and_parents(tool_ctx: ToolContext) -> None:
    res = await AppendFile().run(AppendFile.Input(path="out/a.html", content="<html>\n"), tool_ctx)
    assert res.is_error is False
    assert res.output == "appended 7 bytes to out/a.html (now 7 bytes)"
    assert (tool_ctx.workspace_root / "out" / "a.html").read_text(encoding="utf-8") == "<html>\n"


async def test_append_adds_to_end_without_separator(tool_ctx: ToolContext) -> None:
    target = tool_ctx.workspace_root / "a.txt"
    target.write_text("line1\nhalf", encoding="utf-8")
    res = await AppendFile().run(AppendFile.Input(path="a.txt", content="-line\nline3\n"), tool_ctx)
    assert target.read_text(encoding="utf-8") == "line1\nhalf-line\nline3\n"
    assert res.output == "appended 12 bytes to a.txt (now 22 bytes)"


async def test_write_then_append_builds_file_in_pieces(tool_ctx: ToolContext) -> None:
    await WriteFile().run(WriteFile.Input(path="doc.md", content="# Title\n"), tool_ctx)
    for i in range(3):
        await AppendFile().run(AppendFile.Input(path="doc.md", content=f"part {i}\n"), tool_ctx)
    body = (tool_ctx.workspace_root / "doc.md").read_text(encoding="utf-8")
    assert body == "# Title\npart 0\npart 1\npart 2\n"


async def test_append_escapes_jail(tool_ctx: ToolContext) -> None:
    with pytest.raises(PathEscapeError):
        await AppendFile().run(AppendFile.Input(path="../evil.txt", content="x"), tool_ctx)


def test_classify_is_write_and_git_dir_escalates() -> None:
    assert classify("append_file", {"path": "src/a.py"}, WW) is WW
    assert classify("append_file", {"path": ".git/config"}, WW) is DFA
    assert classify("append_file", {"path": ".gitignore"}, WW) is WW


def test_append_file_is_core_not_deferred() -> None:
    reg = default_registry()
    assert "append_file" in {e["name"] for e in reg.core_schemas()}
    assert "append_file" not in {e["name"] for e in reg.deferred_catalog()}


def test_alias_resolves_to_append_file() -> None:
    assert default_registry().resolve_allowed_tools(["Append"]) == {"append_file"}


def test_condense_treats_append_as_a_change() -> None:
    def use(call_id: str, name: str) -> ConversationMessage:
        block = ToolUseBlock(id=call_id, name=name, input={"path": "a.html"})
        return ConversationMessage(role="assistant", blocks=[block])

    def result(call_id: str, name: str) -> ConversationMessage:
        block = ToolResultBlock(tool_use_id=call_id, tool_name=name, output="ok")
        return ConversationMessage(role="tool", blocks=[block])

    messages = [
        use("r1", "read_file"),
        result("r1", "read_file"),
        use("a1", "append_file"),
        result("a1", "append_file"),
    ]
    out, _ = condense(messages, CondenseConfig(collapse_enabled=False, cluster_enabled=False))
    remaining = {b.id for m in out for b in m.blocks if isinstance(b, ToolUseBlock)}
    assert remaining == {"a1"}


def _blocks(rt: ConversationRuntime) -> dict[str, ToolResultBlock]:
    return {
        b.tool_use_id: b
        for m in rt.session.messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }


async def test_plan_mode_allows_append_to_plan_file_only(tmp_path: Path) -> None:
    session = new_session(tmp_path, "mock-model")
    session.mode = ModeState(main="plan", sub=WW)
    plan_rel = plan_file_relpath(session.session_id)
    (tmp_path / ".satsuma" / "plans").mkdir(parents=True)
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="p1", name="write_file", input={"path": plan_rel, "content": "# P\n"}),
                ToolUse(id="p2", name="append_file", input={"path": plan_rel, "content": "1. a\n"}),
                ToolUse(id="o1", name="append_file", input={"path": "src.txt", "content": "x"}),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = ConversationRuntime(
        session=session,
        provider=provider,
        tools=default_registry(),
        policy=PermissionResolver(default_policy(WW), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    await rt.run_turn("plan")
    by_id = _blocks(rt)
    assert by_id["p1"].is_error is False and by_id["p2"].is_error is False
    assert (tmp_path / plan_rel).read_text(encoding="utf-8") == "# P\n1. a\n"
    assert by_id["o1"].is_error is True and "plan mode is active:" in by_id["o1"].output
    assert not (tmp_path / "src.txt").exists()


def _plain(renderable: object) -> str:
    c = Console(file=io.StringIO(), force_terminal=False, width=100)
    c.print(renderable)
    assert isinstance(c.file, io.StringIO)
    return c.file.getvalue()


def test_change_preview_shows_only_appended_lines_with_line_numbers(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("".join(f"old {i}\n" for i in range(1, 11)), encoding="utf-8")
    call = ToolUse(id="t1", name="append_file", input={"path": "a.txt", "content": "new 11\n"})
    preview = _plain(change_preview(call, tmp_path))
    assert "11 + new 11" in preview
    assert "old 10" in preview and "old 1\n" not in preview
    assert " - " not in preview


def test_change_preview_new_file(tmp_path: Path) -> None:
    call = ToolUse(id="t1", name="append_file", input={"path": "b.txt", "content": "hello\n"})
    preview = _plain(change_preview(call, tmp_path))
    assert "（新規ファイル）" in preview and "1 + hello" in preview


def test_action_preview_summarizes_append() -> None:
    call = ToolUse(id="t1", name="append_file", input={"path": "a.txt", "content": "x\ny\n"})
    card = _plain(action_preview(call, WW.value))
    assert "a.txt" in card and "追記 2 行" in card
    detailed = _plain(action_preview(call, WW.value, detailed=True))
    assert "x" in detailed and "y" in detailed


def test_renderer_labels_append(tmp_path: Path) -> None:
    buf = io.StringIO()
    r = RichRenderer(
        console=Console(file=buf, force_terminal=True, width=100, color_system=None), cwd=tmp_path
    )
    r.start_turn()
    r.on_tool_start(ToolUse(id="t1", name="append_file", input={"path": "a.txt", "content": "x"}))
    r.end_turn()
    out = buf.getvalue()
    assert "・追記" in out and "- a.txt" in out and "append_file" not in out


def test_guidance_points_to_append_file() -> None:
    assert "append_file" in WriteFile.spec.description
    static = SystemPromptBuilder()._static_sections("mock-model", RuntimeConfig())
    assert "write it in pieces: write_file for the first part, then append_file" in static
    note = timeout_note(120.0)
    assert "`append_file` for each following part" in note
    assert "several files" not in note
