"""遅延ツールの直呼びは、読み込んだうえでそのまま実行する。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import HookConfig, RuntimeConfig, ToolsConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder, format_tool_catalog
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider

_ENTRIES = [{"name": "todo_write", "description": "Record the current TODO list. More."}]


def _runtime(
    provider: SequencedMockProvider, root: Path, config: RuntimeConfig | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(root, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), root),
        config=config or RuntimeConfig(),
    )


def _call(name: str, id: str = "c1", **inp: object) -> list[AssistantEvent]:
    return [ToolUse(id=id, name=name, input=inp), MessageStop()]


def _done() -> list[AssistantEvent]:
    return [TextDelta(text="ok"), MessageStop()]


def _result(rt: ConversationRuntime, id: str) -> ToolResultBlock:
    for m in rt.session.messages:
        for b in m.blocks:
            if isinstance(b, ToolResultBlock) and b.tool_use_id == id:
                return b
    raise AssertionError(id)


def _known_tool_names(provider: SequencedMockProvider, index: int) -> set[str]:
    return {str(t["name"]) for t in provider.requests[index].tools}


async def test_first_call_runs_and_exposes_the_schema(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [_call("todo_write", todos=[{"content": "A", "status": "in_progress"}]), _done()]
    )
    rt = _runtime(provider, tmp_path)
    summary = await rt.run_turn("整理して")

    first = _result(rt, "c1")
    assert not first.is_error and "[~] A" in first.output
    assert [t.content for t in rt.session.todos] == ["A"]
    assert summary.deferred_autoloads == 1 and summary.iterations == 2
    assert "todo_write" not in _known_tool_names(provider, 0)
    assert "todo_write" in _known_tool_names(provider, 1)


async def test_wrong_arguments_on_first_call_return_the_field_list(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            _call("todo_write", items=[{"content": "A"}]),
            _call("todo_write", id="c2", todos=[{"content": "A"}]),
            _done(),
        ]
    )
    rt = _runtime(provider, tmp_path)
    summary = await rt.run_turn("整理して")

    first = _result(rt, "c1")
    assert first.is_error and "missing required field `todos`" in first.output
    assert "Required fields: todos" in first.output
    assert "todo_write" in _known_tool_names(provider, 1)
    assert not _result(rt, "c2").is_error
    assert summary.deferred_autoloads == 1


async def test_enter_plan_mode_is_still_not_run(tmp_path: Path) -> None:
    provider = SequencedMockProvider([_call("enter_plan_mode"), _done()])
    rt = _runtime(provider, tmp_path)
    summary = await rt.run_turn("計画して")
    block = _result(rt, "c1")
    assert block.is_error and "deferred tool" in block.output
    assert rt.session.mode.main == "code" and summary.deferred_autoloads == 0


async def test_legacy_flag_loads_but_does_not_run(tmp_path: Path) -> None:
    cfg = RuntimeConfig(tools=ToolsConfig(deferred_run_on_first_call=False))
    provider = SequencedMockProvider([_call("todo_write", todos=[{"content": "A"}]), _done()])
    rt = _runtime(provider, tmp_path, cfg)
    summary = await rt.run_turn("整理して")

    block = _result(rt, "c1")
    assert block.is_error and "not loaded yet, so this call was not run" in block.output
    assert "required: todos" in block.output
    assert rt.session.todos == []
    assert "todo_write" in _known_tool_names(provider, 1) and summary.deferred_autoloads == 1


def test_catalog_text_follows_the_setting() -> None:
    direct = format_tool_catalog(_ENTRIES)
    assert "Call one directly by its name" in direct
    assert "loaded on the first call" in direct
    assert "tool_search" in direct
    assert "- todo_write: Record the current TODO list" in direct

    legacy = format_tool_catalog(_ENTRIES, run_on_first_call=False)
    assert "call tool_search first" in legacy
    assert "directly" not in legacy
    assert format_tool_catalog([], run_on_first_call=False) == ""


async def test_todo_write_without_active_form_succeeds(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            _call(
                "todo_write",
                todos=[
                    {"content": "Add tests", "status": "in_progress"},
                    {"content": "Run them"},
                ],
            ),
            _done(),
        ]
    )
    rt = _runtime(provider, tmp_path)
    await rt.run_turn("整理して")
    block = _result(rt, "c1")
    assert not block.is_error
    assert "[~] Add tests" in block.output and "[ ] Run them" in block.output
    assert [t.active_form for t in rt.session.todos] == ["", ""]
