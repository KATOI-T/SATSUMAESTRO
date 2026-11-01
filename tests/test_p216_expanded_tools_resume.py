"""読み込んだ遅延ツールをセッションに残し、--resume の後も同じツール一覧から始める。"""

from __future__ import annotations

import json
from pathlib import Path

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import Session
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider


def _runtime(
    provider: SequencedMockProvider, tmp_path: Path, session: Session
) -> ConversationRuntime:
    return ConversationRuntime(
        session=session,
        provider=provider,
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


def _tool_names(tools: list[dict[str, object]]) -> list[str]:
    return [str(t.get("name") or t.get("function", {}).get("name")) for t in tools]  # type: ignore[union-attr]


def _first_turn() -> list[list[object]]:
    return [
        [
            ToolUse(id="s1", name="tool_search", input={"query": "select:todo_write"}),
            MessageStop(stop_reason="tool_use"),
        ],
        [TextDelta(text="読み込みました"), MessageStop(stop_reason="end_turn")],
    ]


def _reply() -> list[list[object]]:
    return [[TextDelta(text="はい"), MessageStop(stop_reason="end_turn")]]


def test_expanded_tools_round_trip_through_the_store(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    session = new_session(tmp_path, "m")
    store.save(session)
    path = store.path_for(tmp_path, session.session_id)
    assert '"expanded_tools"' not in path.read_text(encoding="utf-8")
    session.expanded_tools = ["todo_write", "web_fetch"]
    store.save(session)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert {"type": "expanded_tools", "names": ["todo_write", "web_fetch"]} in rows
    assert store.load(path).expanded_tools == ["todo_write", "web_fetch"]


async def test_tool_search_is_kept_in_the_session(tmp_path: Path) -> None:
    session = new_session(tmp_path, "m")
    provider = SequencedMockProvider(_first_turn())
    rt = _runtime(provider, tmp_path, session)
    summary = await rt.run_turn("ToDo を使えるようにして")
    assert rt.session.expanded_tools == ["todo_write"]
    assert summary.prompt_prefix_breaks == 1
    assert "todo_write" not in _tool_names(provider.requests[0].tools)
    assert "todo_write" in _tool_names(provider.requests[1].tools)


async def test_resume_starts_with_the_same_tool_list(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    session = new_session(tmp_path, "m")
    first = SequencedMockProvider(_first_turn())
    await _runtime(first, tmp_path, session).run_turn("ToDo を使えるようにして")

    resumed = store.load(store.path_for(tmp_path, session.session_id))
    second = SequencedMockProvider(_reply())
    summary = await _runtime(second, tmp_path, resumed).run_turn("続けて")
    assert second.requests[0].tools == first.requests[-1].tools
    assert summary.turn_prefix_break is False


async def test_a_lost_tool_list_is_counted_as_a_break(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    session = new_session(tmp_path, "m")
    first = SequencedMockProvider(_first_turn())
    await _runtime(first, tmp_path, session).run_turn("ToDo を使えるようにして")

    resumed = store.load(store.path_for(tmp_path, session.session_id))
    resumed.expanded_tools = []
    second = SequencedMockProvider(_reply())
    summary = await _runtime(second, tmp_path, resumed).run_turn("続けて")
    assert "todo_write" not in _tool_names(second.requests[0].tools)
    assert second.requests[0].system == first.requests[-1].system
    assert summary.turn_prefix_break is True
