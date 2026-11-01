"""会話ループ堅牢化のテスト。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime, _detect_text_tool_call
from satsumaestro.engine.trace import (
    ModelReplied,
    ToolDispatched,
    ToolReturned,
    TurnAborted,
    TurnBegan,
    TurnEnded,
    TurnEvent,
)
from satsumaestro.hooks.runner import HookRunner, append_hook_notes
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionPolicy, default_policy
from satsumaestro.session.models import TokenUsage, ToolUseBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import MaxIterationsError
from tests.fixtures.mock_provider import MockProvider, SequencedMockProvider

_TOOLS = {"read_file", "write_file", "bash", "grep_search"}


class SeenEvents:
    """受け取った出来事を順に溜める受け手（テレメトリの代役）。"""

    def __init__(self) -> None:
        self.events: list[TurnEvent] = []

    def observe(self, event: TurnEvent) -> None:
        self.events.append(event)

    def kinds(self) -> list[type]:
        return [type(event) for event in self.events]


class SpyCompactor(Compactor):
    """``needs_compaction`` の呼び出し回数を記録する Compactor。"""

    def __init__(self, keep_last: int = 6) -> None:
        super().__init__(keep_last)
        self.checks = 0

    def needs_compaction(self, total_usage: TokenUsage, cfg: RuntimeConfig) -> bool:
        self.checks += 1
        return super().needs_compaction(total_usage, cfg)


def _make_runtime(
    provider: object,
    tmp_path: Path,
    *,
    policy: PermissionPolicy | None = None,
    config: RuntimeConfig | None = None,
    compactor: Compactor | None = None,
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=policy or default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config or RuntimeConfig(),
        compactor=compactor,
    )


async def test_terminal_iteration_runs_compaction(tmp_path: Path) -> None:
    spy = SpyCompactor()
    provider = MockProvider([TextDelta(text="plain answer"), MessageStop()])
    rt = _make_runtime(provider, tmp_path, compactor=spy)
    await rt.run_turn("hi")
    assert spy.checks == 1


async def test_terminal_compaction_check_every_iteration(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    spy = SpyCompactor()
    provider = SequencedMockProvider(
        [
            [ToolUse(id="t1", name="read_file", input={"path": "f.txt"}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path, compactor=spy)
    await rt.run_turn("read")
    assert spy.checks == 2


async def test_max_iterations_raises_with_context(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    provider = MockProvider(
        [
            TextDelta(text="still working on it"),
            ToolUse(id="t", name="read_file", input={"path": "f.txt"}),
            MessageStop(),
        ]
    )
    seen = SeenEvents()
    rt = _make_runtime(provider, tmp_path, config=RuntimeConfig(max_iterations=2))
    with pytest.raises(MaxIterationsError) as exc:
        await rt.run_turn("loop", observer=seen)
    assert exc.value.limit == 2
    assert exc.value.last_text == "still working on it"
    assert seen.events[-1] == TurnAborted(iteration=2, error=exc.value)


async def test_parallel_region_preserves_order(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "d.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    calls = [
        ToolUse(id="c1", name="read_file", input={"path": "a.txt"}),
        ToolUse(id="c2", name="read_file", input={"path": "b.txt"}),
        ToolUse(id="c3", name="write_file", input={"path": "c.txt", "content": "c"}),
        ToolUse(id="c4", name="read_file", input={"path": "d.txt"}),
    ]
    provider = SequencedMockProvider(
        [[*calls, MessageStop()], [TextDelta(text="done"), MessageStop()]]
    )
    rt = _make_runtime(provider, tmp_path)
    await rt.run_turn("mixed")
    tool_msg = rt.session.messages[2]
    assert tool_msg.role == "tool"
    assert [b.tool_use_id for b in tool_msg.blocks] == ["c1", "c2", "c3", "c4"]
    assert [b.is_error for b in tool_msg.blocks] == [False, False, False, False]


async def test_parallel_and_serial_mix(tmp_path: Path) -> None:
    rt = _make_runtime(MockProvider([MessageStop()]), tmp_path)
    calls = [
        ToolUseBlock(id="r1", name="read_file", input={}),
        ToolUseBlock(id="r2", name="read_file", input={}),
        ToolUseBlock(id="w1", name="write_file", input={}),
        ToolUseBlock(id="r3", name="read_file", input={}),
    ]
    log: list[tuple[str, str]] = []

    async def fake(call: ToolUseBlock) -> Any:
        log.append(("start", call.id))
        await asyncio.sleep(0.01)
        log.append(("end", call.id))
        from satsumaestro.session.models import ToolResultBlock

        return ToolResultBlock(tool_use_id=call.id, tool_name=call.name, output="ok")

    out = await rt._scheduled_gather(calls, fake)
    assert [b.tool_use_id for b in out] == ["r1", "r2", "w1", "r3"]
    idx = {e: i for i, e in enumerate(log)}
    assert idx[("start", "r1")] < idx[("end", "r2")]
    assert idx[("start", "r2")] < idx[("end", "r1")]
    assert idx[("start", "w1")] > idx[("end", "r1")]
    assert idx[("start", "w1")] > idx[("end", "r2")]
    assert idx[("start", "r3")] > idx[("end", "w1")]


async def test_unknown_tool_is_serial_by_default(tmp_path: Path) -> None:
    rt = _make_runtime(MockProvider([MessageStop()]), tmp_path)

    def call(name: str, **inp: object) -> ToolUseBlock:
        return ToolUseBlock(id="c", name=name, input=dict(inp))

    assert rt._is_parallel_safe(call("read_file")) is True
    assert rt._is_parallel_safe(call("write_file")) is False
    assert rt._is_parallel_safe(call("mcp__x__unknown")) is False
    assert rt._is_parallel_safe(call("lsp", action="hover")) is True
    assert rt._is_parallel_safe(call("lsp", action="format")) is False
    assert rt._is_parallel_safe(call("lsp", action="Formatting")) is False


def test_detect_text_tool_call_openai_array() -> None:
    hit = '{"tool_calls": [{"function": {"name": "write_file", "arguments": "{}"}}]}'
    assert _detect_text_tool_call(hit, _TOOLS) == "write_file"
    flat = '{"tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]}'
    assert _detect_text_tool_call(flat, _TOOLS) == "bash"
    two = (
        '{"tool_calls": [{"name": "bash", "arguments": {}},'
        ' {"name": "read_file", "arguments": {}}]}'
    )
    assert _detect_text_tool_call(two, _TOOLS) is None
    assert _detect_text_tool_call('{"tool_calls": []}', _TOOLS) is None


def test_detect_text_tool_call_fence_variants() -> None:
    payload = '{"name": "write_file", "arguments": {"path": "a", "content": "x"}}'
    assert _detect_text_tool_call(f"```json\n{payload}\n```", _TOOLS) == "write_file"
    assert _detect_text_tool_call(f"```javascript\n\n{payload}\n\n```", _TOOLS) == "write_file"
    assert _detect_text_tool_call(f"```JSON {payload}```", _TOOLS) == "write_file"
    assert _detect_text_tool_call(f"```\n{payload}\n```", _TOOLS) == "write_file"


def test_detect_text_tool_call_no_false_positive() -> None:
    prose = 'You could send {"tool_calls": [{"name": "bash", "arguments": {}}]} to the API.'
    assert _detect_text_tool_call(prose, _TOOLS) is None
    assert _detect_text_tool_call('{"answer": 42}', _TOOLS) is None
    assert _detect_text_tool_call('{"name": "unknown_tool", "arguments": {}}', _TOOLS) is None
    fenced_prose = "```text\nhere is how tool calling works\n```"
    assert _detect_text_tool_call(fenced_prose, _TOOLS) is None


async def test_a_turn_reports_its_events_in_order(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    seen = SeenEvents()
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="t1", name="read_file", input={"path": "f.txt"}),
                ToolUse(id="t2", name="read_file", input={"path": "f.txt"}),
                MessageStop(),
            ],
            [TextDelta(text="done"), Usage(input_tokens=10), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path)
    summary = await rt.run_turn("go", observer=seen)

    assert seen.events[0] == TurnBegan(prompt_chars=2)
    kinds = seen.kinds()
    assert kinds[1] is ModelReplied and kinds[-2] is ModelReplied
    assert kinds.count(ToolDispatched) == 2 and kinds.count(ToolReturned) == 2
    assert kinds.index(ToolDispatched) < kinds.index(ToolReturned)
    first_reply = seen.events[1]
    assert isinstance(first_reply, ModelReplied)
    assert (first_reply.blocks, first_reply.tool_calls) == (2, 2)
    returned = [e for e in seen.events if isinstance(e, ToolReturned)]
    assert [(e.tool, e.failed) for e in returned] == [("read_file", False)] * 2
    assert seen.events[-1] == TurnEnded(summary)
    assert summary.assistant_messages == 2
    assert summary.tool_calls == 2


async def test_observer_failures_do_not_reach_the_turn(tmp_path: Path) -> None:
    class Broken:
        def observe(self, event: TurnEvent) -> None:
            raise RuntimeError("boom")

    provider = MockProvider([TextDelta(text="fine"), MessageStop()])
    rt = _make_runtime(provider, tmp_path)
    summary = await rt.run_turn("hello", observer=Broken())
    assert summary.final_text == "fine"


def test_hook_notes_are_listed_under_a_heading() -> None:
    assert append_hook_notes("out", [], False) == "out"
    assert append_hook_notes("out", ["fix A"], False) == "out\n\nNotes from hooks:\n- fix A"
    assert (
        append_hook_notes("out", ["fix A", "unused import: os\nremove it from app.py\n"], True)
        == "out\n\nNotes from hooks (the call failed):\n- fix A\n"
        "- unused import: os\n  remove it from app.py"
    )
    assert append_hook_notes("", ["fix"], False) == "Notes from hooks:\n- fix"
    assert append_hook_notes("  ", ["fix"], True) == "Notes from hooks (the call failed):\n- fix"
