"""会話ループ run_turn の結線テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import TurnSummary
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionOutcome, PermissionPolicy, default_policy
from satsumaestro.session.models import ConversationMessage, TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import MaxIterationsError, ToolCallingUnsupportedError
from tests.fixtures.mock_provider import MockProvider, SequencedMockProvider


def _make_runtime(
    provider: object,
    tmp_path: Path,
    *,
    policy: PermissionPolicy | None = None,
    config: RuntimeConfig | None = None,
    compactor: Compactor | None = None,
    session_seed: list[ConversationMessage] | None = None,
) -> ConversationRuntime:
    session = new_session(tmp_path, "mock-model")
    if session_seed:
        session.messages.extend(session_seed)
    return ConversationRuntime(
        session=session,
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=policy or default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config or RuntimeConfig(),
        compactor=compactor,
    )


async def test_run_turn_text_only(tmp_path: Path) -> None:
    provider = MockProvider([TextDelta(text="hello"), MessageStop()])
    rt = _make_runtime(provider, tmp_path)
    summary = await rt.run_turn("hi")
    assert isinstance(summary, TurnSummary)
    assert summary.final_text == "hello"
    assert summary.stop_reason == "end_turn"
    assert summary.iterations == 1
    assert [m.role for m in rt.session.messages] == ["user", "assistant"]
    assert len(provider.requests) == 1


async def test_run_turn_tool_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("file body", encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [ToolUse(id="t1", name="read_file", input={"path": "hello.txt"}), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path)
    summary = await rt.run_turn("read it")
    assert summary.iterations == 2
    assert summary.final_text == "done"
    roles = [m.role for m in rt.session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    tool_msg = rt.session.messages[2]
    assert tool_msg.blocks[0].output == "file body"
    assert tool_msg.blocks[0].is_error is False
    assert len(provider.requests) == 2


async def test_max_iterations(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    provider = MockProvider(
        [ToolUse(id="t", name="read_file", input={"path": "f.txt"}), MessageStop()]
    )
    rt = _make_runtime(provider, tmp_path, config=RuntimeConfig(max_iterations=3))
    with pytest.raises(MaxIterationsError):
        await rt.run_turn("loop")


class _RecordingRenderer:
    """on_tool_start と on_tool_result の対応を検証する記録レンダラ。"""

    def __init__(self) -> None:
        self.starts: list[str] = []
        self.results: list[tuple[str, bool]] = []

    def on_assistant_event(self, ev: object) -> None:
        pass

    def on_tool_start(self, call: ToolUse) -> None:
        self.starts.append(call.name)

    def on_tool_result(self, block: object) -> None:
        self.results.append((block.tool_name, block.is_error))  # type: ignore[attr-defined]


async def test_renderer_gets_result_for_denied_and_unknown_tools(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="u1", name="search", input={"query": "x"}),
                ToolUse(id="w1", name="write_file", input={"path": "o.txt", "content": "x"}),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path, policy=default_policy(PermissionMode.READ_ONLY))
    render = _RecordingRenderer()
    summary = await rt.run_turn("go", render=render)
    assert summary.final_text == "done"
    assert render.starts == ["search", "write_file"]
    assert render.results == [("search", True), ("write_file", True)]


async def test_permission_deny_in_loop(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [
                ToolUse(
                    id="w1",
                    name="write_file",
                    input={"path": "out.txt", "content": "nope"},
                ),
                MessageStop(),
            ],
            [TextDelta(text="after deny"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path, policy=default_policy(PermissionMode.READ_ONLY))
    summary = await rt.run_turn("write it")
    assert summary.iterations == 2
    assert summary.final_text == "after deny"
    tool_msg = rt.session.messages[2]
    assert tool_msg.role == "tool"
    assert tool_msg.blocks[0].is_error is True
    assert not (tmp_path / "out.txt").exists()


async def test_session_persist_resume(tmp_path: Path) -> None:
    provider = MockProvider([TextDelta(text="persisted"), MessageStop()])
    rt = _make_runtime(provider, tmp_path)
    await rt.run_turn("remember")
    store = JSONLSessionStore()
    path = store.path_for(tmp_path, rt.session.session_id)
    loaded = store.load(path)
    assert loaded.session_id == rt.session.session_id
    assert loaded.prompt_history == ["remember"]
    assert [m.role for m in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[1].blocks[0].text == "persisted"


async def test_compaction_trigger(tmp_path: Path) -> None:
    seed = [ConversationMessage(role="user", blocks=[TextBlock(text=f"old {i}")]) for i in range(8)]
    provider = MockProvider([TextDelta(text="ok"), Usage(input_tokens=100), MessageStop()])
    rt = _make_runtime(
        provider,
        tmp_path,
        config=RuntimeConfig(compact_token_threshold=1),
        compactor=Compactor(keep_last=2),
        session_seed=seed,
    )
    before = len(rt.session.messages)
    summary = await rt.run_turn("go")
    assert summary.compacted is True
    assert rt.session.compaction is not None
    assert rt.session.compaction.dropped_messages > 0
    assert len(rt.session.messages) < before + 2
    assert rt.session.messages[0].blocks[0].text.startswith("[Earlier conversation summarized")


async def test_tool_execution_error_becomes_is_error_and_continues(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="r1", name="read_file", input={"path": "missing.md"}), MessageStop()],
            [TextDelta(text="recovered"), MessageStop()],
        ]
    )
    rt = _make_runtime(provider, tmp_path)
    summary = await rt.run_turn("read missing.md")
    assert summary.iterations == 2
    assert summary.final_text == "recovered"
    tool_msg = rt.session.messages[2]
    assert tool_msg.role == "tool"
    assert tool_msg.blocks[0].is_error is True
    assert tool_msg.blocks[0].output.startswith("file not found: missing.md")


async def test_rejects_text_tool_call(tmp_path: Path) -> None:
    leaked = '{"name": "write_file", "arguments": {"path": "a", "content": "x"}}'
    provider = MockProvider([TextDelta(text=leaked), MessageStop()])
    rt = _make_runtime(provider, tmp_path)
    with pytest.raises(ToolCallingUnsupportedError):
        await rt.run_turn("write a file")
    assert not (tmp_path / "a").exists()


async def test_rejects_text_tool_call_with_braces_in_string_and_trailing(tmp_path: Path) -> None:
    leaked = (
        '{"name": "write_file", "arguments": {"path": "a", "content": "print(\'}\')"}}'
        " Here is what I did."
    )
    provider = MockProvider([TextDelta(text=leaked), MessageStop()])
    rt = _make_runtime(provider, tmp_path)
    with pytest.raises(ToolCallingUnsupportedError):
        await rt.run_turn("write a file")


async def test_json_answer_that_is_not_a_tool_call_is_ok(tmp_path: Path) -> None:
    provider = MockProvider([TextDelta(text='{"answer": 42}'), MessageStop()])
    rt = _make_runtime(provider, tmp_path)
    summary = await rt.run_turn("give me json")
    assert summary.stop_reason == "end_turn"
    assert summary.final_text == '{"answer": 42}'


async def test_prose_mentioning_json_is_ok(tmp_path: Path) -> None:
    text = 'I would call {"name": "write_file", "arguments": {}} but here is the answer.'
    provider = MockProvider([TextDelta(text=text), MessageStop()])
    rt = _make_runtime(provider, tmp_path)
    summary = await rt.run_turn("explain")
    assert summary.final_text == text


async def test_hook_input_rewrite_reaches_prompter(tmp_path: Path) -> None:
    hook = tmp_path / "rewrite.py"
    hook.write_text(
        "import json,sys\njson.load(sys.stdin)\n"
        "print(json.dumps({'input': {'command': 'echo safe'}}))\n",
        encoding="utf-8",
    )
    captured: dict[str, dict] = {}

    class CapturePrompter:
        def decide(
            self, call: ToolUse, required: PermissionMode, warning: str | None = None
        ) -> PermissionOutcome:
            captured["input"] = dict(call.input)
            return PermissionOutcome(decision="deny", reason="captured")

    provider = MockProvider(
        [ToolUse(id="b1", name="bash", input={"command": "rm -rf /"}), MessageStop()]
    )
    session = new_session(tmp_path, "mock-model")
    rt = ConversationRuntime(
        session=session,
        provider=provider,
        tools=default_registry(),
        policy=PermissionPolicy(mode=PermissionMode.PROMPT),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(pre_tool_use=[f"{sys.executable} {hook}"]), tmp_path),
        config=RuntimeConfig(max_iterations=1),
    )
    with pytest.raises(MaxIterationsError):
        await rt.run_turn("go", prompter=CapturePrompter())
    assert captured["input"] == {"command": "echo safe"}
