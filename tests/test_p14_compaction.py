"""コンテキスト管理強化のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from satsumaestro.config.models import CondenseConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import (
    _CONTINUATION_NOTICE,
    _FOLDED_MARK,
    _SUMMARY_PREFIX,
    Compactor,
)
from satsumaestro.engine.condense import CondenseReport, condense
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.tool_check import check_tool_executor
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import (
    ConversationMessage,
    Session,
    TextBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.registry import ToolRegistry
from tests.fixtures.mock_provider import MockProvider


def _text_msg(role: str, text: str) -> ConversationMessage:
    return ConversationMessage(role=role, blocks=[TextBlock(text=text)])  # type: ignore[arg-type]


def _tool_use_msg(call_id: str, name: str, path: str) -> ConversationMessage:
    return ConversationMessage(
        role="assistant",
        blocks=[ToolUseBlock(id=call_id, name=name, input={"path": path})],
    )


def _tool_result_msg(call_id: str, name: str, output: str = "ok") -> ConversationMessage:
    return ConversationMessage(
        role="tool",
        blocks=[ToolResultBlock(tool_use_id=call_id, tool_name=name, output=output)],
    )


def _session(tmp_path: Path, messages: list[ConversationMessage]) -> Session:
    session = new_session(tmp_path, "mock-model")
    session.messages.extend(messages)
    return session


def _long_history(n: int) -> list[ConversationMessage]:
    return [
        _text_msg("user" if i % 2 == 0 else "assistant", f"message {i} " + "x" * 300)
        for i in range(n)
    ]


def test_needs_compaction_threshold() -> None:
    compactor = Compactor()
    cfg = RuntimeConfig(compact_token_threshold=100)
    assert compactor.needs_compaction(TokenUsage(input_tokens=60, output_tokens=40), cfg) is True
    assert compactor.needs_compaction(TokenUsage(input_tokens=60, output_tokens=39), cfg) is False
    off = RuntimeConfig(compact_token_threshold=0)
    assert compactor.needs_compaction(TokenUsage(input_tokens=10**9), off) is False


async def test_keep_last_window_and_orphan_guard(tmp_path: Path) -> None:
    messages = [
        *_long_history(6),
        _tool_use_msg("t1", "read_file", "a.txt"),
        _tool_result_msg("t1", "read_file"),
        _text_msg("assistant", "done " + "y" * 300),
    ]
    session = _session(tmp_path, messages)
    outcome = await Compactor(keep_last=2).acompact(session)
    assert outcome is not None
    assert [m.role for m in session.messages[1:]] == ["assistant", "tool", "assistant"]
    assert outcome.dropped_messages == 6
    assert session.compaction is not None
    assert session.compaction.dropped_messages == 6


async def test_acompact_short_history_returns_none(tmp_path: Path) -> None:
    session = _session(tmp_path, _long_history(3))
    assert await Compactor(keep_last=6).acompact(session) is None


_ONLY_STALE = CondenseConfig(collapse_enabled=False, cluster_enabled=False)
_ONLY_BRIEF = CondenseConfig(supersede_enabled=False, cluster_enabled=False)
_ONLY_GROUP = CondenseConfig(supersede_enabled=False, collapse_enabled=False)


def test_stale_file_ops_before_the_last_change_are_dropped() -> None:
    messages = [
        _tool_use_msg("r1", "read_file", "src/a.py"),
        _tool_result_msg("r1", "read_file", "old content"),
        _tool_use_msg("w1", "write_file", "src/a.py"),
        _tool_result_msg("w1", "write_file"),
        _tool_use_msg("w2", "edit_file", "src/a.py"),
        _tool_result_msg("w2", "edit_file"),
        _tool_use_msg("r2", "read_file", "src/b.py"),
        _tool_result_msg("r2", "read_file"),
    ]
    out, report = condense(messages, _ONLY_STALE)
    remaining_ids = {b.id for m in out for b in m.blocks if isinstance(b, ToolUseBlock)}
    assert remaining_ids == {"w2", "r2"}
    assert report.stale_dropped == 4
    assert report.messages_after == 4


def test_stale_file_ops_with_an_explanation_are_kept() -> None:
    explained = ConversationMessage(
        role="assistant",
        blocks=[
            TextBlock(text="Reading the parser first because the bug report points there. " * 3),
            ToolUseBlock(id="r1", name="read_file", input={"path": "src/a.py"}),
        ],
    )
    messages = [
        explained,
        _tool_result_msg("r1", "read_file"),
        _tool_use_msg("w1", "edit_file", "src/a.py"),
    ]
    out, report = condense(messages, _ONLY_STALE)
    assert report.stale_dropped == 0 and len(out) == 3


def test_brief_runs_are_folded_into_one_note() -> None:
    brief = [_text_msg("user" if i % 2 == 0 else "assistant", f"short {i}") for i in range(4)]
    tool_chain = [
        _tool_use_msg("t1", "read_file", "a.txt"),
        _tool_result_msg("t1", "read_file"),
        _text_msg("assistant", "tail " + "z" * 300),
    ]
    out, report = condense([*brief, *tool_chain], _ONLY_BRIEF)
    assert (report.brief_runs, report.brief_folded) == (1, 4)
    assert len(out) == 1 + len(tool_chain)
    first = out[0].blocks[0]
    assert isinstance(first, TextBlock)
    assert first.text.splitlines() == [
        "(4 brief messages in a row)",
        "user: short 0",
        "assistant: short 1",
        "user: short 2",
        "assistant: short 3",
    ]


def test_short_brief_runs_stay() -> None:
    brief = [_text_msg("user", f"short {i}") for i in range(3)]
    out, report = condense(brief, _ONLY_BRIEF)
    assert report.brief_runs == 0
    assert len(out) == 3


def test_similar_tool_steps_become_one_line_with_outcomes() -> None:
    steps = [
        _tool_use_msg("g1", "grep_search", "src/big.py"),
        _tool_result_msg("g1", "grep_search"),
        _tool_use_msg("g2", "grep_search", "src/big.py"),
        _tool_result_msg("g2", "grep_search"),
    ]
    request = _text_msg("user", "find where totals are computed")
    out, report = condense([request, *steps], _ONLY_GROUP)
    assert (report.step_groups, report.steps_grouped) == (2, 4)
    assert out[0] is request
    texts = [b.text for m in out[1:] for b in m.blocks if isinstance(b, TextBlock)]
    assert texts == [
        "(2 similar tool steps) tools: grep_search; files: src/big.py",
        "(2 similar tool steps) tools: grep_search; results: 2 ok, 0 failed",
    ]


def test_file_changing_steps_are_never_grouped() -> None:
    edits = [_tool_use_msg(f"e{i}", "edit_file", f"src/m{i}.py") for i in range(4)]
    out, report = condense(edits, _ONLY_GROUP)
    assert report.step_groups == 0 and len(out) == 4


def test_grouping_is_skipped_over_max_candidates() -> None:
    steps = [_tool_use_msg(f"g{i}", "grep_search", "src/big.py") for i in range(4)]
    capped = CondenseConfig(supersede_enabled=False, collapse_enabled=False, cluster_max_messages=3)
    out, report = condense(steps, capped)
    assert report.step_groups == 0
    assert len(out) == 4
    wider = CondenseConfig(supersede_enabled=False, collapse_enabled=False, cluster_max_messages=4)
    out2, report2 = condense(steps, wider)
    assert report2.step_groups == 1
    assert len(out2) == 1


def test_report_line() -> None:
    report = CondenseReport(
        messages_before=10,
        messages_after=4,
        chars_before=1_400,
        chars_after=400,
        stale_dropped=2,
        brief_runs=1,
        brief_folded=4,
        step_groups=1,
        steps_grouped=3,
    )
    assert report.tokens_saved == 250
    assert report.describe() == (
        "condensed 10 -> 4 messages (about 250 tokens fewer): "
        "2 stale file operation(s) dropped, 1 run(s) of brief messages folded (4 messages), "
        "1 group(s) of similar tool steps (3 messages)"
    )


async def test_llm_summary_default_path(tmp_path: Path) -> None:
    provider = MockProvider([TextDelta(text="- decided X\n- changed a.py"), MessageStop()])
    session = _session(tmp_path, _long_history(10))
    compactor = Compactor(keep_last=2, provider=provider, model="mock-model")  # type: ignore[arg-type]
    outcome = await compactor.acompact(session)
    assert outcome is not None
    assert outcome.used_llm is True
    assert outcome.summary.startswith("What the user asked (oldest first):\n  - message 0 ")
    assert outcome.summary.endswith("\n\n- decided X\n- changed a.py")
    assert session.compaction is not None
    assert session.compaction.summary == outcome.summary
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == []
    assert provider.requests[0].thinking is False


async def test_llm_summary_empty_falls_back(tmp_path: Path) -> None:
    provider = MockProvider([TextDelta(text="   "), MessageStop()])
    session = _session(tmp_path, _long_history(10))
    compactor = Compactor(keep_last=2, provider=provider, model="mock-model")  # type: ignore[arg-type]
    outcome = await compactor.acompact(session)
    assert outcome is not None
    assert outcome.used_llm is False
    assert outcome.summary.startswith("Earlier messages folded: 8 (")


async def test_llm_summary_error_falls_back(tmp_path: Path) -> None:
    class BrokenProvider:
        async def stream(self, req: object):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider down")
            yield  # pragma: no cover

    session = _session(tmp_path, _long_history(10))
    compactor = Compactor(keep_last=2, provider=BrokenProvider(), model="mock-model")  # type: ignore[arg-type]
    outcome = await compactor.acompact(session)
    assert outcome is not None
    assert outcome.used_llm is False
    assert outcome.summary.startswith("Earlier messages folded: 8 (")


async def test_provider_none_uses_mechanical(tmp_path: Path) -> None:
    session = _session(tmp_path, _long_history(10))
    outcome = await Compactor(keep_last=2).acompact(session)
    assert outcome is not None
    assert outcome.used_llm is False
    assert outcome.summary.startswith("Earlier messages folded: 8 (")


async def test_continuation_message_prefix(tmp_path: Path) -> None:
    session = _session(tmp_path, _long_history(10))
    outcome = await Compactor(keep_last=2).acompact(session)
    assert outcome is not None
    head = session.messages[0]
    assert head.role == "user"
    block = head.blocks[0]
    assert isinstance(block, TextBlock)
    assert block.text.startswith(_SUMMARY_PREFIX)
    assert _CONTINUATION_NOTICE in block.text
    assert outcome.summary in block.text


async def test_recompaction_flattens_prior_summary(tmp_path: Path) -> None:
    session = _session(tmp_path, _long_history(10))
    compactor = Compactor(keep_last=2)
    assert await compactor.acompact(session) is not None
    session.messages.extend(_long_history(8))
    provider = MockProvider([TextDelta(text="merged summary"), MessageStop()])
    llm_compactor = Compactor(keep_last=2, provider=provider, model="mock-model")  # type: ignore[arg-type]
    assert await llm_compactor.acompact(session) is not None
    head = session.messages[0].blocks[0]
    assert isinstance(head, TextBlock)
    assert head.text.count(_SUMMARY_PREFIX) == 1
    assert head.text.count(_CONTINUATION_NOTICE) == 1
    transcript_msg = provider.requests[0].messages[0].blocks[0]
    assert isinstance(transcript_msg, TextBlock)
    assert _SUMMARY_PREFIX not in transcript_msg.text
    assert _CONTINUATION_NOTICE not in transcript_msg.text
    assert _FOLDED_MARK in transcript_msg.text


def _make_runtime(tmp_path: Path, tools: ToolRegistry | None = None) -> ConversationRuntime:
    provider = MockProvider([TextDelta(text="ok"), MessageStop()])
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=tools if tools is not None else default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


async def test_tool_check_passes_and_fails(tmp_path: Path) -> None:
    rt = _make_runtime(tmp_path)
    rt.session.messages.append(_text_msg("user", "hi"))
    assert await check_tool_executor(rt) is None

    broken = _make_runtime(tmp_path, tools=ToolRegistry())
    broken.session.messages.append(_text_msg("user", "hi"))
    reason = await check_tool_executor(broken)
    assert reason is not None
    assert "no tool is registered" in reason


async def test_tool_check_accepts_empty_session_with_compaction_record(tmp_path: Path) -> None:
    rt = _make_runtime(tmp_path)
    assert (await check_tool_executor(rt)) is not None
    session = _session(tmp_path, _long_history(10))
    assert await Compactor(keep_last=2).acompact(session) is not None
    rt.session.compaction = session.compaction
    rt.session.messages = []
    assert await check_tool_executor(rt) is None


async def test_run_turn_uses_acompact_and_tool_check(tmp_path: Path) -> None:
    from satsumaestro.engine.events import Usage

    seed = _long_history(8)
    provider = MockProvider([TextDelta(text="ok"), Usage(input_tokens=100), MessageStop()])
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(compact_token_threshold=1),
        compactor=Compactor(keep_last=2),
    )
    rt.session.messages.extend(seed)
    summary = await rt.run_turn("go")
    assert summary.compacted is True
    assert summary.health_warning is None
    head = rt.session.messages[0].blocks[0]
    assert isinstance(head, TextBlock)
    assert head.text.startswith(_SUMMARY_PREFIX)


def test_from_config_wires_compaction_settings() -> None:
    cfg = RuntimeConfig.model_validate(
        {
            "compaction": {
                "keep_last": 3,
                "use_llm_summary": False,
                "trident": {"collapse_threshold": 5},
            }
        }
    )
    compactor = Compactor.from_config(cfg)
    assert compactor._keep_last == 3
    assert compactor._use_llm_summary is False
    assert compactor._condense.collapse_threshold == 5
    assert compactor._condense.cluster_min_size == 2


@pytest.mark.parametrize(
    "compaction",
    [
        {"keep_last": -1},
        {"llm_summary_max_tokens": 0},
        {"trident": {"collapse_threshold": 1}},
        {"trident": {"cluster_min_size": 1}},
        {"trident": {"cluster_similarity_threshold": 1.5}},
        {"trident": {"cluster_similarity_threshold": -0.1}},
        {"trident": {"cluster_max_messages": 0}},
    ],
)
def test_compaction_config_rejects_out_of_range(compaction: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({"compaction": compaction})
