"""送る前に要求の量を見積もり、閾値を超えるなら先に畳み、窓を超えるなら送らない。"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from satsumaestro.cli.errors_ui import turn_failure
from satsumaestro.config.models import CompactionConfig, HookConfig, ProviderConfig, RuntimeConfig
from satsumaestro.engine.compaction import _SUMMARY_PREFIX, Compactor
from satsumaestro.engine.context_budget import input_limit, message_tokens, request_tokens
from satsumaestro.engine.context_fit import apply_context_fit
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import NullRenderer
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ConversationMessage, TextBlock, TokenUsage, ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ContextOverflowError
from tests.fixtures.mock_provider import SequencedMockProvider


def _say(role: str, text: str) -> ConversationMessage:
    return ConversationMessage(role=role, blocks=[TextBlock(text=text)])  # type: ignore[arg-type]


def _read_then_answer(input_tokens: int) -> SequencedMockProvider:
    return SequencedMockProvider(
        [
            [
                ToolUse(id="r1", name="read_file", input={"path": "data.txt"}),
                Usage(input_tokens=input_tokens, output_tokens=10),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), Usage(input_tokens=500, output_tokens=5), MessageStop()],
        ]
    )


def _runtime(
    tmp_path: Path, provider: SequencedMockProvider, config: RuntimeConfig
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config,
        compactor=Compactor(keep_last=4),
    )


def _history(rt: ConversationRuntime, pairs: int) -> None:
    for i in range(pairs):
        rt.session.messages.extend([_say("user", f"question {i}"), _say("assistant", f"a {i}")])


class _Notices(NullRenderer):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def on_context_notice(self, text: str) -> None:
        self.lines.append(text)


def test_the_input_limit_comes_from_the_window_only() -> None:
    assert input_limit(131_072) == 117_964
    assert input_limit(0) is None


def test_the_request_estimate_adds_up_the_parts() -> None:
    msg = _say("user", "a" * 300)
    assert message_tokens(msg) == 104
    assert request_tokens("s" * 30, "t" * 30, [msg, msg]) == 10 + 10 + 208


async def test_tool_results_that_cross_the_threshold_are_folded_before_sending(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.txt").write_text("x" * 9_000, encoding="utf-8")
    provider = _read_then_answer(input_tokens=4_000)
    config = RuntimeConfig(
        compact_token_threshold=6_000,
        compaction=CompactionConfig(use_llm_summary=False, max_tool_result_share=1.0),
    )
    rt = _runtime(tmp_path, provider, config)
    _history(rt, 4)
    summary = await rt.run_turn("data.txt を読んで")
    assert summary.presend_compactions == 1
    first = provider.requests[1].messages[0].blocks[0]
    assert isinstance(first, TextBlock) and first.text.startswith(_SUMMARY_PREFIX)
    last = provider.requests[1].messages[-1].blocks[0]
    assert isinstance(last, ToolResultBlock) and last.output == "x" * 9_000


async def test_nothing_is_folded_while_the_estimate_stays_under_the_threshold(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.txt").write_text("x" * 9_000, encoding="utf-8")
    provider = _read_then_answer(input_tokens=1_000)
    config = RuntimeConfig(
        compact_token_threshold=6_000,
        compaction=CompactionConfig(use_llm_summary=False, max_tool_result_share=1.0),
    )
    rt = _runtime(tmp_path, provider, config)
    _history(rt, 4)
    summary = await rt.run_turn("data.txt を読んで")
    assert summary.presend_compactions == 0
    first = provider.requests[1].messages[0].blocks[0]
    assert isinstance(first, TextBlock) and first.text == "question 0"


async def test_results_are_cut_to_fit_the_window_when_folding_cannot_help(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.txt").write_text("line of text\n" * 3_000, encoding="utf-8")
    provider = _read_then_answer(input_tokens=1_000)
    config = RuntimeConfig(
        compact_token_threshold=0,
        compaction=CompactionConfig(max_tool_result_share=0.0, context_window=10_000),
    )
    rt = _runtime(tmp_path, provider, config)
    render = _Notices()
    summary = await rt.run_turn("data.txt を読んで", render=render)
    assert summary.tool_result_cuts == 1 and len(render.lines) == 1
    sent = provider.requests[1]
    result = sent.messages[-1].blocks[0]
    assert isinstance(result, ToolResultBlock)
    assert "[read_file: this result was cut to fit the context window." in result.output
    tail = sent.messages[2:]
    assert 1_000 + sum(message_tokens(m) for m in tail) < 9_000
    assert provider.requests[1].messages[0] is rt.session.messages[0]


async def test_a_request_that_cannot_fit_is_not_sent_and_not_kept(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[TextDelta(text="never"), MessageStop()]])
    config = RuntimeConfig(compaction=CompactionConfig(context_window=8_000))
    rt = _runtime(tmp_path, provider, config)
    _history(rt, 1)
    with pytest.raises(ContextOverflowError) as caught:
        await rt.run_turn("ログを貼ります\n" + "0123456789abcdef\n" * 4_000)
    assert caught.value.dropped_request
    assert "the request was not sent and was removed from the conversation" in str(caught.value)
    assert provider.requests == []
    assert [m.role for m in rt.session.messages] == ["user", "assistant"]
    assert rt.aborted_summary is not None


def test_the_estimate_starts_from_the_last_reply_and_stops_after_history_changes(
    tmp_path: Path,
) -> None:
    rt = _runtime(tmp_path, _read_then_answer(1_000), RuntimeConfig())
    _history(rt, 2)
    rt._note_sent(len(rt.session.messages), TokenUsage(input_tokens=5_000), estimate=10_000)
    assert rt._estimate_ratio == 0.5
    rt.session.messages.append(_say("user", "a" * 300))
    assert rt._projected_tokens() == 5_000 + 104
    rt.session.messages.clear()
    rt.session.messages.extend([_say("user", "q")] * 4)
    assert rt._projected_tokens() is None


async def test_the_detected_window_is_kept_for_the_input_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "m", "context_length": 32768}]})
        return httpx.Response(404)

    cfg = RuntimeConfig(provider=ProviderConfig(base_url="http://localhost:11434/v1"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await apply_context_fit(cfg, "m", {}, client=client)
    assert cfg.compaction.context_window == 32768
    cfg = RuntimeConfig(
        provider=ProviderConfig(base_url="http://localhost:11434/v1"),
        compaction=CompactionConfig(context_window=8192),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await apply_context_fit(cfg, "m", {}, client=client)
    assert cfg.compaction.context_window == 8192


def test_the_repl_explains_the_overflow_and_what_to_do() -> None:
    dropped = turn_failure(ContextOverflowError(150_000, 117_964, dropped_request=True))
    assert dropped.reason == (
        "依頼がモデルの文脈の窓に収まりません（見積もり 約 150,000・上限 約 117,964 トークン）"
    )
    assert dropped.hint.startswith("依頼は送らず、会話にも残していません。")
    kept = turn_failure(ContextOverflowError(150_000, 117_964, dropped_request=False))
    assert kept.reason.startswith("要求がモデルの文脈の窓に収まりません")
    assert "/compact" in kept.hint
