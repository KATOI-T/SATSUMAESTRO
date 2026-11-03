"""読み直しが 120 秒に収まらない見込みのときだけ、最初の中身までの待ちを延ばす。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.events import MessageStop, TextDelta, Usage
from satsumaestro.engine.prefill import DEFAULT_RATE, PrefillClock, keep_alive_seconds
from satsumaestro.engine.runtime import ConversationRuntime, _TurnCounters
from satsumaestro.engine.turn import NullRenderer
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers import timing
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.openai_provider import OpenAICompatProvider, _watch_chunks
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ConversationMessage, TextBlock, TokenUsage
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider


def test_the_curve_matches_the_gpt_oss_measurement() -> None:
    clock = PrefillClock()
    clock.note(0, 7_611, 8.8)
    assert 110 < clock.seconds(0, 63_925) < 130
    assert 30 < clock.seconds(21_096, 39_213) < 37
    assert clock.seconds(60_000, 63_925) < 15


def test_without_a_measurement_the_slow_default_is_used() -> None:
    clock = PrefillClock()
    assert clock.rate == DEFAULT_RATE and not clock.measured
    clock.note(0, 1_500, 5.0)
    clock.note(0, 20_000, 0.4)
    assert not clock.measured


def test_the_rate_is_the_median_of_recent_samples() -> None:
    clock = PrefillClock()
    clock.note(0, 10_000, 100.0)
    clock.note(0, 10_000, 10.0)
    clock.note(0, 10_000, 11.0)
    assert 1_000 < clock.rate < 1_200


def test_the_wait_is_extended_only_for_a_long_reread() -> None:
    clock = PrefillClock()
    clock.note(0, 7_611, 8.8)
    assert clock.first_byte_timeout(0, 20_000, 120.0, 600.0) is None
    wait = clock.first_byte_timeout(0, 90_000, 120.0, 600.0)
    assert wait is not None and 420 < wait < 430
    assert clock.first_byte_timeout(0, 500_000, 120.0, 600.0) == 600.0
    assert clock.first_byte_timeout(85_000, 90_000, 120.0, 600.0) is None


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("30m", 1800.0), ("1h30m", 5400.0), ("300", 300.0), ("90s", 90.0), ("0", 0.0)],
)
def test_keep_alive_durations(value: str, seconds: float) -> None:
    assert keep_alive_seconds(value) == seconds


@pytest.mark.parametrize("value", [None, "", "-1", "-5m", "soon", "5x"])
def test_keep_alive_without_a_deadline(value: str | None) -> None:
    assert keep_alive_seconds(value) is None


def _chunk(**delta: Any) -> Any:
    choice = SimpleNamespace(delta=SimpleNamespace(**delta), finish_reason=None)
    return SimpleNamespace(choices=[choice], usage=None)


async def _stream(items: list[tuple[float, Any]]) -> Any:
    for delay, item in items:
        await asyncio.sleep(delay)
        yield item


async def test_the_first_content_is_marked_and_role_only_chunks_are_not() -> None:
    stream_timing = timing.StreamTiming()
    items = [(0.0, _chunk(role="assistant", content="")), (0.05, _chunk(content="a"))]
    with timing.bind(stream_timing):
        got = [c async for c in _watch_chunks(_stream(items), None)]
    assert len(got) == 2
    assert stream_timing.first_chunk_at is not None
    assert stream_timing.first_chunk_at - stream_timing.started_at >= 0.04


async def test_silence_after_the_reply_began_is_cut_at_the_usual_limit() -> None:
    items = [(0.0, _chunk(reasoning="hmm")), (0.3, _chunk(content="late"))]
    with pytest.raises(httpx.ReadTimeout):
        _ = [c async for c in _watch_chunks(_stream(items), 0.1)]


async def test_the_wait_before_the_first_content_is_not_cut_by_the_silence_limit() -> None:
    items = [(0.2, _chunk(content="first")), (0.0, _chunk(content="next"))]
    got = [c async for c in _watch_chunks(_stream(items), 0.1)]
    assert len(got) == 2


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **call: Any) -> Any:
        self.calls.append(call)
        return _stream([])


def _provider() -> tuple[OpenAICompatProvider, _Completions]:
    provider = OpenAICompatProvider(
        "k",
        "http://localhost:11434/v1",
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
    )
    completions = _Completions()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]
    return provider, completions


def _req(wait: float | None) -> ApiRequest:
    return ApiRequest(model="m", system="s", messages=[], first_byte_timeout_s=wait)


@pytest.mark.parametrize(("wait", "read"), [(300.0, 300.0), (None, None), (100.0, None)])
async def test_the_request_carries_the_longer_read_limit_only_when_extended(
    wait: float | None, read: float | None
) -> None:
    provider, completions = _provider()
    req = _req(wait)
    await provider._create(req, provider._initial_shape(req, [], False, None))
    timeout = completions.calls[0].get("timeout")
    if read is None:
        assert timeout is None
    else:
        assert timeout.read == read and timeout.connect == 10.0 and timeout.write == 30.0


def _runtime(tmp_path: Path, provider: SequencedMockProvider) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(compact_token_threshold=0),
        compactor=Compactor(keep_last=6),
    )


class _Notices(NullRenderer):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def on_context_notice(self, text: str) -> None:
        self.lines.append(text)


def _say(role: str, text: str) -> ConversationMessage:
    return ConversationMessage(role=role, blocks=[TextBlock(text=text)])  # type: ignore[arg-type]


async def test_a_long_reread_gets_a_longer_first_wait_and_a_cached_one_does_not(
    tmp_path: Path,
) -> None:
    provider = SequencedMockProvider(
        [[TextDelta(text="ok"), Usage(input_tokens=60_000, output_tokens=5), MessageStop()]]
    )
    rt = _runtime(tmp_path, provider)
    for i in range(30):
        rt.session.messages.extend(
            [_say("user", f"q{i} " + "x" * 6_000), _say("assistant", f"a{i}")]
        )
    render = _Notices()
    summary = await rt.run_turn("続けて", render=render)
    assert summary.prefill_waits == 1
    wait = provider.requests[0].first_byte_timeout_s
    assert wait is not None and 120 < wait <= 600
    (line,) = render.lines
    assert re.fullmatch(
        r"会話の読み込み（約 [\d,]+ トークン）が長くなる見込みです。"
        r"最初の応答まで最大 \d+ 秒待ちます",
        line,
    )
    await rt.run_turn("ありがとう", render=render)
    assert provider.requests[1].first_byte_timeout_s is None


def test_the_cache_is_not_counted_after_a_model_switch_or_a_long_idle(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, SequencedMockProvider([[MessageStop()]]))
    rt.session.messages.append(_say("user", "q"))
    rt._note_sent(1, TokenUsage(input_tokens=40_000), estimate=50_000, digest="d")
    assert rt._cached_tokens("d") == 40_000
    assert rt._cached_tokens("other") == 0
    rt.session.model = "other-model"
    assert rt._cached_tokens("d") == 0
    rt.session.model = "mock-model"
    assert rt._sent is not None
    rt._sent = rt._sent.__class__(**{**rt._sent.__dict__, "at": rt._sent.at - 3_600})
    assert rt._cached_tokens("d") == 0


def test_the_timeout_tells_how_long_it_actually_waited(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, SequencedMockProvider([[MessageStop()]]))
    counters = _TurnCounters()
    counters.stream_timing = timing.StreamTiming()
    assert rt._waited_s(_req(300.0), counters) == 300.0
    counters.stream_timing.first_chunk_at = counters.stream_timing.started_at + 5
    assert rt._waited_s(_req(300.0), counters) == 120.0
    assert rt._waited_s(_req(None), counters) == 120.0


def test_the_first_content_time_becomes_a_rate_sample(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, SequencedMockProvider([[MessageStop()]]))
    stream = timing.StreamTiming(started_at=100.0, first_chunk_at=108.8)
    rt._note_prefill(0, TokenUsage(input_tokens=7_611), stream)
    assert rt._prefill.measured and 900 < rt._prefill.rate < 1_000


async def test_the_extension_can_be_turned_off(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[TextDelta(text="ok"), MessageStop()]])
    rt = _runtime(tmp_path, provider)
    rt.config.provider.retry.extend_first_wait = False
    for i in range(30):
        rt.session.messages.extend(
            [_say("user", f"q{i} " + "x" * 6_000), _say("assistant", f"a{i}")]
        )
    summary = await rt.run_turn("続けて")
    assert summary.prefill_waits == 0
    assert provider.requests[0].first_byte_timeout_s is None


def test_a_moderate_wait_is_extended_without_a_notice(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, SequencedMockProvider([[MessageStop()]]))
    rt._prefill.note(0, 7_611, 8.1)
    render = _Notices()
    counters = _TurnCounters()
    wait = rt._first_byte_wait(46_732, 66_642, render, counters)
    assert wait is not None and 120 < wait < 220
    assert counters.prefill_waits == 1 and render.lines == []
    assert rt._first_byte_wait(0, 90_000, render, counters) is not None
    (line,) = render.lines
    assert line.startswith("会話の読み込みに約 ") and "最初の応答まで最大 " in line
