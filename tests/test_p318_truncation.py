"""応答の usage が送った量の下限より大きく少なければ、サーバの切り詰めとみて応答を使わない。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from satsumaestro.cli.errors_ui import turn_failure
from satsumaestro.config.models import CompactionConfig, HookConfig, ProviderConfig, RuntimeConfig
from satsumaestro.engine import context_fit
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.context_budget import estimate_lower, request_floor
from satsumaestro.engine.context_fit import refit_window
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import NullRenderer
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ProviderKind
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ConversationMessage, TextBlock, TokenUsage
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ContextTruncatedError
from tests.fixtures.mock_provider import SequencedMockProvider

LOST = "何かお手伝いできることがありましたら、具体的な内容をご指示ください。"


def _provider(*, kind: ProviderKind = ProviderKind.OPENAI) -> SequencedMockProvider:
    return SequencedMockProvider(
        [
            [
                ToolUse(id="r1", name="read_file", input={"path": "data.txt"}),
                Usage(input_tokens=4_000, output_tokens=20),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text=LOST), Usage(input_tokens=3_000, output_tokens=20), MessageStop()],
            [TextDelta(text="最後の行は 9999 です"), Usage(input_tokens=9_000), MessageStop()],
        ],
        kind=kind,
    )


def _runtime(
    tmp_path: Path, provider: SequencedMockProvider, config: RuntimeConfig | None = None
) -> ConversationRuntime:
    (tmp_path / "data.txt").write_text("0123456789abcdef\n" * 4_700, encoding="utf-8")
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config or RuntimeConfig(),
        compactor=Compactor(keep_last=6),
    )


class _Notices(NullRenderer):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def on_context_notice(self, text: str) -> None:
        self.lines.append(text)


def _no_refit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def none(cfg: RuntimeConfig, model: str, env: Mapping[str, str]) -> int | None:
        return None

    monkeypatch.setattr(context_fit, "refit_window", none)


def test_the_lower_estimate_counts_eight_characters_as_one() -> None:
    assert estimate_lower("a" * 800) == 100
    msg = ConversationMessage(role="user", blocks=[TextBlock(text="b" * 80)])
    assert request_floor("s" * 80, "t" * 80, [msg]) == 30


async def test_a_truncated_request_is_not_taken_as_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_refit(monkeypatch)
    provider = _provider()
    rt = _runtime(tmp_path, provider)
    with pytest.raises(ContextTruncatedError) as caught:
        await rt.run_turn("data.txt を全部読んで、最後の行を答えて")
    assert caught.value.read == 3_000 and caught.value.sent > 13_000
    assert all(LOST not in str(m.blocks) for m in rt.session.messages)
    assert rt.aborted_summary is not None and rt.aborted_summary.context_truncations == 1


async def test_a_smaller_loaded_window_is_taken_and_the_request_is_sent_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refit(cfg: RuntimeConfig, model: str, env: Mapping[str, str]) -> int | None:
        cfg.compaction.context_window = 16_384
        return 16_384

    monkeypatch.setattr(context_fit, "refit_window", refit)
    provider = _provider()
    rt = _runtime(tmp_path, provider)
    render = _Notices()
    summary = await rt.run_turn("data.txt を全部読んで、最後の行を答えて", render=render)
    assert summary.final_text == "最後の行は 9999 です"
    assert summary.context_truncations == 1
    assert len(provider.requests) == 3
    assert any("モデルの窓 16,384 に合わせて送り直します" in line for line in render.lines)
    assert summary.tool_result_cuts == 1


async def test_a_second_truncation_in_the_same_turn_stops_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def refit(cfg: RuntimeConfig, model: str, env: Mapping[str, str]) -> int | None:
        nonlocal calls
        calls += 1
        return 120_000

    monkeypatch.setattr(context_fit, "refit_window", refit)
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="r1", name="read_file", input={"path": "data.txt"}),
                Usage(input_tokens=4_000),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text=LOST), Usage(input_tokens=3_000), MessageStop()],
        ],
        kind=ProviderKind.OPENAI,
    )
    rt = _runtime(tmp_path, provider)
    with pytest.raises(ContextTruncatedError):
        await rt.run_turn("data.txt を全部読んで")
    assert calls == 1 and len(provider.requests) == 3


async def test_anthropic_usage_is_not_read_as_a_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_refit(monkeypatch)
    rt = _runtime(tmp_path, _provider(kind=ProviderKind.ANTHROPIC))
    summary = await rt.run_turn("data.txt を全部読んで")
    assert summary.final_text == LOST and summary.context_truncations == 0


def test_the_last_measurement_is_a_floor_only_while_the_head_is_unchanged(
    tmp_path: Path,
) -> None:
    rt = _runtime(tmp_path, _provider())
    rt.session.messages.append(ConversationMessage(role="user", blocks=[TextBlock(text="q")]))
    rt._note_sent(1, TokenUsage(input_tokens=50_000), estimate=60_000, digest="head-a")
    assert rt._sent_floor("s", "t", "head-a") == 47_500
    assert rt._sent_floor("s", "t", "head-b") < 100


def _ps(models: list[dict[str, object]]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": models})
        return httpx.Response(200, json={"model_info": {"x.context_length": 131072}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _cfg(window: int) -> RuntimeConfig:
    return RuntimeConfig(
        provider=ProviderConfig(base_url="http://localhost:11434/v1"),
        compact_token_threshold=104_857,
        compaction=CompactionConfig(context_window=window),
    )


async def test_refit_takes_a_smaller_loaded_window() -> None:
    cfg = _cfg(131_072)
    async with _ps([{"name": "m", "context_length": 4096}]) as client:
        assert await refit_window(cfg, "m", {}, client=client) == 4096
    assert cfg.compaction.context_window == 4096
    assert cfg.compact_token_threshold == 3276


async def test_refit_leaves_a_window_that_is_not_smaller() -> None:
    cfg = _cfg(131_072)
    async with _ps([{"name": "m", "context_length": 131_072}]) as client:
        assert await refit_window(cfg, "m", {}, client=client) is None
    async with _ps([]) as client:
        assert await refit_window(cfg, "m", {}, client=client) is None
    assert (cfg.compaction.context_window, cfg.compact_token_threshold) == (131_072, 104_857)


async def test_refit_does_nothing_off_ollama() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(base_url="http://localhost:8080/v1"))
    async with _ps([{"name": "m", "context_length": 4096}]) as client:
        assert await refit_window(cfg, "m", {}, client=client) is None


def test_the_repl_says_the_answer_was_not_used() -> None:
    failure = turn_failure(ContextTruncatedError(152_000, 3_051))
    assert (
        failure.reason
        == "サーバが要求を切り詰めました（送った量 約 152,000 以上・読まれたのは 3,051 トークン）"
    )
    assert failure.hint.startswith("応答は使っていません。")
