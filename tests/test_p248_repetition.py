"""応答の反復をストリーム中に打ち切る。"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from rich.console import Console

from satsumaestro.cli.errors_ui import turn_failure
from satsumaestro.cli.render import RichRenderer
from satsumaestro.config.models import GuidanceConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta
from satsumaestro.engine.reminders import repetition_note
from satsumaestro.engine.repetition import (
    LINE_WINDOW,
    PARAGRAPH_MAX_SEEN,
    PARAGRAPH_WINDOW,
    RepetitionDetector,
    RepetitionVerdict,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import NullRenderer, TurnSummary
from satsumaestro.evals.agents import AgentAnswer
from satsumaestro.evals.defects import detect_defects
from satsumaestro.evals.models import TaskResult
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import TextBlock, TokenUsage
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.telemetry.recorder import summary_fields
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ProviderError, RepetitionLoopError

FIXTURE = Path(__file__).parent / "fixtures" / "issue248_loop.txt"

_CYCLE = [
    '```python\n_DUCKDUCKGO_URL = "https://duckduckgo.com/"\n```',
    "→ 定義されていると仮定します。",
    '```python\nurl = cfg.provider or _DUCKDUCKGO_URL\nassert url == "duckduckgo"  # Pass\n```',
    '→ `or` は `"duckduckgo"` を返します。',
    "しかし、テストが通っているため、このコードは正しく機能している可能性があります。",
    "`_DUCKDUCKGO_URL` が定義されていない場合、`cfg.provider or _DUCKDUCKGO_URL` は "
    '`cfg.provider` が `"duckduckgo"` の場合に不正な URL になります。',
    "しかし、テストが通っているため、`_DUCKDUCKGO_URL` が定義されていて、かつ `or` が"
    "正しく機能している可能性があります。",
    "`_DUCKDUCKGO_URL` の定義を検索します。",
]
_PREFIX = [
    "`_query_url` は provider が duckduckgo のとき `cfg.base_url or _DUCKDUCKGO_URL` を使います。",
    "テストが落ちる原因を `satsumaestro/tools/builtins/web_search.py` から順に追います。",
    "まず `_with_query` の引数の組み立てを確認します。",
]


def _loop_text(cycles: int) -> str:
    return "\n\n".join([*_PREFIX, *(_CYCLE * cycles)]) + "\n\n"


def _feed(text: str, chunk: int) -> RepetitionVerdict | None:
    detector = RepetitionDetector()
    for i in range(0, len(text), chunk):
        verdict = detector.feed(text[i : i + chunk])
        if verdict is not None:
            return verdict
    return None


def test_the_loop_shape_from_the_real_response_is_detected() -> None:
    verdict = _feed(_loop_text(4), 64)
    assert verdict is not None
    assert verdict.unit == "paragraph"
    assert (verdict.seen, verdict.window) == (PARAGRAPH_MAX_SEEN, PARAGRAPH_WINDOW)
    assert verdict.repeats == 3
    assert verdict.snippet.startswith("```python _DUCKDUCKGO_URL")
    assert "\n" not in verdict.snippet


def test_two_cycles_are_not_enough() -> None:
    assert _feed(_loop_text(2), 64) is None


@pytest.mark.parametrize("chunk", [1, 7, 64, 100_000])
def test_delta_boundaries_do_not_change_the_verdict(chunk: int) -> None:
    """デルタの切れ目（1 文字・途中・丸ごと）で判定が変わらない（``chars`` だけ粒度で揺れる）。"""
    verdict = _feed(FIXTURE.read_text(encoding="utf-8"), chunk)
    assert verdict is not None
    assert (verdict.unit, verdict.seen, verdict.repeats) == ("paragraph", PARAGRAPH_MAX_SEEN, 8)
    assert verdict.snippet == '```python _DUCKDUCKGO_URL = "https://duckduckgo.com/" ```'


def test_real_response_is_cut_at_about_ten_thousand_chars() -> None:
    """実応答の再生: 36,460 字のうち約 10,300 字で止まる（15,728 トークンの 2/3 を救う）。"""
    verdict = _feed(FIXTURE.read_text(encoding="utf-8"), 64)
    assert verdict is not None
    assert 10_000 <= verdict.chars <= 10_500


def test_a_spec_with_boilerplate_paragraphs_is_not_a_loop() -> None:
    """12 エンドポイントの仕様書: 各節に固有の説明と同じ定型段落 3 つ（既出 9/12 で発火しない）。"""
    boilerplate = [
        "認証: `Authorization: Bearer <token>` ヘッダが必須です。無ければ 401 を返します。",
        "レート制限: 1 分あたり 60 リクエスト。超過時は 429 と `Retry-After` を返します。",
        '```json\n{"error": {"code": "unauthorized", "message": "token is missing"}}\n```',
    ]
    sections = []
    for i in range(12):
        sections.append(f"## {i + 1}. GET /api/v1/items/{i}")
        sections.append(
            f"アイテム {i} の詳細を返します。`fields` で返す列を絞り込めます（既定は全列）。"
        )
        sections.append(
            f"| 引数 | 型 | 説明 |\n|---|---|---|\n| id | int | アイテム {i} の識別子 |"
        )
        sections.extend(boilerplate)
    assert _feed("\n\n".join(sections) + "\n\n", 64) is None


def test_fences_rules_and_code_listings_are_not_a_loop() -> None:
    fences = "\n\n".join(f"```python\nvalue_{i} = compute({i})\n```\n\n---" for i in range(40))
    assert _feed(fences + "\n\n", 64) is None
    listing = "\n".join(
        f"    self.assertEqual(result[{i}], expected_values[{i}])" for i in range(80)
    )
    assert _feed(listing + "\n", 64) is None


def test_short_lines_are_not_counted() -> None:
    assert _feed("→ 不正な URL。\n\n" * 200, 64) is None


def test_repeated_lines_without_blank_lines_hit_the_line_rule() -> None:
    line = "しかし、テストが通っているため、このコードは正しく機能している可能性があります。"
    verdict = _feed((line + "\n") * 40, 64)
    assert verdict is not None
    assert verdict.unit == "line"
    assert verdict.window == LINE_WINDOW
    assert verdict.repeats == LINE_WINDOW


class _LoopingProvider:
    """``"loop"`` は終わらない反復本文を流し続ける（``MessageStop`` を出さない）プロバイダ。"""

    def __init__(self, scripts: list[str | list[AssistantEvent]], cycles: int = 200) -> None:
        self.kind = ProviderKind.OPENAI
        self._scripts = scripts
        self._cycles = cycles
        self.requests: list[ApiRequest] = []
        self.aclose_calls = 0
        self.loop_events = 0

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.requests.append(req)
        script = self._scripts[min(len(self.requests) - 1, len(self._scripts) - 1)]
        try:
            if isinstance(script, str):
                text = _loop_text(self._cycles)
                for i in range(0, len(text), 64):
                    self.loop_events += 1
                    yield TextDelta(text=text[i : i + 64])
                yield MessageStop()
            else:
                for event in script:
                    yield event
        finally:
            self.aclose_calls += 1

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


def _runtime(
    provider: _LoopingProvider, tmp_path: Path, cfg: RuntimeConfig | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg or RuntimeConfig(),
    )


def _done() -> list[AssistantEvent]:
    return [
        TextDelta(text="結論: `_DUCKDUCKGO_URL` は定義済みで、失敗は別の原因です"),
        MessageStop(),
    ]


def _injected(rt: ConversationRuntime) -> list[str]:
    return [
        block.text
        for message in rt.session.messages
        if message.role == "user"
        for block in message.blocks
        if isinstance(block, TextBlock) and "<system-reminder>" in block.text
    ]


async def test_repetition_is_cut_once_and_the_model_is_told(tmp_path: Path) -> None:
    provider = _LoopingProvider(["loop", _done()])
    rt = _runtime(provider, tmp_path)
    summary = await rt.run_turn("テストが落ちる原因を調べて")

    assert summary.final_text.startswith("結論:")
    assert summary.iterations == 2
    assert summary.repetition_aborts == 1
    assert summary.reminders == 1
    note = _injected(rt)[0]
    assert "kept repeating itself" in note and "None of it was delivered" in note
    assert "_DUCKDUCKGO_URL" in note
    roles = [m.role for m in rt.session.messages]
    assert roles == ["user", "user", "assistant"]
    texts = [b.text for m in rt.session.messages for b in m.blocks if isinstance(b, TextBlock)]
    assert all(_CYCLE[0] not in text for text in texts)
    assert summary.compacted is False
    assert provider.aclose_calls == 2
    assert provider.loop_events < len(_loop_text(200)) // 64


async def test_second_repetition_in_the_same_turn_fails_the_turn(tmp_path: Path) -> None:
    provider = _LoopingProvider(["loop", "loop", _done()])
    rt = _runtime(provider, tmp_path)
    with pytest.raises(RepetitionLoopError, match="repeated again"):
        await rt.run_turn("hi")
    assert len(_injected(rt)) == 1
    assert provider.aclose_calls == 2


async def test_recovery_is_per_turn_not_per_session(tmp_path: Path) -> None:
    provider = _LoopingProvider(["loop", _done(), "loop", _done()])
    rt = _runtime(provider, tmp_path)
    first = await rt.run_turn("1 つめ")
    second = await rt.run_turn("2 つめ")
    assert (first.repetition_aborts, second.repetition_aborts) == (1, 1)


async def test_no_recovery_on_the_last_iteration(tmp_path: Path) -> None:
    provider = _LoopingProvider(["loop", _done()])
    rt = _runtime(provider, tmp_path, RuntimeConfig(max_iterations=1))
    with pytest.raises(RepetitionLoopError, match="iteration limit"):
        await rt.run_turn("hi")
    assert _injected(rt) == []


async def test_flag_off_lets_the_repetition_through(tmp_path: Path) -> None:
    """A/B の before 条件: 反復を止めず、従来どおり全文を畳む。"""
    provider = _LoopingProvider(["loop", _done()], cycles=6)
    cfg = RuntimeConfig(guidance=GuidanceConfig(repetition_abort=False))
    rt = _runtime(provider, tmp_path, cfg)
    summary = await rt.run_turn("hi")
    assert summary.repetition_aborts == 0
    assert summary.iterations == 1
    assert summary.final_text.count(_CYCLE[0]) == 6
    assert _injected(rt) == []


async def test_history_round_trips_after_recovery(tmp_path: Path) -> None:
    rt = _runtime(_LoopingProvider(["loop", _done()]), tmp_path)
    await rt.run_turn("hi")
    store = JSONLSessionStore()
    restored = store.load(store.path_for(tmp_path, rt.session.session_id))
    assert [m.role for m in restored.messages] == ["user", "user", "assistant"]


async def test_renderer_is_told_with_the_verdict(tmp_path: Path) -> None:
    class _Recording(NullRenderer):
        seen: ClassVar[list[tuple[str, int, bool]]] = []

        def on_repetition_aborted(self, verdict: RepetitionVerdict, recovering: bool) -> None:
            type(self).seen.append((verdict.unit, verdict.repeats, recovering))

    rt = _runtime(_LoopingProvider(["loop", "loop"]), tmp_path)
    with pytest.raises(RepetitionLoopError):
        await rt.run_turn("hi", render=_Recording())
    assert _Recording.seen == [("paragraph", 3, True), ("paragraph", 3, False)]


def _verdict(unit: str = "paragraph", repeats: int = 7) -> RepetitionVerdict:
    return RepetitionVerdict(
        unit=unit,  # type: ignore[arg-type]
        seen=10,
        window=12,
        repeats=repeats,
        snippet="```python _DUCKDUCKGO_URL = ...",
        chars=10_300,
    )


def _rendered(verdict: RepetitionVerdict, recovering: bool) -> str:
    buf = io.StringIO()
    r = RichRenderer(console=Console(file=buf, force_terminal=True, width=160, color_system=None))
    r.start_turn()
    r.on_repetition_aborted(verdict, recovering)
    r.end_turn()
    return buf.getvalue()


def test_recovery_notice_is_one_user_facing_line() -> None:
    out = _rendered(_verdict(), True)
    assert (
        "同じ内容の繰り返し（同じ段落が 7 回）を検知して中断し、結論を 1 度だけ書くよう依頼しました"
        in out
    )


def test_abort_without_recovery_does_not_claim_a_request() -> None:
    out = _rendered(_verdict("line", 30), False)
    assert "同じ内容の繰り返し（同じ行が 30 回）を検知したので中断しました" in out
    assert "依頼しました" not in out


def test_loop_error_reads_as_a_user_facing_reason() -> None:
    again = RepetitionLoopError(7, "```python _DUCKDUCKGO_URL", recovered_before=True)
    assert "repeated again after being asked" in str(again) and "7 times" in str(again)
    first = RepetitionLoopError(7, "x", recovered_before=False)
    assert "iteration limit" in str(first) and "again" not in str(first)
    assert turn_failure(again).reason == "同じ内容の繰り返しが止まりませんでした（同じ段落が 7 回）"


def test_note_names_the_unit_count_and_snippet() -> None:
    note = repetition_note(_verdict())
    assert "the same paragraph came back 7 times" in note
    assert "```python _DUCKDUCKGO_URL = ..." in note
    assert "check it with one tool call" in note


def test_counter_reaches_the_evaluation_records() -> None:
    summary = TurnSummary(
        final_text="x",
        stop_reason="end_turn",
        iterations=2,
        usage=TokenUsage(),
        repetition_aborts=1,
    )
    assert AgentAnswer(repetition_aborts=summary.repetition_aborts).repetition_aborts == 1
    result = TaskResult.model_validate({"task_id": "t", "passed": True, "repetition_aborts": 1})
    assert [d.kind for d in detect_defects(result, "satsuma")] == ["repetition abort"]
    assert TaskResult.model_validate({"task_id": "t", "passed": True}).repetition_aborts == 0


class _FakeSdkStream:
    """openai の ``AsyncStream`` の形（非同期反復 + ``close()``）。"""

    def __init__(self, chunks: int | None = None) -> None:
        self.closed = False
        self._chunks = chunks
        self._sent = 0

    def __aiter__(self) -> _FakeSdkStream:
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._chunks is not None and self._sent >= self._chunks:
            raise StopAsyncIteration
        self._sent += 1
        delta = SimpleNamespace(content=f"line {self._sent}\n", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=delta, finish_reason=None)], usage=None
        )

    async def close(self) -> None:
        self.closed = True


def _provider_with(stream: _FakeSdkStream) -> OpenAICompatProvider:
    provider = OpenAICompatProvider("key", "http://127.0.0.1:1/v1")

    async def _open(req: object, tools: object, *, strict: bool) -> _FakeSdkStream:
        return stream

    provider._open = _open  # type: ignore[method-assign]
    return provider


def _request() -> ApiRequest:
    return ApiRequest(model="fake", system="", messages=[], tools=[])


async def test_abandoned_sdk_stream_is_closed_at_once() -> None:
    """打ち切りで放棄したストリームは、GC を待たずにその場で HTTP 接続を閉じる。"""
    sdk = _FakeSdkStream()
    events = _provider_with(sdk).stream(_request())
    first = await events.__anext__()
    assert isinstance(first, TextDelta)
    assert sdk.closed is False
    await events.aclose()
    assert sdk.closed is True


async def test_sdk_stream_is_closed_when_it_ends_in_an_error() -> None:
    sdk = _FakeSdkStream(chunks=0)
    with pytest.raises(ProviderError):
        async for _ in _provider_with(sdk).stream(_request()):
            pass
    assert sdk.closed is True


def test_counter_reaches_the_telemetry_record() -> None:
    summary = TurnSummary(
        final_text="x",
        stop_reason="end_turn",
        iterations=2,
        usage=TokenUsage(),
        repetition_aborts=1,
    )
    assert summary_fields(summary)["repetition_aborts"] == 1
