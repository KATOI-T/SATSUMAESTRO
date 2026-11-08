"""起動時と doctor に出す、文脈の窓の内訳。"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli import app as cli_app
from satsumaestro.cli.render import RichRenderer
from satsumaestro.config.models import (
    CompactionConfig,
    HookConfig,
    ProviderConfig,
    RuntimeConfig,
)
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.context_fit import ContextFitResult
from satsumaestro.engine.events import MessageStop, TextDelta, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.window_breakdown import (
    OLLAMA_CONTEXT_LENGTH_URL,
    WindowBreakdown,
    display_tokens,
    window_breakdown,
)
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.prompt.instructions import instruction_tokens
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.detect import ModelContextInfo
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider

_ROOT = "http://127.0.0.1:11434"


def _runtime(
    tmp_path: Path, scripts: list[list[object]], *, window: int, threshold: int
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(scripts),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(instruction_tokens=instruction_tokens(window)),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(
            compact_token_threshold=threshold,
            compaction=CompactionConfig(context_window=window, use_llm_summary=False),
        ),
        compactor=Compactor(),
    )


def _reply(text: str) -> list[object]:
    return [TextDelta(text=text), Usage(input_tokens=3_000, output_tokens=5), MessageStop()]


def test_an_8k_window_leaves_about_one_first_request_for_the_conversation() -> None:
    small = WindowBreakdown(window=8_192, threshold=6_553, fixed=3_300, instructions=0)
    assert (small.conversation, small.tight, small.recommended()) == (3_253, True, 16_384)
    medium = WindowBreakdown(window=16_384, threshold=13_107, fixed=3_300, instructions=0)
    assert (medium.conversation, medium.tight) == (9_807, False)
    large = WindowBreakdown(window=131_072, threshold=104_857, fixed=3_300, instructions=0)
    assert not large.tight


def test_a_large_first_request_asks_for_a_larger_window() -> None:
    b = WindowBreakdown(window=16_384, threshold=13_107, fixed=6_000, instructions=2_000)
    assert b.tight
    assert b.recommended() == 32_768


def test_display_tokens_count_japanese_per_character() -> None:
    assert display_tokens("abcd" * 10) == 10
    assert display_tokens("日本語の規約") == 6


def test_no_breakdown_without_a_window_or_with_compaction_off() -> None:
    assert window_breakdown(0, 6_553, "system", "[]", "") is None
    assert window_breakdown(8_192, 0, "system", "[]", "") is None


def test_the_lines_in_both_languages() -> None:
    b = WindowBreakdown(window=8_192, threshold=6_553, fixed=3_300, instructions=900)
    assert b.describe("実効 num_ctx（ロード済み）") == (
        "文脈の窓 8,192 トークン（実効 num_ctx（ロード済み））: 最初の要求 約 3,300"
        "（うち指示ファイル 約 900）・会話に使える量 約 3,253（6,553 から履歴を畳む）"
    )
    plain = WindowBreakdown(window=8_192, threshold=6_553, fixed=3_300, instructions=0)
    assert plain.describe("Modelfile の num_ctx") == (
        "文脈の窓 8,192 トークン（Modelfile の num_ctx）: 最初の要求 約 3,300・"
        "会話に使える量 約 3,253（6,553 から履歴を畳む）"
    )
    assert b.advice() == (
        "窓が小さく、会話に使える量が最初の要求の 2 倍に届きません（書き込みのたびに履歴を"
        "畳みます）。窓を 16384 以上にしてください: Ollama を OLLAMA_CONTEXT_LENGTH=16384 で"
        "起動し直すか、Modelfile に PARAMETER num_ctx 16384 を書いてモデルを作り直します"
        f"（{OLLAMA_CONTEXT_LENGTH_URL}）"
    )
    with pinned_ui_language("en"):
        assert b.describe("effective num_ctx (loaded)") == (
            "context window 8,192 tokens (effective num_ctx (loaded)): first request ~3,300 "
            "(instruction files ~900) · room for the conversation ~3,253 (the history is "
            "compacted from 6,553)"
        )
        assert b.advice() == (
            "The window is small: the room for the conversation is less than 2 times the "
            "first request, so the history is compacted at almost every write. Make the window "
            "16384 or larger: restart Ollama with OLLAMA_CONTEXT_LENGTH=16384, or add "
            "PARAMETER num_ctx 16384 to the Modelfile and recreate the model "
            f"({OLLAMA_CONTEXT_LENGTH_URL})"
        )


def test_llamacpp_advice_gives_the_size_for_its_slots() -> None:
    b = WindowBreakdown(window=8_192, threshold=6_553, fixed=3_300, instructions=0)
    for slots in (None, 1):
        one = b.advice("llamacpp", "gpt-oss-20b", slots)
        assert one.endswith("llama-server を `-c 16384` 以上で起動し直します")
    four = b.advice("llamacpp", "gpt-oss-20b", 4)
    assert four.endswith(
        "`-c 16384` 以上で起動し直します。`--parallel 4` で `--kv-unified` を付けていないなら"
        " `-c 65536` 以上です"
    )
    with pinned_ui_language("en"):
        assert b.advice("llamacpp", "gpt-oss-20b", 4).endswith(
            "`-c 16384` or more, or `-c 65536` or more with `--parallel 4` and no `--kv-unified`"
        )


async def test_the_runtime_counts_the_instruction_files(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, [[]], window=16_384, threshold=13_107)
    bare = await rt.window_breakdown()
    (tmp_path / "SATSUMA.md").write_text("この作業場所の規約。" * 50, encoding="utf-8")
    with_rules = await rt.window_breakdown()
    assert bare is not None and with_rules is not None
    assert (bare.window, bare.threshold, bare.instructions) == (16_384, 13_107, 0)
    assert bare.fixed > 2_000
    assert with_rules.instructions >= 500
    assert with_rules.fixed - bare.fixed >= 500
    unknown = _runtime(tmp_path, [[]], window=0, threshold=120_000)
    assert await unknown.window_breakdown() is None


async def test_the_window_is_rechecked_once_after_the_first_reply(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, [_reply("one"), _reply("two")], window=131_072, threshold=104_857)
    rt.window_unconfirmed = True
    calls: list[int] = []

    async def refit() -> int | None:
        calls.append(1)
        rt.config.compaction.context_window = 8_192
        rt.config.compact_token_threshold = 6_553
        return 8_192

    rt._refit_window = refit  # type: ignore[method-assign]
    await rt.run_turn("hello")
    notes = rt.take_window_notes()
    assert calls == [1]
    assert notes[0] == "読み込んだモデルの窓は 8,192 トークンでした（起動時の見込みは 131,072）"
    assert notes[1].startswith("文脈の窓 8,192 トークン（実効 num_ctx（ロード済み））: ")
    assert "OLLAMA_CONTEXT_LENGTH=16384" in notes[2]
    await rt.run_turn("again")
    assert calls == [1]
    assert rt.take_window_notes() == []


async def test_an_unchanged_window_leaves_no_notes(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, [_reply("one")], window=16_384, threshold=13_107)
    rt.window_unconfirmed = True

    async def refit() -> int | None:
        return None

    rt._refit_window = refit  # type: ignore[method-assign]
    await rt.run_turn("hello")
    assert rt.take_window_notes() == []
    assert not rt.window_unconfirmed


def test_the_turn_end_line_follows_the_threshold() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=100, color_system=None)
    limit = [104_857]
    r = RichRenderer(console=console, context_limit=lambda: limit[0])
    r._refresh_live = lambda: None  # type: ignore[method-assign]
    r.start_turn()
    r.on_assistant_event(TextDelta(text="hello"))
    r.on_assistant_event(Usage(input_tokens=3_900, output_tokens=10))
    r.on_assistant_event(MessageStop())
    limit[0] = 6_553
    r.end_turn()
    last = [ln for ln in buf.getvalue().splitlines() if ln.strip()][-1]
    assert last.endswith("文脈 3.91K/6.55K")


def _fit(effective: int) -> ContextFitResult:
    info = ModelContextInfo(effective=effective, source="ps", loaded=True)
    return ContextFitResult("m", info, 120_000, int(effective * 0.8))


@pytest.mark.parametrize(("effective", "status"), [(8_192, "warn"), (131_072, "ok")])
def test_doctor_shows_the_breakdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effective: int, status: str
) -> None:
    async def probe(cfg: RuntimeConfig, model: str) -> ContextFitResult:
        return _fit(effective)

    monkeypatch.setattr(cli_app, "_probe_ollama", probe)
    monkeypatch.setenv("OLLAMA_HOST", _ROOT)
    cfg = RuntimeConfig(provider=ProviderConfig(base_url=f"{_ROOT}/v1"))
    rows = cli_app._ollama_rows(
        cfg, "m", ProviderKind.OPENAI, probe=True, cwd=tmp_path, trusted=True
    )
    (row,) = (r for r in rows if r[0] == "context window")
    assert row[1] == status
    assert row[2].startswith(f"文脈の窓 {effective:,} トークン（実効 num_ctx（ロード済み））: ")
    assert "（MCP のツールは数えていない）" in row[2]
    assert ("OLLAMA_CONTEXT_LENGTH=16384" in row[2]) is (status == "warn")
    assert all(
        r[0] != "context window"
        for r in cli_app._ollama_rows(cfg, "m", ProviderKind.OPENAI, probe=True)
    )
