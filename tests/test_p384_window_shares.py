"""指示ファイルの量・圧縮で残す直近・要約の要求の量を、モデルの窓から決める。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import CompactionConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import (
    MAX_TRANSCRIPT_CHARS,
    CompactionLimits,
    Compactor,
    _cut_index,
    limits_for,
)
from satsumaestro.engine.context_budget import estimate_upper
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.prompt.instructions import discover_instructions, instruction_tokens
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider


def _say(role: str, text: str) -> ConversationMessage:
    return ConversationMessage(role=role, blocks=[TextBlock(text=text)])  # type: ignore[arg-type]


def _read(call_id: str, output: str) -> list[ConversationMessage]:
    return [
        ConversationMessage(
            role="assistant",
            blocks=[ToolUseBlock(id=call_id, name="read_file", input={"path": "a.py"})],
        ),
        ConversationMessage(
            role="tool",
            blocks=[ToolResultBlock(tool_use_id=call_id, tool_name="read_file", output=output)],
        ),
    ]


def _runtime(tmp_path: Path, window: int, provider: object | None = None) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider or SequencedMockProvider([[TextDelta(text="ok"), MessageStop()]]),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(
            compact_token_threshold=int(window * 0.8),
            compaction=CompactionConfig(context_window=window, use_llm_summary=False),
        ),
        compactor=Compactor(),
    )


def test_the_instruction_budget_is_a_share_of_the_window() -> None:
    assert instruction_tokens(16_384) == 2_048
    assert instruction_tokens(8_192) == 1_024
    assert instruction_tokens(0) is None


def test_a_small_window_cuts_japanese_instructions_by_tokens(tmp_path: Path) -> None:
    body = "".join(f"規約の行 {i:04d}: 変更は小さく分けて確かめる。\n" for i in range(200))
    (tmp_path / "CLAUDE.md").write_text(body, encoding="utf-8")
    whole = discover_instructions(tmp_path)
    assert "[truncated" not in whole and body.rstrip("\n") in whole
    cut = discover_instructions(tmp_path, tokens=2_048)
    shown = cut.split("\n", 1)[1].split("\n[truncated", 1)[0]
    assert "[truncated: only the first" in cut
    assert estimate_upper(shown) <= 2_048
    assert shown.endswith("。")


def test_files_past_the_token_budget_are_listed_with_it(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "pkg"
    sub.mkdir()
    (tmp_path / "CLAUDE.md").write_text("規" * 3_000, encoding="utf-8")
    (sub / "AGENTS.md").write_text("rules\n", encoding="utf-8")
    text = discover_instructions(sub, tokens=1_000)
    assert "# Instruction files not loaded" in text
    assert "(16,000 characters or about 1,000 tokens)" in text


def test_the_builder_passes_the_budget(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("規約。\n" * 2_000, encoding="utf-8")
    small = SystemPromptBuilder(instruction_tokens=1_024)._discover_instructions(tmp_path)
    large = SystemPromptBuilder()._discover_instructions(tmp_path)
    assert "[truncated" in small and "[truncated" not in large


def test_the_tail_is_counted_by_amount() -> None:
    big = "x = 1\n" * 3_000
    messages = [_say("user", "q"), _say("assistant", "a")]
    for i in range(3):
        messages += _read(f"r{i}", big)
    assert _cut_index(messages, 6) == 2
    cut = _cut_index(messages, 6, tail_tokens=4_096)
    assert cut == len(messages) - 2
    assert messages[cut].role == "assistant"


def test_a_large_window_keeps_six_messages() -> None:
    messages = [_say("user", f"q{i}") if i % 2 == 0 else _say("assistant", "a") for i in range(10)]
    assert _cut_index(messages, 6, tail_tokens=32_768) == _cut_index(messages, 6) == 4


def test_keep_last_zero_folds_everything() -> None:
    messages = [_say("user", "q"), _say("assistant", "a"), _say("user", "q2")]
    assert _cut_index(messages, 0) == 3
    assert _cut_index(messages[:1], 0) is None


def test_limits_come_from_the_window() -> None:
    assert limits_for(0, 120_000, 5_000, 1_024) == CompactionLimits()
    large = limits_for(131_072, 104_857, 5_000, 1_024)
    assert (large.tail_tokens, large.transcript_chars) == (32_768, MAX_TRANSCRIPT_CHARS)
    small = limits_for(8_192, 6_553, 5_000, 1_024)
    assert (small.tail_tokens, small.transcript_chars) == (776, 4_300)
    assert limits_for(8_192, 6_553, 0, 1_024).tail_tokens == 2_048


async def test_the_summary_request_fits_a_small_window(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[TextDelta(text="- summary"), MessageStop()]])
    compactor = Compactor(keep_last=2, provider=provider, model="m")  # type: ignore[arg-type]
    session = new_session(tmp_path, "m")
    session.messages = [_say("user", "依頼"), *[_say("assistant", "あ" * 3_000) for _ in range(10)]]
    outcome = await compactor.acompact(session, limits=limits_for(8_192, 6_553, 5_000, 1_024))
    assert outcome is not None and outcome.used_llm
    sent = provider.requests[0].messages[0].blocks[0]
    assert isinstance(sent, TextBlock) and len(sent.text) <= 4_300


async def test_no_room_for_a_transcript_uses_the_rule_summary(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[TextDelta(text="- summary"), MessageStop()]])
    compactor = Compactor(keep_last=2, provider=provider, model="m")  # type: ignore[arg-type]
    session = new_session(tmp_path, "m")
    session.messages = [_say("user", "依頼"), *[_say("assistant", "a") for _ in range(6)]]
    outcome = await compactor.acompact(session, limits=CompactionLimits(None, 0))
    assert outcome is not None and not outcome.used_llm
    assert provider.requests == []


def test_the_runtime_limits_use_the_fixed_part(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, 8_192)
    assert rt.compaction_limits().tail_tokens == 2_048
    rt._fixed_tokens = 5_000
    assert rt.compaction_limits() == limits_for(8_192, 6_553, 5_000, 1_024)


def test_a_folded_step_is_handed_again(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, 16_384)
    rt.session.plan_steps = ["Write the HTML skeleton", "Add the styles"]
    rt.session.plan_step_index = 2
    rt.session.messages = [_say("user", "[summary]"), *_read("r1", "x")]
    rt._rehand_folded_step("en")
    note = rt.session.messages[-1]
    assert note.role == "user"
    text = note.blocks[0].text  # type: ignore[union-attr]
    assert "compacted while you were on step 2 of 2 of the approved plan: Add the styles" in text
    before = len(rt.session.messages)
    rt._rehand_folded_step("en")
    assert len(rt.session.messages) == before


def test_no_step_is_handed_outside_a_plan(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, 16_384)
    rt.session.messages = [_say("user", "[summary]"), _say("assistant", "a")]
    rt._rehand_folded_step("en")
    assert len(rt.session.messages) == 2
