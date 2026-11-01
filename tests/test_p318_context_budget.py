"""1 回の応答のツールの結果を、圧縮の閾値から決めた上限に収める。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import CompactionConfig, HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.context_budget import (
    estimate_upper,
    fit_tool_results,
    result_budget,
)
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import NullRenderer
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock, ToolUseBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.builtins.read_file import FULL_READ_CAP_NOTE
from tests.fixtures.mock_provider import SequencedMockProvider


def _lines(first: int, count: int) -> str:
    return "".join(f"{i:06d} log line with some text\n" for i in range(first, first + count))


def _read(cid: str, **args: object) -> ToolUseBlock:
    return ToolUseBlock(id=cid, name="read_file", input={"path": "big.log", **args})


def _result(cid: str, output: str, tool: str = "read_file") -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=cid, tool_name=tool, output=output)


def test_the_estimate_counts_ascii_by_three_and_other_characters_by_one() -> None:
    assert estimate_upper("a" * 300) == 100
    assert estimate_upper("日本語の文") == 5
    assert estimate_upper("abc日本") == 3


def test_the_budget_is_a_share_of_the_compaction_threshold() -> None:
    assert result_budget(104_857, 0.25) == 26_214
    assert result_budget(2_000, 0.25) == 1_000
    assert result_budget(0, 0.25) is None
    assert result_budget(104_857, 0.0) is None


def test_results_within_the_budget_are_left_alone() -> None:
    results = [_result("a", _lines(1, 10)), _result("b", "ok")]
    fitted, cuts = fit_tool_results(results, [_read("a"), _read("b")], 1_000)
    assert fitted == results and cuts == []


def test_a_whole_file_read_is_cut_at_a_line_with_where_to_read_next() -> None:
    text = _lines(1, 20_000)
    fitted, cuts = fit_tool_results([_result("a", text)], [_read("a")], 2_000)
    out = fitted[0].output
    (cut,) = cuts
    assert cut.lines is not None
    first, last = cut.lines
    assert first == 1 and 100 < last < 20_000
    assert cut.total_lines == 20_000
    body, note = out.split("\n[read_file: ", 1)
    assert body.endswith(f"{last:06d} log line with some text\n")
    assert f"Shown: lines 1-{last:,} of 20,000" in note
    assert f"offset={last + 1} limit={last}" in note
    assert estimate_upper(body) <= 2_000
    assert "20,000 行のうち 1〜" in cut.describe()


def test_an_offset_read_counts_lines_from_the_offset() -> None:
    text = _lines(501, 5_000)
    fitted, cuts = fit_tool_results([_result("a", text)], [_read("a", offset=501)], 1_000)
    (cut,) = cuts
    assert cut.lines is not None and cut.lines[0] == 501
    assert cut.total_lines == 5_500
    assert f"offset={cut.lines[1] + 1}" in fitted[0].output


def test_a_range_read_does_not_claim_the_file_length() -> None:
    text = _lines(1, 5_000)
    fitted, cuts = fit_tool_results([_result("a", text)], [_read("a", limit=5_000)], 1_000)
    (cut,) = cuts
    assert cut.total_lines is None
    assert "of the requested 1-5,000" in fitted[0].output


def test_a_read_cut_at_the_byte_cap_does_not_claim_the_file_length() -> None:
    text = _lines(1, 5_000) + FULL_READ_CAP_NOTE
    fitted, cuts = fit_tool_results([_result("a", text)], [_read("a")], 1_000)
    (cut,) = cuts
    assert cut.total_lines is None
    assert cut.lines is not None and f"of more than {cut.lines[1]:,}" in fitted[0].output


def test_large_results_are_cut_to_the_same_level_and_small_ones_are_kept() -> None:
    small = _result("s", "short error", tool="glob_search")
    results = [_result("a", _lines(1, 3_000)), small, _result("b", _lines(1, 9_000))]
    calls = [_read("a"), ToolUseBlock(id="s", name="glob_search", input={}), _read("b")]
    fitted, cuts = fit_tool_results(results, calls, 2_000)
    assert fitted[1] == small
    assert len(cuts) == 2
    assert abs(cuts[0].kept_tokens - cuts[1].kept_tokens) < 50
    assert sum(estimate_upper(b.output.split("\n[read_file")[0]) for b in fitted) <= 2_000


def test_other_tools_get_a_note_in_characters() -> None:
    page = "word " * 40_000
    calls = [ToolUseBlock(id="w", name="web_fetch", input={"url": "https://example.com"})]
    fitted, cuts = fit_tool_results([_result("w", page, tool="web_fetch")], calls, 1_000)
    (cut,) = cuts
    assert cut.lines is None
    assert "[web_fetch: this result was cut to fit the context window. Shown: the first" in (
        fitted[0].output
    )
    assert "先頭の約" in cut.describe()


class _Notices(NullRenderer):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def on_context_notice(self, text: str) -> None:
        self.lines.append(text)


async def test_the_runtime_cuts_a_huge_read_before_it_joins_the_conversation(
    tmp_path: Path,
) -> None:
    (tmp_path / "big.log").write_text(_lines(1, 20_000), encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="r1", name="read_file", input={"path": "big.log"}),
                Usage(input_tokens=3_000, output_tokens=20),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), Usage(input_tokens=6_000, output_tokens=5), MessageStop()],
        ]
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(
            compact_token_threshold=16_000, compaction=CompactionConfig(use_llm_summary=False)
        ),
        compactor=Compactor(keep_last=6),
    )
    render = _Notices()
    summary = await rt.run_turn("big.log を全部読んで、最後の行の番号を答えて", render=render)
    assert summary.tool_result_cuts == 1
    sent = provider.requests[1].messages[-1].blocks[0]
    assert isinstance(sent, ToolResultBlock)
    assert "[read_file: this result was cut to fit the context window." in sent.output
    assert estimate_upper(sent.output) < 4_200
    assert len(render.lines) == 1 and "20,000 行のうち 1〜" in render.lines[0]


async def test_no_cut_when_the_share_is_zero(tmp_path: Path) -> None:
    (tmp_path / "big.log").write_text(_lines(1, 20_000), encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="r1", name="read_file", input={"path": "big.log"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(
            compact_token_threshold=0, compaction=CompactionConfig(max_tool_result_share=0.0)
        ),
        compactor=Compactor(keep_last=6),
    )
    summary = await rt.run_turn("big.log を全部読んで")
    assert summary.tool_result_cuts == 0
    sent = provider.requests[1].messages[-1].blocks[0]
    assert isinstance(sent, ToolResultBlock) and sent.output == _lines(1, 20_000)
