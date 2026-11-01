"""圧縮の再発火抑制・要約品質強化のテスト。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.compaction import (
    _CONTINUATION_NOTICE,
    _SUMMARY_PREFIX,
    RULE_SUMMARY_MAX_CHARS,
    Compactor,
    _render_transcript,
    _rule_summary,
    _summarize_llm,
)
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import (
    ConversationMessage,
    TextBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider


class UsageSpyCompactor(Compactor):
    """``needs_compaction`` へ渡された usage を記録する。"""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []

    def needs_compaction(self, usage: TokenUsage, cfg: RuntimeConfig) -> bool:
        self.seen.append(usage.input_tokens + usage.output_tokens)
        return super().needs_compaction(usage, cfg)


def _runtime(provider: object, tmp_path: Path, compactor: Compactor) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
        compactor=compactor,
    )


async def test_needs_compaction_receives_latest_usage_not_cumulative(tmp_path: Path) -> None:
    spy = UsageSpyCompactor()
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="c1", name="glob_search", input={"pattern": "*"}),
                Usage(input_tokens=150_000, output_tokens=10),
                MessageStop(),
            ],
            [TextDelta(text="done"), Usage(input_tokens=1_000, output_tokens=5), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, spy)
    await rt.run_turn("q")
    assert len(spy.seen) == 2 and 150_000 < spy.seen[0] < 151_000
    assert spy.seen[-1] == 1_005


async def test_needs_compaction_estimates_when_usage_missing(tmp_path: Path) -> None:
    spy = UsageSpyCompactor()
    provider = SequencedMockProvider([[TextDelta(text="plain"), MessageStop()]])
    rt = _runtime(provider, tmp_path, spy)
    await rt.run_turn("q")
    assert len(spy.seen) == 1
    assert 0 < spy.seen[0] < 120_000


class AllowLlmSpyCompactor(Compactor):
    """``acompact`` へ渡された ``allow_llm`` を記録する（provider 無し = 機械要約）。"""

    def __init__(self) -> None:
        super().__init__(keep_last=1)
        self.allow_llm_seen: list[bool] = []

    async def acompact(self, session, *, allow_llm: bool = True, limits=None):  # type: ignore[no-untyped-def, override]
        self.allow_llm_seen.append(allow_llm)
        return await super().acompact(session, allow_llm=allow_llm, limits=limits)


async def test_llm_summary_at_most_once_per_turn(tmp_path: Path) -> None:
    spy = AllowLlmSpyCompactor()
    over = Usage(input_tokens=150_000, output_tokens=10)
    provider = SequencedMockProvider(
        [
            [ToolUse(id="c1", name="glob_search", input={"pattern": "*"}), over, MessageStop()],
            [ToolUse(id="c2", name="glob_search", input={"pattern": "*"}), over, MessageStop()],
            [TextDelta(text="done"), over, MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, spy)
    rt.session.messages.extend(
        ConversationMessage(role="user", blocks=[TextBlock(text=f"warmup {i}")]) for i in range(4)
    )
    summary = await rt.run_turn("q")
    assert summary.compacted is True
    assert len(spy.allow_llm_seen) >= 2
    assert spy.allow_llm_seen[0] is True
    assert all(flag is False for flag in spy.allow_llm_seen[1:])


def test_render_transcript_clips_each_block() -> None:
    msg = ConversationMessage(
        role="tool",
        blocks=[ToolResultBlock(tool_use_id="c1", tool_name="read_file", output="x" * 5_000)],
    )
    rendered = _render_transcript([msg])
    assert "chars trimmed" in rendered
    assert len(rendered) < 1_200


async def test_llm_transcript_keeps_newest_when_over_limit() -> None:
    class CaptureProvider:
        def __init__(self) -> None:
            self.req = None

        async def stream(self, req):  # type: ignore[no-untyped-def]
            self.req = req
            yield TextDelta(text="s")
            yield MessageStop()

    msgs = [ConversationMessage(role="user", blocks=[TextBlock(text="OLDEST_MARK " + "a" * 990)])]
    msgs += [
        ConversationMessage(role="user", blocks=[TextBlock(text=f"m{i} " + "b" * 990)])
        for i in range(40)
    ]
    msgs.append(ConversationMessage(role="user", blocks=[TextBlock(text="NEWEST_MARK done")]))
    provider = CaptureProvider()
    out = await _summarize_llm(msgs, provider, "m")  # type: ignore[arg-type]
    assert out == "s"
    sent = provider.req.messages[0].blocks[0].text  # type: ignore[union-attr]
    assert "NEWEST_MARK" in sent
    assert "OLDEST_MARK" not in sent


def _call(cid: str, name: str, **inp: object) -> ConversationMessage:
    return ConversationMessage(
        role="assistant", blocks=[ToolUseBlock(id=cid, name=name, input=dict(inp))]
    )


def _result(cid: str, name: str, *, error: bool = False) -> ConversationMessage:
    return ConversationMessage(
        role="tool",
        blocks=[ToolResultBlock(tool_use_id=cid, tool_name=name, output="x", is_error=error)],
    )


def _say(role: str, text: str) -> ConversationMessage:
    return ConversationMessage(role=role, blocks=[TextBlock(text=text)])  # type: ignore[arg-type]


def test_rule_summary_keeps_asks_files_and_steps() -> None:
    msgs = [
        _say("user", "make the importer accept CSV"),
        _call("c1", "read_file", path="app/importer.py"),
        _result("c1", "read_file"),
        _call("c2", "edit_file", path="app/importer.py", old="a", new="b"),
        _result("c2", "edit_file", error=True),
        _call("c3", "edit_file", path="app/importer.py", old="a", new="c"),
        _result("c3", "edit_file"),
        _call("c4", "bash", command="pytest -q tests/test_importer.py"),
        _result("c4", "bash"),
        _say("assistant", "The importer now reads CSV; the header test still needs a fixture."),
    ]
    lines = _rule_summary(msgs).splitlines()
    assert lines[0] == "Earlier messages folded: 10 (user 1, assistant 5, tool results 4)"
    assert "  - make the importer accept CSV" in lines
    assert (
        "Last assistant note: The importer now reads CSV; the header test still needs a fixture."
        in lines
    )
    assert "Files changed: app/importer.py" in lines
    assert "Files read: app/importer.py" in lines
    assert "Tool use: edit_file 2 (1 failed), bash 1, read_file 1" in lines
    assert "  - edit_file app/importer.py: failed" in lines
    assert "  - bash pytest -q tests/test_importer.py: ok" in lines


def test_rule_summary_carries_the_previous_summary_instead_of_quoting_it_as_a_request() -> None:
    previous = _say(
        "user",
        f"{_SUMMARY_PREFIX}\n{_CONTINUATION_NOTICE}\n\n"
        "Earlier messages folded: 8 (user 1, assistant 4, tool results 3)\n"
        "Last assistant note: the parser is done\n"
        "Files changed: app/parse.py",
    )
    summary = _rule_summary([previous, _say("user", "now add a --dry-run flag")])
    assert (
        "Carried over from the previous summary: "
        "Last assistant note: the parser is done Files changed: app/parse.py"
    ) in summary
    assert _SUMMARY_PREFIX not in summary and _CONTINUATION_NOTICE not in summary
    assert summary.count("  - ") == 1


def test_rule_summary_keeps_only_the_latest_items() -> None:
    msgs: list[ConversationMessage] = []
    for i in range(12):
        msgs.append(_say("user", f"request {i}"))
        msgs.append(_call(f"w{i}", "write_file", path=f"out/f{i}.txt", content="x"))
        msgs.append(_result(f"w{i}", "write_file"))
    summary = _rule_summary(msgs)
    assert "request 0" in summary
    assert "request 8" not in summary and "request 9" in summary and "request 11" in summary
    assert "out/f1.txt" not in summary and "out/f2.txt" in summary and "out/f11.txt" in summary
    assert summary.count("  - write_file out/") == 5


def test_rule_summary_stays_within_its_ceiling_for_the_worst_input() -> None:
    long = "w" * 5_000
    msgs = [_say("user", f"{_SUMMARY_PREFIX}\n{long}")]
    for i in range(40):
        msgs.append(_say("user", long))
        msgs.append(_call(f"c{i}", f"tool_{i}_{long[:60]}", path=f"{i}/{long[:400]}"))
        msgs.append(_call(f"e{i}", "edit_file", path=f"{i}/{long[:400]}"))
        msgs.append(_result(f"c{i}", "x", error=True))
        msgs.append(_say("assistant", long))
    msgs.append(_call("r", "read_file", path=long))
    summary = _rule_summary(msgs * 50)
    assert len(summary) <= RULE_SUMMARY_MAX_CHARS
    assert max(len(line) for line in summary.splitlines()) <= 340


def test_rule_summary_of_text_only_history() -> None:
    summary = _rule_summary([_say("user", "hello"), _say("assistant", "hi")])
    assert summary.splitlines() == [
        "Earlier messages folded: 2 (user 1, assistant 1, tool results 0)",
        "What the user asked (oldest first):",
        "  - hello",
        "Last assistant note: hi",
    ]
