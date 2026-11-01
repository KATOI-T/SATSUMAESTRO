"""窓に比べて大きい書き込みの本文を、会話の中で縮める。"""

from __future__ import annotations

import json
from pathlib import Path

from satsumaestro.config.models import CompactionConfig, HookConfig, RuntimeConfig
from satsumaestro.engine import written_bodies
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.base import ProviderKind
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import (
    ConversationMessage,
    OmittedBody,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider


def _page(lines: int) -> str:
    return "".join(f'<div class="card" id="card-{i}">item {i}</div>\n' for i in range(lines))


def _write_then_answer(
    content: str, *, path: str = "index.html", input_tokens: int = 3_000, output_tokens: int = 300
) -> list[list[object]]:
    return [
        [
            ToolUse(id="w1", name="write_file", input={"path": path, "content": content}),
            Usage(input_tokens=input_tokens, output_tokens=output_tokens),
            MessageStop(stop_reason="tool_use"),
        ],
        [
            TextDelta(text="done"),
            Usage(input_tokens=input_tokens + 6_000, output_tokens=5),
            MessageStop(),
        ],
    ]


def _two_writes_then_answer(
    first: object,
    second: dict[str, object],
    *,
    usages: tuple[int, int, int] = (3_000, 9_000, 9_000),
) -> list[list[object]]:
    """1 つ目の書き込み・2 つ目の呼び出し（``second`` が引数）・答え の 3 往復。"""
    one, two, three = usages
    return [
        [
            ToolUse(id="w1", name="write_file", input={"path": "index.html", "content": first}),
            Usage(input_tokens=one, output_tokens=300),
            MessageStop(stop_reason="tool_use"),
        ],
        [
            ToolUse(id="w2", name="append_file", input=second),
            Usage(input_tokens=two, output_tokens=300),
            MessageStop(stop_reason="tool_use"),
        ],
        [TextDelta(text="done"), Usage(input_tokens=three, output_tokens=5), MessageStop()],
    ]


def _runtime(
    tmp_path: Path,
    scripts: list[list[object]],
    *,
    window: int,
    threshold: int = 120_000,
    kind: ProviderKind = ProviderKind.OPENAI,
) -> tuple[ConversationRuntime, SequencedMockProvider]:
    provider = SequencedMockProvider(scripts, kind=kind)  # type: ignore[arg-type]
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(
            compact_token_threshold=threshold,
            compaction=CompactionConfig(context_window=window, use_llm_summary=False),
        ),
        compactor=Compactor(keep_last=6),
    )
    return rt, provider


def _sent_calls(provider: SequencedMockProvider, request: int) -> list[ToolUseBlock]:
    return [
        b
        for m in provider.requests[request].messages
        for b in m.blocks
        if isinstance(b, ToolUseBlock)
    ]


def _sent_call(provider: SequencedMockProvider, request: int) -> ToolUseBlock:
    (call,) = _sent_calls(provider, request)
    return call


def test_the_threshold_is_a_share_of_the_window() -> None:
    assert written_bodies.shrink_threshold(16_384) == 512
    assert written_bodies.shrink_threshold(8_192) == 256
    assert written_bodies.shrink_threshold(0) is None


def test_the_shrunk_input_keeps_the_path_and_the_edges_but_not_the_text() -> None:
    content = "".join(f"line {i}\n" for i in range(1, 188))
    out = written_bodies.omitted_input({"path": "a.html", "content": content}, written=True)
    assert out == {
        "path": "a.html",
        "content_omitted": (
            f"187 lines, {len(content):,} chars; written to the file and not repeated here"
        ),
        "first_line": "line 1",
        "last_lines": "line 183\nline 184\nline 185\nline 186\nline 187",
    }
    failed = written_bodies.omitted_input({"path": "a.html", "content": content}, written=False)
    assert failed is not None
    assert failed["content_omitted"].endswith(
        "not written (see the tool result) and not repeated here"
    )


def test_only_bodies_over_the_threshold_and_not_kept_are_shrunk() -> None:
    big, small = _page(60), _page(3)
    message = ConversationMessage(
        role="assistant",
        blocks=[
            TextBlock(text="writing"),
            ToolUseBlock(id="a", name="Write", input={"path": "a.html", "content": big}),
            ToolUseBlock(id="b", name="write_file", input={"path": "b.html", "content": small}),
            ToolUseBlock(id="c", name="append_file", input={"path": "plan.md", "content": big}),
            ToolUseBlock(id="d", name="read_file", input={"path": "a.html"}),
        ],
    )
    results = {
        "a": ToolResultBlock(tool_use_id="a", tool_name="write_file", output="wrote"),
        "b": ToolResultBlock(tool_use_id="b", tool_name="write_file", output="wrote"),
        "c": ToolResultBlock(tool_use_id="c", tool_name="append_file", output="appended"),
        "d": ToolResultBlock(tool_use_id="d", tool_name="read_file", output=big),
    }
    new, omitted = written_bodies.shrink_message(
        message, results, threshold=512, keep=lambda tool, path: path == "plan.md"
    )
    blocks = [b for b in new.blocks if isinstance(b, ToolUseBlock)]
    assert "content" not in blocks[0].input and blocks[0].input["path"] == "a.html"
    assert blocks[1].input["content"] == small
    assert blocks[2].input["content"] == big
    assert blocks[3].input == {"path": "a.html"}
    assert [(o.tool_use_id, o.tool, o.path, o.content) for o in omitted] == [
        ("a", "write_file", "a.html", big)
    ]
    assert message.blocks[1].input["content"] == big  # type: ignore[union-attr]


def test_a_copied_shape_is_refused_by_name() -> None:
    assert written_bodies.copied_omission("write_file", {"path": "a", "content_omitted": "x"})
    assert written_bodies.copied_omission("append_file", {"path": "a", "content_omitted": "x"})
    assert written_bodies.copied_omission("write_file", {"path": "a", "content": "x"}) is None
    assert written_bodies.copied_omission("edit_file", {"content_omitted": "x"}) is None


async def test_the_latest_write_keeps_its_body_until_the_next_write(tmp_path: Path) -> None:
    page, more = _page(187), _page(3)
    scripts = _two_writes_then_answer(page, {"path": "index.html", "content": more})
    rt, provider = _runtime(tmp_path, scripts, window=16_384)
    summary = await rt.run_turn("index.html を書いて")
    assert _sent_call(provider, 1).input["content"] == page
    first, second = _sent_calls(provider, 2)
    assert "content" not in first.input
    assert first.input["content_omitted"].startswith("187 lines, ")
    assert first.input["first_line"] == '<div class="card" id="card-0">item 0</div>'
    assert second.input["content"] == more
    assert summary.written_bodies_shrunk == 1
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == page + more
    assert [b.content for b in rt.session.omitted_bodies] == [page]


async def test_large_windows_and_unknown_windows_keep_the_body(tmp_path: Path) -> None:
    page = _page(187)
    for window in (131_072, 0):
        workdir = tmp_path / str(window)
        workdir.mkdir()
        scripts = _two_writes_then_answer(page, {"path": "index.html", "content": "x\n"})
        rt, provider = _runtime(workdir, scripts, window=window)
        summary = await rt.run_turn("index.html を書いて")
        assert summary.written_bodies_shrunk == 0
        assert _sent_calls(provider, 2)[0].input["content"] == page


async def test_anthropic_keeps_the_body(tmp_path: Path) -> None:
    page = _page(187)
    scripts = _two_writes_then_answer(page, {"path": "index.html", "content": "x\n"})
    rt, provider = _runtime(tmp_path, scripts, window=16_384, kind=ProviderKind.ANTHROPIC)
    summary = await rt.run_turn("index.html を書いて")
    assert summary.written_bodies_shrunk == 0
    assert _sent_calls(provider, 2)[0].input["content"] == page


async def test_the_plan_file_keeps_its_body(tmp_path: Path) -> None:
    rt, _ = _runtime(tmp_path, _write_then_answer("x"), window=16_384)
    assert rt._keeps_written_body("write_file", rt.plan_file_relpath())
    assert not rt._keeps_written_body("write_file", "index.html")


async def test_a_copied_shape_writes_nothing(tmp_path: Path) -> None:
    scripts = [
        [
            ToolUse(
                id="w1",
                name="write_file",
                input={"path": "index.html", "content_omitted": "187 lines, 6,564 chars"},
            ),
            Usage(input_tokens=3_000, output_tokens=40),
            MessageStop(stop_reason="tool_use"),
        ],
        [TextDelta(text="done"), Usage(input_tokens=3_000, output_tokens=5), MessageStop()],
    ]
    rt, provider = _runtime(tmp_path, scripts, window=16_384)
    summary = await rt.run_turn("index.html を書いて")
    assert summary.omitted_copies == 1
    assert not (tmp_path / "index.html").exists()
    (result,) = (
        b for m in provider.requests[1].messages for b in m.blocks if isinstance(b, ToolResultBlock)
    )
    assert result.is_error
    assert result.output == (
        "Nothing was written: write_file needs the text itself in `content`. `content_omitted` "
        "only marks earlier writes whose text this conversation does not repeat (it is in the "
        "files); it is not an argument, and a description of the text does not write it. "
        'Call write_file again as {"path": "index.html", "content": "<the whole text of the '
        'file>"}.'
    )


def test_the_copied_shape_error_shows_the_call_for_each_tool() -> None:
    out = written_bodies.copied_omission("append_file", {"content_omitted": "40 lines"})
    assert out is not None
    assert out.endswith(
        'Call append_file again as {"path": "<file path>", '
        '"content": "<the text to add at the end of the file>"}.'
    )


async def test_a_copied_shape_does_not_shrink_the_latest_write(tmp_path: Path) -> None:
    page = _page(187)
    copied = {"path": "index.html", "content_omitted": "120 lines, 3,500 chars; appended to file"}
    rt, provider = _runtime(tmp_path, _two_writes_then_answer(page, copied), window=16_384)
    summary = await rt.run_turn("index.html を書いて")
    assert (summary.omitted_copies, summary.written_bodies_shrunk) == (1, 0)
    assert _sent_calls(provider, 2)[0].input["content"] == page


async def test_the_next_write_lowers_the_sent_estimate_by_the_shrunk_body(
    tmp_path: Path,
) -> None:
    page = _page(187)
    scripts = _two_writes_then_answer(page, {"path": "index.html", "content": "x\n"})
    rt, _ = _runtime(tmp_path, scripts, window=16_384)
    seen: list[int] = []
    cached_tokens = rt._cached_tokens

    def record(digest: str) -> int:
        seen.append(cached_tokens(digest))
        return seen[-1]

    rt._cached_tokens = record  # type: ignore[method-assign]
    await rt.run_turn("index.html を書いて")
    assert seen[:2] == [0, 3_000]
    assert 1 < seen[2] < 9_000


async def test_the_request_after_the_shrink_is_not_taken_as_truncated(tmp_path: Path) -> None:
    page = _page(187)
    scripts = _two_writes_then_answer(
        page, {"path": "index.html", "content": "x\n"}, usages=(3_000, 12_000, 3_000)
    )
    rt, _ = _runtime(tmp_path, scripts, window=16_384)
    summary = await rt.run_turn("index.html を書いて")
    assert summary.written_bodies_shrunk == 1
    assert summary.context_truncations == 0
    assert summary.final_text == "done"


async def test_a_tool_reply_over_the_threshold_is_not_compacted_before_the_shrink(
    tmp_path: Path,
) -> None:
    page = _page(187)
    scripts = _two_writes_then_answer(
        page, {"path": "index.html", "content": "x\n"}, usages=(3_000, 10_000, 4_000)
    )
    scripts[1][1] = Usage(input_tokens=10_000, output_tokens=2_500)
    rt, _ = _runtime(tmp_path, scripts, window=16_384, threshold=12_000)
    _add_old_turns(rt)
    summary = await rt.run_turn("index.html を書いて")
    assert summary.written_bodies_shrunk == 1
    assert not summary.compacted
    assert rt.session.compaction is None


async def test_the_shrunk_body_is_not_counted_twice_before_the_next_request(
    tmp_path: Path,
) -> None:
    page = _page(187)
    scripts = _two_writes_then_answer(
        page, {"path": "index.html", "content": "x\n"}, usages=(3_000, 11_900, 9_000)
    )
    rt, _ = _runtime(tmp_path, scripts, window=16_384, threshold=12_000)
    _add_old_turns(rt)
    summary = await rt.run_turn("index.html を書いて")
    assert summary.written_bodies_shrunk == 1
    assert not summary.compacted


async def test_the_kept_body_is_shrunk_instead_of_compacting_before_the_send(
    tmp_path: Path,
) -> None:
    page = _page(187)
    scripts = _write_then_answer(page)
    scripts[1] = [TextDelta(text="done"), Usage(input_tokens=4_000, output_tokens=5), MessageStop()]
    rt, provider = _runtime(tmp_path, scripts, window=16_384, threshold=8_000)
    _add_old_turns(rt)
    summary = await rt.run_turn("index.html を書いて")
    assert not summary.compacted
    assert summary.written_bodies_shrunk == 1
    assert "content" not in _sent_call(provider, 1).input


async def test_the_kept_body_is_shrunk_instead_of_compacting_after_a_reply(
    tmp_path: Path,
) -> None:
    page = _page(187)
    scripts = _write_then_answer(page)
    scripts[1] = [
        TextDelta(text="done"),
        Usage(input_tokens=11_500, output_tokens=600),
        MessageStop(),
    ]
    rt, _ = _runtime(tmp_path, scripts, window=16_384, threshold=12_000)
    _add_old_turns(rt)
    summary = await rt.run_turn("index.html を書いて")
    assert not summary.compacted
    assert summary.written_bodies_shrunk == 1
    (call,) = (b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolUseBlock))
    assert "content" not in call.input


def _add_old_turns(rt: ConversationRuntime) -> None:
    for i in range(4):
        rt.session.messages += [
            ConversationMessage(role="user", blocks=[TextBlock(text=f"question {i}")]),
            ConversationMessage(role="assistant", blocks=[TextBlock(text=f"answer {i}")]),
        ]


async def test_fitting_the_window_shrinks_old_bodies_first(tmp_path: Path) -> None:
    page = _page(187)
    scripts = [[TextDelta(text="ok"), Usage(input_tokens=2_000, output_tokens=5), MessageStop()]]
    rt, provider = _runtime(tmp_path, scripts, window=8_192, threshold=6_553)
    rt.session.messages += [
        ConversationMessage(role="user", blocks=[TextBlock(text="index.html を書いて")]),
        ConversationMessage(
            role="assistant",
            blocks=[
                ToolUseBlock(
                    id="w0", name="write_file", input={"path": "index.html", "content": page}
                ),
                ToolUseBlock(id="w1", name="write_file", input={"path": "b.html", "content": page}),
            ],
        ),
        ConversationMessage(
            role="tool",
            blocks=[
                ToolResultBlock(tool_use_id="w0", tool_name="write_file", output="wrote"),
                ToolResultBlock(tool_use_id="w1", tool_name="write_file", output="wrote"),
            ],
        ),
        ConversationMessage(role="assistant", blocks=[TextBlock(text="done")]),
    ]
    summary = await rt.run_turn("続けて")
    assert summary.written_bodies_shrunk == 2
    assert summary.tool_result_cuts == 0
    sent = [
        b for m in provider.requests[0].messages for b in m.blocks if isinstance(b, ToolUseBlock)
    ]
    assert all("content" not in b.input for b in sent)


def test_omitted_bodies_round_trip_through_the_store(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    session = new_session(tmp_path, "mock-model")
    session.omitted_bodies.append(
        OmittedBody(tool_use_id="w1", tool="write_file", path="a.html", content="x\n")
    )
    store.save(session)
    path = store.path_for(tmp_path, session.session_id)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {
        "type": "omitted_body",
        "tool_use_id": "w1",
        "tool": "write_file",
        "path": "a.html",
        "content": "x\n",
    } in rows
    loaded = store.load(path)
    assert loaded.omitted_bodies == session.omitted_bodies
    assert loaded.messages == []


async def test_fitting_the_window_also_shrinks_the_plan_file(tmp_path: Path) -> None:
    page = _page(187)
    scripts = [[TextDelta(text="ok"), Usage(input_tokens=2_000, output_tokens=5), MessageStop()]]
    rt, provider = _runtime(tmp_path, scripts, window=8_192, threshold=6_553)
    plan = rt.plan_file_relpath()
    rt.session.messages += [
        ConversationMessage(role="user", blocks=[TextBlock(text="計画を書いて")]),
        ConversationMessage(
            role="assistant",
            blocks=[
                ToolUseBlock(id="p0", name="write_file", input={"path": plan, "content": page})
            ],
        ),
        ConversationMessage(
            role="tool",
            blocks=[ToolResultBlock(tool_use_id="p0", tool_name="write_file", output="wrote")],
        ),
        ConversationMessage(role="assistant", blocks=[TextBlock(text="done")]),
    ]
    summary = await rt.run_turn("続けて")
    assert summary.written_bodies_shrunk == 1
    assert _sent_call(provider, 0).input["path"] == plan
    assert "content" not in _sent_call(provider, 0).input
