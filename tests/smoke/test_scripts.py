"""台本（tests/smoke/scripts/*.jsonl）と擬似サーバの判定の単体テスト。"""

from __future__ import annotations

import json
from pathlib import Path

from tests.smoke.fake_ollama import (
    NO_RULE,
    SCRIPTS_DIR,
    Script,
    load_script,
    sse_chunks,
    substitute,
)

PLAN_ID = "0123456789abcdef0123456789abcdef"


def _messages(*items: tuple[str, str] | dict[str, object]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = [
        {"role": "system", "content": f"Plan mode: write the plan to `.satsuma/plans/{PLAN_ID}.md`"}
    ]
    for item in items:
        out.append({"role": item[0], "content": item[1]} if isinstance(item, tuple) else item)
    return out


def test_every_script_loads_and_each_rule_replies() -> None:
    for path in sorted(SCRIPTS_DIR.glob("*.jsonl")):
        rules = load_script(path)
        assert rules, path.name
        for rule in rules:
            kinds = ("text", "tool_calls", "http_error", "stream_error", "overflow")
            replies = (rule.reply.get(k) for k in kinds)
            assert any(replies), (path.name, rule.index)


def test_happy_script_follows_the_plan_flow(tmp_path: Path) -> None:
    script = Script(load_script(SCRIPTS_DIR / "plan_three_steps.jsonl"), workspace=tmp_path)
    reply = script.decide({"messages": _messages(("user", "三つのファイル a.txt…を作成して"))})
    call = reply["tool_calls"][0]
    assert call["name"] == "write_file" and call["args"]["path"] == f".satsuma/plans/{PLAN_ID}.md"
    reply = script.decide(
        {
            "messages": _messages(
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "function": {"name": "write_file"}}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "wrote 90 bytes to .satsuma/plans/x.md",
                },
            )
        }
    )
    assert reply["tool_calls"][0]["name"] == "exit_plan_mode"
    reply = script.decide(
        {
            "messages": _messages(
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c2", "function": {"name": "write_file"}}],
                },
                {"role": "tool", "tool_call_id": "c2", "content": "wrote 2 bytes to a.txt"},
            )
        }
    )
    assert reply["text"].startswith("段 1 完了")
    assert [r["note"] for r in script.requests] == [
        "計画を書く",
        "承認を求める",
        "段 1 の要約（ツール無しで終える）",
    ]


def test_no_matching_rule_is_reported_in_the_reply(tmp_path: Path) -> None:
    script = Script(load_script(SCRIPTS_DIR / "empty_plan.jsonl"), workspace=tmp_path)
    reply = script.decide({"messages": _messages(("user", "関係のない依頼"))})
    assert reply["text"].startswith(NO_RULE)
    assert script.requests[-1]["rule"] is None


def test_times_limits_a_rule(tmp_path: Path) -> None:
    script = Script(load_script(SCRIPTS_DIR / "empty_plan.jsonl"), workspace=tmp_path)
    nudge = _messages(
        {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "exit_plan_mode"}}]},
        {
            "role": "tool",
            "tool_call_id": "x",
            "content": "No plan file yet, so there is nothing to show",
        },
    )
    first = script.decide({"messages": nudge})
    second = script.decide({"messages": nudge})
    assert first["tool_calls"][0]["name"] == "exit_plan_mode"
    assert "tool_calls" not in second and second["text"] == "了解しました。"


def test_substitute_uses_replace_not_format() -> None:
    reply = {"text": "{plan_file} と {x}", "tool_calls": [{"args": {"content": "## {not a var}"}}]}
    out = substitute(reply, {"plan_file": "p.md"})
    assert out["text"] == "p.md と {x}"
    assert out["tool_calls"][0]["args"]["content"] == "## {not a var}"


def test_sse_chunks_carry_text_tool_calls_finish_and_usage() -> None:
    reply = {
        "text": "こんにちは。",
        "tool_calls": [{"name": "write_file", "args": {"path": "a.txt", "content": "a\n"}}],
    }
    chunks = [c.decode() for c in sse_chunks(reply, "m")]
    assert chunks[-1] == "data: [DONE]\n\n"
    payloads = [json.loads(c.removeprefix("data: ")) for c in chunks[:-1]]
    text = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads if p["choices"])
    assert text == "こんにちは。"
    args = "".join(
        tc["function"].get("arguments", "")
        for p in payloads
        if p["choices"]
        for tc in p["choices"][0]["delta"].get("tool_calls", [])
    )
    assert json.loads(args) == {"path": "a.txt", "content": "a\n"}
    finishes = [p["choices"][0]["finish_reason"] for p in payloads if p["choices"]]
    assert finishes[-1] == "tool_calls"
    assert payloads[-1]["choices"] == [] and payloads[-1]["usage"]["completion_tokens"] > 0
    assert payloads[-1]["usage"]["prompt_tokens"] == 10


def test_script_can_pin_the_usage_numbers() -> None:
    """台本の ``reply.usage`` が usage チャンクにそのまま載る（表示の確認用）。"""
    reply = {"text": "数えました。", "usage": {"prompt_tokens": 12345, "completion_tokens": 678}}
    chunks = [c.decode() for c in sse_chunks(reply, "m")]
    usage = json.loads(chunks[-2].removeprefix("data: "))["usage"]
    assert usage == {"prompt_tokens": 12345, "completion_tokens": 678, "total_tokens": 13023}


def test_cut_sends_neither_finish_nor_usage() -> None:
    chunks = [c.decode() for c in sse_chunks({"reasoning": "hmm", "finish": "cut"}, "m")]
    assert all("[DONE]" not in c and '"usage"' not in c for c in chunks)
    assert '"reasoning": "hmm"' in chunks[0]


def _deltas(reply: dict[str, object]) -> list[dict[str, object]]:
    chunks = [c.decode() for c in sse_chunks(reply, "m")]
    payloads = [json.loads(c.removeprefix("data: ")) for c in chunks[:-1]]
    return [p["choices"][0]["delta"] for p in payloads if p["choices"]]


def test_stream_option_chunks_text_and_every_tool_call(tmp_path: Path) -> None:
    """``stream`` があれば本文と全呼び出しの引数を N 字ずつ送る。"""
    reply = {
        "text": "abcdefg",
        "tool_calls": [
            {"name": "write_file", "args": {"path": "a.txt", "content": "aaaa\n"}},
            {"name": "write_file", "args": {"path": "b.txt", "content": "bbbb\n"}},
        ],
        "stream": {"chunk_chars": 3, "delay_ms": 0},
    }
    deltas = _deltas(reply)
    texts = [d["content"] for d in deltas if "content" in d]
    assert texts == ["abc", "def", "g"]
    calls = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
    by_index: dict[int, list[dict[str, object]]] = {}
    for call in calls:
        by_index.setdefault(call["index"], []).append(call)
    for index, pieces in by_index.items():
        assert len(pieces) > 2, index
        assert "id" in pieces[0] and pieces[0]["function"]["name"] == "write_file"
        assert all("id" not in p for p in pieces[1:])
        joined = "".join(p["function"]["arguments"] for p in pieces)
        assert json.loads(joined)["path"] in ("a.txt", "b.txt")
    plain = _deltas({k: v for k, v in reply.items() if k != "stream"})
    second = [
        d["tool_calls"][0] for d in plain if "tool_calls" in d and d["tool_calls"][0]["index"] == 1
    ]
    assert len(second) == 1


def test_respond_returns_seq_and_mark_done_records_the_time(tmp_path: Path) -> None:
    script = Script(load_script(SCRIPTS_DIR / "long_code_block.jsonl"), workspace=tmp_path)
    reply, seq = script.respond({"messages": _messages(("user", "長いコードを本文に書いて"))})
    assert seq == 1 and reply["stream"] == {"chunk_chars": 20, "delay_ms": 40}
    assert "done_at" not in script.requests[0]
    script.mark_done(seq)
    assert script.requests[0]["done_at"] > 0
