"""セッション JSONL の秘密リダクション。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from satsumaestro.session.models import (
    ConversationMessage,
    SessionCompaction,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.session.redact import (
    REDACTED,
    _redact_cached,
    redact_rows,
    redact_secrets,
    secret_values,
)
from satsumaestro.session.store import JSONLSessionStore, new_session

_KEY = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"


def test_assign_values_redacted() -> None:
    assert redact_secrets(f"ANTHROPIC_API_KEY={_KEY}") == f"ANTHROPIC_API_KEY={REDACTED}"
    assert (
        redact_secrets('export OPENAI_API_KEY="sk-abc12345"')
        == f'export OPENAI_API_KEY="{REDACTED}"'
    )
    assert redact_secrets("GITHUB_TOKEN: ghx_value_1") == f"GITHUB_TOKEN: {REDACTED}"
    assert redact_secrets("slack_bot_token = xyz-123456") == f"slack_bot_token = {REDACTED}"
    assert redact_secrets('"client_secret": "abcd"') == f'"client_secret": "{REDACTED}"'
    assert redact_secrets("db_password=hunter2") == f"db_password={REDACTED}"
    assert redact_secrets("A_API_KEY = 'value-9'") == f"A_API_KEY = '{REDACTED}'"


def test_assign_prose_and_counters_not_redacted() -> None:
    for text in (
        "OPENAI_API_KEY is not set",
        "ANTHROPIC_API_KEY: ok",
        "ANTHROPIC_API_KEY: not set",
        "max_tokens = 100000",
        "token = 55555",
        "the secret: keep it quiet",
        "tokens: 12746",
        "password_hash = abcdef",
    ):
        assert redact_secrets(text) == text, text


def test_authorization_header() -> None:
    cmd = 'curl -H "Authorization: Bearer abcdef123456" https://x/'
    assert redact_secrets(cmd) == f'curl -H "Authorization: Bearer {REDACTED}" https://x/'
    assert redact_secrets("authorization: basic dXNlcjpwYXNz") == f"authorization: basic {REDACTED}"


def test_query_param() -> None:
    assert (
        redact_secrets("GET https://x/api?token=abc123&x=1")
        == f"GET https://x/api?token={REDACTED}&x=1"
    )
    assert redact_secrets("https://x/?x=1&api_key=k9") == f"https://x/?x=1&api_key={REDACTED}"


def test_token_shapes() -> None:
    for token in (
        _KEY,
        "sk-" + "a" * 24,
        "xai-" + "b" * 24,
        "ghp_" + "A" * 36,
        "github_pat_" + "c" * 30,
        "AKIAABCDEFGHIJKLMNOP",
        "xoxb-1234567890-abc",
        "AIza" + "d" * 35,
    ):
        assert redact_secrets(f"found {token} here") == f"found {REDACTED} here", token
    for plain in ("task-123", "sk-short", "mask", "ghp_tooshort", "AKIA123"):
        assert redact_secrets(plain) == plain, plain


def test_pem_block() -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\nabc\n-----END RSA PRIVATE KEY-----"
    assert redact_secrets(f"key:\n{pem}\nend") == f"key:\n{REDACTED}\nend"


def test_values_are_redacted_anywhere() -> None:
    out = redact_secrets("cat .env\nFOO=supersecretvalue1\nBAR=x", values=["supersecretvalue1"])
    assert out == f"cat .env\nFOO={REDACTED}\nBAR=x"
    assert redact_secrets("key ollama here", values=["ollama"]) == "key ollama here"


def test_secret_values_from_env() -> None:
    env = {
        "OPENAI_API_KEY": "ollama",
        "ANTHROPIC_API_KEY": _KEY,
        "SATSUMA_EMBED_API_KEY": "short",
        "PATH": "/x",
    }
    assert secret_values(env) == (_KEY,)
    assert secret_values({}) == ()


def test_redact_rows_skips_structural_keys() -> None:
    signature = "AKIAABCDEFGHIJKLMNOP"
    rows = [
        {
            "type": "message",
            "role": "assistant",
            "blocks": [
                {"type": "thinking", "thinking": f"use {_KEY}", "signature": signature},
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "bash",
                    "input": {"command": f"echo {_KEY}", "n": 1, "xs": [f"{_KEY}"]},
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "tool_name": "bash",
                    "output": f"KEY={_KEY}",
                    "is_error": False,
                },
            ],
        },
        {"type": "prompt_history", "text": f"my key is {_KEY}"},
    ]
    out = redact_rows(rows)
    blocks = out[0]["blocks"]
    assert blocks[0] == {"type": "thinking", "thinking": f"use {REDACTED}", "signature": signature}
    assert blocks[1]["name"] == "bash" and blocks[1]["id"] == "t1"
    assert blocks[1]["input"] == {"command": f"echo {REDACTED}", "n": 1, "xs": [REDACTED]}
    assert blocks[2]["output"] == f"KEY={REDACTED}" and blocks[2]["tool_name"] == "bash"
    assert out[1]["text"] == f"my key is {REDACTED}"
    assert rows[0]["blocks"][2]["output"] == f"KEY={_KEY}"


def test_store_save_redacts_file_but_not_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = "plainvalue-without-known-shape-0001"
    monkeypatch.setenv("SATSUMA_EMBED_API_KEY", value)
    store = JSONLSessionStore()
    s = new_session(tmp_path, "m")
    s.prompt_history.append(f"use {value}")
    s.messages.append(ConversationMessage(role="user", blocks=[TextBlock(text="cat .env")]))
    s.messages.append(
        ConversationMessage(
            role="assistant",
            blocks=[
                ThinkingBlock(thinking="reading", signature="sig-keep"),
                ToolUseBlock(id="t1", name="bash", input={"command": "cat .env"}),
            ],
        )
    )
    s.messages.append(
        ConversationMessage(
            role="tool",
            blocks=[
                ToolResultBlock(
                    tool_use_id="t1",
                    tool_name="bash",
                    output=f"EMBED={value}\nANTHROPIC_API_KEY={_KEY}\nPATH=/usr/bin",
                )
            ],
        )
    )
    s.compaction = SessionCompaction(
        compacted_at_ms=1, summary=f"user shared {value}", dropped_messages=0
    )
    store.save(s)
    path = store.path_for(tmp_path, s.session_id)
    raw = path.read_text(encoding="utf-8")
    assert value not in raw and _KEY not in raw
    assert raw.count(REDACTED) >= 4
    assert "PATH=/usr/bin" in raw and '"sig-keep"' in raw
    block = s.messages[2].blocks[0]
    assert isinstance(block, ToolResultBlock) and value in block.output and _KEY in block.output
    assert s.prompt_history == [f"use {value}"]
    assert s.compaction is not None and value in s.compaction.summary
    loaded = store.load(path)
    out = loaded.messages[2].blocks[0]
    assert isinstance(out, ToolResultBlock)
    assert out.output == f"EMBED={REDACTED}\nANTHROPIC_API_KEY={REDACTED}\nPATH=/usr/bin"
    assert loaded.prompt_history == [f"use {REDACTED}"]
    assert loaded.compaction is not None and loaded.compaction.summary == f"user shared {REDACTED}"
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert rows[0]["type"] == "session_meta" and rows[0]["model"] == "m"


def test_store_save_is_cached_across_saves(tmp_path: Path) -> None:
    """1 往復に 2 回ある保存で、同じ文字列の照合を繰り返さない。"""
    store = JSONLSessionStore()
    s = new_session(tmp_path, "m")
    chunk = ("line of ordinary tool output without secrets\n" * 400)[:20_000]
    for i in range(50):
        s.messages.append(
            ConversationMessage(
                role="tool",
                blocks=[
                    ToolResultBlock(
                        tool_use_id=f"t{i}", tool_name="read_file", output=chunk + str(i)
                    )
                ],
            )
        )
    _redact_cached.cache_clear()
    started = time.monotonic()
    store.save(s)
    first = time.monotonic() - started
    misses = _redact_cached.cache_info().misses
    started = time.monotonic()
    store.save(s)
    second = time.monotonic() - started
    info = _redact_cached.cache_info()
    assert info.misses == misses
    assert info.hits >= 50
    assert first < 5.0 and second < 1.0, (first, second)
