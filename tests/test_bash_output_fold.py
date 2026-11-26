"""成功した bash の出力を畳み、``/expand`` で出し直す。

- ``tool_output="summary"``: 4 行以上の出力は出さず、完了の行に行数と ``/expand <番号>`` を
  添える。3 行以下（``SHORT_OUTPUT_LINES``）は畳まずに ``⎿`` の下に出す
- 失敗（``✗`` と詳細の行）と、出力の無いコマンドは今のまま
- 畳んだ出力は番号つきで直近 ``COLLAPSED_KEEP`` 件を持ち、``/expand`` は番号を省くと最後の 1 つ
- ``/expand`` は全文を出す（2000 字で切らない）。``redaction.live`` では伏せた後の出力とコマンド
- レンダラの既定は ``"full"``（すべて出す）
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, cast

import pytest
from rich.console import Console

from satsumaestro.cli.commands import ParsedCommand, SlashDispatcher, parse_slash
from satsumaestro.cli.render import COLLAPSED_KEEP, SHORT_OUTPUT_LINES, RichRenderer
from satsumaestro.config.models import (
    HookConfig,
    RedactionConfig,
    RuntimeConfig,
    ToolOutputMode,
)
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import ScriptedFailureProvider

_TOKEN = "ghp_" + "a1B2c3D4e5" * 3 + "f6G7h8"
_BEARER = "bearer-value-0123456789"


def _renderer(
    tool_output: ToolOutputMode | None = "summary", *, redact: bool = False
) -> tuple[RichRenderer, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100, color_system=None)
    if tool_output is None:
        return RichRenderer(console=console, redact=redact), buf
    return RichRenderer(console=console, tool_output=tool_output, redact=redact), buf


def _lines(tag: str, n: int = SHORT_OUTPUT_LINES + 1) -> str:
    return "".join(f"{tag}-{i}\n" for i in range(1, n + 1))


def _run_bash(
    r: RichRenderer, command: str, output: str, *, cid: str = "1", error: bool = False
) -> None:
    call = ToolUse(id=cid, name="bash", input={"command": command})
    r.on_tool_start(call)
    block = ToolResultBlock(tool_use_id=cid, tool_name="bash", output=output, is_error=error)
    r.on_tool_result(block)


def test_summary_collapses_the_output_into_the_done_line() -> None:
    r, buf = _renderer()
    r.start_turn()
    _run_bash(r, "make test", "collected 3 items\nok\nok\nPASSED\n")
    out = buf.getvalue()
    assert "PASSED" not in out
    assert "⏺ コマンド実行(make test)\n  ⎿  ✓ 完了 · 出力 4 行（/expand 1 で全文）" in out


def test_short_outputs_are_shown_under_the_line() -> None:
    r, buf = _renderer()
    r.start_turn()
    _run_bash(r, "pwd", "/tmp/x\n", cid="1")
    _run_bash(r, "ls", _lines("short", SHORT_OUTPUT_LINES), cid="2")
    out = buf.getvalue()
    assert "⏺ コマンド実行(pwd)\n  ⎿  /tmp/x\n     ✓ 完了" in out
    assert "  ⎿  short-1\n     short-2\n     short-3\n     ✓ 完了" in out
    assert "/expand" not in out and r.collapsed_output() is None
    # 番号は畳んだものだけに振る
    _run_bash(r, "seq 1 4", _lines("long"), cid="3")
    assert "出力 4 行（/expand 1 で全文）" in buf.getvalue()


@pytest.mark.parametrize("mode", ["full", None])
def test_full_shows_every_output(mode: ToolOutputMode | None) -> None:
    r, buf = _renderer(mode)
    r.start_turn()
    _run_bash(r, "make test", _lines("full", 6))
    out = buf.getvalue()
    assert "  ⎿  full-1\n" in out and "     full-6\n     ✓ 完了" in out
    assert "/expand" not in out
    assert r.collapsed_output() is None


def test_failures_and_silent_commands_are_not_collapsed() -> None:
    r, buf = _renderer()
    r.start_turn()
    _run_bash(r, "false", "boom\n(exit code 1)", cid="1", error=True)
    _run_bash(r, "mkdir -p out", "", cid="2")
    out = buf.getvalue()
    assert "✗" in out and "boom" in out
    assert "/expand" not in out
    assert r.collapsed_output() is None


def test_only_the_latest_outputs_are_kept() -> None:
    r, _ = _renderer()
    r.start_turn()
    total = COLLAPSED_KEEP + 2
    for i in range(1, total + 1):
        _run_bash(r, f"echo {i}", _lines(str(i)), cid=str(i))
    assert r.collapsed_output(1) is None and r.collapsed_output(2) is None
    kept = r.collapsed_output(3)
    assert kept is not None and kept.output == _lines("3")
    latest = r.collapsed_output()
    assert latest is not None and latest.number == total and latest.command == f"echo {total}"


def test_english_done_line() -> None:
    with pinned_ui_language("en"):
        r, buf = _renderer()
        r.start_turn()
        _run_bash(r, "ls", _lines("a"))
    assert "output 4 lines (/expand 1 to show them)" in buf.getvalue()


# ── /expand ──


def _dispatcher(render: RichRenderer | None) -> tuple[SlashDispatcher, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100, color_system=None)
    rt = cast(Any, None)  # /expand は会話に触れない
    return SlashDispatcher(rt, console, render=render), buf


async def _expand(d: SlashDispatcher, line: str) -> None:
    cmd = parse_slash(line)
    assert isinstance(cmd, ParsedCommand)
    await d.dispatch(cmd)


async def test_expand_shows_the_full_output() -> None:
    r, _ = _renderer()
    r.start_turn()
    long_output = "".join(f"line {i:04d}\n" for i in range(400))  # 2000 字を超える
    _run_bash(r, "make test\n# 2 行目", long_output, cid="1")
    _run_bash(r, "ls", _lines("f"), cid="2")
    d, buf = _dispatcher(r)

    await _expand(d, "/expand 1")
    out = buf.getvalue()
    assert "#1 $ make test" in out and "2 行目" not in out
    assert "line 0000" in out and "line 0399" in out and "truncated" not in out

    await _expand(d, "/expand")
    assert "#2 $ ls" in buf.getvalue()


async def test_expand_explains_what_it_cannot_show() -> None:
    d, buf = _dispatcher(None)
    await _expand(d, "/expand")
    assert "畳んだ bash の出力はまだありません" in buf.getvalue()

    r, _ = _renderer()
    r.start_turn()
    for i in range(1, 4):
        _run_bash(r, f"echo {i}", _lines(str(i)), cid=str(i))
    d, buf = _dispatcher(r)
    await _expand(d, "/expand 9")
    await _expand(d, "/expand abc")
    out = buf.getvalue()
    assert "番号 9 の出力はありません（出し直せるのは 1〜3。直近 20 件まで持ちます）" in out
    assert "番号を数字で指定してください（例: /expand 3）" in out


async def test_help_lists_expand() -> None:
    d, buf = _dispatcher(None)
    await _expand(d, "/help")
    assert "/expand [n]" in buf.getvalue()


# ── 伏せる ──


async def test_expand_heading_uses_the_redacted_command() -> None:
    r, buf = _renderer(redact=True)
    r.start_turn()
    command = f"curl -H 'Authorization: Bearer {_BEARER}' https://example.invalid"
    _run_bash(r, command, _lines("body"))
    d, shown = _dispatcher(r)
    await _expand(d, "/expand 1")
    assert "#1 $ curl" in shown.getvalue()
    for text in (buf.getvalue(), shown.getvalue()):
        assert _BEARER not in text


async def test_expand_shows_the_output_the_runtime_redacted(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text(f"first\nsecond\ntoken {_TOKEN}\nlast\n", encoding="utf-8")
    steps: list[BaseException | list[AssistantEvent | BaseException]] = [
        [ToolUse(id="b1", name="bash", input={"command": "cat notes.txt"}), MessageStop()],
        [TextDelta(text="done"), MessageStop()],
    ]
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.DANGER_FULL_ACCESS),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(max_iterations=4, redaction=RedactionConfig(live=True)),
    )
    r, buf = _renderer(redact=True)
    await rt.run_turn("go", render=r)
    d, shown = _dispatcher(r)
    await _expand(d, "/expand 1")
    out = shown.getvalue()
    assert "#1 $ cat notes.txt" in out and "token [redacted]" in out and "last" in out
    for text in (buf.getvalue(), out):
        assert _TOKEN not in text
