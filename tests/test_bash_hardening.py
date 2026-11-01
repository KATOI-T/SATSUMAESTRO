"""前景 bash の出力制御・実行安全化テスト。"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins import bash as bash_mod
from satsumaestro.tools.builtins.bash import (
    _KEEP_HEAD,
    _KEEP_TAIL,
    _MAX_OUTPUT_CHARS,
    Bash,
    _clip_output,
    _runs_tests,
)
from satsumaestro.util.proc import UNREAD_TAIL_BYTES, BackgroundProcessRegistry


async def _run(ctx: ToolContext, command: str, **kwargs: object):  # type: ignore[no-untyped-def]
    return await Bash().run(Bash.Input(command=command, **kwargs), ctx)  # type: ignore[arg-type]


def test_truncate_keeps_head_and_tail() -> None:
    text = "H" * _KEEP_HEAD + "M" * 50_000 + "T" * _KEEP_TAIL
    out = _clip_output(text)
    assert out.startswith("H" * 100)
    assert out.endswith("T" * 100)
    assert "output truncated: 50000 characters omitted" in out
    assert len(out) < _MAX_OUTPUT_CHARS + 200


def test_truncate_noop_within_limit() -> None:
    text = "x" * _MAX_OUTPUT_CHARS
    assert _clip_output(text) == text


async def test_bash_output_is_capped(tool_ctx: ToolContext) -> None:
    res = await _run(tool_ctx, "yes 0123456789 | head -c 40000")
    assert res.is_error is False
    assert "output truncated" in res.output
    assert len(res.output) < _MAX_OUTPUT_CHARS + 200


async def test_stdin_devnull_prevents_hang(tool_ctx: ToolContext) -> None:
    started = time.monotonic()
    res = await _run(tool_ctx, "cat", timeout_s=10.0)
    elapsed = time.monotonic() - started
    assert res.is_error is False
    assert res.output == ""
    assert elapsed < 5.0


async def test_nonzero_exit_code_is_annotated(tool_ctx: ToolContext) -> None:
    res = await _run(tool_ctx, "echo failing; exit 3")
    assert res.is_error is True
    assert "failing" in res.output
    assert "(exit code 3)" in res.output


async def test_command_not_found_shows_127(tool_ctx: ToolContext) -> None:
    res = await _run(tool_ctx, "no_such_command_xyz")
    assert res.is_error is True
    assert "(exit code 127)" in res.output


async def test_success_has_no_exit_code_annotation(tool_ctx: ToolContext) -> None:
    res = await _run(tool_ctx, "echo ok")
    assert res.is_error is False
    assert "exit code" not in res.output


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest -q",
        "CARGO  TEST --workspace",
        "npm test --silent",
        "npm run test:unit",
        "go test ./...",
        "make check",
        "cd app && npm install && npm test",
        "python3 -m unittest discover",
        "npx mocha test/task-8.spec.js",
        "npx playwright test",
        "FOO=1 timeout 30 pytest tests/",
        "make check 2>&1 | tail -20",
    ],
)
def test_test_runs_are_recognized(command: str) -> None:
    assert _runs_tests(command)


@pytest.mark.parametrize(
    "command",
    [
        "cargo build",
        "echo hello",
        "grep -r pytest .",
        "echo 'npm test'",
        "rm test.md",
        "cat tests/conftest.py",
        "npm install",
        'printf "unterminated',
    ],
)
def test_other_commands_are_not_test_runs(command: str) -> None:
    assert not _runs_tests(command)


async def test_timeout_plain(tool_ctx: ToolContext) -> None:
    res = await _run(tool_ctx, "sleep 3", timeout_s=0.2)
    assert res.is_error is True
    assert res.output == "timeout after 0.2s"


async def test_timeout_of_a_test_run_points_at_the_stuck_test(tool_ctx: ToolContext) -> None:
    res = await _run(tool_ctx, "sleep 3 && pytest -q", timeout_s=0.2)
    assert res.is_error is True
    assert res.output.startswith("timeout after 0.2s. This was a test run")
    assert "Find the test that does not finish" in res.output


async def test_a_huge_timeout_is_lowered_to_the_cap(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bash_mod, "MAX_TIMEOUT_S", 0.2)
    res = await _run(tool_ctx, "sleep 3", timeout_s=100000)
    assert res.is_error is True
    assert res.output == (
        "timeout after 0.2s. timeout_s is capped at 0.2s; run longer commands with "
        "run_in_background and read them with bash_output."
    )
    res = await _run(tool_ctx, "sleep 3 && pytest -q", timeout_s=100000)
    assert res.output.startswith("timeout after 0.2s. This was a test run")
    assert res.output.endswith("read them with bash_output.")


async def test_a_timeout_within_the_cap_has_no_cap_note(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bash_mod, "MAX_TIMEOUT_S", 0.5)
    res = await _run(tool_ctx, "sleep 3", timeout_s=0.2)
    assert res.output == "timeout after 0.2s"


def test_the_schema_tells_the_cap() -> None:
    assert bash_mod.MAX_TIMEOUT_S == 600.0
    description = Bash.spec.input_schema["properties"]["timeout_s"]["description"]
    assert "at most 600" in description


async def test_read_new_output_caps_to_tail(tmp_path: Path) -> None:
    registry = BackgroundProcessRegistry(tmp_path / "bg")
    env = dict(os.environ)
    task = await registry.spawn("yes 0123456789 | head -c 40000", cwd=tmp_path, env=env)
    for _ in range(100):
        if not task.running:
            break
        await asyncio.sleep(0.05)
    assert not task.running
    out = registry.read_new_output(task)
    assert "bytes of earlier unread output omitted" in out
    assert len(out) <= UNREAD_TAIL_BYTES + 100
    assert registry.read_new_output(task) == ""
    await registry.aclose()
