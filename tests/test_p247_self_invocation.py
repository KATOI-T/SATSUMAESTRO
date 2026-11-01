"""エージェント自身の入れ子起動の絶対禁止。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder, TurnContext
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.security.self_invocation_guard import (
    PROHIBITED_SELF_INVOCATION_PREFIX,
    check_bash,
)
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import SequencedMockProvider

DANGER = PermissionMode.DANGER_FULL_ACCESS


@pytest.mark.parametrize(
    "command",
    [
        "satsumaestro repl",
        "uv run satsumaestro eval evals/file_generation/dashboard-html.toml --agent satsuma",
        "uv run --project /Users/x/SATSUMAESTRO satsumaestro prompt 'hi'",
        "cd /Users/x/SATSUMAESTRO && uv run satsumaestro eval evals/code_analysis",
        "ls -la; satsumaestro doctor --probe",
        ".venv/bin/satsumaestro repl --cwd /tmp/scratch",
        "/Users/x/SATSUMAESTRO/.venv/bin/satsumaestro --version",
        "python -m satsumaestro repl",
        "uv run python -m satsumaestro eval evals/code_analysis",
        "satsuma repl",
        "stro eval evals/code_analysis",
        "FOO=1 satsumaestro repl",
        "nohup satsumaestro eval evals/code_analysis",
    ],
)
def test_self_invocation_is_detected(command: str) -> None:
    reason = check_bash(command)
    assert reason is not None, command
    assert reason.startswith(PROHIBITED_SELF_INVOCATION_PREFIX)


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest -q",
        "uv run ruff check .",
        "ls -la",
        "grep -rn satsumaestro README.md",
        "echo 'satsumaestro repl'",
        "cat satsumaestro/cli/app.py",
        "python -m pytest tests/test_smoke.py",
        "uv run python scripts/eval_ab_summary.py --help",
        "git log --oneline -1",
        "rg 'satsumaestro' docs/",
    ],
)
def test_unrelated_commands_pass(command: str) -> None:
    assert check_bash(command) is None, command


def test_unparsable_command_is_left_to_the_usual_path() -> None:
    """解析できない（引用符が閉じない）コマンドは判定せず、従来の経路へ任せる。"""
    assert check_bash("satsumaestro repl 'unterminated") is None


def test_reason_names_the_subcommand() -> None:
    reason = check_bash("uv run satsumaestro eval evals/code_analysis")
    assert reason is not None and "satsumaestro eval" in reason


def _runtime(provider: object, tmp_path: Path, mode: PermissionMode) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=PermissionResolver(default_policy(mode), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


async def test_runtime_refuses_self_invocation_even_in_danger_mode(tmp_path: Path) -> None:
    """danger-full-access（＝確認なしで何でも通るモード）でも拒否する。"""
    provider = SequencedMockProvider(
        [
            [
                ToolUse(
                    id="b1",
                    name="bash",
                    input={"command": "uv run satsumaestro eval evals/code_analysis"},
                ),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path, DANGER)
    await rt.run_turn("評価を回して")
    blocks = {
        b.tool_use_id: b
        for m in rt.session.messages
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }
    out = blocks["b1"]
    assert out.is_error is True
    assert out.output.startswith(PROHIBITED_SELF_INVOCATION_PREFIX)
    assert "Do not retry" in out.output and "ask the user" in out.output


def _static(surface: str) -> str:
    builder = SystemPromptBuilder(surface=surface)
    return builder.compose(
        "gpt-oss:120b", RuntimeConfig(), None, TurnContext(instructions="", environment="", git="")
    ).static


def test_repl_surface_tells_the_model_it_is_inside_the_repl() -> None:
    """「自分が REPL の中にいる」ことを知らないので入れ子起動を試みていた。"""
    text = _static("repl")
    assert "# Session" in text
    assert "inside an interactive REPL session" in text
    assert "Never start another SATSUMAESTRO process" in text
    assert "ask them to run it" in text


def test_single_run_surface_differs_and_says_no_user_is_watching() -> None:
    text = _static("prompt")
    assert "single non-interactive run" in text
    assert "Never start another SATSUMAESTRO process" in text
    assert "inside an interactive REPL session" not in text


def test_default_surface_keeps_the_prompt_byte_identical() -> None:
    """既定（評価ハーネス・サブエージェント）は従来とバイト同一 ＝ 評価の数値に影響しない。"""
    assert "# Session" not in _static("")
    assert _static("") != _static("repl")


def test_surface_is_part_of_the_static_cache_key() -> None:
    """同一インスタンスの静的前半メモ化が実行面をまたいで誤ヒットしない。"""
    builder = SystemPromptBuilder(surface="repl")
    first = builder._static_sections("gpt-oss:120b", RuntimeConfig())
    assert "# Session" in first
    builder._surface = "prompt"
    assert "single non-interactive run" in builder._static_sections("gpt-oss:120b", RuntimeConfig())
