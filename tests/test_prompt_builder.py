"""SystemPromptBuilder のテスト。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.prompt.builder import SystemPromptBuilder


async def test_static_half_is_byte_stable(tmp_path: Path) -> None:
    b = SystemPromptBuilder()
    cfg = RuntimeConfig()
    p1 = await b.build(tmp_path / "a", "claude-opus-4-8", cfg)
    p2 = await b.build(tmp_path / "b", "claude-opus-4-8", cfg)
    assert p1.static == p2.static


async def test_no_volatile_values_before_boundary(tmp_path: Path) -> None:
    b = SystemPromptBuilder()
    static = (await b.build(tmp_path, "claude-opus-4-8", RuntimeConfig())).static
    assert str(tmp_path) not in static
    assert "Date:" not in static
    assert "# Git" not in static
    assert "# Environment" not in static


async def test_claude_vs_generic_intro(tmp_path: Path) -> None:
    b = SystemPromptBuilder()
    claude = await b.build(tmp_path, "claude-opus-4-8", RuntimeConfig())
    generic = await b.build(tmp_path, "gpt-4o", RuntimeConfig())
    assert "terminal" in claude.static
    assert claude.static != generic.static


async def test_language_matching_instruction_present(tmp_path: Path) -> None:
    b = SystemPromptBuilder()
    static = (await b.build(tmp_path, "claude-opus-4-8", RuntimeConfig())).static
    assert "same language" in static


async def test_instructions_discovered(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("PROJECT RULE X", encoding="utf-8")
    b = SystemPromptBuilder()
    prompt = await b.build(tmp_path, "claude-opus-4-8", RuntimeConfig())
    assert "PROJECT RULE X" in prompt.dynamic


async def test_tool_args_json_only_instruction_present(tmp_path: Path) -> None:
    b = SystemPromptBuilder()
    static = (await b.build(tmp_path, "claude-opus-4-8", RuntimeConfig())).static
    assert "only the JSON object" in static
