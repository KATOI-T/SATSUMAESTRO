"""システムプロンプト・タスク遂行ガイダンスのテスト。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.prompt.builder import SystemPromptBuilder

_ANTHROPIC_MODEL_ID = "claude-opus-4-8"
_GENERIC_MODEL = "gpt-oss:120b"
_SECTIONS = ("# How to work", "# Before you act", "# About this harness")


async def _static(tmp_path: Path, model: str = _ANTHROPIC_MODEL_ID) -> str:
    return (await SystemPromptBuilder().build(tmp_path, model, RuntimeConfig())).static


async def test_static_contains_procedural_sections(tmp_path: Path) -> None:
    static = await _static(tmp_path)
    for heading in _SECTIONS:
        assert heading in static


async def test_procedural_sections_are_static_only(tmp_path: Path) -> None:
    prompt = await SystemPromptBuilder().build(tmp_path, _ANTHROPIC_MODEL_ID, RuntimeConfig())
    for heading in _SECTIONS:
        assert heading in prompt.static
        assert heading not in prompt.dynamic


async def test_generic_model_keeps_procedural_sections(tmp_path: Path) -> None:
    static = await _static(tmp_path, _GENERIC_MODEL)
    for heading in _SECTIONS:
        assert heading in static


async def test_intro_still_switches_by_family(tmp_path: Path) -> None:
    claude = await _static(tmp_path, _ANTHROPIC_MODEL_ID)
    generic = await _static(tmp_path, _GENERIC_MODEL)
    assert claude != generic
    assert claude.split("# How to work", 1)[1] == generic.split("# How to work", 1)[1]


async def test_static_prefix_is_byte_stable(tmp_path: Path) -> None:
    b = SystemPromptBuilder()
    p1 = (await b.build(tmp_path / "a", _ANTHROPIC_MODEL_ID, RuntimeConfig())).static
    p2 = (await b.build(tmp_path / "b", _ANTHROPIC_MODEL_ID, RuntimeConfig())).static
    assert p1 == p2


async def test_no_volatile_value_in_static_half(tmp_path: Path) -> None:
    static = await _static(tmp_path)
    assert str(tmp_path) not in static
    assert "Date:" not in static
    assert "# Git" not in static
    assert "# Environment" not in static


async def test_planning_guidance_off_by_default(tmp_path: Path) -> None:
    static = await _static(tmp_path)
    assert "For multi-step tasks" not in static
    assert "call enter_plan_mode first" not in static
    assert "only when the user explicitly asks" in static


async def test_planning_guidance_on_restores_guidance(tmp_path: Path) -> None:
    prompt = await SystemPromptBuilder().build(
        tmp_path, _ANTHROPIC_MODEL_ID, RuntimeConfig(planning_guidance=True)
    )
    static = prompt.static
    assert "For multi-step tasks" in static
    assert "call enter_plan_mode first" in static
    assert "only when the user explicitly asks" not in static


async def test_conventions_deduped(tmp_path: Path) -> None:
    static = await _static(tmp_path)
    conventions = static.split("Conventions:", 1)[1].split("# How to work", 1)[0]
    assert "Never fabricate" not in conventions
    assert "workspace sandbox" not in conventions
    assert "Say what actually happened" in static
    assert "workspace sandbox" in static


async def test_undo_and_reach_wording(tmp_path: Path) -> None:
    static = await _static(tmp_path)
    actions = static.split("# Before you act", 1)[1].split("# About this harness", 1)[0]
    assert "how hard would this be to undo" in actions
    assert "how far does it reach" in actions


async def test_injection_and_compaction_notice(tmp_path: Path) -> None:
    static = await _static(tmp_path)
    system = static.split("# About this harness", 1)[1]
    assert "prompt injection" in system
    assert "compressed into a summary" in system
    assert "hook blocks or redirects" in system
    assert "permission mode" in system
