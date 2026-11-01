"""指示ファイルの選び方: 根と作業ディレクトリでそれぞれ 1 つ・名前の優先・重複・総量と切った印。"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.prompt.instructions import (
    MAX_INSTRUCTION_CHARS,
    candidates,
    discover_instructions,
    search_dirs,
)


def _git_init(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    return path


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _headers(text: str) -> list[str]:
    return re.findall(r"^# Instructions from (.*)$", text, re.M)


def test_candidates_in_priority_order(tmp_path: Path) -> None:
    assert candidates(tmp_path) == [
        tmp_path / ".satsuma" / "SATSUMA.md",
        tmp_path / "SATSUMA.md",
        tmp_path / ".satsuma" / "CLAUDE.md",
        tmp_path / "CLAUDE.md",
        tmp_path / ".satsuma" / "AGENTS.md",
        tmp_path / "AGENTS.md",
    ]


def test_satsuma_md_wins_over_claude_md_and_agents_md(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "AGENTS.md", "AGENTS RULE")
    _write(root / "CLAUDE.md", "CLAUDE RULE")
    satsuma = _write(root / "SATSUMA.md", "SATSUMA RULE")
    out = discover_instructions(root)
    assert "SATSUMA RULE" in out
    assert "CLAUDE RULE" not in out
    assert "AGENTS RULE" not in out
    satsuma.unlink()
    out = discover_instructions(root)
    assert "CLAUDE RULE" in out
    assert "AGENTS RULE" not in out
    (root / "CLAUDE.md").unlink()
    assert "AGENTS RULE" in discover_instructions(root)


def test_inside_satsuma_dir_first_for_the_same_name(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "CLAUDE.md", "TOP LEVEL CLAUDE")
    _write(root / ".satsuma" / "CLAUDE.md", "SATSUMA DIR CLAUDE")
    out = discover_instructions(root)
    assert "SATSUMA DIR CLAUDE" in out
    assert "TOP LEVEL CLAUDE" not in out
    _write(root / "SATSUMA.md", "TOP LEVEL SATSUMA")
    out = discover_instructions(root)
    assert "TOP LEVEL SATSUMA" in out
    assert "SATSUMA DIR CLAUDE" not in out


def test_root_and_cwd_one_file_each_root_first(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "SATSUMA.md", "ROOT RULE")
    _write(root / "packages" / "CLAUDE.md", "MIDDLE RULE")
    cwd = root / "packages" / "api"
    _write(cwd / ".satsuma" / "AGENTS.md", "CWD RULE")
    out = discover_instructions(cwd)
    assert out.index("ROOT RULE") < out.index("CWD RULE")
    assert "MIDDLE RULE" not in out
    assert search_dirs(cwd) == [root.resolve(), cwd.resolve()]


def test_cwd_at_the_root_is_read_once(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "CLAUDE.md", "ONLY RULE")
    out = discover_instructions(root)
    assert out.count("ONLY RULE") == 1
    assert search_dirs(root) == [root.resolve()]


def test_nothing_above_the_git_root_is_read(tmp_path: Path) -> None:
    _write(tmp_path / "CLAUDE.md", "ABOVE THE REPO")
    root = _git_init(tmp_path / "repo")
    _write(root / "AGENTS.md", "REPO RULE")
    (root / "src").mkdir()
    out = discover_instructions(root / "src")
    assert "REPO RULE" in out
    assert "ABOVE THE REPO" not in out


def test_without_git_only_cwd_is_read(tmp_path: Path) -> None:
    _write(tmp_path / "CLAUDE.md", "PARENT RULE")
    cwd = tmp_path / "work"
    _write(cwd / "AGENTS.md", "WORK RULE")
    out = discover_instructions(cwd)
    assert "WORK RULE" in out
    assert "PARENT RULE" not in out
    assert search_dirs(cwd) == [cwd.resolve()]


def test_other_instruction_files_are_not_read(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "CLAUDE.local.md", "LOCAL RULE")
    _write(root / ".claude" / "CLAUDE.md", "DOT CLAUDE RULE")
    _write(root / ".satsuma" / "rules" / "style.md", "RULES DIR RULE")
    assert discover_instructions(root) == ""


def test_empty_file_falls_through_to_the_next_name(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "SATSUMA.md", "  \n\n")
    (root / ".satsuma").mkdir()
    (root / ".satsuma" / "CLAUDE.md").mkdir()
    _write(root / "CLAUDE.md", "CLAUDE RULE")
    out = discover_instructions(root)
    assert _headers(out) == [str(root.resolve() / "CLAUDE.md")]


def test_same_file_or_content_in_root_and_cwd_is_loaded_once(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "CLAUDE.md", "# Rules\n\nUse tabs.\n")
    cwd = root / "app"
    _write(cwd / "AGENTS.md", "\n# Rules  \n\nUse tabs.\n\n")
    out = discover_instructions(cwd)
    assert out.count("Use tabs.") == 1
    linked = root / "lib"
    linked.mkdir()
    (linked / "CLAUDE.md").symlink_to(root / "CLAUDE.md")
    out = discover_instructions(linked)
    assert _headers(out) == [str(root.resolve() / "CLAUDE.md")]


def test_invalid_utf8_does_not_break_the_prompt(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    (root / "CLAUDE.md").write_bytes(b"RULE \xff\xfe OK")
    out = discover_instructions(root)
    assert "RULE" in out and "OK" in out


def test_within_the_budget_nothing_is_cut(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "CLAUDE.md", "a" * 9_000)
    _write(root / "app" / "AGENTS.md", "b" * 6_000)
    out = discover_instructions(root / "app")
    assert "a" * 9_000 in out and "b" * 6_000 in out
    assert "[truncated" not in out
    assert "not loaded" not in out


def test_one_large_file_is_not_capped_on_its_own(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "CLAUDE.md", "x" * 15_000)
    out = discover_instructions(root)
    assert "x" * 15_000 in out
    assert "[truncated" not in out


def test_over_the_budget_the_file_is_cut_with_a_marker(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    lines = [f"rule {i:05d}: keep it short" for i in range(1_000)]
    _write(root / "SATSUMA.md", "\n".join(lines) + "\n")
    cwd = root / "app"
    _write(cwd / "CLAUDE.md", "APP RULE")
    out = discover_instructions(cwd)
    satsuma = root.resolve() / "SATSUMA.md"
    marker = re.search(r"\[truncated: only the first ([\d,]+) characters of this file", out)
    assert marker is not None
    shown = int(marker.group(1).replace(",", ""))
    assert 0 < shown <= MAX_INSTRUCTION_CHARS
    assert f"Read {satsuma} with read_file for the rest." in out
    body = out.split("\n", 1)[1].split("\n[truncated")[0]
    assert body.endswith("keep it short")
    assert "# Instruction files not loaded" in out
    assert f"- {cwd.resolve() / 'CLAUDE.md'}" in out
    assert "APP RULE" not in out


def test_the_budget_counts_root_and_cwd_together(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / "CLAUDE.md", "c" * 10_000)
    cwd = root / "app"
    _write(cwd / "AGENTS.md", "\n".join("d" * 99 for _ in range(100)))
    out = discover_instructions(cwd)
    assert "c" * 10_000 in out
    assert f"Read {cwd.resolve() / 'AGENTS.md'} with read_file for the rest." in out
    assert "# Instruction files not loaded" not in out


async def test_builder_puts_satsuma_md_in_the_dynamic_half(tmp_path: Path) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / ".satsuma" / "SATSUMA.md", "SATSUMA PROJECT RULE")
    prompt = await SystemPromptBuilder().build(root, "claude-opus-4-8", RuntimeConfig())
    assert "SATSUMA PROJECT RULE" in prompt.dynamic
    assert "SATSUMA PROJECT RULE" not in prompt.static


@pytest.mark.parametrize("name", ["SATSUMA.md", "CLAUDE.md", "AGENTS.md"])
def test_builder_snapshot_matches_discovery(tmp_path: Path, name: str) -> None:
    root = _git_init(tmp_path / "repo")
    _write(root / name, "SNAPSHOT RULE")
    assert SystemPromptBuilder()._discover_instructions(root) == discover_instructions(root)
