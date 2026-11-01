"""glob_search の走査枝刈り・brace 展開・mtime 降順のテスト。"""

from __future__ import annotations

import os
from pathlib import Path

from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.glob_search import (
    GlobSearch,
    _brace_choices,
    _pattern_regex,
)


async def _run(ctx: ToolContext, **kwargs: object) -> list[str]:
    res = await GlobSearch().run(GlobSearch.Input(**kwargs), ctx)  # type: ignore[arg-type]
    return res.output.splitlines() if res.output else []


def test_brace_choices_basic() -> None:
    assert _brace_choices("*.{py,md}") == ["*.py", "*.md"]
    assert _brace_choices("src/{a,b}/*.ts") == ["src/a/*.ts", "src/b/*.ts"]


def test_brace_choices_nested_and_multi() -> None:
    assert _brace_choices("{a,b{1,2}}.txt") == ["a.txt", "b1.txt", "b2.txt"]


def test_brace_choices_unmatched_is_literal() -> None:
    assert _brace_choices("foo{bar") == ["foo{bar"]
    assert _brace_choices("no_brace.py") == ["no_brace.py"]


def test_pattern_regex_star_does_not_cross_separator() -> None:
    rx = _pattern_regex("*.py")
    assert rx.fullmatch("a.py")
    assert not rx.fullmatch("dir/a.py")


def test_pattern_regex_recursive_and_classes() -> None:
    rx = _pattern_regex("**/*.py")
    assert rx.fullmatch("a.py")
    assert rx.fullmatch("x/y/a.py")
    assert _pattern_regex("f[0-2].txt").fullmatch("f1.txt")
    assert not _pattern_regex("f[0-2].txt").fullmatch("f3.txt")
    assert _pattern_regex("f?.txt").fullmatch("f9.txt")
    assert _pattern_regex("src/**").fullmatch("src/deep/leaf.py")


async def test_build_output_dirs_pruned_by_default(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    for d in ("target/debug", "build", "coverage", "src"):
        (root / d).mkdir(parents=True)
    (root / "target" / "debug" / "gen.py").write_text("x", encoding="utf-8")
    (root / "build" / "out.py").write_text("x", encoding="utf-8")
    (root / "coverage" / "cov.py").write_text("x", encoding="utf-8")
    (root / "src" / "a.py").write_text("x", encoding="utf-8")
    assert await _run(tool_ctx, pattern="**/*.py") == ["src/a.py"]


async def test_explicit_pattern_lifts_default_exclude(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "dist").mkdir()
    (root / "dist" / "bundle.js").write_text("x", encoding="utf-8")
    assert await _run(tool_ctx, pattern="**/*.js") == []
    assert await _run(tool_ctx, pattern="dist/*.js") == ["dist/bundle.js"]


async def test_explicit_path_lifts_default_exclude(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "target" / "doc").mkdir(parents=True)
    (root / "target" / "doc" / "index.html").write_text("x", encoding="utf-8")
    assert await _run(tool_ctx, pattern="**/*.html", path="target") == ["target/doc/index.html"]


async def test_results_sorted_by_mtime_desc(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    for i, ts in [(0, 100), (1, 300), (2, 200)]:
        f = root / f"m{i}.txt"
        f.write_text("x", encoding="utf-8")
        os.utime(f, (ts, ts))
    assert await _run(tool_ctx, pattern="m*.txt") == ["m1.txt", "m2.txt", "m0.txt"]


async def test_directories_still_match(tool_ctx: ToolContext) -> None:
    (tool_ctx.workspace_root / "src").mkdir()
    assert await _run(tool_ctx, pattern="src") == ["src"]


async def test_brace_end_to_end(tool_ctx: ToolContext) -> None:
    root = tool_ctx.workspace_root
    (root / "a.py").write_text("x", encoding="utf-8")
    (root / "b.md").write_text("x", encoding="utf-8")
    (root / "c.txt").write_text("x", encoding="utf-8")
    for p in root.iterdir():
        os.utime(p, (1_000, 1_000))
    assert await _run(tool_ctx, pattern="*.{py,md}") == ["a.py", "b.md"]


async def test_symlink_escape_not_listed(tool_ctx: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside96"
    outside.mkdir(exist_ok=True)
    (outside / "secret.py").write_text("x", encoding="utf-8")
    link = tool_ctx.workspace_root / "link"
    link.symlink_to(outside)
    assert await _run(tool_ctx, pattern="**/*.py") == []
