"""glob_search 組み込みツール。"""

from __future__ import annotations

import asyncio
import os
import re
from fnmatch import fnmatch
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.paths import AGENT_DIRNAME

_DEFAULT_EXCLUDE_DIRS = frozenset(
    {".git", ".hg", ".svn", AGENT_DIRNAME}
    | {"node_modules", ".venv", "venv"}
    | {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox"}
    | {"target", "dist", "build", ".build", "coverage"}
)

_WILDCARD_CHARS = frozenset("*?[{")


_BRACE_GROUP = re.compile(r"\{([^{}]*,[^{}]*)\}")


def _brace_choices(pattern: str) -> list[str]:
    """``*.{py,md}`` を ``*.py`` と ``*.md`` に開く（入れ子も可・重複は落として出た順）。"""
    done: list[str] = []
    todo = [pattern]
    while todo:
        current = todo.pop(0)
        group = _BRACE_GROUP.search(current)
        if group is None:
            if current not in done:
                done.append(current)
            continue
        head, tail = current[: group.start()], current[group.end() :]
        todo[:0] = [head + choice + tail for choice in group.group(1).split(",")]
    return done


def _component_regex(comp: str) -> str:
    """1 パス成分の glob をセパレータ非越境の正規表現へ変換する（``fnmatch`` 非依存）。"""
    out: list[str] = []
    i = 0
    while i < len(comp):
        ch = comp[i]
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        elif ch == "[":
            j = i + 1
            if j < len(comp) and comp[j] in "!^":
                j += 1
            if j < len(comp) and comp[j] == "]":
                j += 1
            while j < len(comp) and comp[j] != "]":
                j += 1
            if j >= len(comp):
                out.append(re.escape("["))
            else:
                raw = comp[i + 1 : j]
                inner = "^" + raw[1:] if raw.startswith("!") else raw
                out.append(f"[{inner}]")
                i = j
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


def _pattern_regex(pattern: str) -> re.Pattern[str]:
    """glob パターン（``**`` 対応）を相対 POSIX パス全体一致の正規表現へ変換する。"""
    parts = [p for p in pattern.split("/") if p not in ("", ".")]
    out: list[str] = []
    for idx, comp in enumerate(parts):
        last = idx == len(parts) - 1
        if comp == "**":
            if last:
                out.append(".*")
            else:
                out.append("(?:[^/]+/)*")
        else:
            out.append(_component_regex(comp) + ("" if last else "/"))
    return re.compile("".join(out) or r"\Z")


def _literal_components(pattern: str, base_path: str) -> frozenset[str]:
    """パターン・起点に明示されたリテラル成分（除外の解除対象）。"""
    comps = set(Path(base_path).parts)
    comps.update(c for c in pattern.split("/") if c and not _WILDCARD_CHARS.intersection(c))
    return frozenset(comps)


def _excluded(rel: Path, extra: list[str]) -> bool:
    """ユーザー指定 ``exclude``（fnmatch）に掛かるか。"""
    rel_posix = rel.as_posix()
    return any(fnmatch(rel_posix, pat) for pat in extra)


class GlobSearch(Tool):
    """jail 内の glob パターン検索。"""

    class Input(BaseModel):
        pattern: str = Field(
            description='Glob pattern to match, e.g. "**/*.py" or "*.{py,md}" (brace '
            "expansion supported). Matched against paths under the search root."
        )
        path: str = Field(default=".", description="Workspace-relative directory to search from.")
        exclude: list[str] = Field(
            default_factory=list,
            description="Extra fnmatch patterns to drop, matched against the "
            'workspace-relative path (e.g. "docs/*"). Version-control, dependency, cache and '
            "build-output directories (such as .git, node_modules, .venv, __pycache__, dist, "
            "target) are skipped by default unless named in pattern or path.",
        )
        limit: int = Field(
            default=1000,
            ge=1,
            description="Maximum number of paths to return; the list (newest mtime "
            "first) is truncated after this many entries.",
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="glob_search",
        description=(
            "List paths matching a glob pattern within the workspace as workspace-relative "
            "paths, most recently modified first. Supports brace expansion like *.{py,md}. "
            "Version-control, dependency, cache and build-output directories are skipped by "
            "default (name them in pattern or path to search them). Output is truncated after "
            "limit entries (default 1000) with a note."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, GlobSearch.Input)
        base = ctx.jail.resolve(inp.path)

        def _search() -> list[str]:
            root = ctx.jail.root
            regexes = [_pattern_regex(p) for p in _brace_choices(inp.pattern)]
            prune = _DEFAULT_EXCLUDE_DIRS - _literal_components(inp.pattern, inp.path)
            scored: list[tuple[float, str]] = []
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in prune]
                dir_path = Path(dirpath)
                for name in dirnames + filenames:
                    abs_path = dir_path / name
                    rel_to_base = abs_path.relative_to(base).as_posix()
                    if not any(rx.fullmatch(rel_to_base) for rx in regexes):
                        continue
                    if not ctx.jail.is_inside(abs_path):
                        continue
                    rel = abs_path.resolve().relative_to(root)
                    if _excluded(rel, inp.exclude):
                        continue
                    try:
                        mtime = abs_path.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    scored.append((mtime, str(rel)))
            scored.sort(key=lambda t: (-t[0], t[1]))
            hits = [path for _, path in scored]
            if len(hits) > inp.limit:
                del hits[inp.limit :]
                hits.append(f"... (truncated at {inp.limit} entries)")
            return hits

        results = await asyncio.to_thread(_search)
        return ToolResult(output="\n".join(results))
