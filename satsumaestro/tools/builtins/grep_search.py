"""grep_search 組み込みツール。"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from satsumaestro.security.auto_allow import SecretMatcher
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.errors import ToolError

_MAX_HITS = 1000
_SHOWN_SKIPS = 5


def _skipped_note(paths: list[str]) -> str:
    """飛ばした秘密ファイルの注記（件数と、先頭 ``_SHOWN_SKIPS`` 件の名前）。"""
    shown = ", ".join(paths[:_SHOWN_SKIPS]) + (", …" if len(paths) > _SHOWN_SKIPS else "")
    noun = "file" if len(paths) == 1 else "files"
    return f"(skipped {len(paths)} secret {noun}: {shown})"


def _match_lines(
    text: str, lines: list[str], regex: re.Pattern[str], multiline: bool
) -> tuple[list[int], int]:
    """マッチ行番号（1 始まり・昇順）とマッチ数を返す。"""
    if multiline:
        linenos: set[int] = set()
        count = 0
        for m in regex.finditer(text):
            count += 1
            start = text.count("\n", 0, m.start()) + 1
            end = text.count("\n", 0, max(m.end() - 1, m.start())) + 1
            linenos.update(range(start, end + 1))
        return sorted(linenos), count
    hit_list = [i for i, line in enumerate(lines, start=1) if regex.search(line)]
    return hit_list, len(hit_list)


def _blocks(match_linenos: list[int], before: int, after: int, last: int) -> list[tuple[int, int]]:
    """マッチ行±文脈を行範囲へ展開し、重なり・隣接をマージした塊を返す。"""
    blocks: list[tuple[int, int]] = []
    for ln in match_linenos:
        start = max(1, ln - before)
        end = min(last, ln + after)
        if blocks and start <= blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], end))
        else:
            blocks.append((start, end))
    return blocks


class GrepSearch(Tool):
    """jail 内の正規表現検索。"""

    class Input(BaseModel):
        pattern: str = Field(description="Python regular expression to search for.")
        path: str = Field(default=".", description="Workspace-relative directory to search from.")
        include: str = Field(
            default="*", description='Glob that file names must match, e.g. "*.py".'
        )
        output_mode: Literal["content", "files_with_matches", "count"] = Field(
            default="content",
            description='"content" (default) prints path:line:text for each match, '
            '"files_with_matches" prints only the matching file paths, '
            '"count" prints path:match-count per file.',
        )
        context_before: int = Field(
            default=0,
            ge=0,
            description="Context lines to print before each match (grep -B). "
            'Printed as path-line-text. Only used with output_mode "content".',
        )
        context_after: int = Field(
            default=0,
            ge=0,
            description="Context lines to print after each match (grep -A). "
            'Printed as path-line-text. Only used with output_mode "content".',
        )
        multiline: bool = Field(
            default=False,
            description="Apply the pattern to the whole file with '.' also matching "
            "newlines (re.DOTALL), so a match can span multiple lines.",
        )
        ignore_case: bool = Field(default=False, description="Case-insensitive matching (grep -i).")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="grep_search",
        description=(
            "Search file contents for a Python regular expression within the workspace. "
            'output_mode "content" (default) prints matches as path:line:text; context lines '
            "requested via context_before/context_after are printed as path-line-text with "
            '"--" between blocks. "files_with_matches" lists only file paths and "count" '
            "prints per-file match counts. Output is truncated after 1000 matches "
            '(1000 files for "files_with_matches"). '
            "Use include to restrict which files are searched."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
    )

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(inp, GrepSearch.Input)
        flags = 0
        if inp.ignore_case:
            flags |= re.IGNORECASE
        if inp.multiline:
            flags |= re.DOTALL
        try:
            regex = re.compile(inp.pattern, flags)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}") from e
        base = ctx.jail.resolve(inp.path)
        root = ctx.jail.root
        with_context = inp.context_before > 0 or inp.context_after > 0
        secrets = SecretMatcher.from_config(ctx.config.permission)
        skipped: list[str] = []

        def _secret(p: Path, rel: Path) -> str | None:
            """秘密ファイルなら表示する名前（リンクなら書かれた名前）。"""
            try:
                written = p.relative_to(root).as_posix()
            except ValueError:
                written = p.name
            if secrets.matches(rel.as_posix()) or secrets.matches(written):
                return written
            return None

        def _emit_content(
            rel: Path, lines: list[str], linenos: list[int], results: list[str], hits: int
        ) -> tuple[int, bool]:
            """1 ファイル分の content 出力。"""
            matched = set(linenos)
            first_block = True
            for start, end in _blocks(linenos, inp.context_before, inp.context_after, len(lines)):
                if with_context and not first_block:
                    results.append("--")
                first_block = False
                for ln in range(start, end + 1):
                    line = lines[ln - 1]
                    if ln in matched:
                        results.append(f"{rel}:{ln}:{line}")
                        hits += 1
                        if hits >= _MAX_HITS:
                            return hits, True
                    else:
                        results.append(f"{rel}-{ln}-{line}")
            return hits, False

        def _search() -> list[str]:
            results: list[str] = []
            hits = 0
            for p in sorted(base.rglob(inp.include)):
                if not p.is_file() or not ctx.jail.is_inside(p):
                    continue
                rel = Path(p).resolve().relative_to(root)
                secret = _secret(p, rel)
                if secret is not None:
                    skipped.append(secret)
                    continue
                try:
                    text = p.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if inp.output_mode == "files_with_matches":
                    if inp.multiline:
                        found = regex.search(text) is not None
                    else:
                        found = any(regex.search(line) for line in text.splitlines())
                    if not found:
                        continue
                    results.append(str(rel))
                    hits += 1
                    if hits >= _MAX_HITS:
                        results.append(f"... (truncated at {_MAX_HITS} files)")
                        return results
                    continue
                lines = text.splitlines()
                linenos, count = _match_lines(text, lines, regex, inp.multiline)
                if not linenos:
                    continue
                if inp.output_mode == "count":
                    results.append(f"{rel}:{count}")
                    hits += count
                    if hits >= _MAX_HITS:
                        results.append(f"... (truncated at {_MAX_HITS} matches)")
                        return results
                    continue
                hits, truncated = _emit_content(rel, lines, linenos, results, hits)
                if truncated:
                    results.append(f"... (truncated at {_MAX_HITS} matches)")
                    return results
            return results

        results = await asyncio.to_thread(_search)
        if skipped:
            results.append(_skipped_note(skipped))
        return ToolResult(output="\n".join(results))
