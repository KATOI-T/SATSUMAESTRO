"""新しい派生クラスを書いたときに、基底クラスを直接作っている未変更の場所を示す。"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from satsumaestro.rag.walk import _ALWAYS_SKIP_DIRS, load_ignore_spec
from satsumaestro.tools.base import ToolContext

FLAG_RELATED = "related-code"

_JS_SUFFIXES = frozenset({".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".mts", ".cts"})
_PY_SUFFIXES = frozenset({".py"})
_SCAN_SUFFIXES = _JS_SUFFIXES | _PY_SUFFIXES | frozenset({".html", ".htm", ".vue", ".svelte"})

MAX_SCAN_FILES = 300
MAX_SCAN_BYTES = 2_000_000
MAX_SITES = 3
_LINE_CHAR_CAP = 160

_JS_CLASS_RE = re.compile(
    r"^\s*(?:export\s+(?:default\s+)?)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"
    r"(?:\s*<[^>{]*>)?\s+extends\s+([A-Za-z_$][\w$.]*)",
    re.MULTILINE,
)
_PY_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*:", re.MULTILINE)


@dataclass(frozen=True)
class Site:
    """基底クラスを直接作っている 1 行。"""

    rel: str
    line: int
    text: str


def new_subclasses(before: str | None, after: str, suffix: str) -> list[tuple[str, str]]:
    """書き込みで新しく現れた（派生クラス, 基底クラス）の組。"""
    found = _subclasses(after, suffix)
    if not found:
        return []
    old = set(_subclasses(before, suffix)) if before else set()
    return [pair for pair in found if pair not in old]


def _subclasses(text: str, suffix: str) -> list[tuple[str, str]]:
    suffix = suffix.lower()
    pairs: list[tuple[str, str]] = []
    if suffix in _JS_SUFFIXES:
        for m in _JS_CLASS_RE.finditer(text):
            pairs.append((m.group(1), m.group(2).rsplit(".", 1)[-1]))
    elif suffix in _PY_SUFFIXES:
        for m in _PY_CLASS_RE.finditer(text):
            for raw in m.group(2).split(","):
                base = raw.strip().split("[", 1)[0]
                if not base or "=" in base or base == "object":
                    continue
                pairs.append((m.group(1), base.rsplit(".", 1)[-1]))
    return list(dict.fromkeys(pairs))


def construction_sites(root: Path, base: str, exclude: set[Path]) -> list[Site] | None:
    """``base`` を直接作っている行（``exclude`` のファイルは除く）。"""
    files = _scan_files(root)
    if files is None:
        return None
    js_new = re.compile(rf"\bnew\s+{re.escape(base)}\s*\(")
    py_call = re.compile(rf"(?<![\w.]){re.escape(base)}\s*\(")
    js_def = re.compile(rf"\bclass\s+{re.escape(base)}\b")
    py_def = re.compile(rf"^\s*class\s+{re.escape(base)}\b", re.MULTILINE)
    defined = False
    sites: list[Site] = []
    for path, text in files:
        is_py = path.suffix.lower() in _PY_SUFFIXES
        if (py_def if is_py else js_def).search(text):
            defined = True
        if path in exclude:
            continue
        pattern = py_call if is_py else js_new
        if not pattern.search(text):
            continue
        rel = path.relative_to(root).as_posix()
        for no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if is_py and stripped.startswith(("class ", "def ", "#")):
                continue
            if pattern.search(line):
                sites.append(Site(rel, no, stripped[:_LINE_CHAR_CAP]))
    return sites if defined else []


def _scan_files(root: Path) -> list[tuple[Path, str]] | None:
    files: list[tuple[Path, str]] = []
    total = 0
    for path in _walk(root):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total += size
        if len(files) >= MAX_SCAN_FILES or total > MAX_SCAN_BYTES:
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        files.append((path, text))
    return files


def _walk(root: Path) -> Iterator[Path]:
    """対象の拡張子のファイル。"""
    spec = load_ignore_spec(root)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        rel_dir = here.relative_to(root)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in _ALWAYS_SKIP_DIRS
            and not d.startswith(".")
            and not spec.match_file((rel_dir / d).as_posix() + "/")
        )
        for name in sorted(filenames):
            path = here / name
            if path.suffix.lower() not in _SCAN_SUFFIXES or path.is_symlink():
                continue
            if spec.match_file((rel_dir / name).as_posix()):
                continue
            yield path


def related_note(pairs: list[tuple[str, str]], base: str, sites: list[Site], more: int) -> str:
    """書き込み結果に添える注記（先頭は改行）。"""
    subs = ", ".join(sorted({sub for sub, b in pairs if b == base}))
    lines = [
        f"\nnote: {subs} extends {base}. This code creates {base} directly and was not changed "
        f"in this session; check whether it should handle {subs} too:"
    ]
    lines += [f"  {s.rel}:{s.line}  {s.text}" for s in sites]
    if more:
        lines.append(f"  (+{more} more)")
    return "\n".join(lines)


def after_write(
    ctx: ToolContext, real: Path, before: str | None, after: str
) -> tuple[str, list[str]]:
    """書き込みの後に呼ぶ。"""
    if not ctx.config.tools.related_code_notes:
        return "", []
    ledger = ctx.reads
    if ledger is not None and ledger.first_subclass_check(real):
        pairs = _subclasses(after, real.suffix)
    else:
        pairs = new_subclasses(before, after, real.suffix)
    if not pairs:
        return "", []
    exclude = set(ledger.written) if ledger is not None else set()
    exclude.add(real)
    notes: list[str] = []
    for base in dict.fromkeys(b for _, b in pairs):
        sites = construction_sites(ctx.jail.root, base, exclude)
        if not sites:
            continue
        if ledger is not None:
            sites = [s for s in sites if not ledger.related_shown(base, s.rel)]
            for s in sites:
                ledger.mark_related(base, s.rel)
        if not sites:
            continue
        shown = sites[:MAX_SITES]
        notes.append(related_note(pairs, base, shown, len(sites) - len(shown)))
    return "".join(notes), ([FLAG_RELATED] if notes else [])
