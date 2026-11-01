#!/usr/bin/env python3
"""公式の評価（ベンチマーク）の、AI が採点する題を ``evals/official/<参照>/ai.toml`` にまとめる。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
BENCH = REPO / ".satsuma" / "benchmarks"
OUT = REPO / "evals" / "official"

Value = str | int | float | list[str]
Task = dict[str, Value]


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    url: str
    revision: str
    fetch: Callable[[], None]
    build: Callable[[], list[Task]]
    expected: str
    smoke: tuple[str, ...] = ()


_NOT_LITERAL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def toml_text(s: str) -> str:
    """複数行は読みやすいリテラル文字列にし、書けなければ基本文字列にする。"""
    if "\n" in s and "'''" not in s and not _NOT_LITERAL.search(s):
        return "'''\n" + s + "'''"
    return json.dumps(s, ensure_ascii=False).replace("\x7f", "\\u007f")


def toml_value(v: Value) -> str:
    if isinstance(v, int | float):
        return repr(v)
    if isinstance(v, list):
        if sum(len(x) for x in v) < 70:
            return "[" + ", ".join(toml_text(x) for x in v) + "]"
        return "[\n" + "".join(f"  {toml_text(x)},\n" for x in v) + "]"
    return toml_text(v)


def render_ai(source: Source, tasks: list[Task]) -> str:
    high = sum(t["priority"] == "high" for t in tasks)
    lines = [
        f"# {source.title}: AI が採点する題（{len(tasks)} 題・高 {high}・低 {len(tasks) - high}）",
        f"# 出典: {source.url} @ {source.revision[:8]}（出典とライセンスは README.md）",
        f"# expected: {source.expected}",
        "# 生成: scripts/official/catalog.py（手で直さない）",
        f"source = {toml_text(source.name)}",
        f"revision = {toml_text(source.revision)}",
    ]
    if source.smoke:
        ids = {t["id"] for t in tasks}
        missing = [i for i in source.smoke if i not in ids]
        if missing:
            raise SystemExit(f"{source.name}: smoke tasks not found: {missing}")
        lines.append(f"smoke = {toml_value(list(source.smoke))}")
    for t in tasks:
        lines += ["", "[[tasks]]", *(f"{k} = {toml_value(v)}" for k, v in t.items())]
    return "\n".join(lines) + "\n"


def write_source(source: Source, out_root: Path = OUT) -> list[Task]:
    tasks = source.build()
    ids = [t["id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{source.name}: duplicate task ids")
    folder = out_root / source.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ai.toml").write_text(render_ai(source, tasks), encoding="utf-8")
    return tasks


def _git(*args: str, cwd: Path) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def fetch_git(url: str, revision: str, dest: Path) -> None:
    """``dest`` に ``revision`` を取り出す。"""
    if not dest.exists():
        dest.mkdir(parents=True)
        _git("init", "-q", cwd=dest)
        _git("remote", "add", "origin", url, cwd=dest)
        _git("fetch", "-q", "--depth", "1", "origin", revision, cwd=dest)
        _git("checkout", "-q", "FETCH_HEAD", cwd=dest)
    head = _git("rev-parse", "HEAD", cwd=dest)
    if head != revision:
        raise SystemExit(f"{dest}: version is {head}, expected {revision}")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _names(pattern: re.Pattern[str], paths: list[Path]) -> list[str]:
    """テストの名前を、ファイルに出てくる順に重複なしで拾う。"""
    seen: dict[str, None] = {}
    for p in paths:
        if p.exists():
            for m in pattern.finditer(_read(p)):
                seen.setdefault(m.group(m.lastindex or 0), None)
    return list(seen)


POLYGLOT_URL = "https://github.com/Aider-AI/polyglot-benchmark"
POLYGLOT_REV = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
POLYGLOT_DIR = BENCH / "aider-polyglot"
_PY_TEST = re.compile(r"^\s*def (test_\w+)", re.MULTILINE)
POLYGLOT_LANGS = {
    "python": "high", "javascript": "high", "rust": "high",
    "cpp": "low", "go": "low", "java": "low",
}  # fmt: skip
POLYGLOT_TESTS: dict[str, re.Pattern[str]] = {
    "python": _PY_TEST,
    "javascript": re.compile(r"^\s*x?(?:test|it)\(\s*(['\"`])(.+?)\1", re.MULTILINE),
    "rust": re.compile(r"#\[test\]\s*(?:#\[ignore\]\s*)?fn (\w+)"),
    "cpp": re.compile(r"TEST_CASE\(\s*\"(.+?)\""),
    "go": re.compile(r"^func (Test\w+)", re.MULTILINE),
    "java": re.compile(r"@Test\s+(?:@\w+(?:\([^)]*\))?\s+)*(?:public\s+)?void (\w+)"),
}
_POLYGLOT_DOCS = ("introduction.md", "instructions.md", "instructions.append.md")


POLYGLOT_SMOKE = (
    "python/book-store", "python/bowling", "python/phone-number", "python/wordy",
    "javascript/grade-school", "javascript/say", "javascript/tournament",
    "rust/acronym", "rust/luhn-from", "rust/word-count",
)  # fmt: skip


def build_polyglot(root: Path = POLYGLOT_DIR) -> list[Task]:
    tasks: list[Task] = []
    for lang, priority in POLYGLOT_LANGS.items():
        practice = root / lang / "exercises" / "practice"
        for d in sorted(p for p in practice.iterdir() if p.is_dir()):
            files = json.loads(_read(d / ".meta" / "config.json")).get("files", {})
            prompt = "".join(
                _read(d / ".docs" / n) for n in _POLYGLOT_DOCS if (d / ".docs" / n).exists()
            )
            tests = [d / t for t in files.get("test", [])]
            tasks.append(
                {
                    "id": f"{lang}/{d.name}",
                    "priority": priority,
                    "prompt": prompt,
                    "expected": _names(POLYGLOT_TESTS[lang], tests),
                }
            )
    return tasks


SOURCES: dict[str, Source] = {
    s.name: s
    for s in (
        Source(
            "aider-polyglot", "Aider Polyglot", POLYGLOT_URL, POLYGLOT_REV,
            fetch=lambda: fetch_git(POLYGLOT_URL, POLYGLOT_REV, POLYGLOT_DIR),
            build=build_polyglot, expected="題に付いた単体テストの名前", smoke=POLYGLOT_SMOKE,
        ),
    )
}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="公式の評価の、AI が採点する題を ai.toml にまとめる")
    ap.add_argument("--source", action="append", choices=sorted(SOURCES), help="この参照だけ")
    args = ap.parse_args(argv)
    for name in args.source or list(SOURCES):
        source = SOURCES[name]
        source.fetch()
        tasks = write_source(source)
        hi = sum(t["priority"] == "high" for t in tasks)
        print(f"{name}: {len(tasks)} tasks (high {hi}, low {len(tasks) - hi})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
