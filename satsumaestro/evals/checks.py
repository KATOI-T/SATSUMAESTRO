"""決定的採点。"""

from __future__ import annotations

import re
from pathlib import Path

from satsumaestro.evals.models import EvalChecks


def grade(text: str, checks: EvalChecks, workspace: Path | None = None) -> list[str]:
    """落ちた基準の説明リストを返す（空なら合格）。"""
    haystack = text.lower()
    failures: list[str] = []

    for needle in checks.contains_all:
        if needle.lower() not in haystack:
            failures.append(f"missing: {needle}")

    if checks.contains_any and not any(n.lower() in haystack for n in checks.contains_any):
        failures.append("none of: " + ", ".join(checks.contains_any))

    for pattern in checks.regex:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.ASCII) is None:
            failures.append(f"no match: {pattern}")

    for needle in checks.forbidden:
        if needle.lower() in haystack:
            failures.append(f"forbidden: {needle}")

    if workspace is not None:
        failures.extend(_file_failures(workspace, checks.file_min_lines))

    return failures


def _file_failures(workspace: Path, min_lines: dict[str, int]) -> list[str]:
    """生成物の行数検査。"""
    failures: list[str] = []
    for relpath, minimum in min_lines.items():
        path = workspace / relpath
        try:
            lines = len(path.read_text(encoding="utf-8").splitlines())
        except OSError:
            failures.append(f"missing file: {relpath}")
            continue
        if lines < minimum:
            failures.append(f"too short: {relpath} has {lines} lines (want >= {minimum})")
    return failures
