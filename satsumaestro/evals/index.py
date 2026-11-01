"""評価レポートの一覧。"""

from __future__ import annotations

import datetime
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from satsumaestro.evals.models import EvalReport
from satsumaestro.i18n.text import ui

_SKIP_DIRS = frozenset({"baselines", "defects"})
_SKIP_SUFFIX = ".attempts.json"
_ID_RE = re.compile(r'^id\s*=\s*"([^"]+)"', re.M)


@dataclass(frozen=True)
class IndexRow:
    started: datetime.datetime
    suite: str
    agent: str
    model: str
    passed: int
    total: int
    duration_s: float
    iterations: int
    input_tokens: int
    output_tokens: int
    decision: str
    path: str

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def bundled_suites(evals_dir: Path) -> dict[str, frozenset[str]]:
    """同梱スイート名 → 課題 ID の集合（``evals/<suite>/*.toml`` の ``id``）。"""
    suites: dict[str, frozenset[str]] = {}
    if not evals_dir.is_dir():
        return suites
    for sub in sorted(p for p in evals_dir.iterdir() if p.is_dir()):
        ids: set[str] = set()
        for toml in sub.glob("*.toml"):
            try:
                text = toml.read_text(encoding="utf-8")
            except OSError:
                continue
            m = _ID_RE.search(text)
            if m and "[[cases]]" not in text:
                ids.add(m.group(1))
        if ids:
            suites[sub.name] = frozenset(ids)
    return suites


def infer_suite(task_ids: list[str], suites: dict[str, frozenset[str]]) -> str:
    """課題 ID の集合から同梱スイートを推定する（一致しなければ ``?``）。"""
    ids = {t.removesuffix("+plan") for t in task_ids}
    if not ids:
        return "?"
    for name, members in suites.items():
        if ids == set(members):
            return name
    if len(ids) == 1:
        only = next(iter(ids))
        for name, members in suites.items():
            if only in members:
                return f"{name}/{only}"
    for name, members in suites.items():
        if ids <= set(members):
            return ui("evals.index.tasks", name=name, ids_count=len(ids))
    return "?"


def scan_reports(root: Path, evals_dir: Path | None = None) -> tuple[list[IndexRow], int]:
    """``root`` 配下の共通レポートを読んで行にする。"""
    suites = bundled_suites(evals_dir) if evals_dir is not None else {}
    rows: list[IndexRow] = []
    skipped = 0
    for path in sorted(root.rglob("*.json")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        if path.name.endswith(_SKIP_SUFFIX):
            continue
        try:
            report = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError):
            skipped += 1
            continue
        results = report.results
        rows.append(
            IndexRow(
                started=datetime.datetime.fromtimestamp(report.started_at_ms / 1000),
                suite=report.suite or infer_suite([r.task_id for r in results], suites),
                agent=report.agent,
                model=report.model,
                passed=report.passed,
                total=report.total,
                duration_s=sum(r.duration_ms for r in results) / 1000,
                iterations=sum(r.iterations for r in results),
                input_tokens=sum(r.usage.input_tokens for r in results),
                output_tokens=sum(r.usage.output_tokens for r in results),
                decision=report.comparison.decision if report.comparison is not None else "",
                path=str(path.relative_to(root)),
            )
        )
    rows.sort(key=lambda r: r.started, reverse=True)
    return rows, skipped


def render_markdown(rows: list[IndexRow], *, limit: int = 50, root: str = "") -> str:
    """一覧の Markdown（スイート別の要約 ＋ 新しい順の表）。"""
    made = f"{datetime.datetime.now():%Y-%m-%d %H:%M}"
    lines = [
        ui("evals.index.eval_results"),
        "",
        ui("evals.index.generated_reports_scanned", made=made, rows_count=len(rows), root=root)
        if root
        else ui("evals.index.generated_reports", made=made, rows_count=len(rows)),
        "",
        ui("evals.index.by_suite_latest_5"),
        "",
        ui("evals.index.suite_reports_pass_rate_of"),
        "|---|---:|---:|---:|---|",
    ]
    by_suite: dict[str, list[IndexRow]] = defaultdict(list)
    for r in rows:
        by_suite[r.suite].append(r)
    for suite, items in sorted(by_suite.items(), key=lambda kv: kv[1][0].started, reverse=True):
        recent = items[:5]
        rate = sum(r.passed for r in recent) / max(sum(r.total for r in recent), 1)
        med = statistics.median(r.duration_s for r in items)
        lines.append(
            f"| {suite} | {len(items)} | {rate * 100:.0f}% | {med:.0f} | "
            + ui(
                "evals.index.latest_cell",
                started=items[0].started,
                passed=items[0].passed,
                total=items[0].total,
            )
        )
    lines += [
        "",
        ui("evals.index.newest_first_up_to", limit=limit),
        "",
        ui("evals.index.when_suite_agent_model_passed"),
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows[:limit]:
        model = f"{r.agent} / {r.model}" if r.model else r.agent
        lines.append(
            f"| {r.started:%Y-%m-%d %H:%M} | {r.suite} | {model} | {r.passed}/{r.total} | "
            f"{r.duration_s:.0f} | {r.iterations} | {r.input_tokens:,} | {r.output_tokens:,} | "
            f"{r.decision or '-'} | `{r.path}` |"
        )
    return "\n".join(lines) + "\n"
