"""評価の出力の英語の画面と、不具合記録の英語の固定の文。"""

from __future__ import annotations

import datetime
import io
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli import app
from satsumaestro.evals.defects import detect_defects, save_defect_record
from satsumaestro.evals.index import IndexRow, render_markdown
from satsumaestro.evals.loader import load_tasks, localize_tasks
from satsumaestro.evals.models import Comparison, EvalReport, EvalTask, TaskResult
from satsumaestro.evals.web_routing import summarize
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.util.errors import ConfigError

_JAPANESE = re.compile(r"[　-ヿ一-鿿＀-￯]")


@pytest.fixture
def _english_screen() -> Iterator[None]:
    with pinned_ui_language("en"):
        yield


def _english(text: str) -> str:
    assert not _JAPANESE.search(text), text
    return text


def _report() -> EvalReport:
    results = [
        TaskResult(
            task_id="a",
            passed=True,
            tokens_per_s=40.0,
            timeout_recoveries=2,
            syntax_rejects=1,
            unknown_tool_calls=3,
        ),
        TaskResult(task_id="b", passed=False, connection_lost=True, plan_budget_sendbacks=1),
    ]
    return EvalReport(agent="satsuma", model="m", started_at_ms=0, results=results)


@pytest.mark.usefixtures("_english_screen")
def test_summary_line_in_english() -> None:
    assert app._summary_line(_report()) == (
        "passed 1/2 (50%) · generation 40.0 tok/s (median) · connection failures 1 (counted "
        "as failures) · timeout recoveries 2 · syntax-breaking edits: sent back 1 / broken "
        "writes 0 · unknown tool names 3 · plan send-backs 1"
    )


def test_summary_line_in_japanese() -> None:
    assert app._summary_line(_report()) == (
        "合格 1/2 (50%) ・ 生成 40.0 tok/s（中央値） ・ 接続の失敗 1 件（不合格に含む） ・ "
        "タイムアウト復帰 2 回 ・ 構文を壊す編集 差し戻し 1 回 / 壊した書き込み 0 回 ・ "
        "存在しないツール名 3 回 ・ 計画の差し戻し 1 回"
    )


@pytest.mark.usefixtures("_english_screen")
def test_comparison_and_notices_in_english(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = io.StringIO()
    monkeypatch.setattr(app, "_console", Console(file=out, width=300))
    comp = Comparison(
        baseline="base.json",
        gate_tasks=["a"],
        maintained=["a"],
        regressed=["b"],
        structural_errors=["b: run exception"],
        rerun_cases=["b"],
        decision="rerun",
    )
    app._render_comparison(comp)
    for i in range(5):
        (tmp_path / f"2026092{i}-t.md").write_text(
            "---\nfiled: no\nkinds: run exception\n---\n", encoding="utf-8"
        )
    app._render_repeated_defects(tmp_path)
    printed = _english(out.getvalue())
    assert (
        "baseline: base.json (gate tasks: 1) · kept 1 / improved 0 / regression candidates 1"
        in (printed)
    )
    assert "  structural errors: b: run exception" in printed
    assert "verdict: rerun · run 2 more under the same conditions: --only b" in printed
    assert "The same defect has piled up 5 times or more" in printed
    assert "  - run exception: 5 (latest: " in printed


@pytest.mark.usefixtures("_english_screen")
def test_index_and_web_routing_in_english() -> None:
    row = IndexRow(
        started=datetime.datetime(2026, 10, 1, 12, 0),
        suite="code_analysis",
        agent="satsuma",
        model="m",
        passed=1,
        total=2,
        duration_s=10,
        iterations=3,
        input_tokens=4,
        output_tokens=5,
        decision="pass",
        path="x.json",
    )
    text = _english(render_markdown([row], limit=5, root="/r"))
    assert text.startswith("# Eval results\n")
    assert "| code_analysis | 1 | 50% | 10 | 2026-10-01 12:00 (1/2) |" in text
    assert "## Newest first (up to 5)" in text
    assert summarize([]) == "Web Routing: no records"


@pytest.mark.usefixtures("_english_screen")
def test_load_errors_in_english(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"^No eval tasks \(\*\.toml\) found: "):
        load_tasks(tmp_path)
    with pytest.raises(ConfigError, match=r"^--prompt-lang accepts only en \(given: 'fr'\)$"):
        localize_tasks([], "fr")


def _task() -> EvalTask:
    return EvalTask(id="t", prompt="Where is VALUE defined?", fixture=".")


def test_records_are_written_in_english_on_a_japanese_screen(tmp_path: Path) -> None:
    result = TaskResult(task_id="t", passed=False, error="X: y", syntax_rejects=1)
    record = save_defect_record(
        tmp_path,
        _task(),
        result,
        detect_defects(result, "satsuma"),
        agent_name="satsuma",
        model="m",
        workspace=tmp_path / "none",
    )
    text = _english(record.read_text(encoding="utf-8"))
    assert text.startswith("---\nfiled: no\nkinds: run exception, syntax-breaking edit\n")
    assert "## What happened" in text and "## Findings" in text and "- Grade: failed" in text
    assert "- **syntax-breaking edit**: failed the syntax check right after an edit" in text
