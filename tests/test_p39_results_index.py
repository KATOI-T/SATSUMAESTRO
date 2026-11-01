"""評価レポートの一覧（`satsumaestro eval-index`）。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.evals.index import bundled_suites, infer_suite, render_markdown, scan_reports
from satsumaestro.evals.models import EvalReport, TaskResult

CHECKOUT = Path(__file__).resolve().parents[1]


def _report(
    path: Path, task_ids: list[str], *, started_at_ms: int, suite: str = "", passed: bool = True
) -> None:
    report = EvalReport(
        agent="satsuma",
        model="m",
        started_at_ms=started_at_ms,
        suite=suite,
        results=[
            TaskResult(task_id=t, passed=passed, duration_ms=1000, iterations=2) for t in task_ids
        ],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(), encoding="utf-8")


def _evals_dir(root: Path) -> Path:
    evals = root / "evals"
    (evals / "code_analysis").mkdir(parents=True)
    for t in ("find-definition", "read-config"):
        (evals / "code_analysis" / f"{t}.toml").write_text(
            f'id = "{t}"\nprompt = "p"\nfixture = "."\n', encoding="utf-8"
        )
    (evals / "canary").mkdir()
    (evals / "canary" / "canary.toml").write_text(
        'name = "canary"\n[[cases]]\npath = "../code_analysis"\n', encoding="utf-8"
    )
    return evals


def test_bundled_suites_and_inference(tmp_path: Path) -> None:
    suites = bundled_suites(_evals_dir(tmp_path))
    assert suites == {"code_analysis": frozenset({"find-definition", "read-config"})}
    assert infer_suite(["read-config", "find-definition"], suites) == "code_analysis"
    assert infer_suite(["find-definition+plan"], suites) == "code_analysis/find-definition"
    assert infer_suite(["find-definition"], suites) == "code_analysis/find-definition"
    assert infer_suite(["zzz"], suites) == "?"
    assert infer_suite([], suites) == "?"


def test_scan_skips_baselines_defects_and_attempts(tmp_path: Path) -> None:
    evals = _evals_dir(tmp_path)
    root = tmp_path / ".satsuma" / "evals"
    _report(root / "old-run.json", ["find-definition", "read-config"], started_at_ms=1_000)
    _report(
        root / "web-bench" / "survey-1.json",
        ["init", "task-1"],
        started_at_ms=3_000,
        suite="web-bench/survey",
        passed=False,
    )
    _report(root / "p1" / "code-after-1.json", ["find-definition"], started_at_ms=2_000)
    _report(root / "baselines" / "canary.json", ["x"], started_at_ms=9_000)
    _report(root / "defects" / "r" / "sessions.json", ["x"], started_at_ms=9_000)
    _report(root / "web-bench" / "survey-1.attempts.json", ["x"], started_at_ms=9_000)
    (root / "broken.json").write_text("{not json", encoding="utf-8")

    rows, skipped = scan_reports(root, evals)
    assert skipped == 1
    assert [r.path for r in rows] == [
        "web-bench/survey-1.json",
        "p1/code-after-1.json",
        "old-run.json",
    ]
    assert [r.suite for r in rows] == [
        "web-bench/survey",
        "code_analysis/find-definition",
        "code_analysis",
    ]
    assert rows[0].passed == 0 and rows[0].total == 2 and rows[2].pass_rate == 1.0
    assert rows[2].duration_s == 2.0 and rows[2].iterations == 4


def test_render_markdown_has_summary_and_rows(tmp_path: Path) -> None:
    evals = _evals_dir(tmp_path)
    root = tmp_path / ".satsuma" / "evals"
    _report(root / "a.json", ["find-definition", "read-config"], started_at_ms=2_000)
    _report(root / "b.json", ["find-definition", "read-config"], started_at_ms=1_000, passed=False)
    rows, _ = scan_reports(root, evals)
    text = render_markdown(rows, limit=1, root=str(root))
    assert "## スイート別" in text and "## 新しい順（最大 1 本）" in text
    assert "| code_analysis | 2 | 50% |" in text
    assert text.count("`a.json`") == 1 and "`b.json`" not in text


def test_cli_eval_index_writes_file(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from satsumaestro.cli import app as cli_app

    _evals_dir(tmp_path)
    root = tmp_path / ".satsuma" / "evals"
    _report(root / "a.json", ["find-definition"], started_at_ms=2_000)
    out = tmp_path / "results.md"
    result = CliRunner().invoke(
        cli_app.app, ["eval-index", "--cwd", str(tmp_path), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "レポート 1 本" in result.output
    assert "code_analysis/find-definition" in out.read_text(encoding="utf-8")
    result = CliRunner().invoke(cli_app.app, ["eval-index", "--cwd", str(tmp_path)])
    assert result.exit_code == 0 and "評価結果の一覧" in result.output
    result = CliRunner().invoke(cli_app.app, ["eval-index", "--cwd", str(tmp_path / "nowhere")])
    assert result.exit_code == 1


def test_bundled_evals_dir_is_readable() -> None:
    suites = bundled_suites(CHECKOUT / "evals")
    assert "code_analysis" in suites and len(suites["code_analysis"]) == 8
    assert "canary" not in suites
