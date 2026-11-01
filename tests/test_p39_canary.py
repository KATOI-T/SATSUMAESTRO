"""スイート manifest・履歴基準・対応付け・再実行。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.evals.baseline import (
    DEFAULT_KEEP,
    STABLE_MIN_RUNS,
    Baseline,
    TaskHistory,
    compare,
    describe_history,
)
from satsumaestro.evals.models import EvalReport, TaskResult, TokenUsage
from satsumaestro.evals.suite import PLAN_SUFFIX, is_suite_manifest, load_suite, select_tasks
from satsumaestro.util.errors import ConfigError
from tests.fixtures.mock_provider import MockProvider

CHECKOUT = Path(__file__).resolve().parents[1]


def _write_fixture(root: Path) -> Path:
    fixture = root / "fixtures" / "proj"
    fixture.mkdir(parents=True, exist_ok=True)
    (fixture / "main.py").write_text("VALUE = 42\n", encoding="utf-8")
    return fixture


def _write_task(root: Path, name: str, sub: str = "tasks") -> Path:
    tasks = root / sub
    tasks.mkdir(exist_ok=True)
    path = tasks / f"{name}.toml"
    path.write_text(
        f'id = "{name}"\n'
        'prompt = "VALUE はどこで定義されていますか。"\n'
        'fixture = "../fixtures/proj"\n'
        '[checks]\ncontains_all = ["main.py"]\n',
        encoding="utf-8",
    )
    return path


def _write_manifest(root: Path, body: str, name: str = "suite.toml") -> Path:
    path = root / name
    path.write_text(body, encoding="utf-8")
    return path


def _result(task_id: str, passed: bool, **kwargs: object) -> TaskResult:
    return TaskResult.model_validate({"task_id": task_id, "passed": passed, **kwargs})


def _report(*results: TaskResult, agent: str = "satsuma", model: str = "m") -> EvalReport:
    return EvalReport(agent=agent, model=model, started_at_ms=0, results=list(results))


def _history(passed: list[bool], **series: list[int]) -> TaskHistory:
    n = len(passed)
    return TaskHistory(
        passed=passed,
        duration_s=[1.0] * n,
        iterations=series.get("iterations", [3] * n),
        tool_calls=series.get("tool_calls", [2] * n),
        output_tokens=[10] * n,
    )


def test_manifest_expands_dirs_and_marks_plan_cases(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_task(tmp_path, "a")
    _write_task(tmp_path, "b")
    _write_task(tmp_path, "solo", sub="single")
    manifest = _write_manifest(
        tmp_path,
        'name = "canary-test"\ntarget_minutes = 5\n'
        '[[cases]]\npath = "tasks"\n'
        '[[cases]]\npath = "single/solo.toml"\nplan = true\ncapability = "plan-path"\n',
    )
    assert is_suite_manifest(manifest)
    assert not is_suite_manifest(tmp_path / "tasks" / "a.toml")
    suite = load_suite(manifest)
    assert suite.name == "canary-test" and suite.target_minutes == 5
    assert [t.id for t in suite.plain] == ["a", "b"]
    assert [t.id for t in suite.plan] == [f"solo{PLAN_SUFFIX}"]
    assert suite.order == ["a", "b", "solo+plan"]
    assert Path(suite.plan[0].fixture).is_dir()


def test_manifest_rejects_duplicates_unknown_keys_and_missing_paths(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_task(tmp_path, "a")
    dup = _write_manifest(
        tmp_path, 'name = "x"\n[[cases]]\npath = "tasks"\n[[cases]]\npath = "tasks/a.toml"\n'
    )
    with pytest.raises(ConfigError, match="重複"):
        load_suite(dup)
    ok = _write_manifest(
        tmp_path,
        'name = "x"\n[[cases]]\npath = "tasks"\n[[cases]]\npath = "tasks/a.toml"\nplan = true\n',
        name="ok.toml",
    )
    assert load_suite(ok).order == ["a", "a+plan"]
    unknown = _write_manifest(
        tmp_path, 'name = "x"\n[[cases]]\npath = "tasks"\ngate = true\n', name="unknown.toml"
    )
    with pytest.raises(ConfigError, match="不正"):
        load_suite(unknown)
    missing = _write_manifest(
        tmp_path, 'name = "x"\n[[cases]]\npath = "nowhere"\n', name="missing.toml"
    )
    with pytest.raises(ConfigError):
        load_suite(missing)


def test_select_only_filters_and_rejects_unknown_ids(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_task(tmp_path, "a")
    _write_task(tmp_path, "b")
    manifest = _write_manifest(
        tmp_path,
        'name = "x"\n[[cases]]\npath = "tasks"\n[[cases]]\npath = "tasks/a.toml"\nplan = true\n',
    )
    suite = load_suite(manifest)
    plain, plan = suite.select({"b", "a+plan"})
    assert [t.id for t in plain] == ["b"] and [t.id for t in plan] == ["a+plan"]
    with pytest.raises(ConfigError, match="zzz"):
        suite.select({"zzz"})
    assert [t.id for t in select_tasks(suite.plain, {"a"})] == ["a"]
    with pytest.raises(ConfigError):
        select_tasks(suite.plain, {"nope"})


def test_bundled_canary_manifest_loads() -> None:
    """同梱の canary: 全 case が解決し、plan 経路の課題が 1 つ以上ある。"""
    suite = load_suite(CHECKOUT / "evals" / "canary" / "canary.toml")
    assert suite.name == "canary"
    assert len(suite.order) >= 12
    assert len(suite.order) == len(set(suite.order))
    assert any(task_id.endswith(PLAN_SUFFIX) for task_id in suite.order)
    for flaky in ("dashboard-html", "find-anomalous-handler", "transitive-chain"):
        assert flaky not in suite.order and f"{flaky}+plan" not in suite.order


def test_baseline_accumulates_keeps_recent_and_decides_gate(tmp_path: Path) -> None:
    base = Baseline()
    for i in range(DEFAULT_KEEP + 2):
        base.add_report(_report(_result("t", passed=i != 0, iterations=i)), source=f"r{i}.json")
    history = base.tasks["t"]
    assert history.runs == DEFAULT_KEEP
    assert history.passed == [True] * DEFAULT_KEEP
    assert history.iterations[0] == 2
    assert history.stable and history.majority_pass
    assert base.model == "m" and len(base.sources) == DEFAULT_KEEP + 2
    path = tmp_path / "baselines" / "canary.json"
    base.save(path)
    assert Baseline.load(path).tasks["t"].runs == DEFAULT_KEEP
    with pytest.raises(ConfigError, match="モデル"):
        base.add_report(_report(_result("t", True), model="other"))


@pytest.mark.parametrize(
    ("passed", "stable"),
    [
        ([True] * STABLE_MIN_RUNS, True),
        ([True] * (STABLE_MIN_RUNS - 1), False),
        ([True] * 19 + [False], True),
        ([True] * 18 + [False, False], False),
    ],
)
def test_gate_needs_enough_runs_and_pass_rate(passed: list[bool], stable: bool) -> None:
    assert _history(passed).stable is stable


def test_compare_pairs_each_task_with_its_history() -> None:
    base = Baseline(
        tasks={
            "stable": _history([True] * 6),
            "flaky": _history([True, False, True, True, False, True]),
            "broken": _history([False] * 5),
            "fixed": _history([False] * 5),
        }
    )
    report = _report(
        _result("stable", False),
        _result("flaky", False),
        _result("broken", False),
        _result("fixed", True),
        _result("brand-new", True),
    )
    comp = compare(report, base, path="b.json")
    assert comp.gate_tasks == ["stable"]
    assert comp.regressed == ["stable"]
    assert comp.flaky_failed == ["flaky"]
    assert comp.still_failing == ["broken"]
    assert comp.improved == ["fixed"]
    assert comp.new == ["brand-new"]
    assert comp.maintained == []
    assert comp.rerun_cases == ["stable"] and comp.decision == "rerun"


def test_compare_structural_errors_and_budget_trigger_rerun() -> None:
    base = Baseline(
        tasks={
            "ok": _history([True] * 6, iterations=[2, 3, 4, 3, 5, 3]),
            "flat": _history([True] * 6, iterations=[3] * 6),
            "plan-case+plan": _history([True] * 6),
        },
    )
    report = _report(
        _result("ok", True, iterations=9),
        _result("flat", True, iterations=30),
        _result("plan-case+plan", True, plan_exits=0),
        _result("errored", False, error="ProviderError: x"),
    )
    comp = compare(report, base)
    assert comp.over_budget == ["ok: iterations 9 > 3 + 3×0.5"]
    assert "plan-case+plan: plan path not taken" in comp.structural_errors
    assert "errored: run exception" in comp.structural_errors
    assert comp.rerun_cases == ["ok", "plan-case+plan", "errored"]
    assert comp.decision == "rerun"
    assert comp.maintained == ["ok", "flat", "plan-case+plan"] and comp.new == ["errored"]


def test_compare_ignores_duration_and_passes_when_nothing_is_wrong() -> None:
    base = Baseline(tasks={"t": _history([True] * 6)})
    comp = compare(_report(_result("t", True, duration_ms=999_999)), base)
    assert comp.decision == "pass" and comp.rerun_cases == [] and comp.over_budget == []


def test_describe_history_rows() -> None:
    base = Baseline(tasks={"t": _history([True] * 6, iterations=[1, 2, 3, 4, 5, 6])})
    row = describe_history(base)[0]
    assert row.task == "t" and row.runs == 6 and row.gate is True
    assert row.iterations == 3.5


def _cli(monkeypatch: pytest.MonkeyPatch, text: str = "main.py です") -> object:
    from satsumaestro.cli import app as cli_app

    def fake_build_provider(config: object, model: str, env: object) -> MockProvider:
        return MockProvider([TextDelta(text=text), MessageStop()])

    monkeypatch.setattr(cli_app, "build_provider", fake_build_provider)
    return cli_app


def _canary_dir(tmp_path: Path) -> Path:
    _write_fixture(tmp_path)
    _write_task(tmp_path, "smoke")
    return _write_manifest(
        tmp_path,
        'name = "canary-test"\n[[cases]]\npath = "tasks"\n'
        '[[cases]]\npath = "tasks/smoke.toml"\nplan = true\n',
    )


def test_cli_runs_a_manifest_and_folds_plan_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    cli_app = _cli(monkeypatch)
    manifest = _canary_dir(tmp_path)
    out = tmp_path / "canary.json"
    result = CliRunner().invoke(
        cli_app.app,  # type: ignore[attr-defined]
        ["eval", str(manifest), "--cwd", str(tmp_path), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    report = EvalReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.suite == "canary-test"
    assert [r.task_id for r in report.results] == ["smoke", "smoke+plan"]
    assert report.comparison is None
    assert "canary-test" in result.output

    result = CliRunner().invoke(
        cli_app.app,  # type: ignore[attr-defined]
        ["eval", str(manifest), "--cwd", str(tmp_path), "--agent", "satsuma+plan"],
    )
    assert result.exit_code == 1 and "+plan" in result.output


def test_cli_runs_only_and_multiple_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    cli_app = _cli(monkeypatch)
    manifest = _canary_dir(tmp_path)
    out = tmp_path / "rerun.json"
    result = CliRunner().invoke(
        cli_app.app,  # type: ignore[attr-defined]
        [
            "eval",
            str(manifest),
            "--cwd",
            str(tmp_path),
            "--out",
            str(out),
            "--only",
            "smoke",
            "--runs",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    for run in (1, 2):
        saved = EvalReport.model_validate_json(
            (tmp_path / f"rerun-{run}.json").read_text(encoding="utf-8")
        )
        assert [r.task_id for r in saved.results] == ["smoke"]
    assert not out.exists()
    result = CliRunner().invoke(
        cli_app.app,  # type: ignore[attr-defined]
        ["eval", str(manifest), "--cwd", str(tmp_path), "--only", "nope"],
    )
    assert result.exit_code == 1 and "nope" in result.output


def test_cli_baseline_add_then_compare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    cli_app = _cli(monkeypatch)
    manifest = _canary_dir(tmp_path)
    runner = CliRunner()
    first = tmp_path / "first.json"
    assert (
        runner.invoke(
            cli_app.app,  # type: ignore[attr-defined]
            ["eval", str(manifest), "--cwd", str(tmp_path), "--out", str(first)],
        ).exit_code
        == 0
    )
    baseline = tmp_path / "baselines" / "canary.json"
    result = runner.invoke(
        cli_app.app,  # type: ignore[attr-defined]
        ["eval-baseline", "add", str(first), "--into", str(baseline)],
    )
    assert result.exit_code == 0, result.output
    assert "課題 2" in result.output
    loaded = Baseline.load(baseline)
    assert loaded.suite == "canary-test" and set(loaded.tasks) == {"smoke", "smoke+plan"}

    second = tmp_path / "second.json"
    result = runner.invoke(
        cli_app.app,  # type: ignore[attr-defined]
        [
            "eval",
            str(manifest),
            "--cwd",
            str(tmp_path),
            "--out",
            str(second),
            "--baseline",
            str(baseline),
        ],
    )
    assert result.exit_code == 0, result.output
    report = EvalReport.model_validate_json(second.read_text(encoding="utf-8"))
    assert report.comparison is not None
    assert report.comparison.maintained == ["smoke", "smoke+plan"]
    assert "smoke+plan: plan path not taken" in report.comparison.structural_errors
    assert report.comparison.decision == "rerun"
    assert "判定: rerun" in result.output and "--only smoke+plan" in result.output

    result = runner.invoke(
        cli_app.app,  # type: ignore[attr-defined]
        ["eval-baseline", "show", "--into", str(baseline)],
    )
    assert result.exit_code == 0 and "smoke+plan" in result.output
    result = runner.invoke(
        cli_app.app,  # type: ignore[attr-defined]
        ["eval-baseline", "bogus", "--into", str(baseline)],
    )
    assert result.exit_code == 1


def test_report_roundtrip_keeps_comparison(tmp_path: Path) -> None:
    base = Baseline(tasks={"t": _history([True] * 6)})
    report = _report(_result("t", True))
    report.suite = "canary"
    report.comparison = compare(report, base, path="b.json")
    restored = EvalReport.model_validate_json(report.model_dump_json())
    assert restored.suite == "canary"
    assert restored.comparison is not None and restored.comparison.maintained == ["t"]
    old = EvalReport.model_validate({"agent": "satsuma", "started_at_ms": 0, "results": []})
    assert old.suite == "" and old.comparison is None


def test_task_history_uses_usage_output_tokens() -> None:
    h = TaskHistory()
    h.append(_result("t", True, usage=TokenUsage(input_tokens=5, output_tokens=7)), keep=3)
    assert h.output_tokens == [7]
