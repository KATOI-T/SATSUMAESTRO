"""題材の英語の依頼（``prompt_en``）と ``satsumaestro eval --prompt-lang en``。"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from satsumaestro.cli import app as cli_app
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.evals.baseline import Baseline
from satsumaestro.evals.checks import grade
from satsumaestro.evals.loader import load_tasks, localize_tasks
from satsumaestro.evals.models import EvalChecks, EvalReport, EvalTask
from satsumaestro.evals.suite import load_suite
from satsumaestro.session.models import TextBlock
from satsumaestro.util.errors import ConfigError
from tests.fixtures.mock_provider import MockProvider

REPO = Path(__file__).resolve().parents[1]


def _task(task_id: str, prompt_en: str = "") -> EvalTask:
    return EvalTask(id=task_id, prompt="日本語の依頼", prompt_en=prompt_en, fixture="f")


def _write_tasks(root: Path, *, with_english: bool = True) -> Path:
    fixture = root / "fixtures" / "proj"
    fixture.mkdir(parents=True)
    (fixture / "main.py").write_text("VALUE = 42\n", encoding="utf-8")
    tasks = root / "tasks"
    tasks.mkdir()
    english = 'prompt_en = "Where is VALUE defined?"\n' if with_english else ""
    (tasks / "where.toml").write_text(
        'id = "where"\n'
        'prompt = "VALUE はどこで定義されていますか。"\n'
        f"{english}"
        'fixture = "../fixtures/proj"\n'
        "[checks]\n"
        'contains_all = ["main.py"]\n',
        encoding="utf-8",
    )
    return tasks


def test_localize_keeps_the_original_prompt_without_a_language() -> None:
    tasks = [_task("a", "English")]
    assert localize_tasks(tasks, None) == tasks


def test_localize_swaps_in_the_english_prompt() -> None:
    (task,) = localize_tasks([_task("a", "Where is it?")], "en")
    assert task.prompt == "Where is it?"
    assert task.id == "a"


def test_localize_refuses_tasks_without_an_english_prompt() -> None:
    with pytest.raises(ConfigError, match="b, c"):
        localize_tasks([_task("a", "ok"), _task("b"), _task("c")], "en")


def test_localize_refuses_unknown_languages() -> None:
    with pytest.raises(ConfigError, match="en だけ"):
        localize_tasks([_task("a", "ok")], "fr")


def test_the_loader_reads_prompt_en(tmp_path: Path) -> None:
    (task,) = load_tasks(_write_tasks(tmp_path))
    assert task.prompt_en == "Where is VALUE defined?"


def test_the_canary_and_plan_generation_tasks_have_english_prompts() -> None:
    suite = load_suite(REPO / "evals" / "canary" / "canary.toml")
    plain, plan = suite.select(None)
    localized = localize_tasks(plain, "en") + localize_tasks(plan, "en")
    assert len(localized) == 17
    assert all(t.prompt.isascii() for t in localized)
    (dashboard,) = load_tasks(REPO / "evals" / "file_generation" / "dashboard-html.toml")
    assert localize_tasks([dashboard], "en")[0].prompt.startswith("Create an admin dashboard")


@pytest.mark.parametrize(
    ("pattern", "answer"),
    [
        (r"\b12\b", "index.html には `</div>` タグが **12行**あります。"),
        (r"\b9\b", "feed.xml に閉じタグ `</item>` は 9行 あります。"),
        (r"\b12\b", "閉じタグは 12個 です"),
        (r"\btotal\b", "cart.py の total関数が呼んでいます"),
        (r"(最後|last)", "最後の項目が出力されません"),
    ],
)
def test_word_boundaries_accept_numbers_and_names_next_to_japanese(
    pattern: str, answer: str
) -> None:
    assert grade(answer, EvalChecks(regex=[pattern])) == []


@pytest.mark.parametrize("answer", ["120行あります", "112 行", "subtotal関数"])
def test_word_boundaries_still_reject_other_numbers_and_words(answer: str) -> None:
    patterns = [r"\b12\b"] if "total" not in answer else [r"\btotal\b"]
    assert grade(answer, EvalChecks(regex=patterns)) != []


def test_a_baseline_refuses_reports_of_another_prompt_version() -> None:
    base = Baseline()
    base.add_report(EvalReport(started_at_ms=0, prompt_lang="en"), source="en-1.json")
    assert base.prompt_lang == "en"
    with pytest.raises(ConfigError, match="依頼の版"):
        base.add_report(EvalReport(started_at_ms=0), source="ja-1.json")


def test_cli_eval_sends_the_english_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    provider = MockProvider([TextDelta(text="It is defined in main.py."), MessageStop()])
    monkeypatch.setattr(cli_app, "build_provider", lambda config, model, env: provider)
    tasks = _write_tasks(tmp_path)
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli_app.app,
        ["eval", str(tasks), "--cwd", str(tmp_path), "--out", str(out), "--prompt-lang", "en"],
    )
    assert result.exit_code == 0, result.output
    assert "prompt en" in result.output
    first = provider.requests[0].messages[0].blocks[0]
    assert isinstance(first, TextBlock)
    assert first.text.startswith("Where is VALUE defined?")
    report = EvalReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert report.prompt_lang == "en"
    assert report.passed == 1


def test_cli_eval_refuses_tasks_without_an_english_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli_app, "build_provider", lambda config, model, env: MockProvider([MessageStop()])
    )
    tasks = _write_tasks(tmp_path, with_english=False)
    result = CliRunner().invoke(
        cli_app.app, ["eval", str(tasks), "--cwd", str(tmp_path), "--prompt-lang", "en"]
    )
    assert result.exit_code == 1
    assert "prompt_en" in result.output


def test_cli_eval_refuses_a_baseline_of_another_prompt_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli_app, "build_provider", lambda config, model, env: MockProvider([MessageStop()])
    )
    baseline = tmp_path / "baseline.json"
    Baseline().save(baseline)
    tasks = _write_tasks(tmp_path)
    result = CliRunner().invoke(
        cli_app.app,
        [
            "eval",
            str(tasks),
            "--cwd",
            str(tmp_path),
            "--prompt-lang",
            "en",
            "--baseline",
            str(baseline),
        ],
    )
    assert result.exit_code == 1
    assert "依頼の版" in result.output
