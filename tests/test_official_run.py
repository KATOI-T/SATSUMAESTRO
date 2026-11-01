"""公式の評価の題を選んで回す入口と、Aider Polyglot の採点。"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "official" / "run.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("official_run", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["official_run"] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load()
pg = run.polyglot

_CATALOG = {
    "smoke": ["b"],
    "tasks": [
        {"id": "a", "priority": "high"},
        {"id": "b", "priority": "high"},
        {"id": "c", "priority": "low"},
    ],
}


def _ids(tasks: list[dict[str, object]]) -> list[object]:
    return [t["id"] for t in tasks]


def test_select_by_smoke_priority_ids_and_limit() -> None:
    assert _ids(run.select(_CATALOG, smoke=True)) == ["b"]
    assert _ids(run.select(_CATALOG, priority="high")) == ["a", "b"]
    assert _ids(run.select(_CATALOG, priority="all", limit=2)) == ["a", "b"]
    assert _ids(run.select(_CATALOG, ids=["c", "a"])) == ["c", "a"]
    with pytest.raises(SystemExit, match="目録に無い題: z"):
        run.select(_CATALOG, ids=["z"])
    with pytest.raises(SystemExit, match="smoke がありません"):
        run.select({"tasks": _CATALOG["tasks"]}, smoke=True)


def test_default_out_is_stable_so_the_same_command_resumes() -> None:
    out = run.default_out("aider-polyglot", "gpt-oss:120b", "smoke", "")
    assert out == run.RESULTS / "aider-polyglot" / "gpt-oss_120b-smoke"
    assert run.default_out("aider-polyglot", "x", "high", "reference").name == "dry-reference-high"
    assert run.task_file(out, "python/wordy").name == "python__wordy.json"
    high = run.default_out("aider-polyglot", "gpt-oss:120b", "all", "", "aider", "high")
    assert high.name == "aider-gpt-oss_120b-high-all"


_SOLVED = 'def two_fer(name="you"):\n    return f"One for {name}, one for me."\n'
_TEST = (
    "from two_fer import two_fer\n\n\n"
    "def test_no_name():\n"
    '    assert two_fer() == "One for you, one for me."\n'
)


@pytest.fixture(autouse=True)
def _no_toolchains(monkeypatch: pytest.MonkeyPatch) -> None:
    """採点の道具を通す処理は環境変数を書き換えるので、テストの間は止める（専用のテストで見る）。"""
    monkeypatch.setattr(pg, "use_toolchains", lambda: None)


@pytest.fixture
def exercise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Python の作り物の練習問題 two-fer（解答の見本つき）。"""
    src = tmp_path / "bench" / "python" / "exercises" / "practice" / "two-fer"
    (src / ".meta").mkdir(parents=True)
    (src / ".docs").mkdir()
    files = {
        "solution": ["two_fer.py"],
        "test": ["two_fer_test.py"],
        "example": [".meta/example.py"],
    }
    (src / ".meta" / "config.json").write_text(json.dumps({"files": files}))
    (src / ".meta" / "example.py").write_text(_SOLVED)
    (src / ".docs" / "instructions.md").write_text("Say hi.\n")
    (src / "two_fer.py").write_text('def two_fer(name="you"):\n    pass\n')
    (src / "two_fer_test.py").write_text(_TEST)
    monkeypatch.setattr(pg, "POLYGLOT_DIR", tmp_path / "bench")
    monkeypatch.setattr(pg, "WORK_ROOT", tmp_path / "work")
    return src


def test_exercise_lists_only_the_files_the_model_edits(tmp_path: Path) -> None:
    src = tmp_path / "rust" / "exercises" / "practice" / "acronym"
    (src / ".meta").mkdir(parents=True)
    files = {
        "solution": ["src/lib.rs", "Cargo.toml"],
        "test": ["tests/acronym.rs"],
        "example": [".meta/example.rs"],
    }
    (src / ".meta" / "config.json").write_text(json.dumps({"files": files}))
    pg.POLYGLOT_DIR, saved = tmp_path, pg.POLYGLOT_DIR
    try:
        ex = pg.load_exercise("rust/acronym")
    finally:
        pg.POLYGLOT_DIR = saved
    assert ex.solution == ["src/lib.rs"] and ex.file_list == "lib.rs"
    assert ex.tests == ["tests/acronym.rs"]


def test_prompts_use_the_official_wording(exercise: Path) -> None:
    ex = pg.load_exercise("python/two-fer")
    first = pg.first_prompt("Say hi.\n", ex)
    assert first.startswith("Say hi.\n\n####\n\n")
    assert "modify the supplied files: two_fer.py\n" in first
    assert "Only use standard libraries, don't suggest installing any packages.\n" in first
    again = pg.retry_prompt("E   AssertionError", ex)
    assert again.startswith("E   AssertionError\n####\n\nSee the testing errors above.\n")
    assert again.endswith("Fix the code in two_fer.py to resolve the errors.\n")


def test_workdir_has_no_answer_and_is_its_own_git_repo(exercise: Path, tmp_path: Path) -> None:
    ex = pg.load_exercise("python/two-fer")
    workdir = tmp_path / "w"
    pg.prepare_workdir(ex, workdir)
    assert sorted(p.name for p in workdir.iterdir() if p.name not in {".git", ".satsuma"}) == [
        "two_fer.py",
        "two_fer_test.py",
    ]
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=workdir, capture_output=True, text=True
    )
    assert Path(top.stdout.strip()).resolve() == workdir.resolve()


def test_tests_are_restored_before_grading(exercise: Path, tmp_path: Path) -> None:
    ex = pg.load_exercise("python/two-fer")
    workdir = tmp_path / "w"
    pg.prepare_workdir(ex, workdir)
    (workdir / "two_fer.py").write_text(_SOLVED)
    (workdir / "two_fer_test.py").write_text("def test_x():\n    assert False\n")
    passed, output, _ = pg.run_tests(ex, workdir)
    assert passed, output
    assert (workdir / "two_fer_test.py").read_text() == _TEST
    assert str(workdir) not in output


def test_javascript_skipped_tests_are_turned_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.spec.js").write_text("test('one', () => {});\nxtest('two', () => {});\n")
    ex = pg.Exercise("javascript", "a", src, ["a.js"], ["a.spec.js"], [])
    monkeypatch.setattr(pg, "test_command", lambda lang: ["true"])
    workdir = tmp_path / "w"
    workdir.mkdir()
    assert pg.run_tests(ex, workdir)[0]
    assert (workdir / "a.spec.js").read_text() == "test('one', () => {});\ntest('two', () => {});\n"


def test_cleanup_drops_timings_and_the_workdir_path(tmp_path: Path) -> None:
    text = f"{tmp_path}/x.py failed in 0.12s\n"
    assert pg.cleanup_test_output(text, tmp_path) == f"{tmp_path.name}/x.py failed \n"


@pytest.mark.parametrize(("mode", "passed"), [("reference", True), ("empty", False)])
def test_dry_runs_grade_the_answer_and_the_untouched_exercise(
    exercise: Path, tmp_path: Path, mode: str, passed: bool
) -> None:
    task = {"id": "python/two-fer", "prompt": "Say hi.\n"}
    result, attempts = pg.solve(task, "m", tmp_path / "out", dry_run=mode)
    assert result.passed is passed and len(attempts) == 1


def test_second_try_continues_the_session_with_the_test_output(
    exercise: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_satsuma(
        prompt: str, workdir: Path, model: str, timeout_s: float, log: Path, resume: str = ""
    ) -> tuple[dict[str, object], str, float]:
        calls.append((prompt, resume))
        if len(calls) == 2:
            (workdir / "two_fer.py").write_text(_SOLVED)
        usage = {"input_tokens": 100, "output_tokens": 10}
        return {"session_id": "s1", "iterations": 3, "usage": usage}, "", 1.0

    monkeypatch.setattr(pg.wb, "run_satsuma", fake_satsuma)
    task = {"id": "python/two-fer", "prompt": "Say hi.\n"}
    result, attempts = pg.solve(task, "m", tmp_path / "out")
    assert [a.passed for a in attempts] == [False, True] and result.passed
    assert calls[0][1] == "" and calls[0][0].startswith("Say hi.\n")
    assert calls[1][1] == "s1"
    assert "AssertionError" in calls[1][0] and "See the testing errors above." in calls[1][0]
    assert result.iterations == 6 and result.usage.input_tokens == 200


def test_other_harnesses_get_their_own_results() -> None:
    out = run.default_out("aider-polyglot", "gpt-oss:120b", "smoke", "", "opencode")
    assert out.name == "opencode-gpt-oss_120b-smoke"
    assert run.default_out("aider-polyglot", "gpt-oss:120b", "smoke", "", "satsuma").name == (
        "gpt-oss_120b-smoke"
    )
