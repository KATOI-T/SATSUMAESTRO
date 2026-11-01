"""Web-Bench ドライバ（scripts/eval_web_bench.py）の接続先と異常終了の記録のテスト。"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from satsumaestro.config.loader import ConfigLoader
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.util.paths import find_workspace_root

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_web_bench.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_web_bench", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_web_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dotenv_wins_over_the_shell_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load()
    (tmp_path / ".env").write_text(
        "# comment\nOLLAMA_HOST=http://localhost:11434/v1\nOLLAMA_MODEL=\nOTHER=x\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "REPO", tmp_path)
    monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.10:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "from-shell")
    env = mod.ollama_env()
    assert env["OLLAMA_HOST"] == "http://localhost:11434/v1"
    assert env["OLLAMA_MODEL"] == "from-shell"
    assert "OTHER" not in env or env["OTHER"] != "x"


def test_unreachable_ollama_is_reported_before_any_attempt() -> None:
    mod = _load()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    problem = mod.check_ollama({"OLLAMA_HOST": f"http://127.0.0.1:{port}/v1"})
    assert problem is not None and f"127.0.0.1:{port}" in problem and "/v1" not in problem


def test_crashed_attempt_is_recovered_from_the_session(tmp_path: Path) -> None:
    mod = _load()
    since = time.time() - 1
    sessions = tmp_path / ".satsuma" / "sessions" / "abc"
    sessions.mkdir(parents=True)
    records = [
        {"type": "session_meta"},
        {"type": "message", "role": "user", "blocks": []},
        *[
            {
                "type": "message",
                "role": "assistant",
                "usage": {"input_tokens": 1000, "output_tokens": 50},
                "blocks": [],
            }
            for _ in range(50)
        ],
        {"type": "message", "role": "tool", "usage": None, "blocks": []},
    ]
    (sessions / "s.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    stderr = "error: reached max_iterations=50 without end_turn"
    got = mod.recover_from_session(tmp_path, since, stderr)
    assert got["iterations"] == 50 and got["stop_reason"] == "max_iterations"
    assert got["usage"] == {"input_tokens": 50_000, "output_tokens": 2_500}
    assert mod.recover_from_session(tmp_path, time.time() + 60, stderr) == {}
    assert mod.recover_from_session(tmp_path, since, "boom")["stop_reason"] == "error"


def test_error_context_follows_the_official_evaluator(tmp_path: Path) -> None:
    """stdout だけを使い、ANSI・進捗行を落とし、先頭のアサーションを残す。"""
    mod = _load()
    run_dir = tmp_path / "run"
    workdir = run_dir / "work" / "x" / "src"
    stdout = (
        "\x1b[1A\x1b[2K[3/22] [chromium] › test/task-6.spec.js:26:1 › design to preview\n"
        "  1) [chromium] › test/task-6.spec.js:26:1 › SingleSelectionQuestion design to preview\n"
        "    Error: expect(locator).toHaveCount(expected) failed\n"
        "    Locator: locator('.option-text')\n"
        "    Expected: 3\n"
        "    Received: 0\n"
        f"        at {run_dir}/test/task-6.spec.js:44:5\n"
        "\n"
        "<html>\n"
        "  1 failed\n"
    )
    stderr = (
        "(node:1) [DEP0205] DeprecationWarning: module.register() is deprecated.\n"
        "(Use `node --trace-deprecation ...` to show where the warning was created)\n"
    ) * 15
    text = mod.error_context(stdout, stderr, run_dir, workdir)
    assert "DeprecationWarning" not in text
    assert "[3/22]" not in text and "\x1b" not in text and "<html>" not in text
    assert "locator('.option-text')" in text and "Received: 0" in text
    assert str(run_dir) not in text and "/test/task-6.spec.js:44:5" in text
    assert "DeprecationWarning" in mod.error_context("", stderr, run_dir, workdir)


def test_workdir_is_outside_the_test_env_and_broken_env_is_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """作業ディレクトリでの ``npm install`` がテスト環境を壊さない置き場所・壊れたら戻す。"""
    mod = _load()
    bench, run_root = tmp_path / "bench", tmp_path / "run"
    lib = bench / "libraries" / "test-util"
    (lib / "src").mkdir(parents=True)
    (lib / "src" / "index.js").write_text("x", encoding="utf-8")
    (bench / "projects" / "survey" / "src-init").mkdir(parents=True)
    nm = run_root / "survey" / "node_modules"
    (nm / "@playwright" / "test").mkdir(parents=True)
    (nm / "@web-bench").mkdir()
    (nm / "@web-bench" / "test-util").symlink_to(lib)
    monkeypatch.setattr(mod, "REPO", tmp_path)
    monkeypatch.setattr(mod, "BENCH", bench)
    monkeypatch.setattr(mod, "RUN_ROOT", run_root)
    monkeypatch.setattr(mod, "WORK_ROOT", tmp_path / "work")

    workdir = mod.prepare_workdir("survey", "r1")
    assert not workdir.is_relative_to(run_root / "survey")
    assert "修復" in mod.ensure_test_env("survey")
    util = nm / "@web-bench" / "test-util"
    assert util.is_dir() and not util.is_symlink() and (util / "src" / "index.js").exists()
    assert mod.ensure_test_env("survey") == ""


def _git(workdir: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=workdir, check=True, capture_output=True, text=True)
    return proc.stdout


def _enclosing_repo(tmp_path: Path, mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Path:
    """親のリポジトリ（CLAUDE.md と設定入り）を作り、ドライバの置き場所をその中に向ける。"""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "CLAUDE.md").write_text("# parent instructions\n", encoding="utf-8")
    (repo / ".satsuma").mkdir()
    (repo / ".satsuma" / "settings.local.json").write_text(
        json.dumps(
            {
                "provider": {"default_model": "from-repo"},
                "max_iterations": 7,
                "mcp_servers": {"demo": {"transport": "stdio", "command": ".venv/bin/python"}},
            }
        ),
        encoding="utf-8",
    )
    src_init = repo / "bench" / "projects" / "survey" / "src-init"
    src_init.mkdir(parents=True)
    (src_init / "design.js").write_text("// init\n", encoding="utf-8")
    monkeypatch.setattr(mod, "REPO", repo)
    monkeypatch.setattr(mod, "BENCH", repo / "bench")
    monkeypatch.setattr(mod, "WORK_ROOT", repo / ".satsuma" / "benchmarks" / "work")
    return repo


def test_workdir_does_not_inherit_the_enclosing_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """作業ディレクトリが自分のリポジトリになり、親の CLAUDE.md は読まれない。"""
    mod = _load()
    repo = _enclosing_repo(tmp_path, mod, monkeypatch)
    workdir = mod.prepare_workdir("survey", "r1")
    assert workdir.is_relative_to(repo)

    assert find_workspace_root(workdir) == workdir
    assert SystemPromptBuilder()._discover_instructions(workdir) == ""
    cfg = ConfigLoader(workdir, config_home=tmp_path / "home", trusted=True).load()
    assert cfg.provider.default_model == "from-repo"
    assert cfg.max_iterations == 7
    assert cfg.mcp_servers == {}
    copied = json.loads((workdir / ".satsuma" / "settings.local.json").read_text(encoding="utf-8"))
    assert "mcp_servers" not in copied and copied["provider"] == {"default_model": "from-repo"}

    assert _git(workdir, "status", "--short") == ""
    assert _git(workdir, "log", "--format=%s").split() == ["src-init"]
    assert ".satsuma" not in _git(workdir, "ls-files")
    assert "design.js" in _git(workdir, "ls-files")


def test_every_attempt_is_committed_so_the_next_starts_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load()
    _enclosing_repo(tmp_path, mod, monkeypatch)
    workdir = mod.prepare_workdir("survey", "r1")
    (workdir / "common").mkdir()
    (workdir / "common" / "Question.js").write_text("export class Question {}\n", encoding="utf-8")
    (workdir / "design.js").write_text("// task-1\n", encoding="utf-8")
    (workdir / ".satsuma" / "sessions" / "s").mkdir(parents=True)
    (workdir / ".satsuma" / "sessions" / "s" / "x.jsonl").write_text("{}\n", encoding="utf-8")
    (workdir / "node_modules" / "pkg").mkdir(parents=True)
    (workdir / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")
    (workdir / ".aider.chat.history.md").write_text("x", encoding="utf-8")

    mod.commit_all(workdir, "task-1 attempt 1: pass")

    assert _git(workdir, "status", "--short") == ""
    assert _git(workdir, "log", "--format=%s").split("\n")[0] == "task-1 attempt 1: pass"
    tracked = _git(workdir, "ls-files").split()
    assert "common/Question.js" in tracked and "design.js" in tracked
    assert not any(p.startswith((".satsuma/", "node_modules/", ".aider")) for p in tracked)
    mod.commit_all(workdir, "task-2 attempt 1: FAIL")
    assert _git(workdir, "log", "--format=%s").split("\n")[0] == "task-2 attempt 1: FAIL"


_LAUNCH_TIMEOUT = """Running 9 tests using 9 workers
  1) [chromium] › test/task-1.spec.js:23:1 › common/Question.js ──────
    TimeoutError: browserType.launch: Timeout 180000ms exceeded.
"""
_ASSERTION = """Running 9 tests using 9 workers
  1) [chromium] › test/task-2.spec.js:10:1 › survey title ──────
    Error: expect(received).toBe(expected)
"""


@pytest.mark.parametrize(
    ("output", "mark"),
    [
        (_LAUNCH_TIMEOUT, "browserType.launch"),
        ("Error: listen EADDRINUSE: address already in use :::3211", "EADDRINUSE"),
        (_ASSERTION, ""),
        ("", ""),
    ],
)
def test_infra_failure_is_told_by_fixed_marks(output: str, mark: str) -> None:
    assert _load().infra_failure_mark(output) == mark


def _runs(mod: ModuleType, *outcomes: object) -> tuple[list[int], object]:
    calls: list[int] = []
    queue = list(outcomes)

    def run() -> object:
        calls.append(1)
        return queue.pop(0)

    return calls, run


def test_env_failure_retests_without_counting_an_attempt() -> None:
    mod = _load()
    env = mod.GradeOutcome(False, "TimeoutError: browserType.launch", 4.7, "browserType.launch")

    calls, run = _runs(mod, env, mod.GradeOutcome(True, "", 12.0))
    got = mod.grade_attempt(run)
    assert (got.passed, got.infra, len(calls)) == (True, "", 2)
    assert got.seconds == pytest.approx(16.7)

    calls, run = _runs(mod, env, env)
    got = mod.grade_attempt(run)
    assert (got.passed, got.infra, len(calls)) == (False, "browserType.launch", 2)

    calls, run = _runs(mod, mod.GradeOutcome(False, "Error: expect", 9.0))
    got = mod.grade_attempt(run)
    assert (got.passed, got.error, len(calls)) == (False, "Error: expect", 1)


def test_idle_attempt_is_told_by_the_work_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load()
    _enclosing_repo(tmp_path, mod, monkeypatch)
    workdir = mod.prepare_workdir("survey", "r1")
    assert mod.has_changes(workdir) is False
    (workdir / ".satsuma" / "sessions" / "s").mkdir(parents=True)
    (workdir / ".satsuma" / "sessions" / "s" / "x.jsonl").write_text("{}\n", encoding="utf-8")
    assert mod.has_changes(workdir) is False
    (workdir / "design.js").write_text("// changed\n", encoding="utf-8")
    assert mod.has_changes(workdir) is True
    mod.commit_all(workdir, "task-1 attempt 1: pass")
    assert mod.has_changes(workdir) is False


def test_latest_session_is_the_attempts_own(tmp_path: Path) -> None:
    import os

    mod = _load()
    sessions = tmp_path / ".satsuma" / "sessions" / "h"
    sessions.mkdir(parents=True)
    old, new = sessions / "old.jsonl", sessions / "new.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    new.write_text("{}\n", encoding="utf-8")
    now = time.time()
    os.utime(old, (now - 600, now - 600))
    assert mod.latest_session(tmp_path, now - 60) == new
    assert mod.latest_session(tmp_path, now + 60) is None


def _attempt(mod: ModuleType, n: int, **fields: object) -> object:
    base: dict[str, object] = {
        "task_id": "task-2",
        "attempt": n,
        "passed": False,
        "agent_s": 10.0,
        "test_s": 5.0,
        "iterations": 4,
        "input_tokens": 100,
        "output_tokens": 10,
        "stop_reason": "end_turn",
        "agent_error": "",
        "test_error_head": "Error: expect(received).toBe(expected)",
    }
    return mod.Attempt(**{**base, **fields})


def test_task_error_comes_from_any_attempt() -> None:
    mod = _load()
    capped = "exit 1: error: reached max_iterations=50 without end_turn"
    record = mod.TaskRecord("task-2", "easy", False)
    record.attempts = [_attempt(mod, 1), _attempt(mod, 2, agent_error=capped)]

    result = mod.task_result(record)
    assert result.error == f"attempt 2: {capped}"
    assert (result.iterations, result.duration_ms) == (8, 30_000)

    record.passed = True
    assert mod.task_result(record).error == ""


def test_undetermined_task_is_not_a_failure_of_the_model() -> None:
    mod = _load()
    record = mod.TaskRecord("task-2", "easy", False, undetermined=True)
    record.attempts = [_attempt(mod, 1, test_infra_error="browserType.launch")]
    other = mod.TaskRecord("task-3", "easy", True)
    other.attempts = [_attempt(mod, 1, passed=True, no_changes=True), _attempt(mod, 2)]
    idle = mod.TaskRecord("task-4", "easy", False)
    idle.attempts = [_attempt(mod, 1, no_changes=True)]

    assert mod.task_result(record).failures == ["判定不能: テスト環境の失敗（browserType.launch）"]
    assert mod.run_notes([record, other, idle]) == [
        "判定不能 1 課題（テスト環境の失敗。公式指標では通っていない扱い）: task-2",
        "変更の無い試行 2 回（うち不合格 1 回）",
    ]


def test_a_timeout_stops_the_grandchildren_too(tmp_path: Path) -> None:
    """``uv run`` の下のエージェントや、その bash の計算が親を失って居残らない。"""
    mod = _load()
    late = tmp_path / "late.txt"
    script = f"(sleep 2; echo late > {late}) & sleep 30"
    with pytest.raises(subprocess.TimeoutExpired):
        mod.run_group(["sh", "-c", script], 0.5, stdout=subprocess.PIPE, text=True)
    time.sleep(2.5)
    assert not late.exists()


def test_leftovers_are_stopped_after_a_normal_end(tmp_path: Path) -> None:
    """バックグラウンドで起動したものを残したまま終わっても、次の試行まで居残らない。"""
    mod = _load()
    late = tmp_path / "late.txt"
    script = f"(sleep 2; echo late > {late}) > /dev/null 2>&1 & echo started"
    proc = mod.run_group(["sh", "-c", script], 10, stdout=subprocess.PIPE, text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "started"
    time.sleep(2.5)
    assert not late.exists()


def test_a_prompt_starting_with_a_dash_is_passed_as_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load()
    assert mod.cli_text("--- FAIL: TestX") == "\n--- FAIL: TestX"
    assert mod.cli_text("Fix it.") == "Fix it."
    seen: list[list[str]] = []

    class Stop(Exception):
        pass

    def capture(cmd: list[str], *_args: object, **_kw: object) -> None:
        seen.append(cmd)
        raise Stop

    monkeypatch.setattr(mod, "_run_logged", capture)
    with pytest.raises(Stop):
        mod.run_satsuma("--- FAIL: TestX", tmp_path, "m", 30, tmp_path / "a.log")
    assert "\n--- FAIL: TestX" in seen[0] and "--- FAIL: TestX" not in seen[0]
