"""評価中の異常を不具合記録として残す。"""

from __future__ import annotations

import datetime
import io
from pathlib import Path

import pytest

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.evals.agents import AgentAnswer, SatsumaAgent
from satsumaestro.evals.defects import (
    REPEAT_THRESHOLD,
    detect_defects,
    repeated_defects,
    save_defect_record,
)
from satsumaestro.evals.models import EvalPermissionMode, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.util.errors import EvalAgentError
from tests.fixtures.mock_provider import SequencedMockProvider


def _fixture(root: Path) -> Path:
    fixture = root / "fixtures" / "proj"
    fixture.mkdir(parents=True)
    (fixture / "main.py").write_text("VALUE = 42\n", encoding="utf-8")
    return fixture


def _task(fixture: Path, **kwargs: object) -> EvalTask:
    return EvalTask.model_validate(
        {
            "id": "where-value",
            "prompt": "VALUE はどこ？\n定義位置を答えて",
            "fixture": str(fixture),
            "checks": {"contains_all": ["main.py"]},
            **kwargs,
        }
    )


def _answer() -> list[list[object]]:
    return [[TextDelta(text="main.py で定義されています"), MessageStop()]]


@pytest.mark.parametrize(
    ("fields", "agent", "kinds"),
    [
        ({"passed": True}, "satsuma", []),
        ({"passed": False, "failures": ["missing: main.py"]}, "satsuma", []),
        ({"error": "MaxIterationsError: 50"}, "satsuma", ["run exception"]),
        ({"error": "ProviderError: x"}, "satsuma+plan", ["run exception"]),
        ({"passed": True, "plan_exits": 0}, "satsuma+plan", ["plan path not taken"]),
        ({"passed": True, "plan_exits": 1}, "satsuma+decompose+plan", []),
        ({"passed": True, "plan_exits": 0}, "satsuma", []),
        ({"passed": True, "timeout_recoveries": 1}, "satsuma", ["timeout recovery"]),
        ({"passed": True, "repetition_aborts": 1}, "satsuma", ["repetition abort"]),
        ({"passed": True, "empty_final": True}, "satsuma", ["empty final reply"]),
        ({"passed": True, "tool_call_repairs": 1}, "satsuma", ["broken tool_call"]),
        ({"passed": True, "empty_tool_args": 2}, "satsuma", ["broken tool_call"]),
        (
            {"passed": True, "language_expected": "ja", "language_detected": "en"},
            "satsuma",
            ["language mismatch left"],
        ),
        ({"passed": True, "language_expected": "ja", "language_detected": "und"}, "satsuma", []),
    ],
)
def test_detect_defects(fields: dict[str, object], agent: str, kinds: list[str]) -> None:
    result = TaskResult.model_validate({"task_id": "t", "passed": False, **fields})
    assert [d.kind for d in detect_defects(result, agent)] == kinds


async def test_plan_run_that_skips_exit_plan_mode_is_recorded(tmp_path: Path) -> None:
    provider = SequencedMockProvider(_answer())  # type: ignore[arg-type]
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model", plan=True)  # type: ignore[arg-type]
    defects_dir = tmp_path / "defects"
    report = await EvalRunner(agent, defects_dir).run([_task(_fixture(tmp_path))])

    result = report.results[0]
    record = Path(result.defect_record)
    assert result.passed is True
    assert record.parent == defects_dir and record.suffix == ".md"
    text = record.read_text(encoding="utf-8")
    assert "# Defect record from an eval: where-value (satsuma+plan) — plan path not taken" in text
    assert "filed: no" in text
    assert "Agent / model: satsuma+plan / mock-model" in text
    assert "  > VALUE はどこ？" in text and "  > 定義位置を答えて" in text
    assert "main.py で定義されています" in text
    sessions = list((defects_dir / record.stem / "sessions").glob("*.jsonl"))
    assert len(sessions) == 1
    assert f"`{record.stem}/sessions/{sessions[0].name}`" in text
    assert "exit_plan_mode" in sessions[0].read_text(encoding="utf-8")
    assert not list(tmp_path.glob("satsuma-eval-*"))


async def test_exception_run_keeps_the_session(tmp_path: Path) -> None:
    loop = [ToolUse(id="r1", name="read_file", input={"path": "main.py"}), MessageStop()]
    provider = SequencedMockProvider([loop])
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    defects_dir = tmp_path / "defects"
    task = _task(_fixture(tmp_path), max_iterations=2)
    result = (await EvalRunner(agent, defects_dir).run([task])).results[0]

    assert result.error.startswith("MaxIterationsError")
    text = Path(result.defect_record).read_text(encoding="utf-8")
    assert "**run exception**: MaxIterationsError" in text
    stem = Path(result.defect_record).stem
    assert list((defects_dir / stem / "sessions").glob("*.jsonl"))


async def test_clean_run_and_disabled_recording_write_nothing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    defects_dir = tmp_path / "defects"
    ok = SatsumaAgent(SequencedMockProvider(_answer()), RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    assert (await EvalRunner(ok, defects_dir).run([_task(fixture)])).results[0].defect_record == ""
    assert not defects_dir.exists()

    plan = SatsumaAgent(SequencedMockProvider(_answer()), RuntimeConfig(), "m", plan=True)  # type: ignore[arg-type]
    assert (await EvalRunner(plan).run([_task(fixture)])).results[0].defect_record == ""


class _FailingCliAgent:
    """外部 CLI エージェントの代役（ワークスペースにセッションを残さない）。"""

    name = "claude"
    model = ""

    async def run(
        self,
        prompt: str,
        workspace: Path,
        max_iterations: int | None = None,
        *,
        permission_mode: EvalPermissionMode = "read-only",
    ) -> AgentAnswer:
        raise EvalAgentError("claude CLI exited with 1")


async def test_agent_without_sessions_still_gets_a_record(tmp_path: Path) -> None:
    defects_dir = tmp_path / "defects"
    result = (
        await EvalRunner(_FailingCliAgent(), defects_dir).run([_task(_fixture(tmp_path))])
    ).results[0]
    text = Path(result.defect_record).read_text(encoding="utf-8")
    assert "**run exception**: EvalAgentError: claude CLI exited with 1" in text
    assert "none (this agent leaves no session records in the workspace)" in text


def test_plan_file_is_copied_and_names_do_not_collide(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    (workspace / ".satsuma" / "sessions" / "abc").mkdir(parents=True)
    (workspace / ".satsuma" / "sessions" / "abc" / "s1.jsonl").write_text("{}\n", encoding="utf-8")
    (workspace / ".satsuma" / "plans").mkdir(parents=True)
    (workspace / ".satsuma" / "plans" / "s1.md").write_text("# Plan\n", encoding="utf-8")
    task = _task(tmp_path, id="dash")
    result = TaskResult(task_id="dash", passed=False, error="X: y")
    defects = detect_defects(result, "name=my cli")
    now = datetime.datetime(2026, 9, 20, 2, 15, 30, tzinfo=datetime.UTC)

    kwargs = {"agent_name": "name=my cli", "model": "", "workspace": workspace, "now": now}
    first = save_defect_record(tmp_path / "d", task, result, defects, **kwargs)  # type: ignore[arg-type]
    second = save_defect_record(tmp_path / "d", task, result, defects, **kwargs)  # type: ignore[arg-type]

    assert first.name == "20260920-021530-dash-name_my_cli.md"
    assert second.name == "20260920-021530-dash-name_my_cli-2.md"
    text = first.read_text(encoding="utf-8")
    assert "`20260920-021530-dash-name_my_cli/plans/s1.md`" in text
    assert (tmp_path / "d" / first.stem / "plans" / "s1.md").read_text() == "# Plan\n"
    assert (tmp_path / "d" / first.stem / "sessions" / "s1.jsonl").exists()


def _records(defects_dir: Path, n: int, *, start: int = 0) -> list[Path]:
    """「実行例外」の記録を n 件書く（時刻を 1 秒ずつずらす）。"""
    task = _task(defects_dir, id="t")
    result = TaskResult(task_id="t", passed=False, error="X: y")
    defects = detect_defects(result, "satsuma")
    paths = []
    for i in range(start, start + n):
        now = datetime.datetime(2026, 9, 20, 3, 0, i, tzinfo=datetime.UTC)
        paths.append(
            save_defect_record(
                defects_dir,
                task,
                result,
                defects,
                agent_name="satsuma",
                model="m",
                workspace=defects_dir / "none",
                now=now,
            )
        )
    return paths


def test_repeated_defects_needs_the_threshold(tmp_path: Path) -> None:
    _records(tmp_path, 4)
    assert repeated_defects(tmp_path) == []
    _records(tmp_path, 1, start=10)
    (found,) = repeated_defects(tmp_path)
    assert (found.kind, found.count, REPEAT_THRESHOLD) == ("run exception", 5, 5)


def test_filed_records_and_records_without_a_header_are_not_counted(tmp_path: Path) -> None:
    paths = _records(tmp_path, 5)
    text = paths[0].read_text(encoding="utf-8")
    paths[0].write_text(text.replace("filed: no", "filed: #123", 1), encoding="utf-8")
    (tmp_path / "memo.md").write_text("# 手書きのメモ\nfiled: no\n", encoding="utf-8")
    assert repeated_defects(tmp_path) == []


def test_each_kind_of_a_multi_kind_record_is_counted(tmp_path: Path) -> None:
    result = TaskResult(task_id="t", passed=False, error="X: y", empty_final=True)
    defects = detect_defects(result, "satsuma")
    assert [d.kind for d in defects] == ["run exception", "empty final reply"]
    for i in range(5):
        now = datetime.datetime(2026, 9, 20, 3, 0, i, tzinfo=datetime.UTC)
        save_defect_record(
            tmp_path,
            _task(tmp_path, id="t"),
            result,
            defects,
            agent_name="satsuma",
            model="m",
            workspace=tmp_path / "none",
            now=now,
        )
    assert {r.kind for r in repeated_defects(tmp_path)} == {"run exception", "empty final reply"}


def test_cli_prints_the_notice_after_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rich.console import Console

    from satsumaestro.cli import app

    out = io.StringIO()
    monkeypatch.setattr(app, "_console", Console(file=out, width=200))
    app._render_repeated_defects(tmp_path / "missing")
    _records(tmp_path, 5)
    app._render_repeated_defects(tmp_path)
    printed = out.getvalue()
    assert "同じ不具合が 5 件以上たまっています" in printed
    assert "run exception: 5 件" in printed
