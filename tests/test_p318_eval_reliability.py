"""評価の数字を信用できる形で残す。"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli import app as cli_app
from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.evals import runner as runner_mod
from satsumaestro.evals.agents import AgentAnswer, SatsumaAgent
from satsumaestro.evals.baseline import Baseline, compare
from satsumaestro.evals.defects import detect_defects
from satsumaestro.evals.models import EvalPermissionMode, EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.retry import is_connection_failure
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import (
    EvalAgentError,
    ProviderError,
    ProviderTimeoutError,
    RetriesExhaustedError,
)
from tests.fixtures.mock_provider import ScriptedFailureProvider, SequencedMockProvider


class _Clock:
    """壁時計と単調時計を別々に進める。"""

    def __init__(self) -> None:
        self.wall = 1_000_000_000
        self.mono = 5_000

    def advance(self, *, wall: int, mono: int) -> None:
        self.wall += wall
        self.mono += mono


class _SleepyAgent:
    """1 課題の間に、壁時計だけ ``slept`` ミリ秒多く進める擬似エージェント。"""

    name = "sleepy"
    model = "m"

    def __init__(self, clock: _Clock, *, work: int, slept: int, fail: bool = False) -> None:
        self._clock, self._work, self._slept, self._fail = clock, work, slept, fail

    async def run(
        self,
        prompt: str,
        workspace: Path,
        max_iterations: int | None = None,
        *,
        permission_mode: EvalPermissionMode = "read-only",
    ) -> AgentAnswer:
        self._clock.advance(wall=self._work + self._slept, mono=self._work)
        if self._fail:
            raise EvalAgentError("boom")
        return AgentAnswer(final_text="main.py", iterations=2, tool_calls=1)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(runner_mod, "now_ms", lambda: c.wall)
    monkeypatch.setattr(runner_mod, "monotonic_ms", lambda: c.mono)
    return c


def _fixture(root: Path) -> Path:
    fixture = root / "fixtures" / "proj"
    fixture.mkdir(parents=True)
    (fixture / "main.py").write_text("VALUE = 42\n", encoding="utf-8")
    return fixture


def _task(fixture: Path, **kwargs: object) -> EvalTask:
    return EvalTask.model_validate(
        {
            "id": "where-value",
            "prompt": "VALUE はどこ？",
            "fixture": str(fixture),
            "checks": {"contains_all": ["main.py"]},
            **kwargs,
        }
    )


@pytest.mark.parametrize(
    ("slept", "fail", "want_duration", "want_slept"),
    [
        (595_000, False, 5_000, 595_000),
        (595_000, True, 5_000, 595_000),
        (1_500, False, 5_000, 0),
        (0, False, 5_000, 0),
    ],
)
async def test_task_time_leaves_out_sleep(
    tmp_path: Path, clock: _Clock, slept: int, fail: bool, want_duration: int, want_slept: int
) -> None:
    agent = _SleepyAgent(clock, work=5_000, slept=slept, fail=fail)
    result = (await EvalRunner(agent).run([_task(_fixture(tmp_path))])).results[0]

    assert result.passed is not fail
    assert (result.duration_ms, result.slept_ms) == (want_duration, want_slept)


def test_report_shows_speed_and_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    monkeypatch.setattr(cli_app, "_console", Console(file=out, width=240, color_system=None))
    report = EvalReport(
        agent="satsuma",
        model="m",
        started_at_ms=0,
        results=[
            TaskResult(task_id="a", passed=True, duration_ms=5_000, tokens_per_s=50.0),
            TaskResult(
                task_id="b", passed=True, duration_ms=7_000, tokens_per_s=22.0, slept_ms=595_000
            ),
            TaskResult(task_id="c", passed=True, duration_ms=6_000, tokens_per_s=48.0),
        ],
    )
    cli_app._render_eval_report(report)
    text = out.getvalue()

    assert "生成 48.0 tok/s（中央値）" in text
    assert "機械が眠っていた 595 秒（所要には入れていない）" in text
    assert "slept=595s" in text


def _result(**fields: object) -> TaskResult:
    base: dict[str, object] = {
        "task_id": "t",
        "passed": False,
        "iterations": 50,
        "iteration_limit": 50,
        "tool_calls": 49,
    }
    return TaskResult.model_validate({**base, **fields})


@pytest.mark.parametrize(
    ("fields", "kinds"),
    [
        ({}, ["iteration limit"]),
        ({"passed": True}, []),
        ({"iterations": 49}, []),
        ({"iteration_limit": 0}, []),
        ({"plan_steps_run": 3, "iterations": 60}, []),
        ({"error": "MaxIterationsError: x"}, ["run exception"]),
    ],
)
def test_used_up_iterations_is_a_defect(fields: dict[str, object], kinds: list[str]) -> None:
    assert [d.kind for d in detect_defects(_result(**fields), "satsuma")] == kinds


async def test_used_up_iterations_keeps_the_session(tmp_path: Path) -> None:
    read = [ToolUse(id="r", name="read_file", input={"path": "main.py"}), MessageStop()]
    give_up = [TextDelta(text="GitHub から取得してよいですか"), MessageStop()]
    provider = SequencedMockProvider([read, read, give_up])
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    defects_dir = tmp_path / "defects"
    task = _task(_fixture(tmp_path), max_iterations=3)

    result = (await EvalRunner(agent, defects_dir).run([task])).results[0]

    assert (result.passed, result.error, result.iterations) == (False, "", 3)
    assert result.iteration_limit == 3
    text = Path(result.defect_record).read_text(encoding="utf-8")
    assert (
        "**iteration limit**: failed after using up the limit of 3 iterations (2 tool calls)"
        in text
    )
    assert list((defects_dir / Path(result.defect_record).stem / "sessions").glob("*.jsonl"))


async def test_iteration_limit_follows_the_config(tmp_path: Path) -> None:
    provider = SequencedMockProvider([[TextDelta(text="main.py"), MessageStop()]])
    agent = SatsumaAgent(provider, RuntimeConfig(max_iterations=7), "mock-model")  # type: ignore[arg-type]
    result = (await EvalRunner(agent).run([_task(_fixture(tmp_path))])).results[0]
    assert (result.passed, result.iteration_limit) == (True, 7)


def _named(name: str) -> BaseException:
    """SDK や httpx を import せずに、その名前の例外を作る（判定はクラス名で行う）。"""
    return type(name, (Exception,), {})("x")


def _exhausted(cause: str | None, *, out_of_time: bool = False) -> RetriesExhaustedError:
    """再試行を使い切った例外。"""
    inner = ProviderError("OpenAI 互換 API 呼び出しに失敗: Connection error.", retryable=True)
    if cause is not None:
        inner.__cause__ = _named(cause)
    cls = ProviderTimeoutError if out_of_time else RetriesExhaustedError
    err = cls(9, 122.6, last_error=str(inner))
    err.__cause__ = inner
    return err


@pytest.mark.parametrize(
    ("exc", "lost"),
    [
        (_exhausted("APIConnectionError"), True),
        (_exhausted("ConnectError", out_of_time=True), True),
        (_exhausted("RemoteProtocolError"), True),
        (_exhausted("ReadTimeout"), False),
        (_exhausted("APITimeoutError"), False),
        (_exhausted(None), False),
        (ProviderError("HTTP 500", status_code=500), False),
    ],
)
def test_connection_failure_is_told_by_the_exception_chain(exc: BaseException, lost: bool) -> None:
    assert is_connection_failure(exc) is lost


def _first_round() -> list[object]:
    return [
        Usage(input_tokens=4000, output_tokens=30),
        ToolUse(id="r", name="read_file", input={"path": "main.py"}),
        MessageStop(),
    ]


@pytest.mark.parametrize(
    ("failure", "lost", "error_head"),
    [
        (_exhausted("APIConnectionError"), True, "RetriesExhaustedError"),
        (ProviderError("boom"), False, "ProviderError: boom"),
    ],
)
async def test_failed_turn_keeps_its_numbers(
    tmp_path: Path, failure: BaseException, lost: bool, error_head: str
) -> None:
    provider = ScriptedFailureProvider([_first_round(), failure])  # type: ignore[list-item]
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    defects_dir = tmp_path / "defects"

    result = (await EvalRunner(agent, defects_dir).run([_task(_fixture(tmp_path))])).results[0]

    assert result.error.startswith(error_head)
    assert (result.passed, result.connection_lost, result.failures) == (False, lost, [])
    assert (result.iterations, result.tool_calls) == (2, 1)
    assert result.usage.input_tokens == 4000
    assert result.empty_final is False
    assert "**run exception**" in Path(result.defect_record).read_text(encoding="utf-8")


async def test_max_iterations_keeps_its_numbers(tmp_path: Path) -> None:
    loop = [ToolUse(id="r", name="read_file", input={"path": "main.py"}), MessageStop()]
    agent = SatsumaAgent(SequencedMockProvider([loop]), RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    result = (await EvalRunner(agent).run([_task(_fixture(tmp_path), max_iterations=2)])).results[0]

    assert result.error.startswith("MaxIterationsError")
    assert (result.iterations, result.tool_calls, result.iteration_limit) == (2, 2, 2)


async def test_finished_turn_leaves_no_aborted_summary(tmp_path: Path) -> None:
    steps = [_first_round(), ProviderError("boom"), [TextDelta(text="ok"), MessageStop()]]
    runtime = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=PermissionResolver(default_policy(PermissionMode.READ_ONLY), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    (tmp_path / "main.py").write_text("x\n", encoding="utf-8")

    with pytest.raises(ProviderError):
        await runtime.run_turn("q")
    aborted = runtime.aborted_summary
    assert aborted is not None
    assert (aborted.stop_reason, aborted.iterations, aborted.tool_calls) == ("aborted", 2, 1)

    await runtime.run_turn("q")
    assert runtime.aborted_summary is None


def _report(*results: TaskResult) -> EvalReport:
    return EvalReport(agent="satsuma", model="m", started_at_ms=0, results=list(results))


def _lost(task_id: str) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        passed=False,
        error="RetriesExhaustedError: Connection error.",
        connection_lost=True,
    )


def test_connection_failures_stay_out_of_the_baseline() -> None:
    base = Baseline()
    added = base.add_report(_report(TaskResult(task_id="a", passed=True), _lost("b")))
    assert added == 1
    assert list(base.tasks) == ["a"]


def test_connection_failure_on_a_gate_task_is_a_rerun_not_a_regression() -> None:
    base = Baseline()
    for _ in range(5):
        base.add_report(_report(TaskResult(task_id="a", passed=True, iterations=5, tool_calls=4)))
    comp = compare(_report(_lost("a")), base)

    assert comp.gate_tasks == ["a"]
    assert (comp.regressed, comp.structural_errors) == ([], [])
    assert comp.connection_lost == ["a"]
    assert (comp.decision, comp.rerun_cases) == ("rerun", ["a"])


def test_report_counts_connection_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    monkeypatch.setattr(cli_app, "_console", Console(file=out, width=240, color_system=None))
    cli_app._render_eval_report(_report(TaskResult(task_id="a", passed=True), _lost("b")))
    text = out.getvalue()

    assert "合格 1/2 (50%)" in text
    assert "接続の失敗 1 件（不合格に含む）" in text
    assert "conn" in text
