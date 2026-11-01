"""評価ハーネスのテスト。"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from satsumaestro.config.models import GuidanceConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.evals.agents import (
    CliAgent,
    SatsumaAgent,
    build_cli_agent,
    is_satsuma_agent_spec,
    parse_satsuma_agent_spec,
)
from satsumaestro.evals.checks import grade
from satsumaestro.evals.loader import load_tasks
from satsumaestro.evals.models import EvalChecks, EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.providers.base import ApiRequest
from satsumaestro.util.errors import ConfigError, EvalAgentError, ProviderError
from tests.fixtures.mock_provider import MockProvider, SequencedMockProvider

_REPO_EVALS = Path(__file__).resolve().parent.parent / "evals"


def _write_fixture(root: Path) -> Path:
    fixture = root / "fixtures" / "proj"
    fixture.mkdir(parents=True)
    (fixture / "main.py").write_text("VALUE = 42\n", encoding="utf-8")
    return fixture


def _write_task(
    root: Path,
    name: str,
    *,
    checks: str = 'contains_all = ["main.py"]',
    extra: str = "",
) -> Path:
    tasks = root / "tasks"
    tasks.mkdir(exist_ok=True)
    path = tasks / f"{name}.toml"
    path.write_text(
        f'id = "{name}"\n'
        f'prompt = "VALUE はどこで定義されていますか。"\n'
        f'fixture = "../fixtures/proj"\n'
        f"{extra}\n"
        f"[checks]\n{checks}\n",
        encoding="utf-8",
    )
    return path


def test_load_tasks_from_dir(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_task(tmp_path, "b-second")
    _write_task(tmp_path, "a-first")
    tasks = load_tasks(tmp_path / "tasks")
    assert [t.id for t in tasks] == ["a-first", "b-second"]
    assert all(Path(t.fixture).is_absolute() for t in tasks)
    assert all(Path(t.fixture).name == "proj" for t in tasks)


def test_load_single_file(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    path = _write_task(tmp_path, "solo", extra="max_iterations = 3")
    tasks = load_tasks(path)
    assert len(tasks) == 1
    assert tasks[0].max_iterations == 3


def test_load_rejects_bad_task(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    bad = tmp_path / "tasks"
    bad.mkdir()
    (bad / "no-fixture.toml").write_text(
        'id = "x"\nprompt = "p"\nfixture = "../fixtures/nope"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="fixture"):
        load_tasks(bad / "no-fixture.toml")
    (bad / "no-fixture.toml").unlink()

    _write_task(tmp_path, "dup")
    same = _write_task(tmp_path, "dup2")
    same.write_text(
        same.read_text(encoding="utf-8").replace('id = "dup2"', 'id = "dup"'), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="重複"):
        load_tasks(tmp_path / "tasks")
    same.unlink()

    _write_task(tmp_path, "bad-re", checks='regex = ["([unclosed"]')
    with pytest.raises(ConfigError, match="regex"):
        load_tasks(tmp_path / "tasks" / "bad-re.toml")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigError, match="toml"):
        load_tasks(empty)


def test_grade_matrix() -> None:
    checks = EvalChecks(
        contains_all=["pricing.py", "apply_discount"],
        contains_any=["0.1", "10%"],
        regex=[r"def\s+apply"],
        forbidden=["わかりません"],
    )
    ok = "pricing.py の def apply_discount。税率は 0.1 です。"
    assert grade(ok, checks) == []
    assert grade("PRICING.PY def APPLY_DISCOUNT rate 10%", checks) == []

    failures = grade("cart.py を見ました。わかりません。", checks)
    assert "missing: pricing.py" in failures
    assert "missing: apply_discount" in failures
    assert any(f.startswith("none of:") for f in failures)
    assert any(f.startswith("no match:") for f in failures)
    assert "forbidden: わかりません" in failures

    assert grade("anything", EvalChecks()) == []


def test_grade_checks_generated_files(tmp_path: Path) -> None:
    """「大きなファイルを書き切れたか」は本文からは判定できないので成果物を見る。"""
    (tmp_path / "dashboards").mkdir()
    (tmp_path / "dashboards" / "index.html").write_text("x\n" * 300, encoding="utf-8")
    checks = EvalChecks(file_min_lines={"dashboards/index.html": 200})
    assert grade("作りました", checks, tmp_path) == []

    (tmp_path / "dashboards" / "index.html").write_text("x\n" * 10, encoding="utf-8")
    failures = grade("作りました", checks, tmp_path)
    assert failures == ["too short: dashboards/index.html has 10 lines (want >= 200)"]

    assert grade("作りました", EvalChecks(file_min_lines={"nope.md": 1}), tmp_path) == [
        "missing file: nope.md"
    ]


def test_grade_without_workspace_skips_file_checks(tmp_path: Path) -> None:
    assert grade("ok", EvalChecks(file_min_lines={"nope.md": 1})) == []


def test_task_permission_mode_default_and_bounds(tmp_path: Path) -> None:
    base = {"id": "t", "prompt": "p", "fixture": str(tmp_path)}
    assert EvalTask.model_validate(base).permission_mode == "read-only"
    task = EvalTask.model_validate({**base, "permission_mode": "workspace-write"})
    assert task.permission_mode == "workspace-write"
    with pytest.raises(ValueError):
        EvalTask.model_validate({**base, "permission_mode": "danger-full-access"})


def test_generated_file_checks_require_write_permission(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    checks = 'file_min_lines = { "out.md" = 3 }'
    _write_task(tmp_path, "gen", checks=checks)
    with pytest.raises(ConfigError, match="workspace-write"):
        load_tasks(tmp_path / "tasks" / "gen.toml")
    extra = 'permission_mode = "workspace-write"'
    path = _write_task(tmp_path, "gen-ok", checks=checks, extra=extra)
    assert load_tasks(path)[0].permission_mode == "workspace-write"


def _task(fixture: Path, task_id: str = "t1", **kwargs: object) -> EvalTask:
    return EvalTask.model_validate(
        {
            "id": task_id,
            "prompt": "VALUE はどこ？",
            "fixture": str(fixture),
            "checks": {"contains_all": ["main.py"]},
            **kwargs,
        }
    )


async def test_runner_pass_and_fail(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    provider = SequencedMockProvider(
        [
            [TextDelta(text="main.py で定義されています"), Usage(input_tokens=10), MessageStop()],
            [TextDelta(text="見つかりません"), MessageStop()],
        ]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    runner = EvalRunner(agent)
    report = await runner.run([_task(fixture, "pass-task"), _task(fixture, "fail-task")])

    assert report.agent == "satsuma"
    assert report.model == "mock-model"
    assert report.total == 2 and report.passed == 1
    assert report.pass_rate == 0.5
    ok, ng = report.results
    assert ok.passed is True and ok.failures == []
    assert ok.iterations == 1 and ok.usage.input_tokens == 10
    assert ok.empty_final is False
    assert ng.passed is False and ng.failures == ["missing: main.py"]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("satsuma", ("", False, False)),
        ("satsuma+decompose", ("", True, False)),
        ("satsuma+plan", ("", False, True)),
        ("satsuma+plan+decompose", ("", True, True)),
        ("satsuma:gpt-oss:120b", ("gpt-oss:120b", False, False)),
        ("satsuma:gpt-oss:120b+decompose", ("gpt-oss:120b", True, False)),
        ("satsuma:gpt-oss:120b+plan", ("gpt-oss:120b", False, True)),
        ("satsuma:gpt-oss:120b+decompose+plan", ("gpt-oss:120b", True, True)),
        ("satsuma+plan:gpt-oss:120b", ("gpt-oss:120b", False, True)),
    ],
)
def test_parse_satsuma_agent_spec(spec: str, expected: tuple[str, bool, bool]) -> None:
    assert is_satsuma_agent_spec(spec)
    assert parse_satsuma_agent_spec(spec) == expected


@pytest.mark.parametrize(
    "spec", ["claude", "gemini:foo", "satsumaa", "satsuma2+plan", "x=cmd {prompt}"]
)
def test_is_satsuma_agent_spec_rejects_others(spec: str) -> None:
    assert not is_satsuma_agent_spec(spec)


async def test_satsuma_agent_plan_auto_approves_and_answers(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    requests: list[str] = []
    write_call = ToolUse(id="w1", name="write_file", input={"path": "", "content": "# plan"})

    class SpyProvider(SequencedMockProvider):
        async def stream(self, request: ApiRequest) -> AsyncIterator[AssistantEvent]:  # type: ignore[override]
            dump = request.system_text() + "\n" + "\n".join(str(m) for m in request.messages)
            requests.append(dump)
            m = re.search(r"\.satsuma/plans/[0-9a-f]+\.md", dump)
            if m and not write_call.input["path"]:
                write_call.input["path"] = m.group(0)
            async for ev in super().stream(request):
                yield ev

    provider = SpyProvider(
        [
            [write_call, MessageStop()],
            [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
            [TextDelta(text="main.py で定義されています"), MessageStop()],
        ]
    )
    agent = SatsumaAgent(
        provider,  # type: ignore[arg-type]
        RuntimeConfig(guidance=GuidanceConfig(step_budget=False)),
        "mock-model",
        plan=True,
    )
    assert agent.name == "satsuma+plan"
    report = await EvalRunner(agent).run([_task(fixture, "plan-task")])
    assert report.agent == "satsuma+plan"
    result = report.results[0]
    assert result.passed is True and result.iterations == 3
    assert result.plan_exits == 1
    assert result.language_expected == "ja"
    assert result.language_detected == "ja" and result.language_repaired is False
    assert write_call.input["path"].startswith(".satsuma/plans/")
    assert ".satsuma/plans/" in requests[0]
    assert "Plan mode protocol for this task" in requests[0]
    assert "wrote" in requests[1] and "plan mode is active:" not in requests[1]
    assert "Exited plan mode" in requests[2]
    assert "has not approved" not in requests[2]


async def test_plan_steps_run_reaches_the_task_result(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    plan = "# P\n\n## Steps\n1. read the plan file\n2. answer the question\n"
    write_call = ToolUse(id="w1", name="write_file", input={"path": "", "content": plan})

    class SpyProvider(SequencedMockProvider):
        async def stream(self, request: ApiRequest) -> AsyncIterator[AssistantEvent]:  # type: ignore[override]
            dump = request.system_text() + "\n" + "\n".join(str(m) for m in request.messages)
            m = re.search(r"\.satsuma/plans/[0-9a-f]+\.md", dump)
            if m and not write_call.input["path"]:
                write_call.input["path"] = m.group(0)
            read_call.input["path"] = write_call.input["path"]
            async for ev in super().stream(request):
                yield ev

    read_call = ToolUse(id="r1", name="read_file", input={"path": ""})
    provider = SpyProvider(
        [
            [write_call, MessageStop()],
            [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
            [read_call, MessageStop()],
            [TextDelta(text="段 1 完了"), MessageStop()],
            [read_call, MessageStop()],
            [TextDelta(text="main.py で定義されています"), MessageStop()],
        ]
    )
    agent = SatsumaAgent(
        provider,  # type: ignore[arg-type]
        RuntimeConfig(guidance=GuidanceConfig(step_budget=False)),
        "mock-model",
        plan=True,
    )
    result = (await EvalRunner(agent).run([_task(fixture, "stepped")])).results[0]
    assert result.plan_steps_run == 2
    assert result.passed is True


async def test_satsuma_agent_without_plan_keeps_deny_prompter(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    provider = SequencedMockProvider(
        [
            [ToolUse(id="x1", name="exit_plan_mode", input={}), MessageStop()],
            [TextDelta(text="main.py で定義されています"), MessageStop()],
        ]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    assert agent.name == "satsuma"
    report = await EvalRunner(agent).run([_task(fixture, "no-plan")])
    assert report.results[0].passed is True
    assert report.results[0].plan_exits == 0
    assert "Plan mode protocol" not in str(provider.requests[0].messages[0])


async def test_runner_grants_write_permission_per_task(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    write = ToolUse(
        id="w1", name="write_file", input={"path": "docs/out.md", "content": "a\nb\nc\n"}
    )
    provider = SequencedMockProvider(
        [[write, MessageStop()], [TextDelta(text="作りました"), MessageStop()]]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    task = _task(
        fixture,
        "gen",
        permission_mode="workspace-write",
        checks={"file_min_lines": {"docs/out.md": 3}},
    )
    report = await EvalRunner(agent).run([task])
    assert report.results[0].failures == [] and report.results[0].passed is True

    provider = SequencedMockProvider(
        [[write, MessageStop()], [TextDelta(text="作れませんでした"), MessageStop()]]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    await agent.run("q", tmp_path / "ws-ro")
    assert "permission required" in str(provider.requests[1].messages[-1])
    assert not (tmp_path / "ws-ro" / "docs" / "out.md").exists()


async def test_runner_scores_fallback_on_empty_final(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    provider = SequencedMockProvider(
        [
            [
                TextDelta(text="main.py で定義されています"),
                ToolUse(id="c1", name="read_file", input={"path": "main.py"}),
                MessageStop(),
            ],
            [MessageStop()],
        ]
    )
    cfg = RuntimeConfig(guidance=GuidanceConfig(empty_final_retry=False))
    agent = SatsumaAgent(provider, cfg, "mock-model")  # type: ignore[arg-type]
    report = await EvalRunner(agent).run([_task(fixture, "empty-final")])
    result = report.results[0]
    assert result.passed is True
    assert result.empty_final is True
    assert "main.py" in result.final_text


async def test_runner_flags_all_empty_as_failure(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    provider = SequencedMockProvider([[MessageStop()]])
    cfg = RuntimeConfig(guidance=GuidanceConfig(empty_final_retry=False))
    agent = SatsumaAgent(provider, cfg, "mock-model")  # type: ignore[arg-type]
    report = await EvalRunner(agent).run([_task(fixture, "all-empty")])
    result = report.results[0]
    assert result.passed is False
    assert result.empty_final is True
    assert result.final_text.startswith("(empty final answer")


async def test_runner_isolates_fixture(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    before = {p.name for p in fixture.iterdir()}
    workspaces: list[str] = []

    class SpyProvider(MockProvider):
        async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
            workspaces.append(req.system_text())
            async for ev in super().stream(req):
                yield ev

    provider = SpyProvider([TextDelta(text="main.py"), MessageStop()])
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    runner = EvalRunner(agent)
    report = await runner.run([_task(fixture)])
    assert report.passed == 1
    assert {p.name for p in fixture.iterdir()} == before
    assert str(fixture) not in workspaces[0]
    assert "satsuma-eval-t1-" in workspaces[0]


async def test_runner_read_only(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="w1", name="write_file", input={"path": "hack.py", "content": "x"}),
                MessageStop(),
            ],
            [TextDelta(text="書けませんでした。main.py です。"), MessageStop()],
        ]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    runner = EvalRunner(agent)
    report = await runner.run([_task(fixture)])
    result = report.results[0]
    assert result.error == ""
    assert result.passed is True
    assert result.tool_calls == 1
    assert not (fixture / "hack.py").exists()


async def test_runner_continues_after_error(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    class BoomProvider(MockProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.calls = 0

        async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("connection refused")
            yield TextDelta(text="main.py")
            yield MessageStop()

    provider = BoomProvider()
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    runner = EvalRunner(agent)
    report = await runner.run([_task(fixture, "boom"), _task(fixture, "after")])
    boom, after = report.results
    assert boom.passed is False
    assert "ProviderError" in boom.error
    assert after.passed is True


async def test_runner_respects_task_max_iterations(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    (fixture / "f.txt").write_text("x", encoding="utf-8")
    provider = MockProvider(
        [ToolUse(id="t", name="read_file", input={"path": "f.txt"}), MessageStop()]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    runner = EvalRunner(agent)
    report = await runner.run([_task(fixture, "loop", max_iterations=2)])
    assert report.results[0].passed is False
    assert "MaxIterationsError" in report.results[0].error


def test_report_roundtrip(tmp_path: Path) -> None:
    report = EvalReport(
        model="m",
        started_at_ms=123,
        results=[
            {"task_id": "a", "passed": True},  # type: ignore[list-item]
            {"task_id": "b", "passed": False, "failures": ["missing: x"]},  # type: ignore[list-item]
        ],
    )
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    loaded = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.total == 2 and loaded.passed == 1 and loaded.pass_rate == 0.5


def test_bundled_suite_loads() -> None:
    tasks = load_tasks(_REPO_EVALS / "code_analysis")
    ids = {t.id for t in tasks}
    assert ids == {
        "find-definition",
        "trace-call",
        "locate-bug",
        "read-config",
        "list-methods",
        "find-caller",
        "invalid-rate",
        "compute-total",
    }
    assert all(Path(t.fixture).is_dir() for t in tasks)


def test_bundled_file_generation_suite_can_write() -> None:
    tasks = load_tasks(_REPO_EVALS / "file_generation")
    assert {t.id for t in tasks} == {"dashboard-html", "spec-markdown"}
    assert all(t.permission_mode == "workspace-write" for t in tasks)
    assert all(t.checks.file_min_lines for t in tasks)


def test_bundled_bash_quoting_suite_matches_its_fixture() -> None:
    tasks = {t.id: t for t in load_tasks(_REPO_EVALS / "bash_quoting")}
    assert tasks["count-div-close"].permission_mode == "workspace-write"
    assert tasks["count-item-close"].permission_mode == "read-only"
    fixture = _REPO_EVALS / "fixtures" / "html_site"
    for task_id, name, tag in (
        ("count-div-close", "index.html", "</div>"),
        ("count-item-close", "feed.xml", "</item>"),
    ):
        lines = (fixture / name).read_text(encoding="utf-8").splitlines()
        count = sum(tag in line for line in lines)
        assert all(line.count(tag) <= 1 for line in lines)
        assert tasks[task_id].checks.regex == [rf"\b{count}\b"]


def test_bundled_hard_suite_loads() -> None:
    tasks = load_tasks(_REPO_EVALS / "code_analysis_hard")
    ids = {t.id for t in tasks}
    assert ids == {
        "shared-state-bug",
        "dependency-func",
        "call-sites",
        "tags-writer",
        "empty-pop",
        "tie-break",
        "duplicate-register",
        "resolve-order-result",
        "done-literal",
        "status-writer",
    }
    fixture_dir = Path(next(t.fixture for t in tasks if t.id == "done-literal"))
    with_done = {
        p.name
        for p in fixture_dir.glob("*.py")
        if any(
            isinstance(node, ast.Constant) and node.value == "done"
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8")))
        )
    }
    assert with_done == {"runner.py"}
    fixture = Path(next(iter(tasks)).fixture)
    assert fixture.name == "taskflow"
    legacy = (fixture / "legacy.py").read_text(encoding="utf-8")
    for line in legacy.splitlines():
        if "find_by_tag" in line or "tags.append" in line:
            assert line.lstrip().startswith("#"), line
    assert "find_by_tag(" in (fixture / "runner.py").read_text(encoding="utf-8")
    assert "find_by_tag(" in (fixture / "report.py").read_text(encoding="utf-8")


def test_cli_eval_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from satsumaestro.cli import app as cli_app

    _write_fixture(tmp_path)
    _write_task(tmp_path, "smoke")

    def fake_build_provider(config: object, model: str, env: object) -> MockProvider:
        return MockProvider([TextDelta(text="main.py です"), MessageStop()])

    monkeypatch.setattr(cli_app, "build_provider", fake_build_provider)
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        cli_app.app,
        ["eval", str(tmp_path / "tasks"), "--cwd", str(tmp_path), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert "合格 1/1" in result.output
    loaded = EvalReport.model_validate_json(out.read_text(encoding="utf-8"))
    assert loaded.passed == 1

    def fail_provider(config: object, model: str, env: object) -> MockProvider:
        return MockProvider([TextDelta(text="不明"), MessageStop()])

    monkeypatch.setattr(cli_app, "build_provider", fail_provider)
    result = CliRunner().invoke(
        cli_app.app,
        ["eval", str(tmp_path / "tasks"), "--cwd", str(tmp_path), "--strict"],
    )
    assert result.exit_code == 1


def test_cli_eval_web_routing_flag_uses_dedicated_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from satsumaestro.cli import app as cli_app

    seen: list[tuple[Path, str, set[str] | None]] = []

    async def fake_run(
        suite: Path, model: str, *, only: set[str] | None = None
    ) -> tuple[EvalReport, list[object]]:
        seen.append((suite, model, only))
        report = EvalReport(agent="satsuma", model=model, started_at_ms=0, suite="web_routing")
        report.results.append(TaskResult(task_id="conf-date", passed=True))
        return report, []

    monkeypatch.setattr(cli_app, "run_web_routing", fake_run)
    report = tmp_path / "routing.json"
    result = CliRunner().invoke(
        cli_app.app,
        [
            "eval",
            "--web-routing",
            "--model",
            "test-model",
            "--runs",
            "2",
            "--out",
            str(report),
            "--cwd",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen == [(tmp_path / "evals" / "web_routing", "test-model", None)] * 2
    for run in (1, 2):
        saved = EvalReport.model_validate_json(
            (tmp_path / f"routing-{run}.json").read_text(encoding="utf-8")
        )
        assert saved.suite == "web_routing" and saved.passed == 1
    result = CliRunner().invoke(
        cli_app.app,
        ["eval", "--web-routing", "--agent", "claude", "--cwd", str(tmp_path)],
    )
    assert result.exit_code == 1 and "satsuma" in result.output
    result = CliRunner().invoke(cli_app.app, ["eval", "--cwd", str(tmp_path)])
    assert result.exit_code != 0


def _py_agent(script: str) -> list[str]:
    """テスト用の外部エージェント: python -c でプロンプトを受けて stdout を返す。"""
    import sys

    return [sys.executable, "-c", script, "{prompt}"]


async def test_cli_agent_json_shapes(tmp_path: Path) -> None:
    message_shaped = (
        "import json,sys;"
        "print(json.dumps({'message':'in pricing.py','iterations':3,"
        "'usage':{'input_tokens':100,'output_tokens':20}}))"
    )
    agent = CliAgent("fake-cli", _py_agent(message_shaped), parse_json=True)
    ans = await agent.run("q", tmp_path)
    assert ans.final_text == "in pricing.py"
    assert ans.iterations == 3
    assert ans.usage.input_tokens == 100

    claude_like = (
        "import json,sys;"
        "print(json.dumps({'result':'see main.py','num_turns':2,'usage':{'output_tokens':5}}))"
    )
    agent = CliAgent("fake-claude", _py_agent(claude_like), parse_json=True)
    ans = await agent.run("q", tmp_path)
    assert ans.final_text == "see main.py"
    assert ans.iterations == 2
    assert ans.usage.output_tokens == 5

    agent = CliAgent("plain", _py_agent("print('just text')"), parse_json=True)
    ans = await agent.run("q", tmp_path)
    assert ans.final_text == "just text"
    assert ans.iterations == 0
    assert ans.empty_final is False

    agent = CliAgent("silent", _py_agent("pass"))
    ans = await agent.run("q", tmp_path)
    assert ans.final_text == ""
    assert ans.empty_final is True


async def test_cli_agent_receives_prompt_and_cwd(tmp_path: Path) -> None:
    script = "import sys,os;print(sys.argv[1] + '|' + os.getcwd())"
    agent = CliAgent("echo", _py_agent(script))
    ans = await agent.run("QUESTION", tmp_path)
    prompt_part, cwd_part = ans.final_text.split("|")
    assert prompt_part == "QUESTION"
    assert Path(cwd_part).resolve() == tmp_path.resolve()


async def test_cli_agent_failure_modes(tmp_path: Path) -> None:
    agent = CliAgent("boom", _py_agent("import sys;sys.exit(3)"))
    with pytest.raises(EvalAgentError, match="exit 3"):
        await agent.run("q", tmp_path)

    slow = CliAgent("slow", _py_agent("import time;time.sleep(60)"), timeout_s=0.2)
    with pytest.raises(EvalAgentError, match="タイムアウト"):
        await slow.run("q", tmp_path)

    with pytest.raises(ConfigError, match="prompt"):
        CliAgent("bad", ["echo", "hi"])


async def test_runner_records_cli_agent_error(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    boom = CliAgent("boom", _py_agent("import sys;sys.exit(1)"))
    report = await EvalRunner(boom).run([_task(fixture, "a"), _task(fixture, "b")])
    assert report.agent == "boom"
    assert [r.passed for r in report.results] == [False, False]
    assert all("EvalAgentError" in r.error for r in report.results)
    assert all(r.duration_ms >= 0 for r in report.results)


async def test_cli_agent_reports_json_error_from_stdout(tmp_path: Path) -> None:
    import sys

    agent = CliAgent(
        "other-cli",
        [
            sys.executable,
            "-c",
            'import sys; print(\'{"error":"model not found"}\'); sys.exit(1)',
            "{prompt}",
        ],
    )
    with pytest.raises(EvalAgentError, match="model not found"):
        await agent.run("hello", tmp_path)


async def test_custom_agent_reads_json_output(tmp_path: Path) -> None:
    import sys

    script = (
        "import json;print(json.dumps({'message':'in cart.py','iterations':4,"
        "'usage':{'input_tokens':50,'output_tokens':7}}))"
    )
    agent = build_cli_agent(
        f"other={shlex.quote(sys.executable)} -c {shlex.quote(script)} {{prompt}}"
    )
    ans = await agent.run("q", tmp_path)
    assert (ans.final_text, ans.iterations, ans.usage.output_tokens) == ("in cart.py", 4, 7)
    plain = build_cli_agent(f"plain={shlex.quote(sys.executable)} -c 'print(1+1)' {{prompt}}")
    assert (await plain.run("q", tmp_path)).final_text == "2"


def test_build_cli_agent_specs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent = build_cli_agent("mytool=/bin/echo {prompt}")
    assert agent.name == "mytool"

    with pytest.raises(ConfigError, match="未知のエージェント"):
        build_cli_agent("gemini")
    with pytest.raises(ConfigError, match="name=command"):
        build_cli_agent("=echo {prompt}")

    import satsumaestro.evals.agents as agents_mod

    monkeypatch.setattr(agents_mod.shutil, "which", lambda _: None)
    with pytest.raises(ConfigError, match="claude"):
        build_cli_agent("claude")


def test_cli_eval_multi_agent_comparison(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    from typer.testing import CliRunner

    from satsumaestro.cli import app as cli_app

    _write_fixture(tmp_path)
    _write_task(tmp_path, "cmp")

    def fake_build_provider(config: object, model: str, env: object) -> MockProvider:
        return MockProvider([TextDelta(text="main.py です"), MessageStop()])

    monkeypatch.setattr(cli_app, "build_provider", fake_build_provider)
    echo_spec = f"echoer={sys.executable} -c \"print('nothing found')\" {{prompt}}"
    result = CliRunner().invoke(
        cli_app.app,
        [
            "eval",
            str(tmp_path / "tasks"),
            "--cwd",
            str(tmp_path),
            "--agent",
            "satsuma",
            "--agent",
            echo_spec,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "agents comparison" in result.output
    assert "合格 1/1" in result.output
    assert "合格 0/1" in result.output
    saved = list((tmp_path / ".satsuma" / "evals").glob("*.json"))
    agents = {p.stem.rsplit("-", 1)[-1] for p in saved}
    assert agents == {"satsuma", "echoer"}


async def test_cli_agent_stdin_is_closed(tmp_path: Path) -> None:
    script = "import sys;print('eof' if sys.stdin.read() == '' else 'blocked')"
    agent = CliAgent("stdin-check", _py_agent(script), timeout_s=10)
    ans = await agent.run("q", tmp_path)
    assert ans.final_text == "eof"


def test_taxonomy_category_validation(tmp_path: Path) -> None:
    from satsumaestro.evals.models import TAXONOMY, is_valid_category

    assert set(TAXONOMY) == {"WHAT", "WHY", "WHERE", "HOW"}
    assert sum(len(v) for v in TAXONOMY.values()) == 12
    assert is_valid_category("WHERE/identifier-location") is True
    assert is_valid_category("WHY/design-rationale") is True
    assert is_valid_category("WHERE/unknown") is False
    assert is_valid_category("HOW") is False
    assert is_valid_category("") is False

    _write_fixture(tmp_path)
    _write_task(tmp_path, "tagged", extra='category = "WHAT/concept-definition"')
    tasks = load_tasks(tmp_path / "tasks" / "tagged.toml")
    assert tasks[0].category == "WHAT/concept-definition"
    _write_task(tmp_path, "badcat", extra='category = "NANI/sonzai-shinai"')
    with pytest.raises(ConfigError, match="category"):
        load_tasks(tmp_path / "tasks" / "badcat.toml")


def test_bundled_suites_are_fully_tagged() -> None:
    from satsumaestro.evals.models import is_valid_category

    for suite in ("code_analysis", "code_analysis_hard"):
        for task in load_tasks(_REPO_EVALS / suite):
            assert task.category, f"{suite}/{task.id} に category が無い"
            assert is_valid_category(task.category), f"{suite}/{task.id}: {task.category}"


async def test_runner_carries_category(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    task = _task(fixture, "cat-task", category="HOW/algorithm-implementation")
    provider = MockProvider([TextDelta(text="main.py"), MessageStop()])
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    report = await EvalRunner(agent).run([task])
    assert report.results[0].category == "HOW/algorithm-implementation"


def test_bundled_challenge_suite_loads_and_ground_truths() -> None:
    import ast

    tasks = load_tasks(_REPO_EVALS / "code_analysis_challenge")
    ids = {t.id for t in tasks}
    assert ids == {
        "find-anomalous-handler",
        "exact-count",
        "compute-output",
        "transitive-chain",
        "dead-branch",
        "single-dispatch",
        "validate-importers",
        "results-order",
        "status-flow",
        "retry-attempts",
        "dynamic-import",
        "bonus-source",
    }
    from satsumaestro.evals.models import is_valid_category

    assert all(t.category and is_valid_category(t.category) for t in tasks)

    fixture = Path(next(iter(tasks)).fixture)
    assert fixture.name == "relay"
    emit_calls = 0
    for p in sorted(fixture.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        emit_calls += sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "emit"
        )
    assert emit_calls == 21
    missing = []
    for p in sorted(fixture.glob("handler_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        calls = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if "validate" not in calls:
            missing.append(p.name)
    assert missing == ["handler_13.py"]
    importers = 0
    for p in sorted(fixture.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        if any(isinstance(n, ast.ImportFrom) and n.module == "validate" for n in ast.walk(tree)):
            importers += 1
    assert importers == 19
