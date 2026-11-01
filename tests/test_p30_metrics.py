"""計測プラミングのテスト。"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.evals.agents import SatsumaAgent
from satsumaestro.evals.models import EvalChecks, EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.openai_provider import _assemble_openai, _safe_json_args
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ProviderError
from tests.fixtures.mock_provider import ScriptedFailureProvider, SequencedMockProvider

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "eval_ab_summary.py"


def test_safe_json_args_flags_fallback() -> None:
    assert _safe_json_args('{"a": 1}') == ({"a": 1}, False)
    assert _safe_json_args("") == ({}, False)
    assert _safe_json_args("We list files.{") == ({}, True)
    assert _safe_json_args("[1, 2]") == ({}, True)


async def _chunks(items: list[NS]) -> AsyncIterator[NS]:
    for it in items:
        yield it


def _tool_chunk(args: str, finish: str | None = None) -> NS:
    tc = NS(index=0, id="call_1", function=NS(name="read_file", arguments=args))
    delta = NS(content=None, tool_calls=[tc])
    return NS(choices=[NS(delta=delta, finish_reason=finish)], usage=None)


async def test_assemble_marks_input_fallback() -> None:
    broken = [ev async for ev in _assemble_openai(_chunks([_tool_chunk("prose{", "tool_calls")]))]
    tool = next(e for e in broken if isinstance(e, ToolUse))
    assert tool.input == {} and tool.input_fallback is True

    ok = [ev async for ev in _assemble_openai(_chunks([_tool_chunk('{"path":"a"}', "tool_calls")]))]
    tool = next(e for e in ok if isinstance(e, ToolUse))
    assert tool.input == {"path": "a"} and tool.input_fallback is False


def test_tool_use_default_keeps_equality() -> None:
    assert ToolUse(id="x", name="n", input={}) == ToolUse(id="x", name="n", input={})
    assert ToolUse(id="x", name="n", input_fallback=True) != ToolUse(id="x", name="n")


def _make_runtime(provider: object, tmp_path: Path) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


async def test_turn_summary_counts_empty_tool_args(tmp_path: Path) -> None:
    provider = SequencedMockProvider(
        [
            [ToolUse(id="c1", name="read_file", input={}, input_fallback=True), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    summary = await _make_runtime(provider, tmp_path).run_turn("hi")
    assert summary.final_text == "done"
    assert summary.empty_tool_args == 1
    assert summary.tool_call_repairs == 0


async def test_turn_summary_counts_repairs(tmp_path: Path) -> None:
    provider = ScriptedFailureProvider(
        [
            ProviderError("error parsing tool call", status_code=500),
            [TextDelta(text="ok"), MessageStop()],
        ]
    )
    summary = await _make_runtime(provider, tmp_path).run_turn("hi")
    assert summary.final_text == "ok"
    assert summary.tool_call_repairs == 1
    assert summary.empty_tool_args == 0


async def test_eval_task_result_carries_metrics(tmp_path: Path) -> None:
    fixture = tmp_path / "fx"
    fixture.mkdir()
    (fixture / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [ToolUse(id="c1", name="read_file", input={}, input_fallback=True), MessageStop()],
            [TextDelta(text="main.py"), MessageStop()],
        ]
    )
    task = EvalTask(
        id="t1",
        prompt="where?",
        fixture=str(fixture),
        checks=EvalChecks(contains_all=["main.py"]),
    )
    report = await EvalRunner(SatsumaAgent(provider, RuntimeConfig(), "mock")).run([task])
    result = report.results[0]
    assert result.passed
    assert result.empty_tool_args == 1
    assert result.tool_call_repairs == 0
    assert result.compacted is False


def test_task_result_reads_legacy_json() -> None:
    legacy: dict[str, Any] = {"task_id": "a", "passed": True}
    r = TaskResult.model_validate(legacy)
    assert (r.compacted, r.tool_call_repairs, r.empty_tool_args) == (False, 0, 0)
    assert r.empty_final is False
    fresh = TaskResult(task_id="b", passed=False, tool_call_repairs=2, empty_tool_args=3)
    rt = TaskResult.model_validate_json(fresh.model_dump_json())
    assert (rt.tool_call_repairs, rt.empty_tool_args) == (2, 3)


def _report(path: Path, rows: list[dict[str, Any]]) -> Path:
    report = EvalReport(agent="satsuma", model="m", started_at_ms=0)
    report.results = [TaskResult.model_validate(r) for r in rows]
    data = json.loads(report.model_dump_json())
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _legacy_report(path: Path, rows: list[dict[str, Any]]) -> Path:
    data = json.loads(_report(path, rows).read_text(encoding="utf-8"))
    for r in data["results"]:
        for key in ("compacted", "tool_call_repairs", "empty_tool_args", "empty_final"):
            r.pop(key, None)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_ab_summary_script(tmp_path: Path) -> None:
    before = [
        _legacy_report(
            tmp_path / "b1.json",
            [
                {"task_id": "x", "passed": True, "final_text": "ok", "duration_ms": 1000},
                {"task_id": "y", "passed": False, "final_text": "", "duration_ms": 3000},
            ],
        ),
        _legacy_report(
            tmp_path / "b2.json",
            [
                {"task_id": "x", "passed": False, "final_text": "", "duration_ms": 1000},
                {"task_id": "y", "passed": False, "error": "boom", "duration_ms": 1000},
            ],
        ),
    ]
    after = [
        _report(
            tmp_path / "a1.json",
            [
                {"task_id": "x", "passed": True, "final_text": "ok", "duration_ms": 500},
                {
                    "task_id": "y",
                    "passed": True,
                    "final_text": "ok",
                    "duration_ms": 500,
                    "tool_call_repairs": 1,
                    "empty_tool_args": 2,
                    "compacted": True,
                },
            ],
        ),
    ]
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--title",
            "suite",
            "--before",
            *map(str, before),
            "--after",
            *map(str, after),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    out = proc.stdout
    assert "### `suite`（2/1 回平均 / 2 題）" in out
    assert "| 合格率 | 25.0% | 100.0% | +75.0% |" in out
    assert "| 本文空の終了 (empty_final) | 1.0 | 0.0 | -1.0 |" in out
    assert "| tool_call 修復回数 | 0.0 | 1.0 | +1.0 |" in out
    assert "| 引数 {} fallback 回数 | 0.0 | 2.0 | +2.0 |" in out
    assert "| 圧縮発火タスク数 | 0.0 | 1.0 | +1.0 |" in out
    assert "| x | ○× | ○ | 1.0 | 0.5 |" in out


def test_ab_summary_reads_explicit_empty_final(tmp_path: Path) -> None:
    before = [
        _report(
            tmp_path / "b.json",
            [
                {
                    "task_id": "x",
                    "passed": True,
                    "final_text": "(answer found only in reasoning)\nmain.py",
                    "empty_final": True,
                    "duration_ms": 1000,
                }
            ],
        )
    ]
    after = [
        _report(
            tmp_path / "a.json",
            [{"task_id": "x", "passed": True, "final_text": "ok", "duration_ms": 1000}],
        )
    ]
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--title",
            "suite",
            "--before",
            *map(str, before),
            "--after",
            *map(str, after),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "| 本文空の終了 (empty_final) | 1.0 | 0.0 | -1.0 |" in proc.stdout


def test_ab_summary_script_tolerates_uneven_task_sets(tmp_path: Path) -> None:
    before = [
        _report(
            tmp_path / "b1.json",
            [
                {"task_id": "x", "passed": True, "final_text": "ok", "duration_ms": 1000},
                {"task_id": "y", "passed": True, "final_text": "ok", "duration_ms": 1000},
            ],
        ),
        _report(
            tmp_path / "b2.json",
            [{"task_id": "x", "passed": False, "final_text": "", "duration_ms": 1000}],
        ),
    ]
    after = [
        _report(
            tmp_path / "a1.json",
            [{"task_id": "x", "passed": True, "final_text": "ok", "duration_ms": 500}],
        )
    ]
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--title",
            "suite",
            "--before",
            *map(str, before),
            "--after",
            *map(str, after),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "| y | ○- | - |" in proc.stdout
