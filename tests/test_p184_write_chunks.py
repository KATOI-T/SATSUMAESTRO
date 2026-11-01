"""書き込み 1 回ごとの分量の記録と、分割の目安の一元化。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import satsumaestro.engine.runtime as runtime_module
from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse, Usage
from satsumaestro.engine.plan_steps import PLAN_STEPS_REQUEST, PLAN_STEPS_REQUEST_LEGACY
from satsumaestro.engine.reminders import timeout_note
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.evals.agents import SatsumaAgent
from satsumaestro.evals.models import EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import _STATIC_DOING_TASKS, SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import WriteChunk
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.builtins.append_file import AppendFile
from satsumaestro.tools.builtins.write_file import WriteFile
from satsumaestro.util import chunking
from tests.fixtures.mock_provider import SequencedMockProvider

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"

_HEAD = "<html>\n<body>\n"
_TAIL = "<p>日本語</p>\n</body>\n</html>"


def _runtime(scripts: list[list[AssistantEvent]], tmp_path: Path) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(scripts),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


def _write(path: str, content: str, *, tool: str = "write_file", id: str = "w1") -> ToolUse:
    return ToolUse(id=id, name=tool, input={"path": path, "content": content})


async def test_each_write_is_recorded_with_its_size(tmp_path: Path) -> None:
    rt = _runtime(
        [
            [_write("out/index.html", _HEAD), Usage(output_tokens=40), MessageStop()],
            [
                _write("out/index.html", _TAIL, tool="append_file", id="a1"),
                Usage(output_tokens=60),
                MessageStop(),
            ],
            [TextDelta(text="書きました"), Usage(output_tokens=5), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("HTML を作って")

    assert (tmp_path / "out/index.html").read_text(encoding="utf-8") == _HEAD + _TAIL
    first, second = summary.write_chunks
    assert (first.tool, first.path, first.ok, first.iteration) == (
        "write_file",
        "out/index.html",
        True,
        1,
    )
    assert (first.chars, first.lines, first.bytes) == (14, 2, 14)
    assert first.output_tokens == 40
    assert (second.tool, second.iteration, second.output_tokens) == ("append_file", 2, 60)
    assert (second.chars, second.lines) == (len(_TAIL), 3)
    assert second.bytes == len(_TAIL.encode("utf-8")) == second.chars + 6
    assert all(c.response_s >= 0 for c in summary.write_chunks)
    assert summary.tokens_per_s >= 0


async def test_tokens_per_s_is_the_median_over_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter([0.0, 2.0, 10.0, 14.0, 20.0, 21.0])
    monkeypatch.setattr(runtime_module, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    rt = _runtime(
        [
            [_write("a.html", _HEAD), Usage(output_tokens=40), MessageStop()],
            [_write("a.html", _TAIL, tool="append_file"), Usage(output_tokens=60), MessageStop()],
            [TextDelta(text="done"), Usage(output_tokens=5), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("HTML を作って")
    assert summary.tokens_per_s == 15.0
    assert [c.response_s for c in summary.write_chunks] == [2.0, 4.0]


async def test_failed_write_is_recorded_and_other_tools_are_not(tmp_path: Path) -> None:
    rt = _runtime(
        [
            [
                _write("../escape.txt", "x\n"),
                ToolUse(id="r1", name="read_file", input={"path": "missing.txt"}),
                MessageStop(),
            ],
            [TextDelta(text="失敗しました"), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("書いて")
    assert [(c.tool, c.ok, c.chars) for c in summary.write_chunks] == [("write_file", False, 2)]


async def test_plan_file_writes_are_not_recorded(tmp_path: Path) -> None:
    script: list[list[AssistantEvent]] = [[TextDelta(text="ok"), MessageStop()]]
    rt = _runtime(script, tmp_path)
    plan_path = rt.plan_file_relpath()
    script.insert(
        0, [_write(plan_path, "# Plan\n"), _write("a.txt", "a\n", id="w2"), MessageStop()]
    )
    summary = await rt.run_turn("書いて")
    assert [c.path for c in summary.write_chunks] == ["a.txt"]


async def test_write_chunks_reach_the_report_json(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    provider = SequencedMockProvider(
        [
            [_write("doc.md", "# A\n"), Usage(output_tokens=10), MessageStop()],
            [_write("doc.md", "b\nc\n", tool="append_file", id="a1"), MessageStop()],
            [TextDelta(text="doc.md を書きました"), MessageStop()],
        ]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    task = EvalTask.model_validate(
        {
            "id": "gen",
            "prompt": "doc.md を書いて",
            "fixture": str(fixture),
            "permission_mode": "workspace-write",
            "checks": {"contains_all": ["doc.md"]},
        }
    )
    report = await EvalRunner(agent).run([task])
    result = report.results[0]
    assert [(c.tool, c.lines) for c in result.write_chunks] == [
        ("write_file", 1),
        ("append_file", 2),
    ]

    dumped = json.loads(report.model_dump_json())
    chunks = dumped["results"][0]["write_chunks"]
    assert [c["tool"] for c in chunks] == ["write_file", "append_file"]
    assert "content" not in chunks[0]
    assert EvalReport.model_validate(dumped).results[0].write_chunks == result.write_chunks


def test_old_report_without_the_key_still_loads() -> None:
    old = TaskResult.model_validate({"task_id": "t", "passed": True})
    assert old.write_chunks == [] and old.tokens_per_s == 0.0


def _report(path: Path, chunks: list[list[int]], rates: list[float]) -> Path:
    results = [
        {
            "task_id": f"t{i}",
            "passed": True,
            "duration_ms": 1000,
            "write_chunks": [WriteChunk(tool="append_file", chars=n).model_dump() for n in sizes],
            "tokens_per_s": rate,
        }
        for i, (sizes, rate) in enumerate(zip(chunks, rates, strict=True))
    ]
    path.write_text(json.dumps({"agent": "satsuma", "results": results}), encoding="utf-8")
    return path


def test_ab_summary_folds_write_chunks(tmp_path: Path) -> None:
    before = _report(tmp_path / "b.json", [[1000, 3000], [2000]], [20.0, 30.0])
    after = tmp_path / "a.json"
    after.write_text(
        json.dumps({"results": [{"task_id": "t0", "passed": True}, {"task_id": "t1"}]}),
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "gen", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 書き込み回数 | 3.0 | 0.0 | -3.0 |" in out
    assert "| 書き込み 1 回の文字数 中央値 | 2000 | 0 | -2000 |" in out
    assert "| 書き込み 1 回の文字数 最大 | 3000 | 0 | -3000 |" in out
    assert "| 生成速度の下限 (tok/s) | 25.0 | 0.0 | -25.0 |" in out


def test_guidance_texts_come_from_one_place_and_are_unchanged() -> None:
    assert (chunking.CHUNK_LINES, chunking.CHUNK_KB) == (150, 6)
    assert WriteFile.spec.description == (
        "Write a UTF-8 text file within the workspace (atomic). Keep `content` under ~150 "
        "lines (~6 KB) per call. For a longer file, write the first part here and continue "
        "with append_file."
    )
    assert AppendFile.spec.description == (
        "Append text to the end of a file within the workspace (created if missing). Use it "
        "to build a long file in pieces: write_file for the first ~150 lines, then "
        "append_file for each following piece. Keep each piece under ~6 KB."
    )
    assert (
        "- When a file will be longer than ~150 lines, write it in pieces: write_file for the "
        "first part, then append_file for each remaining part.\n"
    ) in _STATIC_DOING_TASKS
    assert "(at most ~150 lines)" in timeout_note(120.0)
    assert "more than ~150 lines of new content" in PLAN_STEPS_REQUEST
    assert (
        "(no step should create more than ~150 lines of new content)" in PLAN_STEPS_REQUEST_LEGACY
    )
