"""存在しないツール名の呼び出しに、表記ゆれの読み替えと候補の提示で答えるテスト。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from satsumaestro.config.models import HookConfig, RuntimeConfig, ToolsConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.evals.agents import SatsumaAgent
from satsumaestro.evals.defects import detect_defects
from satsumaestro.evals.models import EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, tool_names
from tests.fixtures.mock_provider import SequencedMockProvider

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"
REG = default_registry()
CORE = set(REG.core_names())

REAL_PATTERN_SHAPE = {"path": "common/SurveyDesign.js", "pattern": "addOpenBtn"}
REAL_QUERY_SHAPE = {"path": "", "query": "_randomTitle", "max_results": 20}


def _names(name: str, inp: dict[str, object]) -> list[str]:
    return [c.name for c in tool_names.suggest(name, inp, REG, CORE).candidates]


def test_variants_resolve_to_the_same_tool() -> None:
    assert tool_names.resolve_variant("writeFile?", REG) == "write_file"
    assert tool_names.resolve_variant("Grep", REG) == "grep_search"
    assert tool_names.resolve_variant("read-file", REG) == "read_file"
    assert tool_names.resolve_variant("ReadFile", REG) == "read_file"
    for name in ("greps_search", "search", "list_workspace", "", "   "):
        assert tool_names.resolve_variant(name, REG) is None


def test_search_candidates_follow_the_argument_shape() -> None:
    assert _names("search", REAL_PATTERN_SHAPE) == ["grep_search", "glob_search"]
    assert _names("search", {"path": ".", "pattern": "**/*.js"})[0] == "glob_search"
    assert _names("search", {"pattern": "a.*b", "include": "*.js"})[0] == "grep_search"
    assert "tool_search" not in _names("search", REAL_QUERY_SHAPE)
    assert "web_search" not in _names("search", REAL_QUERY_SHAPE)


def test_real_webbench_search_calls_put_the_next_used_tool_in_the_top_two() -> None:
    assert "grep_search" in _names("search", REAL_PATTERN_SHAPE)[:2]
    assert "glob_search" in _names("search", REAL_QUERY_SHAPE)[:2]


def test_other_real_names_get_sensible_candidates() -> None:
    assert _names("list_workspace", {"path": "."}) == ["list_directory", "glob_search"]
    assert _names("print_tree", {}) == ["list_directory", "glob_search"]
    assert _names("git_status", {}) == ["bash"]
    assert _names("open_file", {"path": "a"}) == ["read_file"]
    assert _names("globlSearch", {"pattern": "*.js"})[0] == "glob_search"
    assert _names("greps_search", {"pattern": "x"})[0] == "grep_search"
    assert _names("globa", {}) == ["glob_search"]
    assert _names("todo", {}) == ["todo_write"]


def test_argument_hints_and_the_available_list() -> None:
    s = tool_names.suggest("search", REAL_QUERY_SHAPE, REG, CORE)
    grep = s.candidates[0]
    assert "use `pattern` instead of `query`" in grep.arg_hint
    assert "`max_results` is not a parameter" in grep.arg_hint
    glob = s.candidates[1]
    assert "use `limit` instead of `max_results`" in glob.arg_hint
    msg = tool_names.unknown_tool_message("search", s, [])
    lines = msg.splitlines()
    assert lines[0] == "unknown tool: search."
    assert lines[2].startswith("- grep_search: ")
    assert lines[-1].startswith("Available tools: read_file") and "tool_search" in lines[-1]
    empty = tool_names.unknown_tool_message("", tool_names.suggest("", {}, REG, CORE), [])
    assert "The tool name was empty" in empty


def _runtime(
    scripts: list[list[AssistantEvent]], tmp_path: Path, config: RuntimeConfig | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(scripts),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config or RuntimeConfig(),
    )


def _call(name: str, id: str = "c1", **inp: object) -> list[AssistantEvent]:
    return [ToolUse(id=id, name=name, input=inp), MessageStop()]


def _result(rt: ConversationRuntime, id: str) -> ToolResultBlock:
    for m in rt.session.messages:
        for b in m.blocks:
            if isinstance(b, ToolResultBlock) and b.tool_use_id == id:
                return b
    raise AssertionError(id)


async def test_variant_name_runs_the_real_tool(tmp_path: Path) -> None:
    rt = _runtime(
        [_call("writeFile?", path="a.txt", content="hi\n"), [TextDelta(text="ok"), MessageStop()]],
        tmp_path,
    )
    summary = await rt.run_turn("a.txt を作って")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hi\n"
    block = _result(rt, "c1")
    assert not block.is_error and block.tool_name == "write_file"
    assert "note: `writeFile?` is not a tool name; ran write_file." in block.output
    assert (summary.tool_name_fixes, summary.unknown_tool_calls) == (1, 0)


async def test_unknown_name_is_not_run_and_gets_candidates(tmp_path: Path) -> None:
    rt = _runtime(
        [_call("search", **REAL_PATTERN_SHAPE), [TextDelta(text="ok"), MessageStop()]], tmp_path
    )
    summary = await rt.run_turn("探して")
    block = _result(rt, "c1")
    assert block.is_error and block.output.startswith("unknown tool: search.")
    assert "- grep_search: " in block.output
    assert (summary.unknown_tool_calls, summary.deferred_autoloads) == (1, 0)


async def test_deferred_top_candidate_is_loaded_for_the_next_call(tmp_path: Path) -> None:
    rt = _runtime([_call("todo", todos=[]), [TextDelta(text="ok"), MessageStop()]], tmp_path)
    assert "todo_write" not in {s["name"] for s in rt._all_schemas()}
    summary = await rt.run_turn("やることを整理して")
    assert "loaded now" in _result(rt, "c1").output
    assert "todo_write" in {s["name"] for s in rt._all_schemas()}
    assert (summary.unknown_tool_calls, summary.deferred_autoloads) == (1, 1)


async def test_direct_call_of_a_deferred_tool_loads_it_but_not_enter_plan_mode(
    tmp_path: Path,
) -> None:
    rt = _runtime(
        [
            _call("todo_write", todos=[{"content": "A"}]),
            _call("enter_plan_mode", id="c2"),
            [TextDelta(text="ok"), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("整理して")
    first = _result(rt, "c1")
    assert not first.is_error and "[ ] A" in first.output
    assert "todo_write" in rt._expanded and summary.deferred_autoloads == 1
    assert "deferred tool" in _result(rt, "c2").output
    assert "enter_plan_mode" not in rt._expanded and rt.session.mode.main == "code"


async def test_without_deferred_loading_nothing_is_loaded(tmp_path: Path) -> None:
    cfg = RuntimeConfig(tools=ToolsConfig(deferred_tool_loading=False))
    rt = _runtime([_call("todo", todos=[]), [TextDelta(text="ok"), MessageStop()]], tmp_path, cfg)
    summary = await rt.run_turn("整理して")
    assert "loaded now" not in _result(rt, "c1").output
    assert (summary.unknown_tool_calls, summary.deferred_autoloads) == (1, 0)


async def test_metrics_reach_the_report_and_the_defect_record(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "a.txt").write_text("x\n", encoding="utf-8")
    calls = [
        _call(name, id=f"s{i}", path=".", query="x")
        for i, name in enumerate(("search", "find", "lookup"))
    ]
    provider = SequencedMockProvider([*calls, [TextDelta(text="a.txt"), MessageStop()]])
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    task = EvalTask.model_validate(
        {
            "id": "t",
            "prompt": "探して",
            "fixture": str(fixture),
            "permission_mode": "workspace-write",
            "checks": {"contains_all": ["a.txt"]},
        }
    )
    report = await EvalRunner(agent).run([task])
    result = report.results[0]
    assert result.unknown_tool_calls == 3
    dumped = json.loads(report.model_dump_json())
    assert EvalReport.model_validate(dumped).results[0].unknown_tool_calls == 3
    assert [d.kind for d in detect_defects(result, "satsuma")] == ["unknown tool name"]
    assert detect_defects(result.model_copy(update={"unknown_tool_calls": 2}), "satsuma") == []
    old = TaskResult.model_validate({"task_id": "t", "passed": True})
    assert (old.unknown_tool_calls, old.tool_name_fixes, old.deferred_autoloads) == (0, 0, 0)


def test_ab_summary_folds_the_counts(tmp_path: Path) -> None:
    before = tmp_path / "b.json"
    before.write_text(
        json.dumps({"results": [{"task_id": "t", "passed": True, "unknown_tool_calls": 5}]}),
        encoding="utf-8",
    )
    after = tmp_path / "a.json"
    after.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "task_id": "t",
                        "passed": True,
                        "unknown_tool_calls": 1,
                        "tool_name_fixes": 1,
                        "deferred_autoloads": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "t", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 存在しないツール名 | 5.0 | 1.0 | -4.0 |" in out
    assert "| ツール名の表記ゆれを読み替え | 0.0 | 1.0 | +1.0 |" in out
    assert "| 遅延ツールの自動読み込み | 0.0 | 1.0 | +1.0 |" in out
