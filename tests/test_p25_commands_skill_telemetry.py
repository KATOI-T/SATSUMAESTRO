"""スラッシュコマンド・Skill・テレメトリのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli.commands import (
    SLASH_COMMANDS,
    CommandOutcome,
    SlashDispatcher,
    parse_slash,
)
from satsumaestro.config.models import HookConfig, RuntimeConfig, TelemetryConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import TurnSummary
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder, format_skill_catalog
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import TokenUsage, WriteChunk
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.skill.resolver import (
    SkillLookupError,
    list_skills,
    read_skill_header,
    resolve_skill,
)
from satsumaestro.telemetry.log import TraceEntry, TraceLog
from satsumaestro.telemetry.recorder import TraceRecorder, insights_from, summary_fields
from satsumaestro.tools import ToolContext, default_registry
from satsumaestro.tools.builtins.skill import Skill
from tests.fixtures.mock_provider import SequencedMockProvider


def _runtime(
    provider: object, tmp_path: Path, cfg: RuntimeConfig | None = None
) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=cfg or RuntimeConfig(),
    )


def _console() -> tuple[Console, list[str]]:
    console = Console(record=True, width=120)
    return console, []


def test_parse_slash_passthrough() -> None:
    assert parse_slash("hello world") is None
    assert parse_slash("  not /a command") is None
    cmd = parse_slash("/help")
    assert cmd is not None and cmd.name == "help" and cmd.args == []
    cmd2 = parse_slash("/effort high")
    assert cmd2 is not None and cmd2.name == "effort" and cmd2.args == ["high"]
    bare = parse_slash("/")
    assert bare is not None and bare.name == "help"


async def test_dispatch_help_lists_specs(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=200)
    d = SlashDispatcher(rt, console)
    out = await d.dispatch(parse_slash("/help"))  # type: ignore[arg-type]
    assert out is CommandOutcome.HANDLED
    text = console.export_text()
    for command in SLASH_COMMANDS:
        assert f"/{command.name}" in text
    assert "/effort low|medium|high|default" in text
    assert "/exit (/quit, /q)" in text
    assert "/lang [auto|ja|en]" in text
    assert "/model [model]" in text


async def test_dispatch_unknown_is_handled(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=120)
    out = await SlashDispatcher(rt, console).dispatch(parse_slash("/no_such"))  # type: ignore[arg-type]
    assert out is CommandOutcome.HANDLED
    assert "/help" in console.export_text()


async def test_dispatch_exit_and_aliases(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=120)
    d = SlashDispatcher(rt, console)
    for line in ("/exit", "/quit", "/q"):
        assert await d.dispatch(parse_slash(line)) is CommandOutcome.EXIT  # type: ignore[arg-type]


async def test_state_flags_effort_fast_plan(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=120)
    d = SlashDispatcher(rt, console)
    await d.dispatch(parse_slash("/effort high"))  # type: ignore[arg-type]
    assert rt.config.provider.sampling.reasoning_effort == "high"
    await d.dispatch(parse_slash("/fast"))  # type: ignore[arg-type]
    assert rt.config.provider.sampling.reasoning_effort == "low"
    await d.dispatch(parse_slash("/fast"))  # type: ignore[arg-type]
    assert rt.config.provider.sampling.reasoning_effort == "high"
    await d.dispatch(parse_slash("/effort default"))  # type: ignore[arg-type]
    assert rt.config.provider.sampling.reasoning_effort is None
    await d.dispatch(parse_slash("/plan on"))  # type: ignore[arg-type]
    assert rt.session.mode.main == "plan"
    await d.dispatch(parse_slash("/plan off"))  # type: ignore[arg-type]
    assert rt.session.mode.main == "code"


async def test_clear_resets_session_in_place(tmp_path: Path) -> None:
    rt = _runtime(SequencedMockProvider([[TextDelta(text="hi"), MessageStop()]]), tmp_path)
    await rt.run_turn("hello")
    assert rt.session.messages
    session_ref = rt.session
    console = Console(record=True, width=120)
    await SlashDispatcher(rt, console).dispatch(parse_slash("/clear"))  # type: ignore[arg-type]
    assert rt.session is session_ref
    assert not rt.session.messages and not rt.session.prompt_history


def _make_skill(root: Path, dirname: str, name: str | None, desc: str, body: str) -> None:
    d = root / dirname
    d.mkdir(parents=True)
    front = "---\n"
    if name:
        front += f'name: "{name}"\n'
    front += f"description: {desc}\n---\n"
    (d / "SKILL.md").write_text(front + body, encoding="utf-8")


def test_skills_are_listed_and_resolved_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(home))
    _make_skill(tmp_path / ".satsuma" / "skills", "deploy", "Deploy", "release steps", "1. build\n")
    _make_skill(tmp_path / ".claude" / "skills", "review", None, "review checklist", "- check\n")
    _make_skill(home / "skills", "deploy-old", "DEPLOY", "old steps", "0. legacy\n")
    _make_skill(home / "skills", "notes", None, "", "- write\n")
    metas = list_skills(tmp_path)
    assert [(m.name, m.description) for m in metas] == [
        ("Deploy", "release steps"),
        ("review", "review checklist"),
        ("notes", ""),
    ]
    assert resolve_skill(tmp_path, "deploy").name == "Deploy"
    assert resolve_skill(tmp_path, "REVIEW").path.name == "SKILL.md"
    assert resolve_skill(tmp_path, "/notes").name == "notes"
    with pytest.raises(SkillLookupError, match="there is no skill named missing"):
        resolve_skill(tmp_path, "missing")
    with pytest.raises(SkillLookupError, match="cannot contain a path"):
        resolve_skill(tmp_path, "../escape")
    with pytest.raises(SkillLookupError, match="no skill name was given"):
        resolve_skill(tmp_path, " / ")


def test_skill_header_variants() -> None:
    assert read_skill_header("---\nname: 'X'\ndescription: y\n---\nbody") == {
        "name": "X",
        "description": "y",
    }
    assert read_skill_header("no header") == {}
    text = '---\nname: a\nname:\ndescription: "use: carefully"\nname: b\n---\nname: c\n'
    assert read_skill_header(text) == {"name": "b", "description": "use: carefully"}
    assert read_skill_header("\ufeff---\nname: z\n---\n") == {"name": "z"}


async def test_skill_tool_injects_body(
    tool_ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tool_ctx.workspace_root / "home"))
    _make_skill(
        tool_ctx.workspace_root / ".satsuma" / "skills",
        "deploy",
        "deploy",
        "steps",
        "1. make check\n",
    )
    res = await Skill().run(Skill.Input(skill="deploy", args="v2"), tool_ctx)
    assert not res.is_error
    assert "# Skill: deploy" in res.output and "1. make check" in res.output and "v2" in res.output
    missing = await Skill().run(Skill.Input(skill="nope"), tool_ctx)
    assert missing.is_error and "deploy" in missing.output


def test_skill_catalog_formatting() -> None:
    text = format_skill_catalog([("deploy", "release steps"), ("review", "")])
    assert "# Available skills" in text and "- deploy: release steps" in text and "- review" in text
    assert format_skill_catalog([]) == ""


async def test_recorder_writes_a_turn_into_the_log(tmp_path: Path) -> None:
    log = TraceLog()
    provider = SequencedMockProvider(
        [
            [
                ToolUse(id="c1", name="glob_search", input={"pattern": "*"}),
                MessageStop(stop_reason="tool_use"),
            ],
            [TextDelta(text="done"), MessageStop()],
        ]
    )
    rt = _runtime(provider, tmp_path)
    await rt.run_turn("hi", observer=TraceRecorder(rt.session.session_id, log))
    names = [e.event for e in log.entries]
    assert names == [
        "turn.begin",
        "model.reply",
        "tool.begin",
        "tool.end",
        "model.reply",
        "turn.end",
    ]
    assert [e.seq for e in log.entries] == list(range(1, 7))
    assert {e.session for e in log.entries} == {rt.session.session_id}
    assert log.entries[0].data == {"prompt_chars": 2}
    assert log.entries[3].data == {"iteration": 0, "tool": "glob_search", "failed": False}
    end = log.entries[-1].data
    assert "final_text" not in end and "last_text" not in end
    assert end["stop_reason"] == "end_turn" and end["iterations"] == 2
    assert end["usage"] == TokenUsage().model_dump()


def test_summary_fields_keep_counts_and_leave_out_free_text() -> None:
    summary = TurnSummary(
        final_text="do-not-log answer",
        last_text="do-not-log",
        last_thinking="do-not-log",
        stop_reason="end_turn",
        iterations=3,
        usage=TokenUsage(input_tokens=5, output_tokens=7),
        health_warning="tool check failed",
        plan_budget_verdicts=["ok", "over"],
        write_chunks=[WriteChunk(tool="write_file", path="a.py", lines=3, chars=30)],
        tokens_per_s=12.5,
    )
    data = summary_fields(summary)
    assert "do-not-log" not in json.dumps(data)
    assert "health_warning" not in data and "decomposition" not in data
    assert data["iterations"] == 3 and data["tokens_per_s"] == 12.5
    assert data["usage"] == {
        "input_tokens": 5,
        "output_tokens": 7,
        "cache_read": 0,
        "cache_write": 0,
    }
    assert data["write_chunks"] == 1
    assert data["plan_budget_verdicts"] == "ok,over"
    assert data["plan_step_records"] == 0
    assert (
        summary_fields(summary.model_copy(update={"plan_budget_verdicts": []}))[
            "plan_budget_verdicts"
        ]
        == ""
    )


def test_log_appends_json_lines_and_keeps_them_in_memory(tmp_path: Path) -> None:
    path = tmp_path / "trace" / "t.jsonl"
    log = TraceLog(path)
    for i in range(3):
        log.add(TraceEntry("s", i + 1, "turn.begin", 1000 + i, {"k": i}))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and len(log.entries) == 3
    assert json.loads(lines[2]) == {
        "session": "s",
        "seq": 3,
        "event": "turn.begin",
        "at_ms": 1002,
        "data": {"k": 2},
    }


def test_log_keeps_entries_when_the_file_cannot_be_written(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    log = TraceLog(blocker / "t.jsonl")
    log.add(TraceEntry("s", 1, "turn.begin", 0, {}))
    assert len(log.entries) == 1


def test_insights_count_turns_tools_and_failures() -> None:
    entries = [
        TraceEntry("s", 1, "turn.begin", 0, {}),
        TraceEntry("s", 2, "tool.end", 0, {"tool": "bash", "failed": False}),
        TraceEntry("s", 3, "tool.end", 0, {"tool": "bash", "failed": True}),
        TraceEntry("s", 4, "turn.end", 0, {"iterations": 3}),
        TraceEntry("s", 5, "turn.abort", 0, {"iteration": 5, "error": "MaxIterationsError"}),
        TraceEntry("s", 6, "recovery", 0, {}),
        TraceEntry("s", 7, "permission.auto_allow", 0, {}),
    ]
    ins = insights_from(entries)
    assert ins.turns == 1 and ins.turns_failed == 1
    assert ins.tools["bash"].calls == 2 and ins.tools["bash"].errors == 1
    assert ins.iterations_avg == 3.0
    assert ins.recoveries == 1 and ins.auto_allowed == 1


async def test_insights_command_renders(tmp_path: Path) -> None:
    log = TraceLog()
    log.add(TraceEntry("s", 1, "turn.end", 0, {"iterations": 2}))
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    console = Console(record=True, width=120)
    await SlashDispatcher(rt, console, trace_log=log).dispatch(parse_slash("/insights"))  # type: ignore[arg-type]
    assert "ターン: 1" in console.export_text()


def test_telemetry_is_built_only_when_enabled(tmp_path: Path) -> None:
    from satsumaestro.cli.app import _build_telemetry

    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path)
    assert _build_telemetry(rt) == (None, None)
    rt2 = _runtime(
        SequencedMockProvider([[MessageStop()]]),
        tmp_path,
        RuntimeConfig(telemetry=TelemetryConfig(enabled=True, jsonl_file="out/trace.jsonl")),
    )
    recorder, log = _build_telemetry(rt2)
    assert isinstance(recorder, TraceRecorder) and isinstance(log, TraceLog)
    assert log.file == tmp_path / "out" / "trace.jsonl"
