"""ツール名の書き方の揺れと、形の壊れた呼び出し。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.cli.tool_display import describe
from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.security.self_invocation_guard import PROHIBITED_SELF_INVOCATION_REPEAT
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.tools.tool_names import echoes_its_name, malformed_name, resolve_variant
from satsumaestro.util.errors import UnknownToolLoopError
from tests.fixtures.mock_provider import SequencedMockProvider


def _runtime(tmp_path: Path, scripts: list[list[AssistantEvent]]) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(scripts),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=PermissionResolver(default_policy(PermissionMode.DANGER_FULL_ACCESS), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


def _results(rt: ConversationRuntime) -> list[ToolResultBlock]:
    return [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


def _call(name: str, **inp: object) -> list[AssistantEvent]:
    return [ToolUse(id="c1", name=name, input=inp), MessageStop()]


_DONE: list[AssistantEvent] = [TextDelta(text="done"), MessageStop()]


@pytest.mark.parametrize(
    "name",
    [
        "run_command_output",
        "run_command",
        "RunCommand",
        "execute_command",
        "run_shell_command",
        "shell",
    ],
)
def test_command_names_of_other_agents_mean_bash(name: str) -> None:
    assert resolve_variant(name, default_registry()) == "bash"


@pytest.mark.parametrize("name", ["run_turn", "run_plan_mode", "command_runner"])
def test_other_names_stay_unknown(name: str) -> None:
    assert resolve_variant(name, default_registry()) is None


async def test_run_command_output_runs_as_bash(tmp_path: Path) -> None:
    call = _call("run_command_output", command="echo hi", description="say hi")
    rt = _runtime(tmp_path, [call, _DONE])

    summary = await rt.run_turn("q")

    (block,) = _results(rt)
    assert not block.is_error
    assert "hi" in block.output
    assert "note: `run_command_output` is not a tool name; ran bash." in block.output
    assert (summary.tool_name_fixes, summary.unknown_tool_calls) == (1, 0)


@pytest.mark.parametrize(
    ("name", "broken"),
    [
        ("run_plan_mode\n</parameter", True),
        ("write_file<parameter=path", True),
        ("read file", True),
        (" read_file ", False),
        ("run_turn", False),
    ],
)
def test_malformed_names_are_told_by_spaces_and_markup(name: str, broken: bool) -> None:
    assert malformed_name(name) is broken


@pytest.mark.parametrize(
    ("name", "meant"),
    [
        ("run_plan_mode\n</parameter", None),
        ("write_file\n</parameter", "write_file"),
    ],
)
async def test_malformed_call_is_not_treated_as_a_misspelled_name(
    tmp_path: Path, name: str, meant: str | None
) -> None:
    rt = _runtime(tmp_path, [_call(name), _DONE])

    summary = await rt.run_turn("q")

    (block,) = _results(rt)
    assert block.is_error
    first, *rest = block.output.splitlines()
    assert first.startswith("malformed tool call: the tool name ")
    assert "nothing was run" in first
    assert "Did you mean" not in block.output
    hint = f"If you meant {meant}, call {meant} with its arguments."
    assert (hint in rest) is (meant is not None)
    assert (summary.malformed_tool_calls, summary.unknown_tool_calls) == (1, 0)


def test_tool_header_stays_on_one_line() -> None:
    shown = describe(ToolUse(id="c1", name="run_plan_mode\n</parameter", input={}))
    assert shown.label == "run_plan_mode …"
    assert describe(ToolUse(id="c1", name="run_turn", input={})).label == "run_turn"


def _outputs(rt: ConversationRuntime) -> list[str]:
    return [b.output for b in _results(rt)]


async def test_second_call_of_the_same_unknown_name_gets_a_firmer_reply(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, [_call("run_turn"), _call("run_turn"), _DONE])

    summary = await rt.run_turn("q")

    first, second = _outputs(rt)
    assert "Did you mean one of these?" in first
    assert second.startswith("unknown tool: run_turn (called again).")
    assert "Did you mean" not in second
    assert "If you call run_turn again in this turn, the turn will be stopped." in second
    assert summary.unknown_tool_calls == 2


async def test_third_call_stops_the_turn(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, [_call("run_turn")] * 3 + [_DONE])

    with pytest.raises(UnknownToolLoopError) as caught:
        await rt.run_turn("q")

    assert (caught.value.name, caught.value.calls) == ("run_turn", 3)
    assert "unknown tool name run_turn was called 3 times in a row" in str(caught.value)
    assert _outputs(rt)[-1] == "unknown tool: run_turn (3 calls in a row). This turn is stopped."
    assert rt.aborted_summary is not None
    assert rt.aborted_summary.unknown_tool_calls == 3


async def test_a_real_tool_in_between_breaks_the_streak(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x\n", encoding="utf-8")
    scripts = [
        _call("run_turn"),
        _call("run_turn"),
        _call("read_file", path="a.txt"),
        _call("run_turn"),
        _call("run_turn"),
        _DONE,
    ]
    rt = _runtime(tmp_path, scripts)

    summary = await rt.run_turn("q")

    outputs = _outputs(rt)
    assert "Did you mean one of these?" in outputs[3]
    assert outputs[4].startswith("unknown tool: run_turn (called again).")
    assert summary.unknown_tool_calls == 4


async def test_a_call_that_only_echoes_its_name_gets_the_firm_reply_at_once(
    tmp_path: Path,
) -> None:
    rt = _runtime(tmp_path, [_call("run_turn", run_turn=""), _DONE])

    await rt.run_turn("q")

    (out,) = _outputs(rt)
    assert out.startswith("unknown tool: run_turn.\nThere is no tool named run_turn")
    assert "Did you mean" not in out


@pytest.mark.parametrize(
    ("name", "inp", "echo"),
    [
        ("run_turn", {"run_turn": ""}, True),
        ("run_turn", {"run_turn": None}, True),
        ("run_turn", {"run_turn": "go"}, False),
        ("run_turn", {}, False),
        ("run_turn", {"run_turn": "", "x": 1}, False),
    ],
)
def test_echoes_its_name(name: str, inp: dict[str, object], echo: bool) -> None:
    assert echoes_its_name(name, inp) is echo


async def test_second_self_invocation_gets_a_firmer_reply(tmp_path: Path) -> None:
    scripts = [
        _call("bash", command="uv run satsumaestro --version"),
        _call("bash", command="uv run satsumaestro dashboard --help"),
        _DONE,
    ]
    rt = _runtime(tmp_path, scripts)

    await rt.run_turn("q")

    first, second = _outputs(rt)
    assert first.startswith("self-invocation prohibited:")
    assert second.startswith("self-invocation prohibited:")
    assert PROHIBITED_SELF_INVOCATION_REPEAT not in first
    assert second.splitlines()[1] == PROHIBITED_SELF_INVOCATION_REPEAT
