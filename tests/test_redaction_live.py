"""``redaction.live``: ツールの結果と画面に出す入力の秘密を伏せる。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RedactionConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import NullRenderer
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionOutcome, default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from tests.fixtures.mock_provider import ScriptedFailureProvider

_TOKEN = "ghp_" + "a1B2c3D4e5" * 3 + "f6G7h8"
_KEY = "plain-looking-provider-key-0123456789"
_BEARER = "bearer-value-0123456789"


class _Render(NullRenderer):
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.results: list[str] = []

    def on_tool_start(self, call: ToolUse) -> None:
        self.started.append(dict(call.input))

    def on_tool_result(self, block: ToolResultBlock) -> None:
        self.results.append(block.output)


class _Prompter:
    def __init__(self) -> None:
        self.shown: list[dict[str, object]] = []

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.shown.append(dict(call.input))
        return PermissionOutcome(decision="deny", reason="no")


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ANTHROPIC_API_KEY", _KEY)
    (tmp_path / "notes.txt").write_text(
        f"token {_TOKEN}\nprovider {_KEY}\nplain line\n", encoding="utf-8"
    )
    return tmp_path


def _runtime(ws: Path, calls: list[ToolUse], *, live: bool) -> ConversationRuntime:
    steps: list[BaseException | list[AssistantEvent | BaseException]] = [
        [call, MessageStop()] for call in calls
    ]
    steps.append([TextDelta(text="done"), MessageStop()])
    return ConversationRuntime(
        session=new_session(ws, "mock-model"),
        provider=ScriptedFailureProvider(steps),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), ws),
        config=RuntimeConfig(max_iterations=8, redaction=RedactionConfig(live=live)),
    )


def _results(rt: ConversationRuntime) -> list[ToolResultBlock]:
    return [b for m in rt.session.messages for b in m.blocks if isinstance(b, ToolResultBlock)]


_READ = ToolUse(id="r1", name="read_file", input={"path": "notes.txt"})
_CURL = ToolUse(
    id="b1",
    name="bash",
    input={"command": f"curl -H 'Authorization: Bearer {_BEARER}' https://example.invalid"},
)


def test_it_is_off_by_default() -> None:
    assert RuntimeConfig().redaction.live is False


async def test_tool_results_go_to_the_model_and_the_screen_redacted(ws: Path) -> None:
    rt = _runtime(ws, [_READ], live=True)
    render = _Render()
    await rt.run_turn("go", render=render)
    [result] = _results(rt)
    for text in (result.output, *render.results):
        assert _TOKEN not in text and _KEY not in text
        assert "[redacted]" in text and "plain line" in text


async def test_tool_results_stay_as_they_are_when_off(ws: Path) -> None:
    rt = _runtime(ws, [_READ], live=False)
    render = _Render()
    await rt.run_turn("go", render=render)
    [result] = _results(rt)
    assert _TOKEN in result.output and _KEY in result.output
    assert _TOKEN in render.results[0]


async def test_the_approval_card_and_the_tool_line_show_the_input_redacted(ws: Path) -> None:
    rt = _runtime(ws, [_CURL], live=True)
    render, prompter = _Render(), _Prompter()
    await rt.run_turn("go", prompter=prompter, render=render)  # type: ignore[arg-type]
    shown = [str(d["command"]) for d in (*prompter.shown, *render.started)]
    assert prompter.shown and all(_BEARER not in s and "[redacted]" in s for s in shown)
    assert _BEARER in str(rt.session.messages[1].blocks[0].model_dump())


async def test_the_approval_card_shows_the_input_as_it_is_when_off(ws: Path) -> None:
    rt = _runtime(ws, [_CURL], live=False)
    prompter = _Prompter()
    await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    assert _BEARER in str(prompter.shown[0]["command"])
