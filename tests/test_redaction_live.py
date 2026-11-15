"""``redaction.live``: ツールの結果と画面に出す入力の秘密を伏せる。"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from satsumaestro.cli.repl import InteractivePrompter
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
from satsumaestro.util.proc import BackgroundProcessRegistry
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


def _runtime(
    ws: Path,
    calls: list[ToolUse],
    *,
    live: bool,
    background: BackgroundProcessRegistry | None = None,
) -> ConversationRuntime:
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
        background=background,
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


async def test_a_quoted_passphrase_reaches_the_model_redacted(ws: Path) -> None:
    (ws / "settings.txt").write_text(
        "DB_PASSWORD='correct horse battery staple'\n", encoding="utf-8"
    )
    read = ToolUse(id="r2", name="read_file", input={"path": "settings.txt"})
    rt = _runtime(ws, [read], live=True)
    await rt.run_turn("go")
    [result] = _results(rt)
    assert "DB_PASSWORD='[redacted]'" in result.output
    assert "horse" not in result.output and "staple" not in result.output


async def test_tool_results_stay_as_they_are_when_off(ws: Path) -> None:
    rt = _runtime(ws, [_READ], live=False)
    render = _Render()
    await rt.run_turn("go", render=render)
    [result] = _results(rt)
    assert _TOKEN in result.output and _KEY in result.output
    assert _TOKEN in render.results[0]


_PASSWORD = ToolUse(
    id="b3", name="bash", input={"command": "PASSWORD='ab$private12345' ./login.sh"}
)


@pytest.mark.parametrize("live", [True, False])
@pytest.mark.parametrize("call", [_CURL, _PASSWORD], ids=["bearer", "quoted-password"])
async def test_the_prompter_and_the_renderer_get_the_real_call(
    ws: Path, live: bool, call: ToolUse
) -> None:
    rt = _runtime(ws, [call], live=live)
    render, prompter = _Render(), _Prompter()
    await rt.run_turn("go", prompter=prompter, render=render)  # type: ignore[arg-type]
    assert prompter.shown == [call.input]
    assert render.started == [call.input]
    assert call.input["command"] in str(rt.session.messages[1].blocks[0].model_dump())


@pytest.mark.parametrize("live", [True, False])
async def test_the_card_hides_the_password_and_the_command_runs_with_it(
    ws: Path, monkeypatch: pytest.MonkeyPatch, live: bool
) -> None:
    tail = " sh -c 'printf %s \"$PASSWORD\" | wc -c'"
    command = "PASSWORD='ab$private12345'" + tail
    rt = _runtime(ws, [ToolUse(id="b4", name="bash", input={"command": command})], live=live)
    buf = StringIO()
    prompter = InteractivePrompter(Console(file=buf, width=200), redact=live)
    monkeypatch.setattr(prompter.console, "input", lambda *a, **k: "y")
    await rt.run_turn("go", prompter=prompter)
    shown = "PASSWORD='[redacted]'" if live else "PASSWORD='ab$private12345'"
    assert shown + tail in buf.getvalue()
    assert ("private12345" in buf.getvalue()) is not live
    [result] = _results(rt)
    assert "15" in result.output.split()


async def _finished(ws: Path, command: str) -> BackgroundProcessRegistry:
    registry = BackgroundProcessRegistry(ws / ".satsuma" / "background" / "sess")
    task = await registry.spawn(command, cwd=ws, env={})
    assert task.waiter is not None
    await task.waiter
    return registry


_GLOB = ToolUse(id="g1", name="glob_search", input={"pattern": "*.none"})


@pytest.mark.parametrize("live", [True, False])
@pytest.mark.parametrize(
    ("command", "secret"),
    [(f"API_TOKEN={_BEARER} true", _BEARER), (f"echo {_KEY}", _KEY)],
    ids=["assignment", "provider-key"],
)
async def test_background_completion_notes_follow_the_setting(
    ws: Path, live: bool, command: str, secret: str
) -> None:
    registry = await _finished(ws, command)
    rt = _runtime(ws, [_GLOB], live=live, background=registry)
    await rt.run_turn("go")
    [result] = _results(rt)
    _, note = result.output.split("<system-reminder>", 1)
    assert "bg-1" in note and "exited with code 0" in note
    assert (secret in note) is not live
    assert ("[redacted]" in note) is live
    await registry.aclose()
