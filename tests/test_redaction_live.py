"""``redaction.live``: ツールの結果と画面に出す入力の秘密を伏せる。"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console, RenderableType

from satsumaestro.cli.render import RichRenderer, change_preview
from satsumaestro.config.models import HookConfig, RedactionConfig, RuntimeConfig
from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    ToolUse,
    ToolUseDelta,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import NullRenderer
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import PermissionOutcome, default_policy
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.session.redact import redact_shown_input, screen_redactor
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


async def test_the_approval_card_shows_what_runs(ws: Path) -> None:
    command = "API_KEY=$(./payload.sh) make test"
    rt = _runtime(ws, [ToolUse(id="b2", name="bash", input={"command": command})], live=True)
    prompter = _Prompter()
    await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    assert prompter.shown == [{"command": command}]


class _Allow(_Prompter):
    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.shown.append(dict(call.input))
        return PermissionOutcome(decision="allow", reason="yes")


async def test_the_approval_card_hides_a_quoted_password(ws: Path) -> None:
    command = "PASSWORD='ab$private12345' ./login.sh"
    rt = _runtime(ws, [ToolUse(id="b3", name="bash", input={"command": command})], live=True)
    prompter = _Prompter()
    await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    assert prompter.shown == [{"command": "PASSWORD='[redacted]' ./login.sh"}]


async def test_the_command_runs_with_the_real_value(ws: Path) -> None:
    tail = " sh -c 'printf %s \"$PASSWORD\" | wc -c'"
    command = "PASSWORD='ab$private12345'" + tail
    rt = _runtime(ws, [ToolUse(id="b4", name="bash", input={"command": command})], live=True)
    prompter = _Allow()
    await rt.run_turn("go", prompter=prompter)  # type: ignore[arg-type]
    assert prompter.shown == [{"command": "PASSWORD='[redacted]'" + tail}]
    [result] = _results(rt)
    assert "15" in result.output.split()


_CONF = "API_KEY=real-secret-0123456789\nname=demo\n"
_WRITE = {"path": "conf.env", "content": "API_KEY=real-secret-0123456789\nname=demo2\n"}
_EDIT = {"path": "conf.env", "old": "name=demo", "new": "name=demo2"}


def _text(renderable: RenderableType) -> str:
    out = StringIO()
    Console(file=out, width=100, color_system=None).print(renderable)
    return out.getvalue()


@pytest.mark.parametrize(("name", "inp"), [("write_file", _WRITE), ("edit_file", _EDIT)])
def test_change_previews_hide_the_current_contents(
    tmp_path: Path, name: str, inp: dict[str, object]
) -> None:
    (tmp_path / "conf.env").write_text(_CONF, encoding="utf-8")
    call = ToolUse(id="w", name=name, input=redact_shown_input(name, inp))
    preview = change_preview(call, tmp_path, screen_redactor(True))
    assert preview is not None
    text = _text(preview)
    assert "real-secret" not in text and "API_KEY=[redacted]" in text
    assert "- API_KEY" not in text and "+ API_KEY" not in text
    assert "- name=demo" in text and "+ name=demo2" in text


def test_previews_hide_current_values_that_hold_shell_characters(tmp_path: Path) -> None:
    (tmp_path / "conf.env").write_text(
        "DB_PASSWORD=abc$example-secret\nname=demo\n", encoding="utf-8"
    )
    call = ToolUse(id="e", name="edit_file", input=redact_shown_input("edit_file", _EDIT))
    preview = change_preview(call, tmp_path, screen_redactor(True))
    assert preview is not None
    text = _text(preview)
    assert "example-secret" not in text and "DB_PASSWORD=[redacted]" in text


@pytest.mark.parametrize("live", [True, False])
def test_the_renderer_shows_the_preview_redacted_only_when_on(tmp_path: Path, live: bool) -> None:
    (tmp_path / "conf.env").write_text(_CONF, encoding="utf-8")
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=100)
    r = RichRenderer(console, cwd=tmp_path, redact=screen_redactor(live))
    r.start_turn()
    r.on_tool_start(ToolUse(id="e", name="edit_file", input=_EDIT))
    assert ("real-secret" in buf.getvalue()) is not live


def test_the_renderer_does_not_stream_tool_arguments_when_it_redacts() -> None:
    console = Console(file=StringIO(), force_terminal=True, width=70)
    r = RichRenderer(console, stream_tool_args=True, redact=screen_redactor(True))
    r.start_turn()
    args = json.dumps({"path": "a.txt", "content": "streamed-content"})
    r.on_assistant_event(ToolUseDelta(index=0, name="write_file", args_delta=args))
    out = StringIO()
    Console(file=out, force_terminal=False, width=70).print(r._live_renderable())
    assert "streamed-content" not in out.getvalue()


def test_the_screen_redactor_reads_the_keys_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    assert screen_redactor(False) is None
    redact = screen_redactor(True)
    assert redact is not None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "later-provider-key-0123456789")
    assert redact("key later-provider-key-0123456789") == "key [redacted]"
