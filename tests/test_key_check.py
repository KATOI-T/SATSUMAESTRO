"""クラウドの鍵をトークンを使わない要求で確かめ、断られた鍵を REPL と doctor に出す。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console

import satsumaestro.cli.app as cli_app
from satsumaestro.cli import repl as repl_mod
from satsumaestro.cli.commands import SlashDispatcher, parse_slash, provider_notes
from satsumaestro.cli.completion import SlashCompleter
from satsumaestro.cli.completion_menu import SuggestionMenu
from satsumaestro.config.models import GuidanceConfig, ProviderConfig, RuntimeConfig
from satsumaestro.providers.key_check import KeyCheck, KeyChecks, check_keys
from tests.fixtures.mock_provider import MockProvider
from tests.test_p165_plan_steps import _runtime
from tests.test_p170_slash_path_passthrough import _FakeRender
from tests.test_recommended_models import _local_runtime

_KEYS = {"OPENAI_API_KEY": "sk-o", "ANTHROPIC_API_KEY": "sk-ant-a", "XAI_API_KEY": "xai-x"}
_INVALID_ANTHROPIC = {
    "type": "error",
    "error": {"type": "authentication_error", "message": "invalid x-api-key"},
}
_INVALID_XAI = {
    "code": "invalid-argument",
    "error": "Incorrect API key provided. You can obtain an API key from https://console.x.ai.",
}


def _transport(
    answers: dict[str, httpx.Response | Exception], seen: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        answer = answers[request.url.host]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return httpx.MockTransport(handle)


def _rejecting() -> httpx.MockTransport:
    return _transport(
        {
            "api.openai.com": httpx.Response(200, json={"object": "list", "data": []}),
            "api.anthropic.com": httpx.Response(401, json=_INVALID_ANTHROPIC),
            "api.x.ai": httpx.Response(400, json=_INVALID_XAI),
        }
    )


async def test_each_key_is_checked_with_a_request_that_spends_no_tokens() -> None:
    seen: list[httpx.Request] = []
    transport = _transport(
        {
            "api.openai.com": httpx.Response(200, json={"object": "list", "data": []}),
            "api.anthropic.com": httpx.Response(401, json=_INVALID_ANTHROPIC),
            "api.x.ai": httpx.Response(400, json=_INVALID_XAI),
        },
        seen,
    )
    found = await check_keys(_KEYS, transport=transport)
    assert found == {
        "openai": KeyCheck("openai", False, 200),
        "anthropic": KeyCheck("anthropic", True, 401, "invalid x-api-key"),
        "xai": KeyCheck("xai", True, 400, _INVALID_XAI["error"]),
    }
    sent = {(r.method, str(r.url)) for r in seen}
    assert sent == {
        ("GET", "https://api.openai.com/v1/models"),
        ("GET", "https://api.anthropic.com/v1/models"),
        ("GET", "https://api.x.ai/v1/api-key"),
    }
    by_host = {r.url.host: r.headers for r in seen}
    assert by_host["api.openai.com"]["authorization"] == "Bearer sk-o"
    assert by_host["api.anthropic.com"]["x-api-key"] == "sk-ant-a"
    assert by_host["api.anthropic.com"]["anthropic-version"] == "2023-06-01"
    assert by_host["api.x.ai"]["authorization"] == "Bearer xai-x"


async def test_a_blocked_xai_key_is_rejected_with_the_names_of_the_blocks() -> None:
    body = {"api_key_blocked": False, "api_key_disabled": True, "team_blocked": True}
    transport = _transport({"api.x.ai": httpx.Response(200, json=body)})
    found = await check_keys({"XAI_API_KEY": "xai-x"}, transport=transport)
    assert found == {"xai": KeyCheck("xai", True, 200, "api_key_disabled, team_blocked")}


async def test_only_set_keys_are_checked_and_unclear_answers_are_left_out() -> None:
    seen: list[httpx.Request] = []
    transport = _transport(
        {
            "api.openai.com": httpx.Response(500, text="oops"),
            "api.anthropic.com": httpx.ConnectError("down"),
        },
        seen,
    )
    env = {"OPENAI_API_KEY": "sk-o", "ANTHROPIC_API_KEY": "sk-ant-a", "XAI_API_KEY": ""}
    assert await check_keys(env, transport=transport) == {}
    assert {r.url.host for r in seen} == {"api.openai.com", "api.anthropic.com"}
    assert await check_keys({}, transport=transport) == {}


async def test_only_rejected_keys_are_kept() -> None:
    checks = KeyChecks(_rejecting())
    assert checks.rejected() == {}
    await checks.run(_KEYS)
    assert set(checks.rejected()) == {"anthropic", "xai"}


class _SlowLines:
    """行を返す前に少し待つ入力欄の代役（裏の確かめが先に終わる）。"""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def prompt_async(self, *_: object, **__: object) -> str:
        await asyncio.sleep(0.05)
        if not self._lines:
            raise EOFError
        return self._lines.pop(0)


async def test_the_repl_checks_at_startup_and_shows_it_only_in_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in _KEYS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _SlowLines(["/provider"]))
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    cfg = RuntimeConfig(
        provider=ProviderConfig(kind="anthropic"), guidance=GuidanceConfig(step_budget=False)
    )
    rt = _runtime(MockProvider([]), tmp_path, cfg=cfg)
    rt.session.model = "claude-sonnet-5-5"
    console = Console(record=True, width=200)
    checks = KeyChecks(_rejecting())
    await repl_mod.run_repl(  # type: ignore[arg-type]
        rt, None, _FakeRender(), console, key_checks=checks
    )
    out = console.export_text()
    assert set(checks.rejected()) == {"anthropic", "xai"}
    assert "受け付けませんでした" not in out
    assert "5) anthropic ✓  無効な鍵（ANTHROPIC_API_KEY）" in out


async def test_the_provider_list_and_completion_mark_rejected_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in _KEYS.items():
        monkeypatch.setenv(name, value)
    checks = KeyChecks(_rejecting())
    await checks.run(_KEYS)
    rt = _local_runtime(tmp_path)
    console = Console(record=True, width=200)
    await SlashDispatcher(rt, console, key_checks=checks).dispatch(  # type: ignore[arg-type]
        parse_slash("/provider")
    )
    out = console.export_text(styles=True)
    assert "\x1b[31m無効な鍵（ANTHROPIC_API_KEY）" in out
    assert "\x1b[31m無効な鍵（XAI_API_KEY）" in out
    assert "鍵あり（OPENAI_API_KEY）" in out

    rejected = checks.rejected()
    notes = provider_notes(rt.config.provider, rt.session.model, _KEYS, rejected)
    assert notes["anthropic"] == "Anthropic · 無効な鍵（ANTHROPIC_API_KEY）"
    assert notes["openai"] == "OpenAI · 鍵あり（OPENAI_API_KEY）"
    completer = SlashCompleter(providers=lambda: notes, rejected=lambda: rejected)
    text = "/provider "
    shown = list(completer.get_completions(Document(text), CompleteEvent()))
    styles = {c.text.strip(): c.style for c in shown}
    assert styles["anthropic"] == styles["xai"] == "fg:ansired"
    assert styles["openai"] == styles["ollama"] == ""

    buffer = Buffer()
    buffer.document = Document(text)
    buffer.complete_state = CompletionState(Document(text), shown)
    rows = SuggestionMenu(buffer, rows=8).lines(120)
    meta = {row[1][1].strip(): row[-1][0] for row in rows}
    assert "fg:ansired" in meta["anthropic"] and "fg:ansired" not in meta["openai"]


def test_doctor_probe_checks_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, str]] = []
    answer: dict[str, KeyCheck] = {}

    async def fake(env: dict[str, str], **_: object) -> dict[str, KeyCheck]:
        calls.append(dict(env))
        return answer

    monkeypatch.setattr(cli_app, "check_keys", fake)
    cfg = RuntimeConfig(provider=ProviderConfig(kind="anthropic"))
    env = {"ANTHROPIC_API_KEY": "sk-ant-a"}
    model = "claude-sonnet-5-5"
    assert cli_app._key_status(cfg, model, env) == ("ok", "ANTHROPIC_API_KEY set")
    assert calls == []
    answer["anthropic"] = KeyCheck("anthropic", True, 401, "invalid x-api-key")
    assert cli_app._key_status(cfg, model, env, probe=True) == (
        "warn",
        "Anthropic が API の鍵を受け付けませんでした（401）: invalid x-api-key",
    )
    assert calls == [env]
    answer["anthropic"] = KeyCheck("anthropic", False, 200)
    assert cli_app._key_status(cfg, model, env, probe=True) == (
        "ok",
        "ANTHROPIC_API_KEY あり（受け付けられました）",
    )
    answer.clear()
    assert cli_app._key_status(cfg, model, env, probe=True) == (
        "warn",
        "ANTHROPIC_API_KEY あり（確かめられませんでした）",
    )
    assert len(calls) == 3
