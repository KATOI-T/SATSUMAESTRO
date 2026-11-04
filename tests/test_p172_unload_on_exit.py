"""REPL を閉じるとき、このプロセスが Ollama で使ったモデルを降ろす。"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import httpx
import pytest
from openai import APIConnectionError
from rich.console import Console

from satsumaestro.cli import app as cli_app
from satsumaestro.cli.repl import InteractivePrompter
from satsumaestro.config.models import RuntimeConfig
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.detect import note_model_used, unload_model, used_models
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.util.errors import ProviderError

_ROOT = "http://ollama.test:11434"
_MODEL = "gpt-oss:120b"


async def test_unload_sends_keep_alive_zero() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"done": True, "done_reason": "unload"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await unload_model(_ROOT, _MODEL, client=client) is True
    assert seen == [("/api/generate", {"model": _MODEL, "keep_alive": 0})]


@pytest.mark.parametrize("failure", ["status", "connect"])
async def test_failed_unload_does_not_raise(failure: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "connect":
            raise httpx.ConnectError("refused")
        return httpx.Response(404, json={"error": "model not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await unload_model(_ROOT, _MODEL, client=client) is False


def _chunks() -> list[NS]:
    return [
        NS(choices=[NS(delta=NS(content="hi", tool_calls=None), finish_reason="stop")], usage=None),
        NS(choices=[], usage=NS(prompt_tokens=1, completion_tokens=1)),
    ]


async def _aiter(items: list[NS]) -> AsyncIterator[NS]:
    for it in items:
        yield it


def _provider(
    monkeypatch: pytest.MonkeyPatch, *, ollama_root: str | None, fail: bool = False
) -> OpenAICompatProvider:
    provider = OpenAICompatProvider("k", f"{_ROOT}/v1", ollama_root=ollama_root, keep_alive=None)

    async def fake_create(**kwargs: Any) -> AsyncIterator[NS]:
        if fail:
            raise APIConnectionError(request=httpx.Request("POST", f"{_ROOT}/v1/chat/completions"))
        return _aiter(_chunks())

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)
    return provider


async def _drain(provider: OpenAICompatProvider, model: str) -> None:
    req = ApiRequest(model=model, system="s", messages=[], tools=[])
    async for _ in provider.stream(req):
        pass


async def test_models_used_against_ollama_are_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(monkeypatch, ollama_root=_ROOT)
    await _drain(provider, _MODEL)
    await _drain(provider, "llama3.1:8b")
    await _drain(provider, _MODEL)
    assert used_models() == [(_ROOT, _MODEL), (_ROOT, "llama3.1:8b")]


async def test_other_endpoints_and_failed_requests_are_not_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _drain(_provider(monkeypatch, ollama_root=None), _MODEL)
    with pytest.raises(ProviderError):
        await _drain(_provider(monkeypatch, ollama_root=_ROOT, fail=True), _MODEL)
    assert used_models() == []


@pytest.fixture
def unloads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    async def fake_unload(root: str, model: str) -> bool:
        calls.append((root, model))
        return model != "broken:1b"

    monkeypatch.setattr(cli_app, "unload_model", fake_unload)
    return calls


@pytest.fixture
def screen(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    out = io.StringIO()
    monkeypatch.setattr(cli_app, "_console", Console(file=out, width=200))
    return out


async def test_every_used_model_is_unloaded(
    unloads: list[tuple[str, str]], screen: io.StringIO
) -> None:
    note_model_used(_ROOT, _MODEL)
    note_model_used(_ROOT + "/", "llama3.1:8b")
    await cli_app._unload_on_exit(RuntimeConfig())
    assert unloads == [(_ROOT, _MODEL), (_ROOT, "llama3.1:8b")]
    assert screen.getvalue().strip() == (
        f"Ollama から {_MODEL}, llama3.1:8b を降ろしました"
        "（残すなら provider.ollama.unload_on_exit: false）"
    )


async def test_setting_off_keeps_the_models(
    unloads: list[tuple[str, str]], screen: io.StringIO
) -> None:
    note_model_used(_ROOT, _MODEL)
    cfg = RuntimeConfig.model_validate({"provider": {"ollama": {"unload_on_exit": False}}})
    await cli_app._unload_on_exit(cfg)
    assert unloads == [] and screen.getvalue() == ""


async def test_nothing_used_nothing_sent(
    unloads: list[tuple[str, str]], screen: io.StringIO
) -> None:
    await cli_app._unload_on_exit(RuntimeConfig())
    assert unloads == [] and screen.getvalue() == ""


async def test_only_unloaded_models_are_reported(
    unloads: list[tuple[str, str]], screen: io.StringIO
) -> None:
    note_model_used(_ROOT, "broken:1b")
    await cli_app._unload_on_exit(RuntimeConfig())
    assert unloads == [(_ROOT, "broken:1b")] and screen.getvalue() == ""


def test_default_is_on() -> None:
    assert RuntimeConfig().provider.ollama.unload_on_exit is True


class _Rt:
    def __init__(self) -> None:
        self.config = RuntimeConfig()
        self.mcp = self.rag = self.orchestrator = self.background = self.lsp = None


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> tuple[_Rt, list[RuntimeConfig]]:
    rt = _Rt()
    calls: list[RuntimeConfig] = []

    async def fake_assemble(*args: Any, **kwargs: Any) -> Any:
        return rt, InteractivePrompter(Console(file=io.StringIO())), NS(start_turn=lambda: None)

    async def fake_run_repl(*args: Any, **kwargs: Any) -> None:
        return None

    async def fake_shutdown(*args: Any) -> None:
        return None

    async def fake_unload_on_exit(cfg: RuntimeConfig) -> None:
        calls.append(cfg)

    monkeypatch.setattr(cli_app, "_assemble_runtime", fake_assemble)
    monkeypatch.setattr(cli_app, "run_repl", fake_run_repl)
    monkeypatch.setattr(cli_app, "_shutdown", fake_shutdown)
    monkeypatch.setattr(cli_app, "_build_telemetry", lambda rt: (None, None))
    monkeypatch.setattr(cli_app, "load_probe", lambda rt, env: None)
    monkeypatch.setattr(cli_app, "_unload_on_exit", fake_unload_on_exit)
    return rt, calls


async def test_repl_exit_unloads(exits: tuple[_Rt, list[RuntimeConfig]], tmp_path: Path) -> None:
    rt, calls = exits
    await cli_app._repl_session(None, tmp_path, trusted=True)
    assert calls == [rt.config]


async def test_resumed_repl_exit_unloads(
    exits: tuple[_Rt, list[RuntimeConfig]], tmp_path: Path
) -> None:
    rt, calls = exits
    await cli_app._run_repl_resume(None, tmp_path, tmp_path / "s.jsonl", trusted=True)
    assert calls == [rt.config]


async def test_repl_unloads_even_when_the_loop_fails(
    exits: tuple[_Rt, list[RuntimeConfig]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt, calls = exits

    async def broken_run_repl(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_app, "run_repl", broken_run_repl)
    with pytest.raises(RuntimeError):
        await cli_app._repl_session(None, tmp_path, trusted=True)
    assert calls == [rt.config]


async def test_one_shot_prompt_keeps_the_model(
    exits: tuple[_Rt, list[RuntimeConfig]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, calls = exits

    async def fake_drain(rt: Any) -> list[dict[str, str]]:
        return []

    class _OneTurn(_Rt):
        async def run_turn(self, *args: Any, **kwargs: Any) -> Any:
            return NS(delegated_usage=None)

    async def assemble_one_turn(*args: Any, **kwargs: Any) -> Any:
        render = NS(start_turn=lambda: None, end_turn=lambda **_: None)
        return _OneTurn(), None, render

    monkeypatch.setattr(cli_app, "_assemble_runtime", assemble_one_turn)
    monkeypatch.setattr(cli_app, "_drain_background", fake_drain)
    await cli_app._run_once("やって", None, "text", tmp_path, trusted=True)
    assert calls == []
