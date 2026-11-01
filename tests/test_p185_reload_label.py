"""2 ターン目以降でモデルが降りていたら、待ちの途中で照会してロード中を出す。"""

from __future__ import annotations

import asyncio
import io
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from rich.console import Console

from satsumaestro.cli import load_watch
from satsumaestro.cli import repl as repl_mod
from satsumaestro.cli.load_watch import ollama_load_probe, watch_model_load, watching_model_load
from satsumaestro.cli.render import RichRenderer
from satsumaestro.config.models import RuntimeConfig
from satsumaestro.engine.events import TextDelta, ToolUse
from satsumaestro.providers.detect import model_loaded
from satsumaestro.session.models import ToolResultBlock
from tests.test_repl_turn_failure import _FakeRuntime, _FakeSession, _NoSlash, _summary

_MODEL = "gpt-oss:120b"
_ROOT = "http://ollama.test:11434"


def _ps_client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ps(*names: str) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(200, json={"models": [{"name": n, "model": n} for n in names]})

    return handler


@pytest.mark.parametrize(
    ("listed", "expected"),
    [((_MODEL,), True), (("llama3.1:8b",), False), ((), False)],
)
async def test_model_loaded_reads_api_ps(listed: tuple[str, ...], expected: bool) -> None:
    async with _ps_client(_ps(*listed)) as client:
        assert await model_loaded(_ROOT, _MODEL, client=client) is expected


async def test_latest_tag_is_the_same_model() -> None:
    async with _ps_client(_ps("llama3.1:latest")) as client:
        assert await model_loaded(_ROOT, "llama3.1", client=client) is True


@pytest.mark.parametrize("failure", ["status", "connect", "json"])
async def test_failed_query_is_unknown(failure: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "connect":
            raise httpx.ConnectError("refused")
        if failure == "json":
            return httpx.Response(200, text="not json")
        return httpx.Response(500, json={"error": "boom"})

    async with _ps_client(handler) as client:
        assert await model_loaded(_ROOT, _MODEL, client=client) is None


def _rt(model: str) -> Any:
    return SimpleNamespace(session=SimpleNamespace(model=model), config=RuntimeConfig())


async def test_probe_follows_the_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[tuple[str, str]] = []

    async def fake_loaded(root: str, model: str) -> bool | None:
        asked.append((root, model))
        return model == "llama3.1:8b"

    monkeypatch.setattr(load_watch, "model_loaded", fake_loaded)
    rt = _rt(_MODEL)
    probe = ollama_load_probe(rt, {"OLLAMA_HOST": _ROOT})

    assert await probe() == _MODEL
    rt.session.model = "llama3.1:8b"
    assert await probe() is None
    assert asked == [(_ROOT, _MODEL), (_ROOT, "llama3.1:8b")]


async def test_probe_does_not_ask_other_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_loaded(root: str, model: str) -> bool | None:
        raise AssertionError("Ollama 以外へは照会しない")

    monkeypatch.setattr(load_watch, "model_loaded", fake_loaded)
    assert await ollama_load_probe(_rt("claude-opus-5-5"), {"ANTHROPIC_API_KEY": "k"})() is None
    remote = {"OPENAI_BASE_URL": "https://api.test/v1"}
    assert await ollama_load_probe(_rt(_MODEL), remote)() is None


async def test_probe_treats_an_unknown_answer_as_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_loaded(root: str, model: str) -> bool | None:
        return None

    monkeypatch.setattr(load_watch, "model_loaded", fake_loaded)
    assert await ollama_load_probe(_rt(_MODEL), {"OLLAMA_HOST": _ROOT})() is None


def _renderer() -> RichRenderer:
    return RichRenderer(console=Console(file=io.StringIO(), force_terminal=True, width=100))


def _status(r: RichRenderer) -> str:
    out = io.StringIO()
    Console(file=out, force_terminal=False, width=100).print(r._live_renderable())
    return out.getvalue()


def test_loading_label_in_the_first_wait_of_a_turn() -> None:
    r = _renderer()
    r.start_turn()
    since = r.waiting_since()
    assert since is not None and "考え中…" in _status(r)

    r.on_model_loading(_MODEL, since)

    assert f"{_MODEL} をロード中…" in _status(r)
    assert r.waiting_since() is None
    r.on_assistant_event(TextDelta(text="はい"))
    assert "ロード中" not in _status(r)


def test_loading_label_in_the_wait_after_a_tool() -> None:
    r = _renderer()
    r.start_turn()
    r.on_assistant_event(TextDelta(text="読みます"))
    call = ToolUse(id="t1", name="read_file", input={"path": "a.txt"})
    r.on_tool_start(call)
    assert r.waiting_since() is None
    r.on_tool_result(ToolResultBlock(tool_use_id="t1", tool_name="read_file", output="x"))
    since = r.waiting_since()
    assert since is not None and "次の操作を考え中…" in _status(r)

    r.on_model_loading(_MODEL, since)

    assert f"{_MODEL} をロード中…" in _status(r)


def test_an_answer_for_an_earlier_wait_is_ignored() -> None:
    r = _renderer()
    r.start_turn()
    since = r.waiting_since()
    assert since is not None
    r.on_assistant_event(TextDelta(text="はい"))
    r.on_model_loading(_MODEL, since)
    assert "ロード中" not in _status(r)
    r.on_tool_start(ToolUse(id="t1", name="read_file", input={"path": "a.txt"}))
    r.on_tool_result(ToolResultBlock(tool_use_id="t1", tool_name="read_file", output="x"))
    assert r.waiting_since() != since
    r.on_model_loading(_MODEL, since)
    assert "ロード中" not in _status(r)


def test_loading_label_is_not_carried_to_the_next_turn() -> None:
    r = _renderer()
    r.start_turn()
    since = r.waiting_since()
    assert since is not None
    r.on_model_loading(_MODEL, since)
    r.end_turn()
    r.start_turn()
    assert "考え中…" in _status(r) and "ロード中" not in _status(r)
    assert r.waiting_since() is not None


class _Waits:
    """待ちの区間を手で進める描画の代役。"""

    def __init__(self, since: float | None) -> None:
        self.since = since
        self.loading: list[tuple[str, float]] = []

    def waiting_since(self) -> float | None:
        return self.since

    def on_model_loading(self, model: str, since: float) -> None:
        self.loading.append((model, since))


class _Probe:
    def __init__(self, *answers: str | None | Exception) -> None:
        self._answers = list(answers)
        self.calls = 0

    async def __call__(self) -> str | None:
        self.calls += 1
        answer = self._answers.pop(0) if self._answers else None
        if isinstance(answer, Exception):
            raise answer
        return answer


async def _watch(render: _Waits, probe: _Probe, *, after_s: float, steps: list[Any]) -> None:
    task = asyncio.create_task(watch_model_load(render, probe, after_s=after_s, poll_s=0.001))
    try:
        for step in [None, *steps]:
            if step is not None:
                step()
            await asyncio.sleep(0.02)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_a_short_wait_is_not_queried() -> None:
    render, probe = _Waits(time.monotonic()), _Probe(_MODEL)
    await _watch(render, probe, after_s=60.0, steps=[])
    assert probe.calls == 0 and render.loading == []


async def test_a_long_wait_is_queried_once() -> None:
    since = time.monotonic() - 10
    render, probe = _Waits(since), _Probe(_MODEL, _MODEL)
    await _watch(render, probe, after_s=3.0, steps=[])
    assert probe.calls == 1
    assert render.loading == [(_MODEL, since)]


async def test_each_wait_is_queried_again() -> None:
    first = time.monotonic() - 10
    render, probe = _Waits(first), _Probe(None, Exception("boom"), _MODEL)
    second, third = first + 1, first + 2

    def move(since: float | None) -> Any:
        def step() -> None:
            render.since = since

        return step

    await _watch(render, probe, after_s=3.0, steps=[move(second), move(None), move(third)])
    assert probe.calls == 3
    assert render.loading == [(_MODEL, third)]


async def test_no_probe_no_task() -> None:
    before = len(asyncio.all_tasks())
    async with watching_model_load(_Waits(None), None):
        assert len(asyncio.all_tasks()) == before


async def test_the_watcher_stops_with_the_block() -> None:
    before = asyncio.all_tasks()
    async with watching_model_load(_Waits(None), _Probe()):
        started = asyncio.all_tasks() - before
        assert len(started) == 1
    (task,) = started
    assert task.cancelled()


class _SlowRuntime(_FakeRuntime):
    async def run_turn(self, line: str, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.05)
        return await super().run_turn(line, *args, **kwargs)


class _WaitingRender(_Waits):
    def __init__(self) -> None:
        super().__init__(time.monotonic() - 10)
        self.turns = 0

    def start_turn(self) -> None:
        self.turns += 1

    def end_turn(self, *, delegated: object = None) -> None:
        self.since = None


async def test_repl_watches_only_while_a_turn_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _FakeSession(["こんにちは"]))
    monkeypatch.setattr(repl_mod, "SlashDispatcher", lambda *a, **k: _NoSlash())
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    before = asyncio.all_tasks()
    render, probe = _WaitingRender(), _Probe(_MODEL)
    await repl_mod.run_repl(
        _SlowRuntime([_summary("こんにちは")]),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        render,  # type: ignore[arg-type]
        Console(file=io.StringIO()),
        load_probe=probe,
    )
    assert probe.calls == 1
    assert [model for model, _ in render.loading] == [_MODEL]
    assert asyncio.all_tasks() - before == set()


async def test_repl_without_a_probe_does_not_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _FakeSession(["こんにちは"]))
    monkeypatch.setattr(repl_mod, "SlashDispatcher", lambda *a, **k: _NoSlash())
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    render = _WaitingRender()
    await repl_mod.run_repl(
        _SlowRuntime([_summary("こんにちは")]),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        render,  # type: ignore[arg-type]
        Console(file=io.StringIO()),
    )
    assert render.loading == []
