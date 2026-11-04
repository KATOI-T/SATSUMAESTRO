"""存在しないモデルを起動時と ``/model`` で弾く。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

import satsumaestro.cli.app as cli_app
from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.config.models import GuidanceConfig, ProviderConfig, RuntimeConfig
from satsumaestro.engine.context_fit import (
    ModelCheck,
    check_model,
    describe_missing_model,
    list_models,
    missing_model_notice,
)
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.session.store import JSONLSessionStore
from tests.fixtures.mock_provider import MockProvider, SequencedMockProvider
from tests.test_p165_plan_steps import _runtime

_ROOT = "http://localhost:11434"
_TAGS = ["gpt-oss:120b", "qwen3-coder-next:q4_K_M", "llama3.1:8b"]
_ENV = {"OLLAMA_HOST": _ROOT}

Handler = Callable[[httpx.Request], httpx.Response]


def _ollama(tags: list[str] = _TAGS, *, show_status: int | None = None) -> Handler:
    """``/api/show`` は tags にある名前だけ 200（``show_status`` で固定もできる）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n, "model": n} for n in tags]})
        if request.url.path == "/api/show":
            name = json.loads(request.content)["model"]
            status = show_status or (200 if name.casefold() in map(str.casefold, tags) else 404)
            if status == 404:
                return httpx.Response(404, json={"error": f"model '{name}' not found"})
            return httpx.Response(status, json={"model_info": {}})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(404)

    return handler


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _cfg() -> RuntimeConfig:
    return RuntimeConfig(provider=ProviderConfig(base_url=f"{_ROOT}/v1"))


async def test_show_404_is_missing_with_the_tag_list() -> None:
    async with _client(_ollama()) as http:
        check = await check_model(_cfg(), "gpt-oss:999b", _ENV, client=http)
    assert check == ModelCheck(_ROOT, missing=True, available=_TAGS)


async def test_show_200_is_present_even_with_other_case() -> None:
    async with _client(_ollama()) as http:
        check = await check_model(_cfg(), "GPT-OSS:120b", _ENV, client=http)
    assert check is not None and not check.missing


@pytest.mark.parametrize("status", [500, 400])
async def test_other_statuses_are_undecided(status: int) -> None:
    async with _client(_ollama(show_status=status)) as http:
        assert await check_model(_cfg(), "gpt-oss:999b", _ENV, client=http) is None


async def test_unreachable_is_undecided() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with _client(down) as http:
        assert await check_model(_cfg(), "gpt-oss:999b", _ENV, client=http) is None
        assert await list_models(_cfg(), "gpt-oss:120b", _ENV, client=http) is None


async def test_not_ollama_is_not_checked() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(base_url="https://api.example.com/v1"))
    async with _client(_ollama()) as http:
        assert await check_model(cfg, "gpt-oss:999b", {}, client=http) is None
        assert await list_models(cfg, "gpt-oss:999b", {}, client=http) is None


def test_describe_missing_model() -> None:
    many = [f"m{i}:1b" for i in range(12)] + ["gpt-oss:120b"]
    text = describe_missing_model("gpt-oss:12b", ModelCheck(_ROOT, missing=True, available=many))
    assert text.startswith(f"モデル 'gpt-oss:12b' が接続先の Ollama（{_ROOT}）に見つかりません。")
    assert "近い名前: gpt-oss:120b" in text
    assert "m9:1b ほか 3 件" in text and "m10:1b" not in text.split("ほか")[0]
    assert "`ollama pull gpt-oss:12b`" in text and "OLLAMA_MODEL" in text
    repl = describe_missing_model("x", ModelCheck(_ROOT, missing=True, available=[]), startup=False)
    assert "利用可能" not in repl and "OLLAMA_MODEL" not in repl and "モデル名を確認" in repl


def test_missing_model_text_colors_only_the_headline_red() -> None:
    """全文を赤にすると読みにくい。"""
    many = ["gpt-oss:120b"] + [f"m{i}:1b" for i in range(12)]
    notice = missing_model_notice("gpt-oss:12b", ModelCheck(_ROOT, missing=True, available=many))
    text = cli_app.missing_model_text(notice, prefix="error: ")

    def styles(part: str) -> set[str]:
        start = text.plain.index(part)
        return {str(s.style) for s in text.spans if s.start <= start < s.end}

    assert text.plain.splitlines()[0] == "error: " + notice.headline
    assert str(text.style) == ""
    assert styles("error: モデル") == {"red"}
    assert styles("近い名前:") == {"yellow"} and styles("利用可能:") == {"yellow"}
    assert styles("ollama pull gpt-oss:12b") == {"yellow"}
    for plain in ("gpt-oss:120b, m0:1b", "ほか 3 件", "で取得するか"):
        assert styles(plain) == set()
    assert "--model / .env の OLLAMA_MODEL" in text.plain


def _route_http(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    real = httpx.AsyncClient

    def fake(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake)
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)


def _patch_provider(monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> None:
    def build(config: RuntimeConfig, model: str, env: object) -> MockProvider:
        return MockProvider([TextDelta(text="ok"), MessageStop()])

    async def fit(cfg: RuntimeConfig, model: str, env: object) -> None:
        seen.append(model)

    monkeypatch.setattr(cli_app, "build_provider", build)
    monkeypatch.setattr(cli_app, "apply_context_fit", fit)


def test_startup_stops_before_the_context_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitted: list[str] = []
    _patch_provider(monkeypatch, fitted)
    _route_http(monkeypatch, _ollama())
    result = CliRunner().invoke(
        cli_app.app, ["prompt", "hi", "-m", "gpt-oss:999b", "--cwd", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "モデル 'gpt-oss:999b' が接続先の Ollama" in result.output
    assert "近い名前: gpt-oss:120b" in result.output
    assert fitted == []


def test_repl_startup_prints_the_notice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_provider(monkeypatch, [])
    _route_http(monkeypatch, _ollama())
    result = CliRunner().invoke(cli_app.app, ["repl", "-m", "gpt-oss:999b", "--cwd", str(tmp_path)])
    assert result.exit_code == 1
    out = result.output.replace("\n", "")
    assert "error: モデル 'gpt-oss:999b' が接続先の Ollama" in out
    assert "近い名前: gpt-oss:120b" in out
    assert "ollama pull gpt-oss:999b で取得するか" in out


def test_startup_with_an_existing_model_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitted: list[str] = []
    _patch_provider(monkeypatch, fitted)
    _route_http(monkeypatch, _ollama())
    result = CliRunner().invoke(
        cli_app.app, ["prompt", "hi", "-m", "gpt-oss:120b", "--cwd", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert fitted == ["gpt-oss:120b"]


async def test_resume_replaces_a_missing_saved_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_provider(monkeypatch, [])
    _route_http(monkeypatch, _ollama())
    store = JSONLSessionStore()
    rt, _, _ = await cli_app._assemble_runtime(
        "gpt-oss:120b", tmp_path, interactive=False, trusted=True
    )
    rt.session.model = "gpt-oss:999b"
    store.save(rt.session)
    path = store.path_for(tmp_path, rt.session.session_id)
    resumed, _, _ = await cli_app._assemble_runtime(
        "gpt-oss:120b", tmp_path, interactive=False, resume_path=path, trusted=True
    )
    assert resumed.session.model == "gpt-oss:120b"


def _dispatch_runtime(tmp_path: Path) -> Any:
    cfg = RuntimeConfig(
        provider=ProviderConfig(base_url=f"{_ROOT}/v1"),
        guidance=GuidanceConfig(step_budget=False),
    )
    rt = _runtime(SequencedMockProvider([[MessageStop()]]), tmp_path, cfg=cfg)
    rt.session.model = "gpt-oss:120b"
    return rt


async def _model(rt: Any, line: str) -> str:
    console = Console(record=True, width=200)
    await SlashDispatcher(rt, console).dispatch(parse_slash(line))  # type: ignore[arg-type]
    return console.export_text()


async def test_model_switch_to_a_missing_name_keeps_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_http(monkeypatch, _ollama())
    rt = _dispatch_runtime(tmp_path)
    out = await _model(rt, "/model gpt-oss:999b")
    assert rt.session.model == "gpt-oss:120b"
    assert "見つかりません" in out and "近い名前: gpt-oss:120b" in out
    assert "gpt-oss:120b のままです" in out and "model →" not in out


async def test_model_switch_to_an_existing_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_http(monkeypatch, _ollama())
    rt = _dispatch_runtime(tmp_path)
    out = await _model(rt, "/model qwen3-coder-next:q4_K_M")
    assert rt.session.model == "qwen3-coder-next:q4_K_M" and "model →" in out


async def test_model_switch_when_undecided_switches_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_http(monkeypatch, _ollama(show_status=500))
    rt = _dispatch_runtime(tmp_path)
    await _model(rt, "/model gpt-oss:999b")
    assert rt.session.model == "gpt-oss:999b"


async def test_model_without_args_lists_the_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_http(monkeypatch, _ollama())
    rt = _dispatch_runtime(tmp_path)
    out = await _model(rt, "/model")
    assert "model: gpt-oss:120b" in out
    assert "* gpt-oss:120b" in out and "  qwen3-coder-next:q4_K_M" in out
