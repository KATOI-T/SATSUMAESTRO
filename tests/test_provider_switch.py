"""会話の途中でプロバイダを切り替える（``switch_provider`` と ``/provider``）。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

import satsumaestro.cli.app  # noqa: F401
from satsumaestro.cli.commands import SlashDispatcher, parse_slash, provider_args
from satsumaestro.config.models import HookConfig, ProviderConfig, RuntimeConfig
from satsumaestro.engine.provider_switch import (
    MissingModelError,
    ModelNeededError,
    switch_provider,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.subagent import RuntimeFactory
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers.anthropic_provider import _to_wire
from satsumaestro.providers.base import Provider
from satsumaestro.providers.detect import note_model_used, used_models
from satsumaestro.providers.local import Unloaded, unload_used
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ConversationMessage, ToolResultBlock, ToolUseBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ConfigError
from tests.fixtures.mock_provider import MockProvider

_LLAMA = "http://127.0.0.1:9931"
_OLLAMA = "http://10.0.0.9:11434"
_TAGS = ["gpt-oss:120b", "qwen3:8b"]
_ENV = {"OLLAMA_HOST": "10.0.0.9"}


class _Servers:
    """Ollama と 1 つのモデルの llama-server の擬似。"""

    def __init__(self) -> None:
        self.unloaded: list[str] = []
        self.down: set[str] = set()

    def __call__(self, req: httpx.Request) -> httpx.Response:
        host = f"{req.url.host}:{req.url.port}"
        if host in self.down:
            raise httpx.ConnectError("Connection refused")
        if req.url.port == 9931:
            body = {
                "default_generation_settings": {"params": {}, "n_ctx": 65536},
                "total_slots": 1,
                "model_alias": "gpt-oss-20b",
                "model_path": "/models/gpt-oss-20b.gguf",
            }
            return httpx.Response(200, json=body)
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n, "model": n} for n in _TAGS]})
        if req.url.path == "/api/show":
            name = json.loads(req.content)["model"]
            if name not in _TAGS:
                return httpx.Response(404, json={"error": f"model '{name}' not found"})
            return httpx.Response(200, json={"model_info": {}})
        if req.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if req.url.path == "/api/generate":
            self.unloaded.append(json.loads(req.content)["model"])
            return httpx.Response(200, json={})
        return httpx.Response(404)


@pytest.fixture
def servers(monkeypatch: pytest.MonkeyPatch) -> _Servers:
    fake = _Servers()
    real = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(fake)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return fake


class _Decomposition:
    def __init__(self) -> None:
        self.used: list[tuple[Provider, str]] = []

    async def before_turn(self, user_text: str) -> str | None:
        return None

    def take_record(self) -> None:
        return None

    def use_provider(self, provider: Provider, model: str) -> None:
        self.used.append((provider, model))


Built = list[tuple[str, str]]


def _builder(built: Built) -> Callable[[RuntimeConfig, str, object], Provider]:
    def build(cfg: RuntimeConfig, model: str, env: object) -> Provider:
        built.append((cfg.provider.kind, model))
        return MockProvider([])  # type: ignore[return-value]

    return build


def _runtime(tmp_path: Path, cfg: RuntimeConfig, model: str) -> ConversationRuntime:
    provider = MockProvider([])
    tools = default_registry()
    policy = default_policy(PermissionMode.WORKSPACE_WRITE)
    hooks = HookRunner(HookConfig(), tmp_path)
    factory = RuntimeFactory(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        config=cfg,
        tools=tools,
        policy=policy,
        hooks=hooks,
        cwd=tmp_path,
        provider_builder=lambda m: MockProvider([]),  # type: ignore[arg-type,return-value]
    )
    return ConversationRuntime(
        session=new_session(tmp_path, model),
        provider=provider,  # type: ignore[arg-type]
        tools=tools,
        policy=policy,
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=hooks,
        config=cfg,
        runtime_factory=factory,
        decomposition=_Decomposition(),
    )


def _llamacpp_cfg() -> RuntimeConfig:
    provider = ProviderConfig(base_url=f"{_LLAMA}/v1", local={"server": "llamacpp"})
    return RuntimeConfig(provider=provider)


async def test_llamacpp_to_ollama_and_back(servers: _Servers, tmp_path: Path) -> None:
    rt = _runtime(tmp_path, _llamacpp_cfg(), "gpt-oss-20b")
    built: Built = []
    build = _builder(built)
    compactor = rt.compactor

    with pytest.raises(ModelNeededError) as needed:
        await switch_provider(rt, "ollama", env=_ENV, build=build)
    assert needed.value.models == _TAGS and rt.session.model == "gpt-oss-20b" and built == []

    first = await switch_provider(rt, "ollama", env=_ENV, model="gpt-oss:120b", build=build)
    assert (first.choice, first.model, first.previous) == ("ollama", "gpt-oss:120b", "llamacpp")
    assert first.base_url == f"{_OLLAMA}/v1" and first.endpoint.source == "OLLAMA_HOST"  # type: ignore[union-attr]
    assert first.remember["llamacpp"].base_url == f"{_LLAMA}/v1"
    assert first.remember["llamacpp"].model == "gpt-oss-20b"
    provider = rt.config.provider
    assert (provider.kind, provider.base_url, provider.default_model) == (
        "ollama",
        f"{_OLLAMA}/v1",
        "gpt-oss:120b",
    )
    assert built == [("ollama", "gpt-oss:120b")]
    assert rt.session.model == "gpt-oss:120b" and rt.compactor is not compactor
    factory = rt._runtime_factory
    assert factory is not None and factory.provider is rt.provider
    assert factory.model == "gpt-oss:120b"
    assert rt.decomposition.used == [(rt.provider, "gpt-oss:120b")]  # type: ignore[union-attr]
    assert first.unloaded == ()

    note_model_used(_OLLAMA, "gpt-oss:120b")
    back = await switch_provider(rt, "llamacpp", env=_ENV, build=build)
    assert (back.model, back.endpoint.source) == ("gpt-oss-20b", "profile")  # type: ignore[union-attr]
    assert rt.config.provider.base_url == f"{_LLAMA}/v1"
    assert back.unloaded == (Unloaded("ollama", ("gpt-oss:120b",)),)
    assert servers.unloaded == ["gpt-oss:120b"] and used_models() == []

    again = await switch_provider(rt, "ollama", env={}, build=build)
    assert (again.model, again.base_url) == ("gpt-oss:120b", f"{_OLLAMA}/v1")
    assert again.unloaded == ()


async def test_a_failed_switch_changes_nothing(servers: _Servers, tmp_path: Path) -> None:
    cfg = _llamacpp_cfg()
    rt = _runtime(tmp_path, cfg, "gpt-oss-20b")
    provider, settings, built = rt.provider, cfg.provider, []
    build = _builder(built)
    servers.down.add("10.0.0.9:11434")
    cases: list[tuple[str, dict[str, Any], str]] = [
        ("anthropic", {"model": "claude-sonnet-5-5"}, "ANTHROPIC_API_KEY が設定されていません"),
        ("lmstudio", {}, "LMSTUDIO_HOST"),
        ("ollama", {"model": "gpt-oss:120b"}, "10.0.0.9:11434"),
        ("xai", {"url": "10.0.0.5"}, "URL"),
        ("nope", {}, "nope"),
    ]
    for choice, kw, text in cases:
        with pytest.raises(ConfigError, match=text):
            await switch_provider(rt, choice, env=_ENV, build=build, **kw)
    servers.down.clear()
    with pytest.raises(MissingModelError):
        await switch_provider(rt, "ollama", env=_ENV, model="nope:1b", build=build)
    with pytest.raises(ModelNeededError):
        await switch_provider(rt, "anthropic", env={"ANTHROPIC_API_KEY": "k"}, build=build)
    assert rt.provider is provider and rt.config.provider is settings
    assert rt.session.model == "gpt-oss-20b" and settings.kind == "auto"
    assert built == [("ollama", "nope:1b")]


async def test_a_cloud_keeps_the_local_url_in_the_settings(
    servers: _Servers, tmp_path: Path
) -> None:
    rt = _runtime(tmp_path, _llamacpp_cfg(), "gpt-oss-20b")
    env = {"ANTHROPIC_API_KEY": "k"}
    result = await switch_provider(
        rt, "anthropic", env=env, model="claude-sonnet-5-5", build=_builder([])
    )
    assert result.base_url is None and result.endpoint is None
    assert rt.config.provider.kind == "anthropic"
    assert rt.config.provider.base_url == f"{_LLAMA}/v1"
    assert rt.config.provider.profiles["llamacpp"].model == "gpt-oss-20b"


async def test_unloading_is_limited_to_one_server() -> None:
    note_model_used(_OLLAMA, "gpt-oss:120b")
    note_model_used("http://10.0.0.8:11434", "qwen3:8b")

    async def unload(root: str, model: str) -> bool:
        return True

    import satsumaestro.providers.local as local_servers

    original = local_servers.unload_model
    local_servers.unload_model = unload  # type: ignore[assignment]
    try:
        done = await unload_used(ProviderConfig(), root=_OLLAMA)
    finally:
        local_servers.unload_model = original
    assert done == [Unloaded("ollama", ("gpt-oss:120b",))]
    assert used_models() == [("http://10.0.0.8:11434", "qwen3:8b")]


@pytest.mark.parametrize(
    ("line", "parsed"),
    [
        (
            ["llamacpp", "http://192.168.1.20:9931/v1"],
            ("llamacpp", "http://192.168.1.20:9931/v1", None),
        ),
        (
            ["3", "192.168.1.20:9931", "gpt-oss-20b"],
            ("llamacpp", "192.168.1.20:9931", "gpt-oss-20b"),
        ),
        (["Ollama", "qwen3:8b"], ("ollama", None, "qwen3:8b")),
        (["ollama", "studio.local:11434"], ("ollama", "studio.local:11434", None)),
        (["anthropic", "claude-sonnet-5-5"], ("anthropic", None, "claude-sonnet-5-5")),
        (["ollama", "a:1b", "b:2b"], None),
    ],
)
def test_the_arguments_are_read(line: list[str], parsed: object) -> None:
    assert provider_args(line) == parsed


async def _dispatch(rt: ConversationRuntime, line: str) -> str:
    console = Console(record=True, width=200)
    await SlashDispatcher(rt, console).dispatch(parse_slash(line))
    return console.export_text()


async def test_the_command_lists_switches_and_saves(
    servers: _Servers, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(home))
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.9")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    rt = _runtime(tmp_path, _llamacpp_cfg(), "gpt-oss-20b")

    listed = await _dispatch(rt, "/provider")
    assert "provider: auto → llamacpp（http://127.0.0.1:9931/v1 · gpt-oss-20b）" in listed
    assert "1) ollama       http://10.0.0.9:11434/v1（OLLAMA_HOST）" in listed
    assert "2) lmstudio     未設定" in listed
    assert "3) llamacpp ✓   http://127.0.0.1:9931/v1（settings）" in listed
    assert "5) anthropic    鍵あり（ANTHROPIC_API_KEY）" in listed
    assert "6) xai          鍵なし（XAI_API_KEY）" in listed

    out = await _dispatch(rt, "/provider 1 gpt-oss:120b")
    assert "provider → ollama（http://10.0.0.9:11434/v1 · gpt-oss:120b）" in out
    saved = json.loads((home / "settings.json").read_text(encoding="utf-8"))["provider"]
    assert saved == {
        "kind": "ollama",
        "default_model": "gpt-oss:120b",
        "base_url": "http://10.0.0.9:11434/v1",
        "profiles": {
            "llamacpp": {"base_url": "http://127.0.0.1:9931/v1", "model": "gpt-oss-20b"},
            "ollama": {"base_url": "http://10.0.0.9:11434/v1", "model": "gpt-oss:120b"},
        },
    }

    kept = await _dispatch(rt, "/provider lmstudio")
    assert "LM Studio の接続先がありません" in kept and "プロバイダは ollama のままです" in kept


def test_anthropic_gets_unique_tool_ids() -> None:
    def use(tid: str) -> ConversationMessage:
        block = ToolUseBlock(id=tid, name="read_file", input={})
        return ConversationMessage(role="assistant", blocks=[block])

    def result(tid: str) -> ConversationMessage:
        block = ToolResultBlock(tool_use_id=tid, tool_name="read_file", output="ok")
        return ConversationMessage(role="user", blocks=[block])

    history = [
        use("call_synth_0"),
        result("call_synth_0"),
        use("call_synth_0"),
        result("call_synth_0"),
        use("call.1"),
        result("call.1"),
    ]
    wire = _to_wire(history)
    ids = [b["id"] for m in wire for b in m["content"] if b["type"] == "tool_use"]
    refs = [b["tool_use_id"] for m in wire for b in m["content"] if b["type"] == "tool_result"]
    assert ids == ["call_synth_0", "toolu_0", "toolu_1"] and refs == ids
    assert history[2].blocks[0].id == "call_synth_0"  # type: ignore[union-attr]
