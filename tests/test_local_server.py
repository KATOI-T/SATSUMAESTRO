"""ローカルのサーバの判定（Ollama の判定と同じ結果になること）と LM Studio のネイティブ API。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import openai
import pytest

from satsumaestro.config.models import (
    HookConfig,
    LocalConfig,
    OllamaConfig,
    ProviderConfig,
    RuntimeConfig,
)
from satsumaestro.engine import context_fit
from satsumaestro.engine.compaction import Compactor
from satsumaestro.engine.context_fit import check_model, local_server_for, missing_model_notice
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, Usage
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import NullRenderer
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.providers import local as local_servers
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.providers.detect import OllamaDetectMode, is_ollama_endpoint, ollama_root
from satsumaestro.providers.factory import build_bare
from satsumaestro.providers.local import (
    LlamaCppServer,
    LMStudioServer,
    OllamaServer,
    looks_local,
    resolve_server,
    server_for_endpoint,
)
from satsumaestro.providers.local import _probe as real_probe
from satsumaestro.providers.local.llamacpp import (
    unload_models_loaded_here as unload_llamacpp_models_loaded_here,
)
from satsumaestro.providers.local.lmstudio import unload_models_loaded_here
from satsumaestro.providers.openai_provider import OpenAICompatProvider
from satsumaestro.providers.retry import ReportedOverflow, is_context_overflow, reported_overflow
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ConversationMessage, TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ContextOverflowError, ProviderError

_CASES: list[tuple[str, dict[str, str], OllamaDetectMode]] = [
    ("http://localhost:11434/v1", {}, "auto"),
    ("http://192.168.0.5:11434/v1", {}, "auto"),
    ("http://gpu.lan:8000/v1", {"OLLAMA_HOST": "gpu.lan:8000"}, "auto"),
    ("http://gpu.lan:8000/v1", {"OLLAMA_HOST": "other.lan:8000"}, "auto"),
    ("http://localhost:1234/v1", {}, "auto"),
    ("http://localhost:8080/v1", {}, "auto"),
    ("http://localhost:8080/v1", {}, "on"),
    ("http://localhost:11434/v1", {}, "off"),
    ("https://api.example.com/v1", {}, "auto"),
]

_ROOT = "http://localhost:1234"
_OVERFLOW = (
    "The number of tokens to keep from the initial prompt is greater than the context length. "
    "Try to load the model with a larger context length, or provide a shorter input"
)


def _provider_cfg(
    detect: OllamaDetectMode = "auto", server: str = "auto", ttl: str | None = None
) -> ProviderConfig:
    return ProviderConfig(
        ollama=OllamaConfig(detect=detect),
        local=LocalConfig.model_validate({"server": server, "ttl": ttl}),
    )


def _v1(*, loaded: bool, context: int = 8192, key: str = "openai/gpt-oss-20b") -> dict[str, Any]:
    instances = (
        [
            {
                "id": key,
                "config": {"context_length": context, "parallel": 4},
                "remaining_ttl_seconds": 300,
            }
        ]
        if loaded
        else []
    )
    return {
        "models": [
            {
                "type": "llm",
                "key": key,
                "loaded_instances": instances,
                "max_context_length": 131072,
            },
            {
                "type": "embedding",
                "key": "nomic-embed",
                "loaded_instances": [],
                "max_context_length": 2048,
            },
        ]
    }


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(("endpoint", "env", "detect"), _CASES)
def test_the_server_matches_the_ollama_rule(
    endpoint: str, env: dict[str, str], detect: OllamaDetectMode
) -> None:
    server = server_for_endpoint(endpoint, env, _provider_cfg(detect))
    if is_ollama_endpoint(endpoint, env, detect):
        assert server == OllamaServer(ollama_root(endpoint))
        assert server.kind == "ollama"
    else:
        assert server is None


def test_only_the_openai_compatible_kind_has_a_local_server() -> None:
    cfg = RuntimeConfig()
    cfg.provider.base_url = "http://localhost:11434/v1"
    assert local_server_for(cfg, "gpt-oss:120b", {}) == OllamaServer("http://localhost:11434")
    assert local_server_for(RuntimeConfig(), "claude-opus-5-5", {"ANTHROPIC_API_KEY": "k"}) is None
    assert local_server_for(RuntimeConfig(), "m", {}) is None


def test_the_setting_picks_the_server_without_asking() -> None:
    env = {"LM_API_TOKEN": "tok"}
    lmstudio = server_for_endpoint("http://gpu.lan:9000/v1", env, _provider_cfg(server="lmstudio"))
    assert lmstudio == LMStudioServer("http://gpu.lan:9000", token="tok")
    ollama = server_for_endpoint("http://gpu.lan:9000/v1", {}, _provider_cfg(server="ollama"))
    assert ollama == OllamaServer("http://gpu.lan:9000")
    generic = _provider_cfg(server="generic")
    assert server_for_endpoint("http://localhost:11434/v1", {}, generic) is None


def test_contradicting_settings_are_refused() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        _provider_cfg(detect="on", server="lmstudio")
    with pytest.raises(ValueError, match="conflicts"):
        _provider_cfg(detect="off", server="ollama")
    with pytest.raises(ValueError, match="ttl"):
        _provider_cfg(ttl="soon")


@pytest.mark.parametrize(
    ("endpoint", "local"),
    [
        ("http://localhost:1234/v1", True),
        ("http://127.0.0.1:1234/v1", True),
        ("http://[::1]:1234/v1", True),
        ("http://192.168.188.110:1234/v1", True),
        ("http://10.0.0.2:8000/v1", True),
        ("http://studio.local:1234/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://api.x.ai/v1", False),
        ("http://8.8.8.8:1234/v1", False),
        ("http://gpu-box:1234/v1", False),
    ],
)
def test_only_local_looking_hosts_are_asked(endpoint: str, local: bool) -> None:
    assert looks_local(endpoint) is local


async def test_lmstudio_is_found_by_asking_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_servers, "_probe", real_probe)
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(200, json=_v1(loaded=False))

    endpoint = f"{_ROOT}/v1"
    async with _client(handler) as client:
        found = await resolve_server(endpoint, {}, _provider_cfg(), client=client)
        again = await resolve_server(endpoint, {}, _provider_cfg(), client=client)
    assert found == again == LMStudioServer(_ROOT)
    assert paths == ["/api/v1/models"]
    assert server_for_endpoint(endpoint, {}, _provider_cfg()) == LMStudioServer(_ROOT)


async def test_an_older_lmstudio_is_found_through_v0(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_servers, "_probe", real_probe)

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/v0/models":
            data = [
                {
                    "id": "qwen3-8b",
                    "type": "llm",
                    "state": "not-loaded",
                    "max_context_length": 32768,
                }
            ]
            return httpx.Response(200, json={"data": data})
        return httpx.Response(404, json={"error": "not found"})

    async with _client(handler) as client:
        found = await resolve_server(f"{_ROOT}/v1", {}, _provider_cfg(), client=client)
    assert found == LMStudioServer(_ROOT)


async def test_another_server_stays_generic_and_is_not_asked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_servers, "_probe", real_probe)
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="not found")

    async with _client(handler) as client:
        assert (
            await resolve_server("http://localhost:8000/v1", {}, _provider_cfg(), client=client)
            is None
        )
        assert (
            await resolve_server("http://localhost:8000/v1", {}, _provider_cfg(), client=client)
            is None
        )
    assert calls == 3


async def test_a_server_that_is_down_is_asked_again_later(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_servers, "_probe", real_probe)
    up = False

    def handler(req: httpx.Request) -> httpx.Response:
        if not up:
            raise httpx.ConnectError("refused", request=req)
        return httpx.Response(200, json=_v1(loaded=False))

    async with _client(handler) as client:
        assert await resolve_server(f"{_ROOT}/v1", {}, _provider_cfg(), client=client) is None
        up = True
        assert await resolve_server(f"{_ROOT}/v1", {}, _provider_cfg(), client=client) is not None


@pytest.mark.parametrize(
    ("endpoint", "provider"),
    [
        ("https://api.openai.com/v1", _provider_cfg()),
        ("http://localhost:11434/v1", _provider_cfg()),
        ("http://localhost:1234/v1", _provider_cfg(server="generic")),
        ("http://localhost:1234/v1", _provider_cfg(detect="off")),
    ],
)
async def test_no_question_is_sent_when_the_answer_is_known(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, provider: ProviderConfig
) -> None:
    monkeypatch.setattr(local_servers, "_probe", real_probe)

    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"asked {req.url}")

    async with _client(handler) as client:
        await resolve_server(endpoint, {}, provider, client=client)


async def test_the_loaded_window_is_read_from_the_instance() -> None:
    seen: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers.get("authorization"))
        return httpx.Response(200, json=_v1(loaded=True, context=8192))

    async with _client(handler) as client:
        info = await LMStudioServer(_ROOT, token="t0k").context_info(
            "openai/gpt-oss-20b", client=client
        )
    assert info.server == "lmstudio" and info.confirmed and info.loaded
    assert info.effective == 8192 and info.native == 131072 and info.context_length == 8192
    assert info.expires_at is not None
    assert seen == ["Bearer t0k"]


async def test_a_model_not_loaded_yet_gives_its_largest_window() -> None:
    async with _client(lambda req: httpx.Response(200, json=_v1(loaded=False))) as client:
        info = await LMStudioServer(_ROOT).context_info("gpt-oss-20b", client=client)
    assert info.unconfirmed and not info.loaded and info.context_length == 131072


async def test_the_model_is_matched_by_key_or_its_last_part() -> None:
    server = LMStudioServer(_ROOT)
    async with _client(lambda req: httpx.Response(200, json=_v1(loaded=False))) as client:
        assert await server.has_model("openai/gpt-oss-20b", client=client) is True
        assert await server.has_model("GPT-OSS-20B", client=client) is True
        assert await server.has_model("no/such-model", client=client) is False
        assert await server.list_models(client=client) == ["openai/gpt-oss-20b"]


async def test_a_missing_model_points_at_lms_get() -> None:
    cfg = RuntimeConfig()
    cfg.provider.base_url = f"{_ROOT}/v1"
    cfg.provider.local.server = "lmstudio"
    async with _client(lambda req: httpx.Response(200, json=_v1(loaded=False))) as client:
        check = await check_model(cfg, "openai/gpt-oss-120b", {}, client=client)
    assert check is not None and check.missing and check.server == "lmstudio"
    notice = missing_model_notice("openai/gpt-oss-120b", check)
    assert "LM Studio" in notice.headline
    assert notice.pull_command == "lms get openai/gpt-oss-120b"
    assert "openai/gpt-oss-20b" in notice.close


async def test_an_unreachable_server_tells_how_to_start_it() -> None:
    cfg = RuntimeConfig()
    cfg.provider.base_url = f"{_ROOT}/v1"
    cfg.provider.local.server = "lmstudio"

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    async with _client(handler) as client:
        problem = await context_fit.check_reachable(cfg, "m", {}, client=client)
    assert problem is not None and "lms server start" in problem and "LM_API_TOKEN" in problem


async def test_only_models_loaded_by_this_process_are_unloaded() -> None:
    loaded: set[str] = {"openai/gpt-oss-20b"}
    unloaded: list[str] = []
    models = {"openai/gpt-oss-20b": 8192, "qwen/qwen3-8b": 32768}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/v1/models/unload":
            instance = json.loads(req.content)["instance_id"]
            unloaded.append(instance)
            loaded.discard(instance)
            return httpx.Response(200, json={"instance_id": instance})
        entries = [
            {
                "type": "llm",
                "key": key,
                "max_context_length": 131072,
                "loaded_instances": (
                    [{"id": key, "config": {"context_length": ctx}}] if key in loaded else []
                ),
            }
            for key, ctx in models.items()
        ]
        return httpx.Response(200, json={"models": entries})

    server = LMStudioServer(_ROOT)
    async with _client(handler) as client:
        await server.context_info("openai/gpt-oss-20b", client=client)
        await server.context_info("qwen/qwen3-8b", client=client)
        loaded.add("qwen/qwen3-8b")
        server.note_used("openai/gpt-oss-20b")
        server.note_used("qwen/qwen3-8b")
        done = await unload_models_loaded_here(client=client)
    assert done == ["qwen/qwen3-8b"] and unloaded == ["qwen/qwen3-8b"]
    assert loaded == {"openai/gpt-oss-20b"}


def test_the_provider_sends_the_lmstudio_token_and_ttl() -> None:
    cfg = RuntimeConfig()
    cfg.provider.base_url = f"{_ROOT}/v1"
    cfg.provider.local.server = "lmstudio"
    cfg.provider.local.ttl = "10m"
    provider = build_bare(cfg, "openai/gpt-oss-20b", {"LM_API_TOKEN": "lm-tok"})
    assert provider._client.api_key == "lm-tok"  # type: ignore[attr-defined]
    assert provider._extra_body == {"ttl": 600}  # type: ignore[attr-defined]
    assert provider._ollama_root is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("key", "sent"),
    [("sk-proj-abc", "lm-studio"), ("xai-abc", "lm-studio"), ("local-secret", "local-secret")],
)
def test_a_cloud_key_is_not_sent_to_lmstudio(key: str, sent: str) -> None:
    cfg = RuntimeConfig()
    cfg.provider.base_url = f"{_ROOT}/v1"
    cfg.provider.local.server = "lmstudio"
    provider = build_bare(cfg, "m", {"OPENAI_API_KEY": key})
    assert provider._client.api_key == sent  # type: ignore[attr-defined]
    assert provider._extra_body is None  # type: ignore[attr-defined]


def test_lmstudio_overflow_is_recognized() -> None:
    assert is_context_overflow(ProviderError(f"OpenAI-compatible API call failed: {_OVERFLOW}"))
    assert not is_context_overflow(ProviderError("rate limited"))


class _OverflowOnce:
    """1 回目はサーバが窓あふれで断り、2 回目は答えるプロバイダ。"""

    kind = ProviderKind.OPENAI

    def __init__(self, failures: int = 1) -> None:
        self.requests: list[ApiRequest] = []
        self._failures = failures

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.requests.append(req)
        if len(self.requests) <= self._failures:
            raise ProviderError(f"OpenAI-compatible API call failed: {_OVERFLOW}")
        yield TextDelta(text="ok")
        yield Usage(input_tokens=5_000, output_tokens=2)
        yield MessageStop()

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


def _runtime(tmp_path: Path, provider: _OverflowOnce) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "openai/gpt-oss-20b"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
        compactor=Compactor(keep_last=6),
    )


class _Notices(NullRenderer):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def on_context_notice(self, text: str) -> None:
        self.lines.append(text)


async def test_an_overflow_refits_the_window_and_sends_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refit(cfg: RuntimeConfig, model: str, env: object) -> int | None:
        cfg.compaction.context_window = 8192
        return 8192

    monkeypatch.setattr(context_fit, "refit_window", refit)
    provider = _OverflowOnce()
    rt = _runtime(tmp_path, provider)
    render = _Notices()
    summary = await rt.run_turn("こんにちは", render=render)
    assert summary.final_text == "ok" and len(provider.requests) == 2
    assert rt.config.compaction.context_window == 8192
    assert any("8,192" in line for line in render.lines)


async def test_a_second_overflow_stops_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refit(cfg: RuntimeConfig, model: str, env: object) -> int | None:
        return 8192

    monkeypatch.setattr(context_fit, "refit_window", refit)
    provider = _OverflowOnce(failures=2)
    with pytest.raises(ProviderError, match="greater than the context length"):
        await _runtime(tmp_path, provider).run_turn("こんにちは")
    assert len(provider.requests) == 2


async def test_an_overflow_without_a_smaller_window_is_not_sent_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refit(cfg: RuntimeConfig, model: str, env: object) -> int | None:
        return None

    monkeypatch.setattr(context_fit, "refit_window", refit)
    provider = _OverflowOnce()
    with pytest.raises(ProviderError):
        await _runtime(tmp_path, provider).run_turn("こんにちは")
    assert len(provider.requests) == 1


_LLAMA = "http://localhost:9931"


def _props(*, n_ctx: int = 65536, slots: int = 4, sleeping: bool = False) -> dict[str, Any]:
    return {
        "default_generation_settings": {"params": {}, "n_ctx": n_ctx},
        "total_slots": slots,
        "model_alias": "gpt-oss-20b",
        "model_path": "/models/gpt-oss-20b.gguf",
        "is_sleeping": sleeping,
    }


_ROUTER_PROPS = {
    "role": "router",
    "max_instances": 4,
    "models_autoload": True,
    "model_alias": "llama-server",
    "model_path": "none",
    "default_generation_settings": {"params": {}, "n_ctx": 0},
}


def _router_model(
    name: str, status: str, *, n_ctx: int | None = None, aliases: tuple[str, ...] = ()
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": name,
        "aliases": list(aliases),
        "object": "model",
        "owned_by": "llamacpp",
        "status": {"value": status, "args": []},
        "context_length": 131072,
    }
    if n_ctx is not None:
        entry["meta"] = {"n_ctx": n_ctx, "n_ctx_train": 131072}
    return entry


def _router(models: list[dict[str, Any]]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/props":
            return httpx.Response(200, json=_ROUTER_PROPS)
        if req.url.path == "/models":
            return httpx.Response(200, json={"data": models, "object": "list"})
        return httpx.Response(404, json={"error": "not found"})

    return handler


def _llamacpp_cfg(**local: object) -> RuntimeConfig:
    cfg = RuntimeConfig()
    cfg.provider.base_url = f"{_LLAMA}/v1"
    cfg.provider.local = LocalConfig.model_validate({"server": "llamacpp", **local})
    return cfg


async def test_llamacpp_is_found_by_asking_props(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_servers, "_probe", real_probe)
    paths: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        if req.url.path == "/props":
            return httpx.Response(200, json=_props())
        return httpx.Response(404, json={"error": {"code": 404, "type": "not_found_error"}})

    endpoint = f"{_LLAMA}/v1"
    async with _client(handler) as client:
        found = await resolve_server(endpoint, {}, _provider_cfg(), client=client)
        again = await resolve_server(endpoint, {}, _provider_cfg(), client=client)
    assert found == again == LlamaCppServer(_LLAMA)
    assert paths == ["/api/v1/models", "/props"]
    assert server_for_endpoint(endpoint, {}, _provider_cfg()) == LlamaCppServer(_LLAMA)


async def test_a_llamacpp_router_is_found_too(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_servers, "_probe", real_probe)
    async with _client(_router([])) as client:
        found = await resolve_server(f"{_LLAMA}/v1", {}, _provider_cfg(), client=client)
    assert found == LlamaCppServer(_LLAMA)


def test_the_setting_picks_llamacpp_without_asking() -> None:
    env = {"LLAMA_API_KEY": "key"}
    server = server_for_endpoint("http://gpu.lan:9000/v1", env, _provider_cfg(server="llamacpp"))
    assert server == LlamaCppServer("http://gpu.lan:9000", token="key")
    assert server is not None and server.kind == "llamacpp"
    with pytest.raises(ValueError, match="conflicts"):
        _provider_cfg(detect="on", server="llamacpp")


async def test_the_slot_window_is_read_from_props() -> None:
    seen: list[str | None] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.headers.get("authorization"))
        return httpx.Response(200, json=_props(n_ctx=16384))

    async with _client(handler) as client:
        info = await LlamaCppServer(_LLAMA, token="k3y").context_info("anything", client=client)
    assert info.server == "llamacpp" and info.source == "llamacpp.slot"
    assert info.confirmed and info.loaded and info.context_length == 16384
    assert seen == ["Bearer k3y"]


async def test_a_sleeping_server_still_knows_its_window() -> None:
    async with _client(lambda req: httpx.Response(200, json=_props(sleeping=True))) as client:
        info = await LlamaCppServer(_LLAMA).context_info("m", client=client)
    assert info.confirmed and not info.loaded and info.context_length == 65536


async def test_a_one_model_server_takes_any_name_and_lists_its_model() -> None:
    server = LlamaCppServer(_LLAMA)
    async with _client(lambda req: httpx.Response(200, json=_props())) as client:
        assert await server.has_model("whatever", client=client) is True
        assert await server.list_models(client=client) == ["gpt-oss-20b"]
        assert await server.slots(client=client) == 4


@pytest.mark.parametrize(
    ("caps", "expected"),
    [({"supports_tools": False}, False), ({"supports_tools": True}, True), (None, None)],
)
async def test_the_template_says_whether_it_takes_tools(
    caps: dict[str, bool] | None, expected: bool | None
) -> None:
    body = {**_props(), **({"chat_template_caps": caps} if caps is not None else {})}
    async with _client(lambda req: httpx.Response(200, json=body)) as client:
        assert await LlamaCppServer(_LLAMA).tools_supported(client=client) is expected
    async with _client(_router([])) as client:
        assert await LlamaCppServer(_LLAMA).tools_supported(client=client) is None


async def test_the_router_reads_the_window_of_a_loaded_model() -> None:
    models = [_router_model("gpt-oss-20b", "loaded", n_ctx=32768, aliases=("oss",))]
    async with _client(_router(models)) as client:
        info = await LlamaCppServer(_LLAMA).context_info("oss", client=client)
        slots = await LlamaCppServer(_LLAMA).slots(client=client)
    assert info.confirmed and info.loaded and info.context_length == 32768
    assert info.native == 131072 and slots is None


async def test_a_model_the_router_has_not_loaded_gives_its_trained_window() -> None:
    async with _client(_router([_router_model("qwen3-8b", "unloaded")])) as client:
        info = await LlamaCppServer(_LLAMA).context_info("qwen3-8b", client=client)
    assert info.unconfirmed and not info.loaded and info.context_length == 131072
    assert info.source == "llamacpp.train"


async def test_the_router_matches_names_and_aliases_exactly() -> None:
    server = LlamaCppServer(_LLAMA)
    models = [
        _router_model("gpt-oss-20b", "unloaded", aliases=("oss",)),
        _router_model("x", "incomplete"),
    ]
    async with _client(_router(models)) as client:
        assert await server.has_model("gpt-oss-20b", client=client) is True
        assert await server.has_model("oss", client=client) is True
        assert await server.has_model("GPT-OSS-20B", client=client) is False
        assert await server.list_models(client=client) == ["gpt-oss-20b"]


async def test_a_model_missing_from_the_router_points_at_models_dir() -> None:
    async with _client(_router([_router_model("gpt-oss-20b", "unloaded")])) as client:
        check = await check_model(_llamacpp_cfg(), "gpt-oss-120b", {}, client=client)
    assert check is not None and check.missing and check.server == "llamacpp"
    notice = missing_model_notice("gpt-oss-120b", check)
    assert "llama.cpp" in notice.headline and notice.pull_command == "--models-dir"
    assert notice.pull_command in notice.pull_hint(notice.pull_command)
    assert "gpt-oss-20b" in notice.close and "`--models-dir`" in notice.text()


@pytest.mark.parametrize(("status", "reachable"), [(200, True), (503, True), (401, False)])
async def test_llamacpp_is_reached_while_it_loads_but_not_without_its_key(
    status: int, reachable: bool
) -> None:
    async with _client(lambda req: httpx.Response(status, json={})) as client:
        cause = await LlamaCppServer(_LLAMA).reachable(client=client)
    assert (cause is None) is reachable


async def test_an_unreachable_llamacpp_tells_how_to_start_it() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=req)

    async with _client(handler) as client:
        problem = await context_fit.check_reachable(_llamacpp_cfg(), "m", {}, client=client)
    assert problem is not None and "llama-server" in problem and "LLAMA_API_KEY" in problem


async def test_the_router_unloads_only_models_it_loaded_for_this_process() -> None:
    status = {"gpt-oss-20b": "loaded", "qwen3-8b": "unloaded"}
    unloaded: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/models/unload":
            name = json.loads(req.content)["model"]
            unloaded.append(name)
            status[name] = "unloaded"
            return httpx.Response(200, json={"success": True})
        models = [
            _router_model(n, s, n_ctx=32768 if s == "loaded" else None) for n, s in status.items()
        ]
        return _router(models)(req)

    server = LlamaCppServer(_LLAMA)
    async with _client(handler) as client:
        await server.context_info("gpt-oss-20b", client=client)
        await server.context_info("qwen3-8b", client=client)
        status["qwen3-8b"] = "loaded"
        server.note_used("gpt-oss-20b")
        server.note_used("qwen3-8b")
        done = await unload_llamacpp_models_loaded_here(client=client)
    assert done == ["qwen3-8b"] and unloaded == ["qwen3-8b"]
    assert status["gpt-oss-20b"] == "loaded"


async def test_a_one_model_server_is_never_unloaded() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/models/unload":
            raise AssertionError("unloaded a one-model server")
        return httpx.Response(200, json=_props(sleeping=True))

    server = LlamaCppServer(_LLAMA)
    async with _client(handler) as client:
        await server.context_info("m", client=client)
        server.note_used("m")
        assert await unload_llamacpp_models_loaded_here(client=client) == []


@pytest.mark.parametrize(
    ("env", "sent"),
    [
        ({"LLAMA_API_KEY": "llama-key"}, "llama-key"),
        ({"OPENAI_API_KEY": "sk-proj-abc"}, "llama.cpp"),
        ({"OPENAI_API_KEY": "local-secret"}, "local-secret"),
    ],
)
def test_the_provider_sends_the_llamacpp_key(env: dict[str, str], sent: str) -> None:
    provider = build_bare(_llamacpp_cfg(), "m", env)
    assert provider._client.api_key == sent  # type: ignore[attr-defined]
    assert provider._extra_body is None  # type: ignore[attr-defined]
    assert provider._ollama_root is None  # type: ignore[attr-defined]


def _llamacpp_overflow_body(window: int, tokens: int) -> dict[str, Any]:
    return {
        "error": {
            "code": 400,
            "message": (
                f"request ({tokens} tokens) exceeds the available context size ({window} tokens), "
                "try increasing it"
            ),
            "type": "exceed_context_size_error",
            "n_prompt_tokens": tokens,
            "n_ctx": window,
        }
    }


async def test_a_llamacpp_overflow_is_raised_once_with_its_numbers() -> None:
    sent: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content))
        return httpx.Response(400, json=_llamacpp_overflow_body(8192, 9500))

    provider = OpenAICompatProvider("k", f"{_LLAMA}/v1")
    provider._client = openai.AsyncOpenAI(  # type: ignore[attr-defined]
        api_key="k",
        base_url=f"{_LLAMA}/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    tool = {
        "name": "a",
        "description": "",
        "input_schema": {
            "type": "object",
            "properties": {"p": {"type": "string", "description": "d"}},
            "required": ["p"],
        },
    }
    req = ApiRequest(model="m", system="s", messages=[], tools=[tool], strict_tools=True)
    with pytest.raises(ProviderError) as caught:
        async for _ in provider.stream(req):
            pass
    assert len(sent) == 1
    assert is_context_overflow(caught.value)
    assert reported_overflow(caught.value) == ReportedOverflow(window=8192, prompt_tokens=9500)
    assert reported_overflow(ProviderError(f"failed: {_OVERFLOW}")) == ReportedOverflow()


def _llamacpp_overflow(window: int, tokens: int) -> ProviderError:
    body = _llamacpp_overflow_body(window, tokens)
    request = httpx.Request("POST", f"{_LLAMA}/v1/chat/completions")
    response = httpx.Response(400, json=body, request=request)
    cause = openai.BadRequestError(
        f"Error code: 400 - {body}", response=response, body=body["error"]
    )
    error = ProviderError(f"OpenAI-compatible API call failed: Error code: 400 - {body}")
    error.__cause__ = cause
    return error


class _ReportedOverflowOnce:
    """初めは llama.cpp の形の窓あふれで断り、その後は答えるプロバイダ。"""

    kind = ProviderKind.OPENAI

    def __init__(self, window: int, tokens: int, failures: int = 1) -> None:
        self.requests: list[ApiRequest] = []
        self._error = _llamacpp_overflow(window, tokens)
        self._failures = failures

    async def stream(self, req: ApiRequest) -> AsyncIterator[AssistantEvent]:
        self.requests.append(req)
        if len(self.requests) <= self._failures:
            raise self._error
        yield TextDelta(text="ok")
        yield Usage(input_tokens=5_000, output_tokens=2)
        yield MessageStop()

    async def count_tokens(self, req: ApiRequest) -> int:
        return 0


def _history(pairs: int) -> list[ConversationMessage]:
    filler = "これは前の作業の記録です。" * 60
    messages: list[ConversationMessage] = []
    for i in range(pairs):
        messages.append(
            ConversationMessage(role="user", blocks=[TextBlock(text=f"依頼 {i} {filler}")])
        )
        messages.append(
            ConversationMessage(role="assistant", blocks=[TextBlock(text=f"答え {i} {filler}")])
        )
    return messages


def _windowed_runtime(
    tmp_path: Path, provider: _ReportedOverflowOnce, *, window: int, pairs: int
) -> ConversationRuntime:
    rt = _runtime(tmp_path, provider)  # type: ignore[arg-type]
    rt.config.compaction.context_window = window
    rt.config.compact_token_threshold = int(window * 0.75)
    rt.session.messages.extend(_history(pairs))
    return rt


async def test_a_reported_overflow_is_compacted_with_the_counted_size_and_sent_again(
    tmp_path: Path,
) -> None:
    provider = _ReportedOverflowOnce(window=65536, tokens=70000)
    rt = _windowed_runtime(tmp_path, provider, window=65536, pairs=20)
    render = _Notices()
    summary = await rt.run_turn("続けて", render=render)
    assert summary.final_text == "ok" and len(provider.requests) == 2
    assert len(provider.requests[1].messages) < len(provider.requests[0].messages)
    assert rt.config.compaction.context_window == 65536
    assert any("65,536" in line for line in render.lines)


async def test_a_reported_smaller_window_is_used_without_asking_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def refit(cfg: RuntimeConfig, model: str, env: object) -> int | None:
        raise AssertionError("asked the server")

    monkeypatch.setattr(context_fit, "refit_window", refit)
    provider = _ReportedOverflowOnce(window=32768, tokens=40000)
    rt = _windowed_runtime(tmp_path, provider, window=65536, pairs=20)
    summary = await rt.run_turn("続けて")
    assert summary.final_text == "ok" and len(provider.requests) == 2
    assert rt.config.compaction.context_window == 32768


async def test_a_request_that_cannot_fit_after_a_reported_overflow_is_dropped(
    tmp_path: Path,
) -> None:
    provider = _ReportedOverflowOnce(window=8192, tokens=20000)
    rt = _windowed_runtime(tmp_path, provider, window=8192, pairs=0)
    with pytest.raises(ContextOverflowError) as caught:
        await rt.run_turn("長い依頼")
    assert caught.value.dropped_request and len(provider.requests) == 1
    assert rt.session.messages == []


async def test_a_second_reported_overflow_stops_the_turn(tmp_path: Path) -> None:
    provider = _ReportedOverflowOnce(window=65536, tokens=70000, failures=2)
    rt = _windowed_runtime(tmp_path, provider, window=65536, pairs=20)
    with pytest.raises(ProviderError, match="exceed_context_size_error"):
        await rt.run_turn("こんにちは")
    assert len(provider.requests) == 2
