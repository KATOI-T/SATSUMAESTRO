"""ローカルのサーバの判定（Ollama の判定と同じ結果になること）と LM Studio のネイティブ API。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
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
    LMStudioServer,
    OllamaServer,
    looks_local,
    resolve_server,
    server_for_endpoint,
)
from satsumaestro.providers.local import _probe as real_probe
from satsumaestro.providers.local.lmstudio import unload_models_loaded_here
from satsumaestro.providers.retry import is_context_overflow
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ProviderError

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
    assert calls == 2


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
