"""プロバイダの種類の選び方・種類ごとの接続先・設定への保存。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from satsumaestro.config.loader import ConfigLoader
from satsumaestro.config.models import ProviderConfig, RuntimeConfig
from satsumaestro.providers import local as local_servers
from satsumaestro.providers.base import ProviderKind
from satsumaestro.providers.choice import (
    Endpoint,
    current_choice,
    host_endpoint,
    missing_text,
    provider_statuses,
    resolve_endpoint,
    switch_endpoint,
)
from satsumaestro.providers.detect import choose_provider_kind
from satsumaestro.providers.factory import build_bare
from satsumaestro.providers.local import LlamaCppServer, LMStudioServer, OllamaServer
from satsumaestro.util.errors import ProviderError


@pytest.mark.parametrize(
    ("choice", "raw", "url"),
    [
        ("llamacpp", "10.0.0.5", "http://10.0.0.5:9931/v1"),
        ("llamacpp", "10.0.0.5:8080", "http://10.0.0.5:8080/v1"),
        ("llamacpp", "http://h:1/v1", "http://h:1/v1"),
        ("lmstudio", "studio.local", "http://studio.local:1234/v1"),
        ("ollama", "192.168.1.9", "http://192.168.1.9:11434/v1"),
    ],
)
def test_each_host_variable_has_its_default_port(choice: str, raw: str, url: str) -> None:
    var = {"ollama": "OLLAMA_HOST", "lmstudio": "LMSTUDIO_HOST", "llamacpp": "LLAMACPP_HOST"}
    assert host_endpoint(choice, {var[choice]: raw}) == Endpoint(url, var[choice])


def _provider(**kw: object) -> ProviderConfig:
    return ProviderConfig.model_validate(kw)


def test_a_local_kind_reads_the_settings_and_then_its_own_variable() -> None:
    env = {"LLAMACPP_HOST": "10.0.0.5", "OLLAMA_HOST": "10.0.0.9"}
    with_url = _provider(kind="llamacpp", base_url="http://127.0.0.1:9931/v1")
    assert resolve_endpoint(with_url, env) == Endpoint("http://127.0.0.1:9931/v1", "settings")
    no_url = _provider(kind="llamacpp")
    assert resolve_endpoint(no_url, env) == Endpoint("http://10.0.0.5:9931/v1", "LLAMACPP_HOST")
    assert resolve_endpoint(_provider(kind="lmstudio"), env) is None


def test_the_clouds_ignore_the_local_endpoints() -> None:
    env = {"OLLAMA_HOST": "10.0.0.9", "OPENAI_BASE_URL": "http://10.0.0.9:8000/v1"}
    base = "http://127.0.0.1:9931/v1"
    openai = _provider(kind="openai", base_url=base)
    assert resolve_endpoint(openai, env) == Endpoint("https://api.openai.com/v1", "fixed")
    assert resolve_endpoint(_provider(kind="anthropic", base_url=base), env) is None
    assert resolve_endpoint(_provider(kind="xai", base_url=base), env) is None


def test_auto_keeps_todays_order() -> None:
    host = {"OLLAMA_HOST": "10.0.0.9"}
    copied = _provider(base_url="http://10.0.0.9:11434")
    assert resolve_endpoint(copied, host) == Endpoint("http://10.0.0.9:11434/v1", "OLLAMA_HOST")
    settings = _provider(base_url="http://127.0.0.1:9931/v1")
    assert resolve_endpoint(settings, host).source == "settings"  # type: ignore[union-attr]
    both = {"OLLAMA_HOST": "10.0.0.9", "OPENAI_BASE_URL": "http://h:8000/v1"}
    assert resolve_endpoint(_provider(), both) == Endpoint("http://h:8000/v1", "OPENAI_BASE_URL")
    assert resolve_endpoint(_provider(), {}) is None


def test_an_explicit_kind_is_not_changed_by_the_model_or_ollama_host() -> None:
    env = {"OLLAMA_HOST": "10.0.0.9", "ANTHROPIC_API_KEY": "k"}
    assert choose_provider_kind("gpt-oss:20b", env, choice="anthropic") is ProviderKind.ANTHROPIC
    assert choose_provider_kind("grok-4", env, choice="xai") is ProviderKind.XAI
    assert choose_provider_kind("claude-opus-5-5", env, choice="llamacpp") is ProviderKind.OPENAI
    assert choose_provider_kind("claude-opus-5-5", env) is ProviderKind.OPENAI
    assert choose_provider_kind("claude-opus-5-5", {}) is ProviderKind.ANTHROPIC


async def test_an_explicit_local_kind_is_never_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def probe(*servers: object) -> None:
        raise AssertionError("probed")

    monkeypatch.setattr(local_servers, "_probe", probe)
    for kind, cls in (
        ("llamacpp", LlamaCppServer),
        ("lmstudio", LMStudioServer),
        ("ollama", OllamaServer),
    ):
        provider = _provider(kind=kind, local={"server": "generic"})
        server = await local_servers.resolve_server("http://127.0.0.1:9931/v1", {}, provider)
        assert isinstance(server, cls)
    cloud = _provider(kind="openai")
    assert await local_servers.resolve_server("http://127.0.0.1:9931/v1", {}, cloud) is None


def test_an_explicit_openai_needs_its_key_and_uses_the_official_endpoint() -> None:
    cfg = RuntimeConfig(provider=_provider(kind="openai", base_url="http://127.0.0.1:9931/v1"))
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        build_bare(cfg, "gpt-5", {})
    provider = build_bare(cfg, "gpt-5", {"OPENAI_API_KEY": "k"})
    assert str(provider._client.base_url).rstrip("/") == "https://api.openai.com/v1"  # type: ignore[attr-defined]


def test_a_local_kind_without_an_endpoint_names_its_variable() -> None:
    cfg = RuntimeConfig(provider=_provider(kind="llamacpp"))
    with pytest.raises(ProviderError, match="LLAMACPP_HOST"):
        build_bare(cfg, "gpt-oss-20b", {})


def _home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: dict[str, object]) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(home))
    return home


def test_ollama_host_from_dotenv_does_not_change_an_explicit_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "unused")
    monkeypatch.setenv("OLLAMA_MODEL", "unused")
    monkeypatch.delenv("OLLAMA_HOST")
    monkeypatch.delenv("OLLAMA_MODEL")
    (tmp_path / ".env").write_text(
        "OLLAMA_HOST=10.0.0.9\nOLLAMA_MODEL=qwen3:8b\n", encoding="utf-8"
    )
    _home(tmp_path, monkeypatch, {"provider": {"kind": "llamacpp"}})
    cfg = ConfigLoader(tmp_path, trusted=True).load()
    assert cfg.provider.kind == "llamacpp"
    assert cfg.provider.base_url is None
    assert cfg.provider.default_model != "qwen3:8b"
    assert os.environ["OLLAMA_HOST"] == "10.0.0.9"


def test_ollama_host_still_fills_auto(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "10.0.0.9")
    _home(tmp_path, monkeypatch, {})
    cfg = ConfigLoader(tmp_path, trusted=True).load()
    assert cfg.provider.kind == "auto"
    assert cfg.provider.base_url == "http://10.0.0.9:11434"


def test_the_switch_endpoint_prefers_the_active_then_the_profile_then_the_variable() -> None:
    provider = _provider(
        kind="llamacpp",
        base_url="http://127.0.0.1:9931/v1",
        profiles={"ollama": {"base_url": "http://a:11434", "model": "m1"}},
    )
    env = {"OLLAMA_HOST": "10.0.0.9", "LMSTUDIO_HOST": "10.0.0.7"}
    active = switch_endpoint(provider, "llamacpp", env, current="llamacpp")
    assert active == Endpoint("http://127.0.0.1:9931/v1", "settings")
    profile = switch_endpoint(provider, "ollama", env, current="llamacpp")
    assert profile == Endpoint("http://a:11434/v1", "profile")
    variable = switch_endpoint(provider, "lmstudio", env, current="llamacpp")
    assert variable == Endpoint("http://10.0.0.7:1234/v1", "LMSTUDIO_HOST")


def test_the_statuses_tell_the_endpoint_or_the_key() -> None:
    provider = _provider(
        kind="llamacpp",
        base_url="http://127.0.0.1:9931/v1",
        profiles={"anthropic": {"model": "claude-sonnet-5-5"}},
    )
    env = {"OLLAMA_HOST": "10.0.0.9", "ANTHROPIC_API_KEY": "k"}
    rows = {s.choice: s for s in provider_statuses(provider, "gpt-oss-20b", env)}
    assert list(rows) == ["ollama", "lmstudio", "llamacpp", "openai", "anthropic", "xai"]
    assert rows["llamacpp"].current and rows["llamacpp"].model == "gpt-oss-20b"
    assert rows["ollama"].ready and not rows["lmstudio"].ready
    assert rows["anthropic"].ready and rows["anthropic"].model == "claude-sonnet-5-5"
    assert not rows["openai"].ready and rows["openai"].key_env == "OPENAI_API_KEY"


def test_auto_guesses_the_current_kind() -> None:
    assert current_choice(_provider(), "claude-opus-5-5", {}) == "anthropic"
    settings = _provider(base_url="http://127.0.0.1:9931/v1", local={"server": "llamacpp"})
    assert current_choice(settings, "gpt-oss-20b", {}) == "llamacpp"
    assert current_choice(_provider(), "gpt-oss:20b", {"OLLAMA_HOST": "h"}) == "ollama"
    assert current_choice(_provider(), "gpt-oss:20b", {}) is None


def test_the_missing_texts_tell_what_to_set() -> None:
    assert "LLAMACPP_HOST" in missing_text("llamacpp") and "<port>" in missing_text("llamacpp")
    assert "LMSTUDIO_HOST" in missing_text("lmstudio") and ":1234/v1" in missing_text("lmstudio")
    assert "OLLAMA_HOST" in missing_text("ollama") and ":11434" in missing_text("ollama")
    assert missing_text("anthropic") == "ANTHROPIC_API_KEY が設定されていません"
