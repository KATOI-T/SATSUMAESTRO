"""4 層マージ・.env 反映テスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from satsumaestro.config.loader import ConfigLoader, deep_merge
from satsumaestro.util.errors import ConfigError

PROVIDER_ENV_VARS = [
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
    "SATSUMA_EMBED_BASE_URL",
    "SATSUMA_EMBED_MODEL",
    "SATSUMA_ORCHESTRATION",
    "SATSUMA_DECOMPOSITION",
    "SATSUMA_WEB_SEARCH_PROVIDER",
    "SATSUMA_WEB_SEARCH_BASE_URL",
    "SATSUMA_WEB_FETCH_ENABLED",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in PROVIDER_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SATSUMA_CONFIG_HOME", raising=False)


def _put_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_deep_merge_scalars_and_nested() -> None:
    base = {"a": 1, "n": {"x": 1, "y": 2}, "lst": [1, 2]}
    override = {"a": 9, "n": {"y": 20, "z": 30}, "lst": [3]}

    assert deep_merge(base, override) == {
        "a": 9,
        "n": {"x": 1, "y": 20, "z": 30},
        "lst": [3],
    }


def test_four_layer_merge_last_wins(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "proj"
    cwd.mkdir()

    _put_json(
        home / "settings.json",
        {"max_iterations": 10, "provider": {"default_model": "global"}},
    )
    _put_json(cwd / ".satsuma.json", {"max_iterations": 20})
    _put_json(cwd / ".satsuma" / "settings.json", {"provider": {"default_model": "project"}})
    _put_json(cwd / ".satsuma" / "settings.local.json", {"max_iterations": 99})

    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()

    assert cfg.max_iterations == 99
    assert cfg.provider.default_model == "project"


def test_env_fills_missing_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = tmp_path / "proj"
    cwd.mkdir()
    monkeypatch.setenv("OLLAMA_HOST", "http://host:11434/v1")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setenv("SATSUMA_EMBED_MODEL", "bge-custom")

    cfg = ConfigLoader(cwd, config_home=tmp_path / "empty_home", trusted=True).load()

    assert cfg.provider.base_url == "http://host:11434/v1"
    assert cfg.provider.default_model == "qwen2.5-coder:7b"
    assert cfg.rag.embed_model == "bge-custom"


def test_explicit_config_wins_over_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = tmp_path / "proj"
    cwd.mkdir()
    _put_json(cwd / ".satsuma" / "settings.json", {"provider": {"base_url": "http://explicit/v1"}})
    monkeypatch.setenv("OLLAMA_HOST", "http://env/v1")

    cfg = ConfigLoader(cwd, config_home=tmp_path / "empty_home", trusted=True).load()

    assert cfg.provider.base_url == "http://explicit/v1"


def test_env_toggles_orchestration_and_decomposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SATSUMA_ORCHESTRATION / SATSUMA_DECOMPOSITION で並列実行と自動分解を切り替える。"""
    cwd = tmp_path / "proj"
    cwd.mkdir()
    home = tmp_path / "empty_home"

    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.orchestration.enabled is False
    assert cfg.decomposition.enabled is False

    monkeypatch.setenv("SATSUMA_ORCHESTRATION", "TRUE")
    monkeypatch.setenv("SATSUMA_DECOMPOSITION", "on")
    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.orchestration.enabled is True
    assert cfg.decomposition.enabled is True

    monkeypatch.setenv("SATSUMA_DECOMPOSITION", "0")
    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.decomposition.enabled is False

    _put_json(cwd / ".satsuma" / "settings.json", {"decomposition": {"enabled": False}})
    monkeypatch.setenv("SATSUMA_DECOMPOSITION", "1")
    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.decomposition.enabled is False

    monkeypatch.setenv("SATSUMA_ORCHESTRATION", "enable")
    with pytest.raises(ConfigError, match="SATSUMA_ORCHESTRATION"):
        ConfigLoader(cwd, config_home=home, trusted=True).load()


def test_env_selects_web_search_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SATSUMA_WEB_SEARCH_PROVIDER / _BASE_URL でバックエンドを選ぶ。"""
    cwd = tmp_path / "proj"
    cwd.mkdir()
    home = tmp_path / "empty_home"

    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.web_search.provider is None

    monkeypatch.setenv("SATSUMA_WEB_SEARCH_PROVIDER", "SearXNG")
    monkeypatch.setenv("SATSUMA_WEB_SEARCH_BASE_URL", "http://localhost:8081")
    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.web_search.provider == "searxng"
    assert cfg.web_search.base_url == "http://localhost:8081"

    monkeypatch.setenv("SATSUMA_WEB_SEARCH_PROVIDER", "none")
    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.web_search.provider is None

    _put_json(
        cwd / ".satsuma" / "settings.json",
        {"web_search": {"provider": "duckduckgo", "base_url": "http://explicit:8080"}},
    )
    monkeypatch.setenv("SATSUMA_WEB_SEARCH_PROVIDER", "searxng")
    monkeypatch.setenv("SATSUMA_WEB_SEARCH_BASE_URL", "http://env:9999")
    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.web_search.provider == "duckduckgo"
    assert cfg.web_search.base_url == "http://explicit:8080"

    _put_json(cwd / ".satsuma" / "settings.json", {})
    monkeypatch.setenv("SATSUMA_WEB_SEARCH_PROVIDER", "duckduck")
    with pytest.raises(ConfigError, match="SATSUMA_WEB_SEARCH_PROVIDER"):
        ConfigLoader(cwd, config_home=home, trusted=True).load()


def test_env_enables_web_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SATSUMA_WEB_FETCH_ENABLED で web_fetch を有効にする（未設定なら無効）。"""
    cwd = tmp_path / "proj"
    cwd.mkdir()
    home = tmp_path / "empty_home"

    assert ConfigLoader(cwd, config_home=home, trusted=True).load().web_fetch.enabled is False

    monkeypatch.setenv("SATSUMA_WEB_FETCH_ENABLED", "true")
    assert ConfigLoader(cwd, config_home=home, trusted=True).load().web_fetch.enabled is True

    _put_json(
        cwd / ".satsuma" / "settings.json", {"web_fetch": {"enabled": False, "max_bytes": 10}}
    )
    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.web_fetch.enabled is False and cfg.web_fetch.max_bytes == 10
    _put_json(cwd / ".satsuma" / "settings.json", {"web_fetch": {"max_bytes": 10}})
    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert cfg.web_fetch.enabled is True and cfg.web_fetch.max_bytes == 10

    monkeypatch.setenv("SATSUMA_WEB_FETCH_ENABLED", "ture")
    with pytest.raises(ConfigError, match="SATSUMA_WEB_FETCH_ENABLED"):
        ConfigLoader(cwd, config_home=home, trusted=True).load()


def test_defaults_when_no_files_no_env(tmp_path: Path) -> None:
    cwd = tmp_path / "proj"
    cwd.mkdir()

    cfg = ConfigLoader(cwd, config_home=tmp_path / "empty_home", trusted=True).load()

    assert cfg.provider.default_model == ""
    assert cfg.provider.base_url is None
    assert cfg.rag.embed_model == "bge-m3"
    assert cfg.permission.default_mode == "workspace-write"
