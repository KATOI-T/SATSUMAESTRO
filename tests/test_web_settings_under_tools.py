"""web_fetch と web_search の設定は tools の下（最上位の前の書き方も当面は読む）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from satsumaestro.config.loader import ConfigLoader
from satsumaestro.config.models import RuntimeConfig

_ENV = (
    "SATSUMA_WEB_SEARCH_PROVIDER",
    "SATSUMA_WEB_SEARCH_BASE_URL",
    "SATSUMA_WEB_FETCH_ENABLED",
    "SATSUMA_CONFIG_HOME",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)


def _put(path: Path, data: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    home, cwd = tmp_path / "home", tmp_path / "proj"
    home.mkdir()
    cwd.mkdir()
    return home, cwd


def _load(cwd: Path, home: Path, *, trusted: bool = True) -> tuple[RuntimeConfig, list[str]]:
    loader = ConfigLoader(cwd, config_home=home, trusted=trusted)
    return loader.load(), loader.notes()


def test_web_settings_live_under_tools(tmp_path: Path) -> None:
    home, cwd = _dirs(tmp_path)
    tools = {
        "web_fetch": {"enabled": True, "allowed_hosts": ["example.com"]},
        "web_search": {"provider": "duckduckgo"},
    }
    _put(cwd / ".satsuma" / "settings.local.json", {"tools": tools})
    cfg, notes = _load(cwd, home)
    assert cfg.tools.web_fetch.enabled and cfg.tools.web_fetch.allowed_hosts == ["example.com"]
    assert cfg.tools.web_search.provider == "duckduckgo"
    assert notes == []


@pytest.mark.parametrize("key", ["web_fetch", "web_search"])
def test_the_settings_model_has_no_top_level_web_settings(key: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate({key: {}})


@pytest.mark.parametrize("trusted", [True, False], ids=["trusted", "untrusted"])
def test_the_old_top_level_is_still_read_with_a_note(tmp_path: Path, trusted: bool) -> None:
    home, cwd = _dirs(tmp_path)
    old = {"web_fetch": {"enabled": True}, "web_search": {"provider": "duckduckgo"}}
    user = _put(home / "settings.json", old)
    cfg, notes = _load(cwd, home, trusted=trusted)
    assert cfg.tools.web_fetch.enabled and cfg.tools.web_search.provider == "duckduckgo"
    [note] = notes
    assert str(user) in note and "tools.web_fetch" in note and "tools.web_search" in note


def test_in_one_file_the_tools_place_wins(tmp_path: Path) -> None:
    home, cwd = _dirs(tmp_path)
    both = {
        "web_fetch": {"enabled": True, "max_bytes": 10},
        "tools": {"web_fetch": {"max_bytes": 20}, "syntax_check": False},
    }
    _put(cwd / ".satsuma.json", both)
    cfg, _ = _load(cwd, home)
    assert cfg.tools.web_fetch.enabled and cfg.tools.web_fetch.max_bytes == 20
    assert cfg.tools.syntax_check is False


def test_a_later_file_still_wins_whichever_place_it_uses(tmp_path: Path) -> None:
    home, cwd = _dirs(tmp_path)
    first = {"tools": {"web_fetch": {"enabled": True, "allowed_hosts": ["a.example"]}}}
    _put(home / "settings.json", first)
    shared = _put(cwd / ".satsuma.json", {"web_fetch": {"allowed_hosts": ["b.example"]}})
    local = _put(cwd / ".satsuma" / "settings.local.json", {"web_search": {"max_results": 3}})
    cfg, notes = _load(cwd, home)
    assert cfg.tools.web_fetch.enabled and cfg.tools.web_fetch.allowed_hosts == ["b.example"]
    assert cfg.tools.web_search.max_results == 3
    assert [str(shared) in notes[0], str(local) in notes[1]] == [True, True]


def test_environment_variables_fill_the_tools_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, cwd = _dirs(tmp_path)
    monkeypatch.setenv("SATSUMA_WEB_FETCH_ENABLED", "true")
    monkeypatch.setenv("SATSUMA_WEB_SEARCH_PROVIDER", "searxng")
    monkeypatch.setenv("SATSUMA_WEB_SEARCH_BASE_URL", "http://localhost:8081")
    _put(cwd / ".satsuma.json", {"tools": {"web_fetch": {"max_bytes": 10}}})
    cfg, notes = _load(cwd, home)
    assert cfg.tools.web_fetch.enabled and cfg.tools.web_fetch.max_bytes == 10
    assert cfg.tools.web_search.provider == "searxng"
    assert cfg.tools.web_search.base_url == "http://localhost:8081"
    assert notes == []


def test_a_setting_in_the_old_place_beats_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, cwd = _dirs(tmp_path)
    monkeypatch.setenv("SATSUMA_WEB_FETCH_ENABLED", "true")
    _put(cwd / ".satsuma.json", {"web_fetch": {"enabled": False}})
    cfg, _ = _load(cwd, home)
    assert cfg.tools.web_fetch.enabled is False
