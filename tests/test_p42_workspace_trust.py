"""信頼したディレクトリの記録と、信頼に応じた設定の読み方。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from satsumaestro.config.loader import ConfigLoader
from satsumaestro.security.trust import TrustStore, trust_root
from satsumaestro.util.paths import find_git_root, find_workspace_root

_ENV_VARS = ("OLLAMA_HOST", "OLLAMA_MODEL", "SATSUMA_TRUST_PROBE", "SATSUMA_CONFIG_HOME")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_VARS:
        monkeypatch.setenv(key, "_")
        monkeypatch.delenv(key)


def _put_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _project(root: Path) -> Path:
    """②〜④ の層と .env をすべて置いたディレクトリ。"""
    root.mkdir(parents=True, exist_ok=True)
    _put_json(root / ".satsuma.json", {"max_iterations": 20})
    _put_json(root / ".satsuma" / "settings.json", {"provider": {"default_model": "project"}})
    _put_json(root / ".satsuma" / "settings.local.json", {"hooks": {"pre_tool_use": ["echo x"]}})
    (root / ".env").write_text(
        "OLLAMA_HOST=http://from-dotenv:11434/v1\nSATSUMA_TRUST_PROBE=loaded\n", encoding="utf-8"
    )
    return root


def test_trust_is_recorded_and_covers_the_tree(tmp_path: Path) -> None:
    store = TrustStore(tmp_path / "home" / "trusted.json")
    app = tmp_path / "srv" / "app"
    (app / "api").mkdir(parents=True)
    assert not store.is_trusted(app)

    store.trust(app)
    store.trust(app)
    assert store.roots() == [app.resolve()]
    assert json.loads(store.path.read_text(encoding="utf-8")) == {
        "version": 1,
        "roots": [str(app.resolve())],
    }
    assert store.is_trusted(app) and store.is_trusted(app / "api")
    assert not store.is_trusted(tmp_path / "srv" / "application")
    assert not store.is_trusted(tmp_path / "srv")

    assert store.untrust(app) is True and store.untrust(app) is False
    assert store.roots() == [] and not store.is_trusted(app)


@pytest.mark.parametrize(
    "content",
    ["", "{broken", "[]", '{"roots": "/x"}', '{"roots": ["relative/dir", 3]}'],
)
def test_an_unreadable_record_trusts_nothing(tmp_path: Path, content: str) -> None:
    path = tmp_path / "trusted.json"
    path.write_text(content, encoding="utf-8")
    store = TrustStore(path)
    assert store.roots() == [] and not store.is_trusted(tmp_path)


def test_the_store_defaults_to_the_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    assert TrustStore().path == tmp_path / "home" / "trusted.json"


def test_the_root_is_the_git_root_or_the_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / ".git").mkdir()
    assert find_git_root(repo / "sub") == trust_root(repo / "sub") == repo.resolve()

    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    assert trust_root(wt) == wt.resolve()

    plain = tmp_path / "plain" / "x"
    plain.mkdir(parents=True)
    assert find_git_root(plain) is None
    assert trust_root(plain) == plain.resolve()


def test_trusted_reads_every_layer_and_dotenv(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _put_json(home / "settings.json", {"max_iterations": 10})
    cwd = _project(tmp_path / "proj")

    cfg = ConfigLoader(cwd, config_home=home, trusted=True).load()

    assert cfg.max_iterations == 20
    assert cfg.provider.default_model == "project"
    assert cfg.hooks.pre_tool_use == ["echo x"]
    assert cfg.provider.base_url == "http://from-dotenv:11434/v1"
    assert os.environ["SATSUMA_TRUST_PROBE"] == "loaded"


def test_untrusted_reads_only_the_user_settings_and_the_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    _put_json(home / "settings.json", {"max_iterations": 10})
    cwd = _project(tmp_path / "proj")
    monkeypatch.setenv("OLLAMA_MODEL", "from-shell")

    cfg = ConfigLoader(cwd, config_home=home, trusted=False).load()

    assert cfg.max_iterations == 10
    assert cfg.hooks.pre_tool_use == []
    assert cfg.provider.default_model == "from-shell"
    assert cfg.provider.base_url != "http://from-dotenv:11434/v1"
    assert "SATSUMA_TRUST_PROBE" not in os.environ


def test_outside_git_the_parents_are_not_searched(tmp_path: Path) -> None:
    _project(tmp_path / "shared")
    cwd = tmp_path / "shared" / "x"
    cwd.mkdir()

    cfg = ConfigLoader(cwd, config_home=tmp_path / "home", trusted=True).load()

    assert cfg.max_iterations != 20 and cfg.provider.default_model != "project"
    assert cfg.hooks.pre_tool_use == []


def test_inside_git_the_search_stops_at_the_git_root(tmp_path: Path) -> None:
    _project(tmp_path / "outer")
    repo = tmp_path / "outer" / "repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "pkg" / "sub"
    sub.mkdir(parents=True)

    cfg = ConfigLoader(sub, config_home=tmp_path / "home", trusted=True).load()
    assert cfg.max_iterations != 20 and cfg.hooks.pre_tool_use == []

    _put_json(repo / ".satsuma" / "settings.json", {"max_iterations": 7})
    cfg = ConfigLoader(sub, config_home=tmp_path / "home", trusted=True).load()
    assert cfg.max_iterations == 7


def test_the_config_home_is_not_read_again_as_a_project(tmp_path: Path) -> None:
    home = tmp_path / ".satsuma"
    _put_json(home / "settings.json", {"max_iterations": 10})
    _put_json(home / "settings.local.json", {"max_iterations": 5})
    _put_json(tmp_path / ".satsuma.json", {"max_iterations": 20})

    cfg = ConfigLoader(tmp_path, config_home=home, trusted=True).load()

    assert cfg.max_iterations == 20


def test_the_config_home_is_not_a_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / ".satsuma"))
    (tmp_path / ".satsuma").mkdir()
    proj = tmp_path / "proj"
    proj.mkdir()
    assert find_workspace_root(proj) == proj.resolve()

    (tmp_path / "proj" / ".satsuma").mkdir()
    assert find_workspace_root(proj / ".satsuma") == proj.resolve()
