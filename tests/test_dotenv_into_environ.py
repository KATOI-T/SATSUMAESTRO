"""``.env`` を ``os.environ`` に書かずに読む設定。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from satsumaestro.config.loader import ConfigLoader
from satsumaestro.config.models import RuntimeConfig, SandboxConfig
from satsumaestro.orchestration.worker_supervisor import worker_environment
from satsumaestro.providers.factory import build_bare
from satsumaestro.security.sandbox import child_environment
from satsumaestro.session.redact import secret_values
from satsumaestro.util.environ import ENVIRON, WORKER_DOTENV_ENV, adopt_passed_dotenv
from satsumaestro.util.errors import ConfigError

_KEY = "sk-ant-api03-dotenvonlyabcdefghijklmnopqrstuvwxyz"
_DOTENV = (
    "DOTENV_PROBE=from-dotenv\n"
    "SHARED_PROBE=from-dotenv\n"
    "OLLAMA_HOST=http://10.0.0.5:11434/v1\n"
    "SATSUMA_EMBED_MODEL=embed-from-dotenv\n"
    f"ANTHROPIC_API_KEY={_KEY}\n"
)


def _project(tmp_path: Path, settings: dict[str, object] | None) -> Path:
    cwd = tmp_path / "proj"
    (cwd / ".satsuma").mkdir(parents=True)
    (cwd / ".env").write_text(_DOTENV, encoding="utf-8")
    if settings is not None:
        (cwd / ".satsuma" / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return cwd


def _load(tmp_path: Path, settings: dict[str, object] | None = None) -> RuntimeConfig:
    cwd = _project(tmp_path, {"dotenv": {"into_environ": False}} if settings is None else settings)
    return ConfigLoader(cwd, config_home=tmp_path / "home", trusted=True).load()


@pytest.fixture(autouse=True)
def _clean_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DOTENV_PROBE", "SHARED_PROBE", "SATSUMA_EMBED_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_into_environ_is_on_by_default() -> None:
    assert RuntimeConfig().dotenv.into_environ is True


def test_the_values_are_read_without_writing_os_environ(tmp_path: Path) -> None:
    cfg = _load(tmp_path)
    assert "DOTENV_PROBE" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ
    assert ENVIRON["DOTENV_PROBE"] == "from-dotenv" and ENVIRON["ANTHROPIC_API_KEY"] == _KEY
    assert cfg.provider.base_url == "http://10.0.0.5:11434/v1"
    assert cfg.rag.embed_model == "embed-from-dotenv"


def test_the_shell_wins_except_for_ollama(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARED_PROBE", "from-shell")
    monkeypatch.setenv("OLLAMA_HOST", "http://192.0.2.10:11434")
    cfg = _load(tmp_path)
    assert ENVIRON["SHARED_PROBE"] == "from-shell"
    assert ENVIRON["OLLAMA_HOST"] == "http://10.0.0.5:11434/v1"
    assert cfg.provider.base_url == "http://10.0.0.5:11434/v1"
    assert os.environ["OLLAMA_HOST"] == "http://192.0.2.10:11434"


@pytest.mark.parametrize("into_environ", [True, False])
def test_references_resolve_the_same_either_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, into_environ: bool
) -> None:
    monkeypatch.setenv("HOST_PROBE", "prod")
    monkeypatch.setenv("OLLAMA_HOST", "http://shell:11434/v1")
    monkeypatch.delenv("URL_PROBE", raising=False)
    cwd = _project(tmp_path, {"dotenv": {"into_environ": into_environ}})
    (cwd / ".env").write_text(
        "HOST_PROBE=dev\n"
        "URL_PROBE=https://${HOST_PROBE}/v1\n"
        "OLLAMA_HOST=http://${HOST_PROBE}:11434/v1\n",
        encoding="utf-8",
    )
    ConfigLoader(cwd, config_home=tmp_path / "home", trusted=True).load()
    assert ENVIRON["HOST_PROBE"] == "prod"
    assert ENVIRON["URL_PROBE"] == "https://prod/v1"
    assert ENVIRON["OLLAMA_HOST"] == "http://dev:11434/v1"


def test_the_user_settings_can_turn_it_off(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.json").write_text(
        json.dumps({"dotenv": {"into_environ": False}}), encoding="utf-8"
    )
    cwd = _project(tmp_path, None)
    ConfigLoader(cwd, config_home=home, trusted=True).load()
    assert "DOTENV_PROBE" not in os.environ and ENVIRON["DOTENV_PROBE"] == "from-dotenv"


def test_load_env_alone_follows_the_setting(tmp_path: Path) -> None:
    cwd = _project(tmp_path, {"dotenv": {"into_environ": False}})
    ConfigLoader(cwd, config_home=tmp_path / "home", trusted=True).load_env()
    assert "DOTENV_PROBE" not in os.environ and ENVIRON["DOTENV_PROBE"] == "from-dotenv"


def test_an_untrusted_folder_reads_no_dotenv(tmp_path: Path) -> None:
    cwd = _project(tmp_path, {"dotenv": {"into_environ": False}})
    ConfigLoader(cwd, config_home=tmp_path / "home", trusted=False).load()
    assert "DOTENV_PROBE" not in ENVIRON


def test_an_untrusted_folder_does_not_reuse_the_previous_values(tmp_path: Path) -> None:
    _load(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    cfg = ConfigLoader(other, config_home=tmp_path / "home", trusted=False).load()
    assert "DOTENV_PROBE" not in ENVIRON and "ANTHROPIC_API_KEY" not in ENVIRON
    assert cfg.rag.embed_model != "embed-from-dotenv"


def test_a_removed_dotenv_is_forgotten(tmp_path: Path) -> None:
    cwd = _project(tmp_path, {"dotenv": {"into_environ": False}})
    loader = ConfigLoader(cwd, config_home=tmp_path / "home", trusted=True)
    loader.load()
    (cwd / ".env").unlink()
    loader.load()
    assert "DOTENV_PROBE" not in ENVIRON


def test_a_bad_value_is_a_settings_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        _load(tmp_path, {"dotenv": {"into_environ": "sometimes"}})


def test_child_processes_do_not_get_the_values(tmp_path: Path) -> None:
    _load(tmp_path)
    env = child_environment(SandboxConfig())
    assert "DOTENV_PROBE" not in env and "SATSUMA_EMBED_MODEL" not in env


def test_workers_get_only_the_names_the_harness_reads(tmp_path: Path) -> None:
    _load(tmp_path)
    env = worker_environment()
    passed = json.loads(env[WORKER_DOTENV_ENV])
    assert passed == {
        "ANTHROPIC_API_KEY": _KEY,
        "OLLAMA_HOST": "http://10.0.0.5:11434/v1",
        "SATSUMA_EMBED_MODEL": "embed-from-dotenv",
    }
    assert "ANTHROPIC_API_KEY" not in env and "SATSUMA_EMBED_MODEL" not in env
    assert env.get("OLLAMA_HOST") == os.environ.get("OLLAMA_HOST")
    assert env["PATH"] == os.environ["PATH"]


def test_a_worker_keeps_the_passed_values_out_of_its_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _load(tmp_path)
    monkeypatch.setenv(WORKER_DOTENV_ENV, worker_environment()[WORKER_DOTENV_ENV])
    ConfigLoader(tmp_path, config_home=tmp_path / "home", trusted=False).load()
    adopt_passed_dotenv()
    assert WORKER_DOTENV_ENV not in os.environ
    assert ENVIRON["ANTHROPIC_API_KEY"] == _KEY
    assert ENVIRON["OLLAMA_HOST"] == "http://10.0.0.5:11434/v1"
    cfg = ConfigLoader(tmp_path, config_home=tmp_path / "home", trusted=False).load()
    assert cfg.provider.base_url == "http://10.0.0.5:11434/v1"
    assert _KEY in secret_values()
    env = child_environment(SandboxConfig())
    assert "SATSUMA_EMBED_MODEL" not in env and WORKER_DOTENV_ENV not in env
    assert env.get("OLLAMA_HOST") == os.environ.get("OLLAMA_HOST")


def test_the_key_from_dotenv_is_redacted_in_records(tmp_path: Path) -> None:
    _load(tmp_path)
    assert _KEY in secret_values()


def test_the_provider_takes_the_key_from_dotenv(tmp_path: Path) -> None:
    cfg = _load(tmp_path, {"dotenv": {"into_environ": False}, "provider": {"kind": "anthropic"}})
    provider = build_bare(cfg, "claude-opus-5-5")
    assert type(provider).__name__ == "AnthropicProvider"
