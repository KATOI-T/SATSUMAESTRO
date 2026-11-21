"""Ollama・LM Studio・llama.cpp に届かないときの文は、接続先の出どころと直す所を出す。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from satsumaestro.config.loader import ConfigLoader, base_url_file, save_provider
from satsumaestro.config.models import ProviderConfig, ProviderProfile, RuntimeConfig
from satsumaestro.engine.context_fit import check_reachable
from satsumaestro.engine.provider_switch import switch_provider
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.providers.choice import HOST_ENV
from satsumaestro.util.environ import dotenv_file_of, note_dotenv_file
from satsumaestro.util.errors import ConfigError
from satsumaestro.util.paths import default_config_home
from tests.test_provider_switch import _runtime

_DOWN = "http://127.0.0.1:11434"
_ELSEWHERE = "10.0.0.5"


@pytest.fixture(autouse=True)
def _japanese_screen() -> Iterator[None]:
    with pinned_ui_language("ja"):
        yield


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("Connection refused")


async def _problem(cfg: RuntimeConfig, env: dict[str, str], root: str = _DOWN) -> str:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_refused)) as client:
        problem = await check_reachable(cfg, "qwen3:8b", env, client=client)
    assert problem is not None and root in problem
    return problem


async def test_a_url_from_the_settings_points_to_the_settings() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(kind="ollama", base_url=f"{_DOWN}/v1"))
    problem = await _problem(cfg, {"OLLAMA_HOST": _ELSEWHERE})
    assert "provider.base_url" in problem
    assert str(default_config_home() / "settings.json") in problem
    assert "/provider ollama <URL>" in problem
    assert "OLLAMA_HOST" not in problem and ".env" not in problem


async def test_ollama_host_from_dotenv_names_the_file(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    note_dotenv_file(dotenv, ["OLLAMA_HOST"])
    cfg = RuntimeConfig(provider=ProviderConfig(base_url=_DOWN))
    problem = await _problem(cfg, {"OLLAMA_HOST": _DOWN})
    assert f"{dotenv} の OLLAMA_HOST" in problem
    assert "シェルの環境変数より優先" in problem
    assert "provider.base_url" not in problem


async def test_ollama_host_from_the_shell_does_not_mention_dotenv() -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(kind="ollama"))
    problem = await _problem(cfg, {"OLLAMA_HOST": _DOWN})
    assert "環境変数 OLLAMA_HOST" in problem
    assert ".env" not in problem


async def test_openai_base_url_on_the_ollama_port_names_it() -> None:
    problem = await _problem(RuntimeConfig(), {"OPENAI_BASE_URL": f"{_DOWN}/v1"})
    assert "OPENAI_BASE_URL" in problem
    assert "OLLAMA_HOST" not in problem and "provider.base_url" not in problem


@pytest.fixture
def _all_down(monkeypatch: pytest.MonkeyPatch) -> None:
    real = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(_refused)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)


@pytest.mark.usefixtures("_all_down")
async def test_a_url_typed_after_provider_gets_no_hint(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, RuntimeConfig(provider=ProviderConfig(kind="anthropic")), "m")
    with pytest.raises(ConfigError) as caught:
        await switch_provider(rt, "ollama", env={}, url="127.0.0.1", model="qwen3:8b")
    text = str(caught.value)
    assert _DOWN in text
    assert "接続先は" not in text


@pytest.mark.usefixtures("_all_down")
async def test_a_remembered_url_points_to_provider(tmp_path: Path) -> None:
    profiles = {"ollama": ProviderProfile(base_url=f"{_DOWN}/v1", model="qwen3:8b")}
    cfg = RuntimeConfig(provider=ProviderConfig(kind="anthropic", profiles=profiles))
    rt = _runtime(tmp_path, cfg, "m")
    with pytest.raises(ConfigError) as caught:
        await switch_provider(rt, "ollama", env={"OLLAMA_HOST": _ELSEWHERE})
    text = str(caught.value)
    assert _DOWN in text and "`/provider` が覚えた URL" in text
    assert "provider.base_url" not in text and "OLLAMA_HOST" not in text


@pytest.mark.parametrize("into_environ", [True, False])
def test_the_loader_notes_ollama_host_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, into_environ: bool
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", _ELSEWHERE)
    (tmp_path / ".env").write_text(f"OLLAMA_HOST={_DOWN}\nOLLAMA_MODEL=\n", encoding="utf-8")
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text(
        f'{{"dotenv": {{"into_environ": {str(into_environ).lower()}}}}}', encoding="utf-8"
    )
    ConfigLoader(tmp_path, trusted=True).load()
    assert dotenv_file_of("OLLAMA_HOST") == tmp_path / ".env"
    assert dotenv_file_of("OLLAMA_MODEL") is None
    ConfigLoader(tmp_path, trusted=False).load()
    assert dotenv_file_of("OLLAMA_HOST") is None


def _layers(tmp_path: Path, *, user: bool, project: bool) -> tuple[Path, Path]:
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    (work / ".satsuma").mkdir(parents=True)
    body = json.dumps({"provider": {"kind": "ollama", "base_url": f"{_DOWN}/v1"}})
    if user:
        (home / "settings.json").write_text(body, encoding="utf-8")
    if project:
        (work / ".satsuma" / "settings.local.json").write_text(body, encoding="utf-8")
    return home, work


@pytest.mark.parametrize(
    ("user", "project", "trusted", "expected"),
    [
        (False, True, True, "project"),
        (True, True, True, "user"),
        (True, False, True, "user"),
        (False, True, False, None),
    ],
)
def test_the_loader_notes_the_file_that_set_base_url(
    tmp_path: Path, user: bool, project: bool, trusted: bool, expected: str | None
) -> None:
    home, work = _layers(tmp_path, user=user, project=project)
    cfg = ConfigLoader(work, config_home=home, trusted=trusted).load()
    files = {
        "user": home / "settings.json",
        "project": (work / ".satsuma" / "settings.local.json").resolve(),
    }
    found = base_url_file(cfg.provider.base_url)
    assert (found.resolve() if found else None) == (files[expected] if expected else None)


async def test_a_url_from_the_project_settings_names_that_file(tmp_path: Path) -> None:
    home, work = _layers(tmp_path, user=False, project=True)
    cfg = ConfigLoader(work, config_home=home, trusted=True).load()
    problem = await _problem(cfg, {})
    assert "settings.local.json の provider.base_url" in problem
    assert str(home / "settings.json") not in problem


def test_provider_notes_the_user_settings_it_writes(tmp_path: Path) -> None:
    home, work = _layers(tmp_path, user=False, project=True)
    ConfigLoader(work, config_home=home, trusted=True).load()
    url = "http://10.0.0.7:11434/v1"
    save_provider("ollama", "qwen3:8b", url, config_home=home)
    assert base_url_file(url) == home / "settings.json"
    assert base_url_file(f"{_DOWN}/v1") is None


_OTHERS = [
    pytest.param("lmstudio", "http://127.0.0.1:1234", "LM Studio に接続できません", id="lmstudio"),
    pytest.param("llamacpp", "http://127.0.0.1:9931", "llama.cpp に接続できません", id="llamacpp"),
]


@pytest.mark.parametrize(("kind", "root", "head"), _OTHERS)
async def test_lmstudio_and_llamacpp_point_to_the_settings(kind: str, root: str, head: str) -> None:
    cfg = RuntimeConfig(provider=ProviderConfig(kind=kind, base_url=f"{root}/v1"))
    problem = await _problem(cfg, {}, root)
    assert head in problem
    assert f"{default_config_home() / 'settings.json'} の provider.base_url" in problem
    assert f"/provider {kind} <URL>" in problem


@pytest.mark.parametrize(("kind", "root", "head"), _OTHERS)
async def test_lmstudio_and_llamacpp_name_their_host_variable(
    kind: str, root: str, head: str
) -> None:
    var = HOST_ENV[kind]
    problem = await _problem(RuntimeConfig(provider=ProviderConfig(kind=kind)), {var: root}, root)
    assert head in problem and f"接続先は環境変数か .env の {var} です" in problem
    assert "provider.base_url" not in problem


@pytest.mark.usefixtures("_all_down")
@pytest.mark.parametrize(("kind", "root", "head"), _OTHERS)
async def test_a_remembered_lmstudio_or_llamacpp_url_points_to_provider(
    tmp_path: Path, kind: str, root: str, head: str
) -> None:
    profiles = {kind: ProviderProfile(base_url=f"{root}/v1", model="m")}
    cfg = RuntimeConfig(provider=ProviderConfig(kind="anthropic", profiles=profiles))
    rt = _runtime(tmp_path, cfg, "m")
    with pytest.raises(ConfigError) as caught:
        await switch_provider(rt, kind, env={})
    text = str(caught.value)
    assert head in text and "`/provider` が覚えた URL" in text
    assert f"/provider {kind} <URL>" in text
