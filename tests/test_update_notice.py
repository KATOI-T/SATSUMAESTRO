"""REPL の起動時の新しい版の知らせ（保存した結果から出し、裏で 1 日 1 回取り直す）。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from satsumaestro.cli import repl as repl_mod
from satsumaestro.cli import updates
from satsumaestro.cli.updates import CachedCheck, UpdateCheck, load_cached, save_cached
from satsumaestro.config.loader import ConfigLoader
from satsumaestro.config.models import RuntimeConfig
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import ConfigError
from satsumaestro.util.install import Install
from tests.fixtures.mock_provider import MockProvider
from tests.test_key_check import _SlowLines
from tests.test_p165_plan_steps import _runtime
from tests.test_p170_slash_path_passthrough import _FakeRender

_UV = Install("uv", ("uv", "tool", "upgrade", "satsumaestro"), "/opt/bin/uv")
_BREW = Install("brew", ("brew", "upgrade", "satsumaestro"), "/opt/homebrew/bin/brew")


def _pypi(version: str, seen: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"info": {"version": version}})

    return httpx.MockTransport(handle)


def _check(path: Path, install: Install = _UV, **kwargs: object) -> UpdateCheck:
    return UpdateCheck(install, path, current="0.5.3", **kwargs)  # type: ignore[arg-type]


# ── 知らせ ──────────────────────────────────────────────


def test_a_newer_stored_version_gives_one_line_pointing_to_the_update_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "update_check.json"
    save_cached(path, CachedCheck(1, "pypi", "0.5.4"))
    monkeypatch.setattr(updates.sys, "argv", ["/opt/homebrew/bin/stro"])

    with pinned_ui_language("en"):
        english = _check(path).notice()
    with pinned_ui_language("ja"):
        japanese = _check(path).notice()

    assert english == "SATSUMAESTRO 0.5.4 is available (you have 0.5.3). To upgrade: stro update"
    assert japanese == "SATSUMAESTRO 0.5.4 が出ています（今は 0.5.3）。更新: stro update"


@pytest.mark.parametrize(
    ("argv0", "name"),
    [
        ("/opt/homebrew/bin/satsuma", "satsuma"),
        ("satsumaestro", "satsumaestro"),
        ("C:/Users/me/.local/bin/stro.exe", "stro"),
        ("/repo/satsumaestro/__main__.py", "satsumaestro"),
        ("/usr/bin/pytest", "satsumaestro"),
    ],
)
def test_the_update_command_uses_the_name_the_user_started_with(argv0: str, name: str) -> None:
    assert updates.program_name(argv0) == name


@pytest.mark.parametrize(
    "cached",
    [
        None,
        CachedCheck(1, "pypi", "0.5.3"),
        CachedCheck(1, "pypi", "0.5.2"),
        CachedCheck(1, "pypi", None),
        CachedCheck(1, "tap", "0.5.4"),
    ],
)
def test_no_line_without_a_newer_version_from_the_same_source(
    tmp_path: Path, cached: CachedCheck | None
) -> None:
    path = tmp_path / "update_check.json"
    if cached is not None:
        save_cached(path, cached)

    assert _check(path).notice() is None


def test_homebrew_reads_the_version_from_the_tap(tmp_path: Path) -> None:
    path = tmp_path / "update_check.json"
    save_cached(path, CachedCheck(1, "tap", "0.5.4"))

    with pinned_ui_language("en"):
        line = _check(path, _BREW).notice()

    assert line is not None and line.startswith("SATSUMAESTRO 0.5.4 is available")


# ── 裏の取り直し ────────────────────────────────────────


async def test_refresh_asks_again_after_a_day_and_not_before(tmp_path: Path) -> None:
    path = tmp_path / "update_check.json"
    seen: list[httpx.Request] = []
    save_cached(path, CachedCheck(now_ms(), "pypi", "0.5.3"))

    await _check(path, transport=_pypi("0.5.4", seen)).refresh()
    assert seen == []

    save_cached(path, CachedCheck(1, "pypi", "0.5.3"))
    await _check(path, transport=_pypi("0.5.4", seen)).refresh()
    assert len(seen) == 1
    cached = load_cached(path)
    assert cached is not None and cached.latest == "0.5.4"


async def test_refresh_swallows_any_failure(tmp_path: Path) -> None:
    path = tmp_path / "update_check.json"

    def boom(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom")

    await _check(path, transport=httpx.MockTransport(boom)).refresh()

    assert load_cached(path) is None


# ── 確かめる所と止め方 ──────────────────────────────────


def test_the_repl_check_runs_only_in_a_terminal_outside_workers_and_editable_installs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(updates, "detect_install", lambda: _UV)
    terminal = Console(force_terminal=True)

    made = UpdateCheck.for_repl(RuntimeConfig(), terminal, {})
    assert made is not None and made.path.name == "update_check.json"
    assert UpdateCheck.for_repl(RuntimeConfig(update_check=False), terminal, {}) is None
    assert UpdateCheck.for_repl(RuntimeConfig(), Console(file=io.StringIO()), {}) is None
    assert UpdateCheck.for_repl(RuntimeConfig(), terminal, {"SATSUMA_WORKER_CHILD": "1"}) is None

    monkeypatch.setattr(updates, "detect_install", lambda: Install("editable", ()))
    assert UpdateCheck.for_repl(RuntimeConfig(), terminal, {}) is None


def _load(tmp_path: Path, settings: dict[str, object] | None = None) -> RuntimeConfig:
    home, cwd = tmp_path / "home", tmp_path / "proj"
    home.mkdir(exist_ok=True)
    cwd.mkdir(exist_ok=True)
    if settings is not None:
        (home / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    return ConfigLoader(cwd, config_home=home, trusted=True).load()


def test_the_check_is_on_by_default_and_the_setting_turns_it_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SATSUMA_NO_UPDATE_CHECK", raising=False)

    assert _load(tmp_path).update_check is True
    assert _load(tmp_path, {"update_check": False}).update_check is False


def test_the_environment_variable_turns_it_off_over_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SATSUMA_NO_UPDATE_CHECK", "1")
    assert _load(tmp_path, {"update_check": True}).update_check is False

    monkeypatch.setenv("SATSUMA_NO_UPDATE_CHECK", "0")
    assert _load(tmp_path, {"update_check": True}).update_check is True

    monkeypatch.setenv("SATSUMA_NO_UPDATE_CHECK", "maybe")
    with pytest.raises(ConfigError):
        _load(tmp_path)


# ── REPL ────────────────────────────────────────────────


async def test_the_repl_shows_the_stored_notice_and_refreshes_in_the_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _SlowLines([]))
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    path = tmp_path / "update_check.json"
    save_cached(path, CachedCheck(1, "pypi", "0.5.4"))
    seen: list[httpx.Request] = []
    console = Console(record=True, width=200)

    with pinned_ui_language("en"):
        await repl_mod.run_repl(  # type: ignore[arg-type]
            _runtime(MockProvider([]), tmp_path),
            None,
            _FakeRender(),
            console,
            updates=_check(path, transport=_pypi("0.5.5", seen)),
        )

    out = console.export_text()
    assert "SATSUMAESTRO 0.5.4 is available (you have 0.5.3)" in out
    assert "0.5.5" not in out
    line = next(ln for ln in out.splitlines() if "is available" in ln)
    lines = out.splitlines()
    assert "/help" in lines[lines.index(line) - 1]  # 操作の案内のすぐ下
    left, right = len(line) - len(line.lstrip()), 200 - len(line.rstrip())
    assert left > 0 and abs(left - right) <= 1, line
    assert len(seen) == 1
    cached = load_cached(path)
    assert cached is not None and cached.latest == "0.5.5"


async def test_the_repl_without_a_check_prints_no_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repl_mod, "PromptSession", lambda **_: _SlowLines([]))
    monkeypatch.setattr(repl_mod, "mode_prompt_fragments", lambda _: "> ")
    console = Console(record=True, width=200)

    with pinned_ui_language("en"):
        await repl_mod.run_repl(  # type: ignore[arg-type]
            _runtime(MockProvider([]), tmp_path), None, _FakeRender(), console
        )

    assert "is available" not in console.export_text()
