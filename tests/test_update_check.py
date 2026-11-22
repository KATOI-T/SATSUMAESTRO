"""入れ方の判定と、最新の版の確かめ（PyPI・tap の formula・1 日のキャッシュ）。"""

from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import httpx
import pytest

from satsumaestro.cli import updates
from satsumaestro.cli.updates import (
    CHECK_INTERVAL_MS,
    PYPI_URL,
    TAP_FORMULA_URL,
    CachedCheck,
    check_now,
    is_due,
    is_newer,
    latest_version,
    load_cached,
    parse_version,
    save_cached,
    source_for,
)
from satsumaestro.util import install as install_mod
from satsumaestro.util.install import Install, detect_install, installed_editable, is_editable

_FORMULA = """\
class Satsumaestro < Formula
  include Language::Python::Virtualenv

  desc "Local-first coding agent"
  url "https://files.pythonhosted.org/packages/97/14/abc/satsumaestro-0.5.4.tar.gz"
  sha256 "0000"

  resource "anthropic" do
    url "https://files.pythonhosted.org/packages/aa/bb/anthropic-0.99.0.tar.gz"
    sha256 "1111"
  end
end
"""


def _no_tools(_name: str) -> str | None:
    return None


def _tools(name: str) -> str | None:
    return f"/usr/local/bin/{name}"


# ── 入れ方の判定 ────────────────────────────────────────


def test_a_keg_under_the_cellar_is_a_homebrew_install(tmp_path: Path) -> None:
    keg = tmp_path / "Cellar" / "satsumaestro" / "0.5.3" / "libexec"
    keg.mkdir(parents=True)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "brew").write_text("")

    found = detect_install(keg, editable=False, windows=False, which=_tools)

    assert found.kind == "brew"
    assert found.shown == "brew upgrade satsumaestro"
    assert found.argv == [str((tmp_path / "bin" / "brew").resolve()), "upgrade", "satsumaestro"]


def test_homebrew_falls_back_to_the_brew_on_path_and_then_only_shows_the_command(
    tmp_path: Path,
) -> None:
    keg = tmp_path / "Cellar" / "satsumaestro" / "0.5.3" / "libexec"
    keg.mkdir(parents=True)

    on_path = detect_install(keg, editable=False, windows=False, which=_tools)
    missing = detect_install(keg, editable=False, windows=False, which=_no_tools)

    assert on_path.argv == ["/usr/local/bin/brew", "upgrade", "satsumaestro"]
    assert missing.kind == "brew"
    assert not missing.runnable
    assert missing.argv == []


def test_another_formula_in_the_cellar_is_not_taken_for_satsumaestro(tmp_path: Path) -> None:
    other = tmp_path / "Cellar" / "python@3.15" / "3.15.0" / "libexec"
    other.mkdir(parents=True)

    assert detect_install(other, editable=False, windows=False, which=_tools).kind == "pip"


@pytest.mark.parametrize(
    ("marker", "kind", "shown"),
    [
        ("uv-receipt.toml", "uv", "uv tool upgrade satsumaestro"),
        ("pipx_metadata.json", "pipx", "pipx upgrade satsumaestro"),
    ],
)
def test_uv_tool_and_pipx_are_found_by_the_file_they_leave_in_the_environment(
    tmp_path: Path, marker: str, kind: str, shown: str
) -> None:
    (tmp_path / marker).write_text("")

    found = detect_install(tmp_path, editable=False, windows=False, which=_tools)
    on_windows = detect_install(tmp_path, editable=False, windows=True, which=_tools)
    no_tool = detect_install(tmp_path, editable=False, windows=False, which=_no_tools)

    assert (found.kind, found.shown, found.runnable) == (kind, shown, True)
    assert found.argv[0] == f"/usr/local/bin/{shown.split()[0]}"
    assert (on_windows.kind, on_windows.runnable) == (kind, False)
    assert (no_tool.kind, no_tool.runnable) == (kind, False)


def test_any_other_environment_gets_the_pip_command_without_running_it(tmp_path: Path) -> None:
    found = detect_install(tmp_path, editable=False, windows=False, which=_tools)

    assert found == Install("pip", ("pip", "install", "-U", "satsumaestro"))
    assert not found.runnable


def test_an_editable_install_has_no_command(tmp_path: Path) -> None:
    (tmp_path / "uv-receipt.toml").write_text("")

    found = detect_install(tmp_path, editable=True, windows=False, which=_tools)

    assert (found.kind, found.command, found.runnable) == ("editable", (), False)


@pytest.mark.parametrize(
    ("text", "editable"),
    [
        ('{"url": "file:///src/satsumaestro", "dir_info": {"editable": true}}', True),
        ('{"url": "file:///src/satsumaestro", "dir_info": {}}', False),
        ('{"url": "https://github.com/x/y", "vcs_info": {"vcs": "git"}}', False),
        ("not json", False),
        ("[]", False),
        (None, False),
    ],
)
def test_direct_url_marks_an_editable_install(text: str | None, editable: bool) -> None:
    assert is_editable(text) is editable


def test_running_without_installed_metadata_counts_as_editable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> metadata.Distribution:
        raise metadata.PackageNotFoundError("satsumaestro")

    monkeypatch.setattr(install_mod.metadata, "distribution", missing)

    assert installed_editable() is True


def test_brew_checks_the_tap_and_the_others_check_pypi() -> None:
    assert source_for(Install("brew", ())) == "tap"
    assert {source_for(Install(k, ())) for k in ("uv", "pipx", "pip")} == {"pypi"}


# ── 版の比べ方 ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("latest", "current", "newer"),
    [
        ("0.5.4", "0.5.3", True),
        ("0.5.10", "0.5.9", True),
        ("0.6", "0.5.9", True),
        ("0.5.3", "0.5.3", False),
        ("0.5.3.0", "0.5.3", False),
        ("0.5.2", "0.5.3", False),
        ("0.6.0rc1", "0.5.3", False),
        ("0.5.4", "0.5.3.dev1", False),
    ],
)
def test_only_plain_numeric_versions_are_compared(latest: str, current: str, newer: bool) -> None:
    assert is_newer(latest, current) is newer


def test_parse_version_ignores_surrounding_space_and_rejects_pre_releases() -> None:
    assert parse_version(" 0.5.3\n") == (0, 5, 3)
    assert parse_version("1.0.0b2") is None
    assert parse_version("") is None


# ── 最新の版を聞く ──────────────────────────────────────


def _answering(
    answer: httpx.Response | Exception, seen: list[httpx.Request]
) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if isinstance(answer, Exception):
            raise answer
        return answer

    return httpx.MockTransport(handle)


async def test_pypi_answers_with_the_version_in_info() -> None:
    seen: list[httpx.Request] = []
    body = {"info": {"version": "0.5.4"}, "releases": {"0.6.0rc1": []}}
    transport = _answering(httpx.Response(200, json=body), seen)

    assert await latest_version("pypi", transport=transport) == "0.5.4"
    assert str(seen[0].url) == PYPI_URL
    assert seen[0].headers["user-agent"].startswith("satsumaestro/")


async def test_the_tap_answers_with_the_version_in_the_formula_url() -> None:
    seen: list[httpx.Request] = []
    transport = _answering(httpx.Response(200, text=_FORMULA), seen)

    assert await latest_version("tap", transport=transport) == "0.5.4"
    assert str(seen[0].url) == TAP_FORMULA_URL


@pytest.mark.parametrize(
    ("source", "answer"),
    [
        ("pypi", httpx.Response(404, json={"message": "Not Found"})),
        ("pypi", httpx.Response(200, text="<html>")),
        ("pypi", httpx.Response(200, json={"info": {"version": "0.6.0rc1"}})),
        ("pypi", httpx.Response(200, json=["0.5.4"])),
        ("tap", httpx.Response(200, text="class Other < Formula\nend\n")),
        ("tap", httpx.Response(503, text="busy")),
        ("pypi", httpx.ConnectError("offline")),
        ("tap", httpx.ReadTimeout("slow")),
    ],
)
async def test_an_unreadable_or_failed_answer_gives_no_version(
    source: updates.Source, answer: httpx.Response | Exception
) -> None:
    transport = _answering(answer, [])

    assert await latest_version(source, transport=transport) is None


# ── 1 日のキャッシュ ────────────────────────────────────


def test_the_cache_round_trips_and_ignores_broken_files(tmp_path: Path) -> None:
    path = tmp_path / "update_check.json"
    assert load_cached(path) is None

    save_cached(path, CachedCheck(123, "tap", "0.5.4"))
    assert load_cached(path) == CachedCheck(123, "tap", "0.5.4")
    assert json.loads(path.read_text()) == {"checked_at": 123, "source": "tap", "latest": "0.5.4"}

    for broken in (
        "{",
        "[]",
        '{"checked_at": "x", "source": "pypi", "latest": null}',
        '{"checked_at": 1, "source": "github", "latest": null}',
        '{"checked_at": 1, "source": "pypi", "latest": "0.6.0rc1"}',
    ):
        path.write_text(broken)
        assert load_cached(path) is None, broken


def test_a_check_is_due_once_a_day_or_when_the_source_changes() -> None:
    day = CHECK_INTERVAL_MS
    cached = CachedCheck(10 * day, "pypi", "0.5.3")

    assert is_due(None, "pypi", now=10 * day)
    assert not is_due(cached, "pypi", now=10 * day + day - 1)
    assert is_due(cached, "pypi", now=11 * day)
    assert is_due(cached, "tap", now=10 * day + 1)
    assert is_due(cached, "pypi", now=10 * day - 1)


async def test_check_now_writes_the_answer(tmp_path: Path) -> None:
    path = tmp_path / "update_check.json"
    transport = _answering(httpx.Response(200, text=_FORMULA), [])

    check = await check_now("tap", path, transport=transport)

    assert (check.source, check.latest) == ("tap", "0.5.4")
    assert load_cached(path) == check


async def test_a_failed_check_keeps_the_last_version_from_the_same_source(tmp_path: Path) -> None:
    path = tmp_path / "update_check.json"
    save_cached(path, CachedCheck(1, "pypi", "0.5.4"))
    offline = _answering(httpx.ConnectError("offline"), [])

    kept = await check_now("pypi", path, transport=offline)
    switched = await check_now("tap", path, transport=offline)

    assert (kept.source, kept.latest) == ("pypi", "0.5.4")
    assert kept.checked_at > 1
    assert (switched.source, switched.latest) == ("tap", None)
    assert load_cached(path) == switched
