"""``satsumaestro update``（確かめてから更新を回す）と、``/update``（案内だけ）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

import satsumaestro.cli.app as cli_app
from satsumaestro.__about__ import __version__
from satsumaestro.cli import updates
from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.cli.updates import CachedCheck, cache_path, load_cached, run_update, save_cached
from satsumaestro.i18n.text import pinned_ui_language
from satsumaestro.util.install import Install
from tests.fixtures.mock_provider import MockProvider
from tests.test_p165_plan_steps import _runtime

_UV = Install("uv", ("uv", "tool", "upgrade", "satsumaestro"), "/opt/bin/uv")
_BREW = Install("brew", ("brew", "upgrade", "satsumaestro"), "/opt/homebrew/bin/brew")
_PIP = Install("pip", ("pip", "install", "-U", "satsumaestro"))


class _Fake:
    """入れ方・最新の版・問いの答え・コマンドの結果の代役。"""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        install: Install,
        latest: str | None,
        *,
        answers: tuple[str, ...] = (),
        code: int | Exception = 0,
    ) -> None:
        self.sources: list[str] = []
        self.ran: list[list[str]] = []
        self.asked = 0
        self._answers = list(answers)

        async def latest_version(source: str, **_: object) -> str | None:
            self.sources.append(source)
            return latest

        def run(argv: list[str], check: bool) -> SimpleNamespace:
            self.ran.append(argv)
            if isinstance(code, Exception):
                raise code
            return SimpleNamespace(returncode=code)

        def ask(_console: Console, _prompt: str) -> str:
            self.asked += 1
            return self._answers.pop(0)

        monkeypatch.setattr(updates, "detect_install", lambda: install)
        monkeypatch.setattr(updates, "latest_version", latest_version)
        monkeypatch.setattr(updates.subprocess, "run", run)
        monkeypatch.setattr(updates, "ask_line", ask)
        monkeypatch.setattr(updates.sys, "argv", ["/usr/local/bin/stro"])


def _update(*, yes: bool = False, interactive: bool = True) -> tuple[int, str]:
    console = Console(record=True, width=200)
    with pinned_ui_language("en"):
        code = run_update(console, yes=yes, interactive=interactive, current="0.5.3")
    return code, console.export_text()


# ── satsumaestro update ─────────────────────────────────


def test_a_newer_version_is_installed_after_a_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Fake(monkeypatch, _UV, "0.5.4", answers=("y",))

    code, out = _update()

    assert code == 0
    assert fake.ran == [["/opt/bin/uv", "tool", "upgrade", "satsumaestro"]]
    assert fake.asked == 1
    assert "SATSUMAESTRO 0.5.4 is available (you have 0.5.3)." in out
    assert "Command: uv tool upgrade satsumaestro" in out
    assert "Running: uv tool upgrade satsumaestro" in out
    assert "The upgrade finished. To check it: stro --version" in out
    cached = load_cached(cache_path())
    assert cached is not None and (cached.source, cached.latest) == ("pypi", "0.5.4")


def test_a_no_runs_nothing_and_other_answers_ask_again(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Fake(monkeypatch, _UV, "0.5.4", answers=("maybe", "n"))

    code, out = _update()

    assert (code, fake.ran, fake.asked) == (0, [], 2)
    assert "Not upgraded." in out


def test_ctrl_c_at_the_question_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Fake(monkeypatch, _UV, "0.5.4")

    def interrupted(_console: Console, _prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(updates, "ask_line", interrupted)

    code, out = _update()

    assert (code, fake.ran) == (0, [])
    assert "Not upgraded." in out


def test_the_latest_version_needs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Fake(monkeypatch, _UV, "0.5.3")

    code, out = _update()

    assert (code, fake.ran, fake.asked) == (0, [], 0)
    assert "SATSUMAESTRO 0.5.3 is the latest version." in out


@pytest.mark.parametrize(
    ("install", "source", "where"),
    [(_UV, "pypi", "PyPI"), (_BREW, "tap", "the Homebrew tap (GitHub)")],
)
def test_no_answer_shows_the_command_and_fails(
    monkeypatch: pytest.MonkeyPatch, install: Install, source: str, where: str
) -> None:
    fake = _Fake(monkeypatch, install, None)

    code, out = _update()

    assert (code, fake.sources, fake.ran) == (1, [source], [])
    assert f"Could not get the latest version from {where}." in out
    assert f"Command: {install.shown}" in out
    assert load_cached(cache_path()) is None


@pytest.mark.parametrize("install", [_PIP, Install("brew", _BREW.command)])
def test_installs_it_cannot_run_only_show_the_command(
    monkeypatch: pytest.MonkeyPatch, install: Install
) -> None:
    fake = _Fake(monkeypatch, install, "0.5.4")

    code, out = _update(yes=True)

    assert (code, fake.ran, fake.asked) == (0, [], 0)
    assert f"Command: {install.shown}" in out
    assert "does not run this command for this install. Run it yourself." in out


def test_an_editable_install_asks_for_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Fake(monkeypatch, Install("editable", ()), "0.5.4")

    code, out = _update()

    assert (code, fake.sources, fake.ran) == (0, [], [])
    assert "This is an editable install." in out


def test_without_a_terminal_it_runs_only_with_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Fake(monkeypatch, _BREW, "0.5.4")

    code, out = _update(interactive=False)
    assert (code, fake.ran, fake.asked) == (1, [], 0)
    assert "Add --yes to upgrade" in out

    code, _ = _update(yes=True, interactive=False)
    assert (code, fake.asked) == (0, 0)
    assert fake.ran == [["/opt/homebrew/bin/brew", "upgrade", "satsumaestro"]]


@pytest.mark.parametrize(("result", "code"), [(3, 3), (FileNotFoundError("uv"), 127)])
def test_a_failed_upgrade_says_so_with_its_exit_code(
    monkeypatch: pytest.MonkeyPatch, result: int | Exception, code: int
) -> None:
    _Fake(monkeypatch, _UV, "0.5.4", code=result)

    got, out = _update(yes=True)

    assert got == code
    assert f"The upgrade failed (exit code {code})." in out
    assert "The upgrade finished" not in out


def test_the_command_line_passes_yes_and_the_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _Fake(monkeypatch, _UV, "99.0.0")
    monkeypatch.setattr(cli_app, "_on_terminal", lambda: False)
    runner = CliRunner()

    asked = runner.invoke(cli_app.app, ["update"])
    assert (asked.exit_code, fake.ran) == (1, [])

    upgraded = runner.invoke(cli_app.app, ["update", "-y"])
    assert upgraded.exit_code == 0
    assert fake.ran == [["/opt/bin/uv", "tool", "upgrade", "satsumaestro"]]


# ── /update ─────────────────────────────────────────────


async def _slash_update(tmp_path: Path) -> str:
    console = Console(record=True, width=200)
    with pinned_ui_language("en"):
        await SlashDispatcher(_runtime(MockProvider([]), tmp_path), console).dispatch(
            parse_slash("/update")  # type: ignore[arg-type]
        )
    return console.export_text()


async def test_slash_update_shows_the_versions_and_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, _BREW, "99.0.0")

    out = await _slash_update(tmp_path)

    assert fake.sources == ["tap"]
    assert fake.ran == []
    assert "SATSUMAESTRO 99.0.0 is available" in out
    assert "To upgrade, leave the REPL and run: stro update" in out


async def test_slash_update_on_the_latest_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Fake(monkeypatch, _UV, "0.0.1")

    out = await _slash_update(tmp_path)

    assert "is the latest version." in out
    assert "leave the REPL" not in out


async def test_slash_update_without_an_answer_still_points_to_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Fake(monkeypatch, _UV, None)

    out = await _slash_update(tmp_path)

    assert "Could not get the latest version from PyPI." in out
    assert "To upgrade, leave the REPL and run: stro update" in out


async def test_slash_update_offline_does_not_trust_or_touch_the_saved_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = CachedCheck(123, "pypi", __version__)
    save_cached(cache_path(), saved)
    _Fake(monkeypatch, _UV, None)

    out = await _slash_update(tmp_path)

    assert "Could not get the latest version from PyPI." in out
    assert "is the latest version" not in out
    assert load_cached(cache_path()) == saved


async def test_slash_update_in_an_editable_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, Install("editable", ()), "99.0.0")

    out = await _slash_update(tmp_path)

    assert fake.sources == []
    assert "This is an editable install." in out
