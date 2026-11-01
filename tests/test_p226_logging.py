"""ログは画面（stderr）ではなくファイルへ。"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from satsumaestro.util.logging_setup import LOGGER_NAME, configure_logging, log_path

_log = logging.getLogger("satsumaestro.providers.retry")


@pytest.fixture(autouse=True)
def _restore_handlers() -> object:
    logger = logging.getLogger(LOGGER_NAME)
    before = list(logger.handlers)
    yield
    for h in list(logger.handlers):
        if h not in before:
            logger.removeHandler(h)
            h.close()


def test_warnings_go_to_the_file_not_the_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SATSUMA_LOG_FILE")
    path = configure_logging(config_home=tmp_path)
    assert path == tmp_path / "logs" / "satsumaestro.log"
    _log.warning("error parsing tool call: raw=…")
    assert "error parsing tool call" not in capsys.readouterr().err
    assert "error parsing tool call" in path.read_text(encoding="utf-8")


def test_configuring_twice_does_not_duplicate_handlers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SATSUMA_LOG_FILE")
    configure_logging(config_home=tmp_path)
    configure_logging(config_home=tmp_path)
    ours = [
        h
        for h in logging.getLogger(LOGGER_NAME).handlers
        if getattr(h, "_satsuma_log_setup", False)
    ]
    assert len(ours) == 1


def test_debug_also_writes_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SATSUMA_LOG_FILE")
    configure_logging(stderr=True, config_home=tmp_path)
    _log.warning("retry detail")
    _log.info("info stays in the file")
    err = capsys.readouterr().err
    assert "retry detail" in err and "info stays" not in err


def test_env_overrides_and_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SATSUMA_LOG_FILE", str(tmp_path / "x.log"))
    assert log_path() == tmp_path / "x.log"
    monkeypatch.setenv("SATSUMA_LOG_FILE", "off")
    assert log_path() is None
    assert configure_logging() is None


def test_unwritable_location_does_not_stop_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("SATSUMA_LOG_FILE", str(blocker / "logs" / "satsumaestro.log"))
    assert configure_logging() is None
    _log.warning("still quiet")
    assert "still quiet" not in capsys.readouterr().err
