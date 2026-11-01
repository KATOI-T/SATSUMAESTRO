"""ロギングの行き先を決める。"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from satsumaestro.util.paths import default_config_home

LOGGER_NAME = "satsumaestro"
LOG_FILE_NAME = "satsumaestro.log"
LOG_ROTATE_BYTES = 1_000_000
BACKUP_COUNT = 3
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MARK = "_satsuma_log_setup"


def log_path(config_home: Path | None = None) -> Path | None:
    """ログファイルの場所（``SATSUMA_LOG_FILE=off`` なら None）。"""
    override = os.environ.get("SATSUMA_LOG_FILE", "").strip()
    if override.lower() == "off":
        return None
    if override:
        return Path(override).expanduser()
    return (config_home or default_config_home()) / "logs" / LOG_FILE_NAME


def configure_logging(*, stderr: bool = False, config_home: Path | None = None) -> Path | None:
    """``satsumaestro`` ロガーの行き先を設定し、ログファイルの場所を返す（書かないなら None）。"""
    logger = logging.getLogger(LOGGER_NAME)
    for old in [h for h in logger.handlers if getattr(h, _MARK, False)]:
        logger.removeHandler(old)
        old.close()
    logger.setLevel(logging.INFO)

    path = log_path(config_home)
    handler: logging.Handler
    if path is None:
        handler = logging.NullHandler()
    else:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                path, maxBytes=LOG_ROTATE_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter(_FORMAT))
        except OSError:
            handler = logging.NullHandler()
            path = None
    setattr(handler, _MARK, True)
    logger.addHandler(handler)

    if stderr:
        err = logging.StreamHandler(sys.stderr)
        err.setLevel(logging.WARNING)
        err.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        setattr(err, _MARK, True)
        logger.addHandler(err)
    return path


def wants_stderr(debug: bool) -> bool:
    """stderr にも出すか（``--debug`` か ``SATSUMA_LOG_STDERR=1``）。"""
    return debug or os.environ.get("SATSUMA_LOG_STDERR", "").strip() in ("1", "true", "yes")
