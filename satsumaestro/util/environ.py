"""ハーネスが設定と鍵を読む環境（``os.environ`` に書かずに持つ ``.env`` の値を重ねる）。"""

from __future__ import annotations

import json
import os
from collections import ChainMap
from collections.abc import Mapping

WORKER_DOTENV_ENV = "SATSUMA_WORKER_DOTENV"

_FIRST: dict[str, str] = {}
_PASSED: dict[str, str] = {}
_LAST: dict[str, str] = {}

ENVIRON: Mapping[str, str] = ChainMap(_FIRST, _PASSED, os.environ, _LAST)


def adopt_passed_dotenv() -> None:
    """親から渡された ``.env`` の値を ``os.environ`` から外して持つ（子プロセスには渡さない）。"""
    raw = os.environ.pop(WORKER_DOTENV_ENV, "")
    try:
        values = json.loads(raw) if raw else {}
    except ValueError:
        return
    if isinstance(values, dict):
        _PASSED.update({k: v for k, v in values.items() if isinstance(v, str)})


def hold_dotenv(values: Mapping[str, str | None], preferred: Mapping[str, str]) -> None:
    """``.env`` の値を ``os.environ`` に書かずに持つ（``preferred`` はシェルより優先する）。"""
    forget_dotenv()
    _FIRST.update(preferred)
    _LAST.update({name: value for name, value in values.items() if value is not None})


def forget_dotenv() -> None:
    """持っている ``.env`` の値を捨てる。"""
    _FIRST.clear()
    _LAST.clear()
