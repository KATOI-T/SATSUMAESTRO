"""ハーネスが設定と鍵を読む環境（``os.environ`` に書かずに持つ ``.env`` の値を重ねる）。"""

from __future__ import annotations

import os
from collections import ChainMap
from collections.abc import Mapping

_FIRST: dict[str, str] = {}
_LAST: dict[str, str] = {}

ENVIRON: Mapping[str, str] = ChainMap(_FIRST, os.environ, _LAST)


def hold_dotenv(values: Mapping[str, str | None]) -> None:
    """``.env`` の値を ``os.environ`` に書かずに持つ（空でない ``OLLAMA_*`` はシェルより優先）。"""
    forget_dotenv()
    for name, value in values.items():
        if value is None:
            continue
        if name.startswith("OLLAMA_") and value.strip():
            _FIRST[name] = value.strip()
        else:
            _LAST[name] = value


def forget_dotenv() -> None:
    """持っている ``.env`` の値を捨てる。"""
    _FIRST.clear()
    _LAST.clear()
