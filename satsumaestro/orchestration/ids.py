"""登録簿の ID と、無い ID を引いたときの例外。"""

from __future__ import annotations

from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import OrchestrationError


def make_id(kind: str, serial: int) -> str:
    """``{kind}_{serial}_{unix 秒}`` の ID を返す。"""
    return f"{kind}_{serial}_{now_ms() // 1000}"


class UnknownIdError(OrchestrationError):
    """登録簿に無い ID を引いた。"""

    def __init__(self, kind: str, ident: str) -> None:
        super().__init__(f"unknown {kind} id: {ident}")
        self.kind = kind
        self.ident = ident
