"""トークン概算。"""

from __future__ import annotations

_ASCII_CHARS_PER_TOKEN = 3


def estimate_upper(text: str) -> int:
    """``text`` のトークン数を、多めに出る側で見積もる。"""
    ascii_chars = len(text.encode("ascii", "ignore"))
    return ascii_chars // _ASCII_CHARS_PER_TOKEN + (len(text) - ascii_chars)


def estimate_tokens(text: str) -> int:
    """文字数 / 4 の粗いトークン概算を返す。"""
    return len(text) // 4
