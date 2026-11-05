"""起動時と doctor に出す、文脈の窓の内訳。"""

from __future__ import annotations

from dataclasses import dataclass

from satsumaestro.i18n.text import ui
from satsumaestro.providers.detect import LocalKind

TIGHT_FACTOR = 2
_MIN_RECOMMENDED = 4096
OLLAMA_CONTEXT_LENGTH_URL = "https://docs.ollama.com/context-length"
_ASCII_CHARS_PER_TOKEN = 4


def display_tokens(text: str) -> int:
    """画面に出す目安のトークン数（ASCII は 4 字、それ以外は 1 字を 1 トークン）。"""
    ascii_chars = len(text.encode("ascii", "ignore"))
    return ascii_chars // _ASCII_CHARS_PER_TOKEN + (len(text) - ascii_chars)


@dataclass(frozen=True)
class WindowBreakdown:
    """窓（トークン）・圧縮の閾値・最初の要求・うち指示ファイル。"""

    window: int
    threshold: int
    fixed: int
    instructions: int

    @property
    def conversation(self) -> int:
        """会話に使える量（閾値から最初の要求を引いた残り）。"""
        return max(0, self.threshold - self.fixed)

    @property
    def tight(self) -> bool:
        """会話に使える量が、最初の要求の :data:`TIGHT_FACTOR` 倍に満たない。"""
        return self.conversation < TIGHT_FACTOR * self.fixed

    def recommended(self) -> int:
        """案内する窓: 会話に使える量が最初の要求の :data:`TIGHT_FACTOR` 倍に届く、2 の冪の最小。"""
        share = self.threshold / self.window
        need = self.fixed * (1 + TIGHT_FACTOR) / share
        size = _MIN_RECOMMENDED
        while size < need or size <= self.window:
            size *= 2
        return size

    def describe(self, source: str) -> str:
        """内訳の 1 行。"""
        window, fixed = f"{self.window:,}", f"{self.fixed:,}"
        conversation, threshold = f"{self.conversation:,}", f"{self.threshold:,}"
        if not self.instructions:
            return ui(
                "engine.window_breakdown.breakdown",
                window=window,
                source=source,
                fixed=fixed,
                conversation=conversation,
                threshold=threshold,
            )
        return ui(
            "engine.window_breakdown.breakdown_with_instructions",
            window=window,
            source=source,
            fixed=fixed,
            instructions=f"{self.instructions:,}",
            conversation=conversation,
            threshold=threshold,
        )

    def advice(self, server: LocalKind = "ollama", model: str = "") -> str:
        """窓が小さいときの案内（サーバごとの広げ方）。"""
        size = self.recommended()
        if server == "lmstudio":
            return ui(
                "engine.window_breakdown.window_is_tight_lmstudio",
                factor=TIGHT_FACTOR,
                size=size,
                model=model,
            )
        return ui(
            "engine.window_breakdown.window_is_tight",
            factor=TIGHT_FACTOR,
            size=size,
            url=OLLAMA_CONTEXT_LENGTH_URL,
        )


def window_breakdown(
    window: int, threshold: int, system: str, tools_json: str, instructions: str
) -> WindowBreakdown | None:
    """窓の内訳。"""
    if window <= 0 or threshold <= 0:
        return None
    return WindowBreakdown(
        window=window,
        threshold=threshold,
        fixed=display_tokens(system) + display_tokens(tools_json),
        instructions=display_tokens(instructions),
    )


__all__ = [
    "OLLAMA_CONTEXT_LENGTH_URL",
    "TIGHT_FACTOR",
    "WindowBreakdown",
    "display_tokens",
    "window_breakdown",
]
