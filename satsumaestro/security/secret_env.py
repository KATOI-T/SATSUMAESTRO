"""子プロセスへ渡す環境変数から、ハーネスが使う秘密を外す純関数。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase

PROVIDER_KEY_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "SATSUMA_EMBED_API_KEY",
    "LM_API_TOKEN",
    "LLAMA_API_KEY",
)

DEFAULT_STRIP_GLOBS: tuple[str, ...] = ("*_API_KEY",)


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def strip_secret_env(
    env: Mapping[str, str],
    *,
    extra_strip: Iterable[str] = (),
    keep: Iterable[str] = (),
) -> dict[str, str]:
    """``env`` から秘密の名前を外した写しを返す（元の mapping は変えない）。"""
    strip_patterns = (*PROVIDER_KEY_VARS, *DEFAULT_STRIP_GLOBS, *extra_strip)
    keep_patterns = tuple(keep)
    out: dict[str, str] = {}
    for name, value in env.items():
        if _matches(name, strip_patterns) and not _matches(name, keep_patterns):
            continue
        out[name] = value
    return out
