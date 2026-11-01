"""圧縮の直後に、ツールの実行経路がまだ動くかを確かめる。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from satsumaestro.engine.runtime import ConversationRuntime

_log = logging.getLogger(__name__)

_CHECKS: tuple[tuple[str, dict[str, object]], ...] = (
    ("glob_search", {"pattern": "satsuma-tool-check-*.none"}),
    ("list_directory", {"path": ".", "limit": 1}),
)


async def check_tool_executor(runtime: ConversationRuntime) -> str | None:
    """ツールが 1 回通れば ``None``、通らなければ理由の文を返す。"""
    session = runtime.session
    if not session.messages and session.compaction is None:
        return "the session is empty and has no compaction record"
    for name, payload in _CHECKS:
        tool = runtime.tools.get(name)
        if tool is None:
            continue
        try:
            args = tool.Input.model_validate(payload)
            result = await runtime.tools.run(name, args, runtime.tool_context)
        except Exception as exc:
            _log.debug("tool check raised", exc_info=True)
            return f"running {name} raised {type(exc).__name__}: {exc}"
        if result.is_error:
            return f"{name} returned an error: {result.output}"
        return None
    if not (runtime.tools.core_names() or runtime.tools.deferred_names()):
        return "no tool is registered, so no tool can run"
    _log.debug("tool check skipped: none of %s is registered", [n for n, _ in _CHECKS])
    return None


__all__ = ["check_tool_executor"]
