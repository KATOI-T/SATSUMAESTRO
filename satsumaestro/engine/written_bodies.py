"""会話に残る書き込みの本文を縮める。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from satsumaestro.engine.context_budget import estimate_upper
from satsumaestro.session.models import (
    ConversationMessage,
    OmittedBody,
    ToolResultBlock,
    ToolUseBlock,
)

WRITE_TOOLS = frozenset({"write_file", "append_file"})
OMITTED_KEY = "content_omitted"
WINDOW_DIVISOR = 32
_TAIL_LINES = 5


def shrink_threshold(window: int) -> int | None:
    """これを超える本文を縮める量（トークン）。"""
    return window // WINDOW_DIVISOR if window > 0 else None


def omitted_input(arguments: Mapping[str, Any], *, written: bool) -> dict[str, Any] | None:
    """``content`` を外した引数。"""
    content = arguments.get("content")
    if not isinstance(content, str):
        return None
    lines = content.splitlines()
    state = (
        "written to the file and not repeated here"
        if written
        else "not written (see the tool result) and not repeated here"
    )
    out = {k: v for k, v in arguments.items() if k != "content"}
    out[OMITTED_KEY] = f"{len(lines):,} lines, {len(content):,} chars; {state}"
    if lines:
        out["first_line"] = lines[0]
    if len(lines) > 1:
        out["last_lines"] = "\n".join(lines[max(1, len(lines) - _TAIL_LINES) :])
    return out


def shrink_message(
    message: ConversationMessage,
    results: Mapping[str, ToolResultBlock],
    *,
    threshold: int | None,
    keep: Callable[[str, str], bool],
) -> tuple[ConversationMessage, list[OmittedBody]]:
    """応答の書き込みの本文を縮める。"""
    blocks: list[Any] = []
    omitted: list[OmittedBody] = []
    for block in message.blocks:
        result = results.get(block.id) if isinstance(block, ToolUseBlock) else None
        if not isinstance(block, ToolUseBlock) or result is None:
            blocks.append(block)
            continue
        content = block.input.get("content")
        path = block.input.get("path")
        path = path if isinstance(path, str) else ""
        if (
            result.tool_name not in WRITE_TOOLS
            or not isinstance(content, str)
            or (threshold is not None and estimate_upper(content) <= threshold)
            or keep(result.tool_name, path)
        ):
            blocks.append(block)
            continue
        shrunk = omitted_input(block.input, written=not result.is_error)
        if shrunk is None:
            blocks.append(block)
            continue
        blocks.append(block.model_copy(update={"input": shrunk}))
        omitted.append(
            OmittedBody(tool_use_id=block.id, tool=result.tool_name, path=path, content=content)
        )
    if not omitted:
        return message, []
    return message.model_copy(update={"blocks": blocks}), omitted


def writes_text(message: ConversationMessage, results: Mapping[str, ToolResultBlock]) -> bool:
    """応答に、本文のある書き込みがあるか（縮めた形を真似た呼び出しは数えない）。"""
    return any(
        isinstance(block, ToolUseBlock)
        and (result := results.get(block.id)) is not None
        and result.tool_name in WRITE_TOOLS
        and isinstance(block.input.get("content"), str)
        for block in message.blocks
    )


def shrink_history(
    messages: list[ConversationMessage],
    *,
    keep: Callable[[str, str], bool],
    threshold: int | None = None,
    upto: int | None = None,
) -> list[OmittedBody]:
    """会話の書き込みの本文を縮める（``messages`` をその場で）。"""
    results = {
        b.tool_use_id: b
        for m in messages
        if m.role == "tool"
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }
    omitted: list[OmittedBody] = []
    for i, message in enumerate(messages[:upto]):
        if message.role != "assistant":
            continue
        new, shrunk = shrink_message(message, results, threshold=threshold, keep=keep)
        if shrunk:
            messages[i] = new
            omitted += shrunk
    return omitted


_CONTENT_SHAPE = {
    "write_file": "<the whole text of the file>",
    "append_file": "<the text to add at the end of the file>",
}


def copied_omission(tool: str, arguments: Mapping[str, Any]) -> str | None:
    """縮めた形を真似た呼び出し（本文が無く ``content_omitted`` がある）なら、返す誤りの文。"""
    if tool not in WRITE_TOOLS or OMITTED_KEY not in arguments or "content" in arguments:
        return None
    path = arguments.get("path")
    shape = json.dumps(
        {
            "path": path if isinstance(path, str) and path else "<file path>",
            "content": _CONTENT_SHAPE[tool],
        },
        ensure_ascii=False,
    )
    return (
        f"Nothing was written: {tool} needs the text itself in `content`. `{OMITTED_KEY}` only "
        "marks earlier writes whose text this conversation does not repeat (it is in the "
        "files); it is not an argument, and a description of the text does not write it. "
        f"Call {tool} again as {shape}."
    )


__all__ = [
    "OMITTED_KEY",
    "WINDOW_DIVISOR",
    "WRITE_TOOLS",
    "copied_omission",
    "omitted_input",
    "shrink_history",
    "shrink_message",
    "shrink_threshold",
    "writes_text",
]
