"""権限カードの ``d``: 会話から「何をしようとしているか」をモデルに説明させる。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from satsumaestro.engine.events import ToolUse, closing_stream, fold_events
from satsumaestro.engine.reminders import language_note
from satsumaestro.providers.base import ApiRequest, Provider
from satsumaestro.providers.fallback import FallbackProvider
from satsumaestro.session.models import ConversationMessage, TextBlock, ToolUseBlock

_log = logging.getLogger(__name__)

EXPLAIN_MAX_TOKENS = 1024
EXPLAIN_TIMEOUT_S = 30.0
_INPUT_CHARS = 2_000
_TEXT_CHARS = 1_000
_ANSWER_CHARS = 800

_INSTRUCTION = (
    "Before I approve or reject the tool call below (you requested it; it has NOT run yet), "
    "tell me what it is about to do. Answer in plain text only, in 2-4 short sentences: "
    "concretely what this call will do (which files, commands or targets it touches and what "
    "will change) and why it is needed for my request. Do NOT call any tool in this reply, "
    "do not ask questions, and do not repeat the raw arguments verbatim."
)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def explain_question(
    call: ToolUse,
    input_dict: dict[str, Any],
    *,
    warning: str | None,
    step: tuple[int, int, str] | None,
    model_text: str,
    language: str,
) -> str:
    """説明を求める user の本文（指示・呼び出しの中身・文脈。言語指示は末尾）。"""
    args = _clip(json.dumps(input_dict, ensure_ascii=False, indent=2, default=str), _INPUT_CHARS)
    lines = [_INSTRUCTION, "", f"Tool: {call.name}", "Arguments:", args]
    if model_text.strip():
        said = _clip(model_text.strip(), _TEXT_CHARS)
        lines += ["", "What you wrote just before this call:", said]
    if step is not None:
        index, total, text = step
        lines += ["", f"Current plan step {index} of {total}: {text}"]
    if warning:
        lines += ["", f"Warning shown to the user on the prompt: {warning}"]
    note = language_note(language)
    body = "\n".join(lines)
    return f"{body}\n\n{note}" if note else body


def explain_request(
    base: ApiRequest, history: list[ConversationMessage], question: str
) -> ApiRequest:
    """本体の要求 ``base`` と同じ接頭辞に、説明を求める user を 1 つ足した要求。"""
    message = ConversationMessage(role="user", blocks=[TextBlock(text=question)])
    return base.model_copy(
        update={"messages": [*history, message], "max_tokens": EXPLAIN_MAX_TOKENS}
    )


async def explain_call(
    provider: Provider, req: ApiRequest, *, timeout_s: float = EXPLAIN_TIMEOUT_S
) -> tuple[str | None, str | None]:
    """説明を 1 回だけ求め、``(説明, 失敗の理由)`` を返す。"""
    stream = (
        provider.stream_without_switch(req)
        if isinstance(provider, FallbackProvider)
        else provider.stream(req)
    )

    async def run() -> ConversationMessage:
        async with closing_stream(stream) as events:
            return await fold_events(events)

    try:
        folded = await asyncio.wait_for(run(), timeout=timeout_s)
    except TimeoutError:
        return None, "timeout"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _log.info("could not get the explanation for the permission card: %s", exc)
        return None, "error"
    text = "".join(b.text for b in folded.blocks if isinstance(b, TextBlock)).strip()
    if not text:
        called = any(isinstance(b, ToolUseBlock) for b in folded.blocks)
        _log.info(
            "no explanation for the permission card (%s)",
            "only tool calls" if called else "empty",
        )
        return None, "error"
    return _clip(text, _ANSWER_CHARS), None
