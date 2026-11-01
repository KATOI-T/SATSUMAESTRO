"""ツールの呼び出しの引数を、サーバが描き直した形で見積もる。"""

from __future__ import annotations

import json

from satsumaestro.engine import written_bodies
from satsumaestro.engine.context_budget import (
    estimate_upper,
    message_floor,
    message_tokens,
    tool_input_extra,
)
from satsumaestro.session.models import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

_HTML = "".join(
    f'<div class="card" id="card-{i}"><span class="label">item {i}</span>'
    f" &amp; <b>{i * 3}</b></div>\n"
    for i in range(120)
)


def _write(content: str) -> ConversationMessage:
    return ConversationMessage(
        role="assistant",
        blocks=[
            ToolUseBlock(id="w1", name="write_file", input={"path": "a.html", "content": content})
        ],
    )


def test_the_probe_page_has_the_measured_shape() -> None:
    assert (len(_HTML), sum(_HTML.count(c) for c in "<>&")) == (10_662, 1_560)


def test_escaped_characters_in_tool_arguments_are_counted() -> None:
    message = _write(_HTML)
    assert tool_input_extra(message) == 5 * 1_560
    text = "write_file" + json.dumps(message.blocks[0].input, ensure_ascii=False)  # type: ignore[union-attr]
    assert message_tokens(message) == 4 + estimate_upper(text) + 5 * 1_560


def test_the_estimate_covers_both_measured_models() -> None:
    assert message_tokens(_write(_HTML)) >= 11_624


def test_text_and_tool_results_are_not_escaped() -> None:
    as_text = ConversationMessage(role="assistant", blocks=[TextBlock(text=_HTML)])
    as_result = ConversationMessage(
        role="tool",
        blocks=[ToolResultBlock(tool_use_id="r1", tool_name="read_file", output=_HTML)],
    )
    assert tool_input_extra(as_text) == tool_input_extra(as_result) == 0
    assert message_tokens(as_text) == 4 + estimate_upper(_HTML)


def test_the_lower_estimate_does_not_count_escapes() -> None:
    message = _write(_HTML)
    text = "write_file" + json.dumps(message.blocks[0].input, ensure_ascii=False)  # type: ignore[union-attr]
    assert message_floor(message) == len(text) // 8


def test_the_shrink_threshold_compares_the_plain_body() -> None:
    message = _write(_HTML)
    results = {"w1": ToolResultBlock(tool_use_id="w1", tool_name="write_file", output="wrote")}
    shrunk, omitted = written_bodies.shrink_message(
        message,
        results,
        threshold=written_bodies.shrink_threshold(131_072),
        keep=lambda tool, path: False,
    )
    assert estimate_upper(_HTML) < 4_096 < message_tokens(message)
    assert omitted == [] and shrunk is message
