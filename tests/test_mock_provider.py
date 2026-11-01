"""MockProvider テスト。"""

from __future__ import annotations

from satsumaestro.engine.events import (
    AssistantEvent,
    MessageStop,
    TextDelta,
    Thinking,
    ToolUse,
    Usage,
    fold_events,
)
from satsumaestro.providers.base import ApiRequest, ProviderKind
from satsumaestro.session.models import (
    ConversationMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from tests.fixtures.mock_provider import MockProvider


def _req() -> ApiRequest:
    return ApiRequest(
        model="mock-model",
        system="you are a test",
        messages=[ConversationMessage(role="user", blocks=[TextBlock(text="hi")])],
    )


async def test_mock_provider_yields_script_in_order() -> None:
    script: list[AssistantEvent] = [
        Thinking(text="think", signature="sig"),
        TextDelta(text="Hel"),
        TextDelta(text="lo"),
        ToolUse(id="t1", name="bash", input={"cmd": "ls"}),
        Usage(input_tokens=5, output_tokens=3),
        MessageStop(stop_reason="tool_use"),
    ]
    provider = MockProvider(script, kind=ProviderKind.ANTHROPIC)

    got = [ev async for ev in provider.stream(_req())]

    assert got == script
    assert provider.kind == ProviderKind.ANTHROPIC
    assert len(provider.requests) == 1
    assert provider.requests[0].model == "mock-model"


async def test_mock_provider_stream_folds() -> None:
    script: list[AssistantEvent] = [
        Thinking(text="think", signature="sig"),
        TextDelta(text="Hel"),
        TextDelta(text="lo"),
        ToolUse(id="t1", name="bash", input={"cmd": "ls"}),
        MessageStop(stop_reason="tool_use"),
    ]
    provider = MockProvider(script)

    msg = await fold_events(provider.stream(_req()))

    assert msg.role == "assistant"
    thinking, text, tool = msg.blocks
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.signature == "sig"
    assert isinstance(text, TextBlock)
    assert text.text == "Hello"
    assert isinstance(tool, ToolUseBlock)
    assert tool.name == "bash"


async def test_mock_provider_count_tokens() -> None:
    provider = MockProvider([])
    assert await provider.count_tokens(_req()) > 0
