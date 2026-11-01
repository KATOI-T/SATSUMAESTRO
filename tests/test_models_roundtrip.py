"""モデルの往復テスト。"""

from __future__ import annotations

from satsumaestro.session.models import (
    ConversationMessage,
    Session,
    SessionCompaction,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolResultBlock,
    ToolUseBlock,
)


def test_content_blocks_roundtrip_preserve_type() -> None:
    msg = ConversationMessage(
        role="assistant",
        blocks=[
            TextBlock(text="hello"),
            ThinkingBlock(thinking="考え中", signature="sig-123"),
            ToolUseBlock(id="t1", name="bash", input={"cmd": "ls"}),
            ToolResultBlock(tool_use_id="t1", tool_name="bash", output="ok"),
        ],
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )
    restored = ConversationMessage.model_validate(msg.model_dump())

    assert restored == msg
    assert isinstance(restored.blocks[0], TextBlock)
    assert isinstance(restored.blocks[1], ThinkingBlock)
    assert isinstance(restored.blocks[2], ToolUseBlock)
    assert isinstance(restored.blocks[3], ToolResultBlock)
    assert restored.blocks[1].signature == "sig-123"


def test_session_roundtrip() -> None:
    session = Session(
        session_id="s-1",
        created_at_ms=1000,
        updated_at_ms=2000,
        workspace_root="/ws",
        model="qwen2.5-coder:7b",
        messages=[ConversationMessage(role="user", blocks=[TextBlock(text="hi")])],
        prompt_history=["hi"],
        compaction=SessionCompaction(compacted_at_ms=1500, summary="s", dropped_messages=3),
    )
    restored = Session.model_validate(session.model_dump())

    assert restored == session
    assert restored.version == 1
    assert isinstance(restored.messages[0].blocks[0], TextBlock)
