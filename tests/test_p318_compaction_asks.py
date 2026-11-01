"""圧縮を重ねても、利用者の最初の依頼を要約に残す。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.engine.compaction import (
    _SUMMARY_PREFIX,
    RULE_SUMMARY_MAX_CHARS,
    Compactor,
    _rule_summary,
)
from satsumaestro.engine.events import MessageStop, TextDelta
from satsumaestro.engine.plan_request import ensure_plan_request
from satsumaestro.engine.reminders import format_reminder
from satsumaestro.session.models import (
    ConversationMessage,
    Session,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from satsumaestro.session.store import new_session
from satsumaestro.util.lang import with_language_directive
from tests.fixtures.mock_provider import MockProvider

REQUEST = (
    "files/ の part1.py から part6.py まで、6 つのファイルを read_file で"
    " 1 つずつ順に全部読んでください。読み終えたら、part1.py の先頭の NOTE の番号と、"
    "真ん中あたりの NOTE の番号を答えてください。"
)


def _say(role: str, text: str) -> ConversationMessage:
    return ConversationMessage(role=role, blocks=[TextBlock(text=text)])  # type: ignore[arg-type]


def _read(i: int) -> list[ConversationMessage]:
    cid = f"r{i}"
    return [
        ConversationMessage(
            role="assistant",
            blocks=[ToolUseBlock(id=cid, name="read_file", input={"path": f"files/part{i}.py"})],
        ),
        ConversationMessage(
            role="tool",
            blocks=[ToolResultBlock(tool_use_id=cid, tool_name="read_file", output="x" * 2_000)],
        ),
    ]


def _summary_text(session: Session) -> str:
    head = session.messages[0].blocks[0]
    assert isinstance(head, TextBlock) and head.text.startswith(_SUMMARY_PREFIX)
    return head.text


async def test_the_first_request_survives_repeated_compactions(tmp_path: Path) -> None:
    session = new_session(tmp_path, "m")
    session.messages.append(_say("user", REQUEST))
    compactor = Compactor(keep_last=2)
    for i in range(1, 13):
        session.messages.extend(_read(i))
        if i % 2 == 0:
            assert await compactor.acompact(session) is not None
    text = _summary_text(session)
    lines = text.splitlines()
    head = lines.index("What the user asked (oldest first):")
    assert lines[head + 1].startswith("  - files/ の part1.py から part6.py まで")
    assert "Carried over from the previous summary: Earlier messages folded" not in text
    assert text.count("Earlier messages folded:") == 1


def test_harness_notes_are_not_counted_as_requests() -> None:
    plan_turn = with_language_directive(
        ensure_plan_request("ダッシュボードを作って", ".satsuma/plans/x.md"), "ja"
    )
    msgs = [
        _say("user", plan_turn),
        _say("user", format_reminder(["Your previous response was cancelled: …"])),
        _say("user", "続けて\n\n" + format_reminder(["Plan mode is no longer active: …"])),
    ]
    summary = _rule_summary(msgs)
    assert "  - ダッシュボードを作って" in summary.splitlines()
    assert "  - 続けて" in summary.splitlines()
    assert "Plan mode" not in summary and "cancelled" not in summary
    assert "Reply in" not in summary
    assert summary.count("  - ") == 2


def test_requests_carried_from_the_previous_summary_keep_their_order() -> None:
    previous = _say(
        "user",
        f"{_SUMMARY_PREFIX}\n\nEarlier messages folded: 9 (user 4, assistant 3, tool results 2)\n"
        "What the user asked (oldest first):\n  - build the importer\n  - add CSV\n  - add TSV\n"
        "Files changed: app/importer.py",
    )
    summary = _rule_summary([previous, _say("user", "add JSON"), _say("user", "add XML")])
    lines = summary.splitlines()
    head = lines.index("What the user asked (oldest first):")
    assert lines[head + 1 : head + 5] == [
        "  - build the importer",
        "  - add TSV",
        "  - add JSON",
        "  - add XML",
    ]
    assert "Carried over from the previous summary: Files changed: app/importer.py" in lines


async def test_a_rule_summary_after_an_llm_summary_keeps_the_first_request(tmp_path: Path) -> None:
    provider = MockProvider([TextDelta(text="- read part1 and part2"), MessageStop()])
    compactor = Compactor(keep_last=2, provider=provider, model="m")  # type: ignore[arg-type]
    session = new_session(tmp_path, "m")
    session.messages.append(_say("user", REQUEST))
    for i in (1, 2):
        session.messages.extend(_read(i))
    first = await compactor.acompact(session)
    assert first is not None and first.used_llm
    for i in (3, 4, 5):
        session.messages.extend(_read(i))
    second = await compactor.acompact(session, allow_llm=False)
    assert second is not None and not second.used_llm
    assert "  - files/ の part1.py から part6.py まで" in second.summary
    assert "- read part1 and part2" in second.summary


def test_the_ceiling_still_holds_with_a_long_first_request() -> None:
    long = "w" * 5_000
    previous = _say("user", f"{_SUMMARY_PREFIX}\n\nWhat the user asked (oldest first):\n  - {long}")
    summary = _rule_summary([previous, *[_say("user", long) for _ in range(10)]])
    assert len(summary) <= RULE_SUMMARY_MAX_CHARS
    assert max(len(line) for line in summary.splitlines()) <= 340
