"""転写の検査（tests/smoke/transcript.py）の単体テスト。"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from tests.smoke import transcript as tr

HAPPY = """\
SATSUMAESTRO REPL — /help でコマンド一覧、/exit で終了
[code] satsuma> a b c を作って
╭──────────── プラン承認 ────────────╮
│ 計画                               │
│                                    │
│ Steps                              │
│                                    │
│  1 a.txt を作る (~5 lines)         │
│  2 b.txt を作る (~5 lines)         │
│  3 c.txt を作る (~5 lines)         │
╰────────────────────────────────────╯
（3 段に分けて 1 段ずつ実行します）
（1 段の上限 100 行 ＝ 既定の 20 tok/s（この会話ではまだ未計測） × 120 秒 × 安全率 0.5）
プランを承認して実行に進みますか？ [y=承認 / a=承認して一言添える / n=計画を直させる]: y
— 段 1/3: a.txt を作る (~5 lines)
— 段 1/3 完了: 見積 5 行 / 実績 1 行 / 最大 1 回 1 行 / 最長応答 0 秒
— 段 2/3: b.txt を作る (~5 lines)
— 段 2/3 完了: 見積 5 行 / 実績 1 行 / 最大 1 回 1 行 / 最長応答 0 秒
— 段 3/3: c.txt を作る (~5 lines)
— 段 3/3 完了: 見積 5 行 / 実績 1 行 / 最大 1 回 1 行 / 最長応答 0 秒
すべての段を終えました。
— 出力 450 · 入力 1.2K（3 回） · 文脈 1.65K/52K
"""


def _render(*renderables: object) -> str:
    console = Console(width=60, record=True, force_terminal=False, legacy_windows=False)
    for r in renderables:
        console.print(r)
    return console.export_text()


def test_strip_ansi_removes_csi_osc_and_normalizes_newlines() -> None:
    line = b"\xe2\x80\x94 \xe5\x87\xba\xe5\x8a\x9b 3"
    raw = b"\x1b]0;title\x07\x1b[1;33m" + line + b"\x1b[0m\r\nnext\rline\x1b[?25h"
    assert tr.strip_ansi(raw) == "— 出力 3\nnext\nline"


def test_clean_lines_drops_spinner_then_collapses_duplicates() -> None:
    text = "a\n⠋ 考え中…\nb\n⠙ 考え中…\nb\nb  \nc"
    assert tr.clean_lines(text) == ["a", "b", "c"]


def test_flat_ignores_all_whitespace() -> None:
    assert tr.flat("このセッションで作成した\n ファイルを消します") == (
        "このセッションで作成したファイルを消します"
    )


def test_panels_are_extracted_from_a_real_rich_render() -> None:
    text = _render(
        Panel(Markdown("# 計画\n\n## Steps\n1. a.txt を作る (~5 lines)"), title="プラン承認"),
        Panel(
            Text("a.txt  削除は元に戻せません。この操作は毎回確認します"),
            title="権限確認: delete_path (danger-full-access)",
            expand=False,
        ),
    )
    lines = tr.clean_lines(text)
    cards = tr.approval_cards(lines)
    assert len(cards) == 1 and "Steps" in tr.flat(cards[0]) and "(~5lines)" in tr.flat(cards[0])
    perms = tr.permission_cards(lines, "delete_path")
    assert len(perms) == 1 and "削除は元に戻せません" in tr.flat(perms[0])
    assert tr.permission_cards(lines, "bash") == []


def test_wrapped_japanese_matches_after_flat() -> None:
    text = _render(
        Panel(
            Text("⚠ このセッションで作成したファイルを消します: dashboards/index.html（3 行）"),
            title="権限確認: delete_path (danger-full-access)",
        )
    )
    body = tr.permission_cards(tr.clean_lines(text), "delete_path")[0]
    assert "このセッションで作成したファイルを消します" in tr.flat(body)


def test_markers_on_the_happy_transcript() -> None:
    assert tr.declared_steps(HAPPY) == 3
    assert tr.step_starts(HAPPY) == [(1, 3), (2, 3), (3, 3)]
    assert tr.completed_steps(HAPPY) == [1, 2, 3]
    assert tr.has_budget_line(HAPPY)
    assert tr.sendbacks(HAPPY) == []
    assert tr.turn_ends(HAPPY) == 1
    assert tr.failure_lines(HAPPY) == []


def test_completed_steps_are_a_set() -> None:
    text = "— 段 1/2 完了: x\n— 段 1/2 完了: x\n— 段 2/2 完了: x\n"
    assert tr.completed_steps(text) == [1, 2]


def test_delete_notes() -> None:
    text = "│ ⚠ このセッションで作成したファイルを消します: a.txt（1 行） │\n"
    assert tr.delete_notes(text) == ["a.txt（1 行）"]


def test_check_passes_the_happy_transcript() -> None:
    report = tr.check(HAPPY)
    assert report.problems == []
    assert report.declared == 3 and report.completed == [1, 2, 3] and len(report.cards) == 1


def test_check_reports_an_empty_approval_card() -> None:
    for body in ("...", "…", "", "（計画ファイルがありません）"):
        text = f"╭── プラン承認 ──╮\n│ {body} │\n╰──────────────╯\n"
        assert "approval-card-empty" in tr.check(text).problems, repr(body)


def test_check_reports_a_missing_card_and_declaration() -> None:
    problems = tr.check("[code] satsuma> hi\n— 出力 10\n").problems
    assert "no-approval-card" in problems and "no-step-declaration" in problems


def test_check_reports_step_mismatch() -> None:
    text = HAPPY.replace(
        "— 段 3/3 完了: 見積 5 行 / 実績 1 行 / 最大 1 回 1 行 / 最長応答 0 秒\n", ""
    )
    assert "steps-mismatch: declared 3, completed [1, 2]" in tr.check(text).problems


def test_check_reports_each_failure_line() -> None:
    for line in (
        "⚠ ターンを最後まで進められませんでした: 反復の上限（8 回）に達しました",
        "⚠ ターンが失敗しました: plan モードではないのに exit_plan_mode を 3 回続けて呼んだ",
        "reached max_iterations=8 without end_turn",
        "Traceback (most recent call last):",
        "— 120 秒応答がなかったので中断し、分けて出し直すよう依頼しました",
        "— 応答が途中で切れました",
        "[fake] no rule matched: role=tool",
    ):
        report = tr.check(HAPPY + line + "\n")
        assert any(p.startswith("failure: ") and line[:20] in p for p in report.problems), line


def test_transcript_object_rebuilds_text_after_append() -> None:
    t = tr.Transcript()
    t.append(b"\x1b[32m[code] satsuma> \x1b[0m")
    assert t.text == "[code] satsuma>"
    t.append(b"\r\n\xe2\x80\x94 \xe5\x87\xba\xe5\x8a\x9b 3\r\n")
    assert t.lines[-1] == "— 出力 3" and t.tail(1) == "— 出力 3"


ENGLISH_STEPS = """\
— Step 1/2: Write a.txt (~5 lines)
— Step 1/2 done: estimate 5 lines / actual 1 line / largest write 1 line / longest response 0 s
— Plan sent back: step 2 is over budget
— Step 2/2: Write b.txt (~5 lines)
— Step 2/2 done: estimate 5 lines / actual 1 line / largest write 1 line / longest response 0 s
— output 450 · input 1.2K (3 requests) · context 1.65K/52K
"""


def test_markers_on_the_english_screen() -> None:
    """英語の画面の段の行・差し戻し・ターン末尾の内訳・失敗行も数える。"""
    assert tr.step_starts(ENGLISH_STEPS) == [(1, 2), (2, 2)]
    assert tr.completed_steps(ENGLISH_STEPS) == [1, 2]
    assert tr.sendbacks(ENGLISH_STEPS) == ["step 2 is over budget"]
    assert tr.turn_ends(ENGLISH_STEPS) == 1
    for line in (
        "— No response for 120 s, so it was stopped and asked to resend in smaller parts",
        "— The response was cut off",
    ):
        assert tr.failure_lines(ENGLISH_STEPS + line + "\n") == [line]
    card = "╭── Permission: write_file (workspace-write) ──╮\n│ a.txt │\n╰──────────╯\n"
    assert tr.permission_cards(tr.clean_lines(card), "write_file") == ["a.txt"]
