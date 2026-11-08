"""擬似プロバイダ＋pty の REPL 煙テスト。"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.smoke import transcript as tr
from tests.smoke.fake_ollama import FAKE_MODEL, SCRIPTS_DIR, FakeOllama
from tests.smoke.repl_driver import (
    DELETE_REQUEST,
    REQUEST,
    DriverError,
    ReplDriver,
    ScenarioResult,
    run_scenario,
)

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(sys.platform == "win32", reason="pty が無い"),
]

TIME_LIMIT_S = 60.0

Runner = Callable[..., ScenarioResult]


@pytest.fixture
def scenario(tmp_path: Path) -> Runner:
    """台本を流して結果を返す。"""

    def run(
        script: str,
        *,
        settings: dict[str, Any] | None = None,
        turns: Sequence[str] = (REQUEST, DELETE_REQUEST),
        plan: bool = True,
        commands: Sequence[tuple[str, str]] = (),
        answer: str | Sequence[str] = "y",
        during: Callable[[int, ReplDriver, FakeOllama], None] | None = None,
        rows: int = 40,
        cols: int = 120,
        trust_answer: str | None = None,
        locale: str = "ja_JP.UTF-8",
        screen_answer: str = "auto",
        window: int = 131072,
        loaded_window: int | None = None,
        flavor: str = "ollama",
    ) -> ScenarioResult:
        try:
            result = run_scenario(
                SCRIPTS_DIR / script,
                workspace=tmp_path / "ws",
                home=tmp_path / "home",
                settings=settings,
                turns=turns,
                plan=plan,
                commands=commands,
                answer=answer,
                during=during,
                rows=rows,
                cols=cols,
                trust_answer=trust_answer,
                locale=locale,
                screen_answer=screen_answer,
                window=window,
                loaded_window=loaded_window,
                flavor=flavor,
            )
        except DriverError as e:
            _dump(e.transcript, e.requests)
            raise
        _dump(result.transcript, result.requests)
        print(f"--- elapsed {result.elapsed:.1f}s ---")
        assert result.elapsed < TIME_LIMIT_S, f"{result.elapsed:.1f}s >= {TIME_LIMIT_S}s"
        return result

    return run


def _dump(transcript: tr.Transcript, requests: list[dict[str, Any]]) -> None:
    print("--- transcript ---")
    print(transcript.text)
    print("--- fake_ollama requests ---")
    for r in requests:
        print(json.dumps(r, ensure_ascii=False))


def test_plan_three_steps_happy_path(scenario: Runner, tmp_path: Path) -> None:
    result = scenario("plan_three_steps.jsonl")
    text = result.transcript.text
    lines = result.transcript.lines
    assert result.report.problems == []
    assert re.search(r"\[plan\] satsuma> ?三つのファイル", result.transcript.plain)

    cards = result.report.cards
    assert len(cards) == 1
    assert "Steps" in tr.flat(cards[0]) and "(~5lines)" in tr.flat(cards[0])
    assert tr.has_budget_line(text)

    assert result.report.declared == 3
    assert result.report.completed == [1, 2, 3]
    assert tr.step_starts(text) == [(1, 3), (2, 3), (3, 3)]
    assert tr.sendbacks(text) == []

    perms = tr.permission_cards(lines, "delete_path")
    assert len(perms) == 3
    for body in perms:
        assert "削除は元に戻せません" in tr.flat(body)
        assert "このセッションで作成したファイルを消します" in tr.flat(body)
    plain = result.transcript.plain
    assert plain.count("[y=許可 / n=拒否 / d=詳細]") == 3
    assert "a=永続許可" not in plain
    assert tr.flat(plain).count(tr.flat("削除を伴うため永続許可できません")) == 3
    ws = tmp_path / "ws"
    for name in ("a.txt", "b.txt", "c.txt"):
        assert not (ws / name).exists(), name
    plans = list((ws / ".satsuma" / "plans").glob("*.md"))
    assert len(plans) == 1 and "## Steps" in plans[0].read_text(encoding="utf-8")


def test_plan_three_steps_in_english(scenario: Runner, tmp_path: Path) -> None:
    """英語の発話では画面も英語になり、plan → 承認 → 段 → 完走 → 削除が同じ駆動で通る。"""
    result = scenario(
        "plan_three_steps_en.jsonl",
        turns=(
            "Create three files a.txt, b.txt and c.txt with one line each",
            "Delete the files you created",
        ),
        locale="en_US.UTF-8",
    )
    text, lines, plain = result.transcript.text, result.transcript.lines, result.transcript.plain
    assert result.report.problems == []
    assert len(result.report.cards) == 1
    assert tr.has_budget_line(text)
    assert result.report.declared == 3 and result.report.completed == [1, 2, 3]
    assert tr.step_starts(text) == [(1, 3), (2, 3), (3, 3)]
    perms = tr.permission_cards(lines, "delete_path")
    assert len(perms) == 3
    for body in perms:
        assert tr.flat("Deletion cannot be undone") in tr.flat(body)
        assert tr.flat("Deletes files created in this session") in tr.flat(body)
    assert plain.count("[y=allow / n=deny / d=details]") == 3
    assert tr.flat(plain).count(tr.flat("(This cannot be remembered: it deletes files)")) == 3
    for japanese in ("プラン承認", "段に分けて", "許可しますか", "権限確認", "完了", "— 出力"):
        assert japanese not in plain, japanese
    ws = tmp_path / "ws"
    for name in ("a.txt", "b.txt", "c.txt"):
        assert not (ws / name).exists(), name


def test_empty_plan_card_is_reported(scenario: Runner, tmp_path: Path) -> None:
    """計画を書かずに exit すると、承認カードは「（計画ファイルがありません）」になる。"""
    result = scenario("empty_plan.jsonl", turns=(REQUEST,))
    assert "approval-card-empty" in result.report.problems
    assert "計画ファイルがありません" in tr.flat(result.transcript.text)
    assert not list((tmp_path / "ws" / ".satsuma" / "plans").glob("*.md"))


def test_step_budget_exhausted_is_reported(scenario: Runner) -> None:
    """段 2 で読み続けて反復の上限に当たる → 宣言 3 段 / 完了 [1] の不一致と失敗行。"""
    result = scenario(
        "step_budget_exhausted.jsonl", settings={"max_iterations": 8}, turns=(REQUEST,)
    )
    problems = result.report.problems
    assert "steps-mismatch: declared 3, completed [1]" in problems
    assert any("反復の上限（8 回）に達しました（段 2/3 の中）" in p for p in problems), problems
    flat = tr.flat(result.transcript.text)
    assert "計画の段2/3が未完です" in flat
    assert "reached max_iterations" not in flat
    assert "依頼を分けて頼み直すか" not in flat


def test_absolute_path_reaches_the_model(scenario: Runner) -> None:
    """行頭が絶対パスの依頼はスラッシュコマンドではなく会話としてモデルへ届く。"""
    line = "/Users/alice/example/dashboard を削除して"
    result = scenario("slash_path_passthrough.jsonl", turns=(line,), plan=False)
    users = [r for r in result.requests if r["role"] == "user"]
    assert users and users[0]["head"].startswith("/Users/"), result.requests
    text = result.transcript.text
    assert "未知のコマンドです" not in text
    assert "パスを受け取りました" in text
    assert re.search(r"\[code\] satsuma> ?/Users/", result.transcript.plain)


def test_model_switch_to_a_missing_model_is_refused(scenario: Runner) -> None:
    """/model に存在しない名前 → 切り替えを拒み、次の依頼は元のモデルで届く。"""
    result = scenario(
        "model_switch_unknown.jsonl",
        turns=("こんにちは",),
        plan=False,
        commands=(("/model", r"\* gpt-oss:120b"), ("/model gpt-oss:999b", r"のままです")),
    )
    text = tr.flat(result.transcript.text)
    assert tr.flat("モデル 'gpt-oss:999b' が接続先の Ollama") in text
    assert tr.flat("近い名前: gpt-oss:120b") in text
    assert tr.flat("model →") not in text
    users = [r for r in result.requests if r["role"] == "user"]
    assert users and all(r["model"] == "gpt-oss:120b" for r in users), result.requests
    assert "元のモデルで応答しました" in text


def test_user_deny_stops_the_turn(scenario: Runner, tmp_path: Path) -> None:
    """権限カードで n → モデルが同じ呼び出しを繰り返しても、カードは 1 回でターンが止まる。"""
    result = scenario(
        "user_deny_repeat.jsonl",
        turns=("built.txt を作って",),
        plan=False,
        answer="n",
        settings={"max_iterations": 6},
    )
    assert len(tr.permission_cards(result.transcript.lines, "bash")) == 1
    assert result.report.failures == [], result.report.failures
    assert "このターンを止めました" in tr.flat(result.transcript.text)
    assert [r["role"] for r in result.requests] == ["user"], result.requests
    assert not (tmp_path / "ws" / "built.txt").exists()


def test_detail_explains_what_the_call_does(scenario: Runner, tmp_path: Path) -> None:
    """d → モデルが会話から説明 → y / n / a だけで受け直す → 許可して実行。"""
    result = scenario(
        "permission_detail.jsonl",
        turns=("built.txt を作って",),
        plan=False,
        answer=("d", "y"),
    )
    text = result.transcript.plain
    assert text.count("[y=許可 / n=拒否 / a=永続許可 / d=詳細]") == 1
    assert text.count("[y=許可 / n=拒否 / a=永続許可]") == 1
    assert tr.flat("空の built.txt を作ります") in tr.flat(text)
    assert result.report.failures == [], result.report.failures
    assert (tmp_path / "ws" / "built.txt").exists()
    assert [r["role"] for r in result.requests] == ["user", "user", "tool"], result.requests


def test_destructive_card_has_no_always(scenario: Runner, tmp_path: Path) -> None:
    """削除のカードは a を出さず、a と打たれたら聞き直す。"""
    result = scenario(
        "no_remember.jsonl",
        turns=("built.txt を作って gone.txt を消して",),
        plan=False,
        answer=("a", "a", "y"),
    )
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    assert text.count("[y=許可 / n=拒否 / a=永続許可 / d=詳細]") == 1
    assert text.count("[y=許可 / n=拒否 / d=詳細]") == 2
    assert tr.flat("削除を伴うため永続許可できません") in tr.flat(text)
    assert tr.flat("この操作は永続許可できません") in tr.flat(text)
    ws = tmp_path / "ws"
    assert (ws / "built.txt").exists()
    memory = (ws / ".satsuma" / "memory.md").read_text(encoding="utf-8")
    rules = [line for line in memory.splitlines() if line.startswith("- ")]
    assert rules == ["- bash when command re: \\Atouch\\ built\\.txt\\Z"], rules


def test_deferred_tool_runs_on_first_direct_call(scenario: Runner) -> None:
    """遅延ツールの直呼びは、読み込んだうえでそのまま実行される（往復を捨てない）。"""
    result = scenario(
        "deferred_first_call.jsonl",
        turns=("次の作業を TODO にまとめて: a.txt と b.txt を作る",),
        plan=False,
        settings={"tools": {"core": []}},
    )
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    assert "not loaded yet" not in text and "✗" not in text
    assert text.count("・ToDo 更新") == 1
    assert tr.flat("▶ a.txt を作る") in tr.flat(text)
    assert "TODO を 2 件登録しました" in text
    tools = [r for r in result.requests if r["role"] == "tool"]
    assert len(result.requests) == 2 and len(tools) == 1
    assert "[~] a.txt を作る" in tools[0]["head"]


def test_args_only_json_reply_is_pushed_back_and_the_tool_runs(
    scenario: Runner, tmp_path: Path
) -> None:
    """最初の応答がツール名の無い引数だけの JSON でも、差し戻しを受けた呼び直しで完走する。"""
    result = scenario("args_only_reply.jsonl", turns=("a.txt に a と書いて",), plan=False)
    assert result.report.failures == [], result.report.failures
    heads = [r["head"] for r in result.requests if r["role"] == "user"]
    assert any("looks like the arguments of the tool `glob_search`" in h for h in heads), heads
    tools = [r for r in result.requests if r["role"] == "tool"]
    assert len(tools) == 1 and "a.txt" in tools[0]["head"]
    assert (tmp_path / "ws" / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert "a.txt を作りました" in result.transcript.plain


def test_report_without_a_write_is_pushed_back_and_the_file_gets_written(
    scenario: Runner, tmp_path: Path
) -> None:
    """読み取りだけで完了を報告しても、差し戻しを受けた書き込みで完走する（要求ログで確かめる）。"""
    result = scenario("no_change_report.jsonl", turns=("a.txt に a と書いて",), plan=False)
    assert result.report.failures == [], result.report.failures
    heads = [r["head"] for r in result.requests if r["role"] == "user"]
    assert any("but this turn changed nothing" in h and "`a.txt`" in h for h in heads), heads
    tools = [r["tool"] for r in result.requests if r["role"] == "tool"]
    assert tools == ["list_directory", "write_file"]
    assert (tmp_path / "ws" / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert "a.txt を作りました" in result.transcript.plain


def test_blocked_step_is_not_resumed_by_an_unrelated_request(scenario: Runner) -> None:
    """BLOCKED の直後に無関係な依頼を送っても段は再開せず、「続けて」で再開して完走する。"""
    result = scenario(
        "blocked_then_unrelated.jsonl",
        turns=(REQUEST, DELETE_REQUEST, "続けて"),
    )
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    assert text.count("BLOCKED: a.txt の中身") == 1
    notes = [r["note"] for r in result.requests]
    assert "無関係な依頼: 段は再開されない" in notes, notes
    assert notes.count("再開した段 1") == 1, notes
    assert "削除するものはありません" in text
    assert tr.flat(text).count(tr.flat("計画の段 1/3 が未完です")) == 2
    assert tr.completed_steps(text) == [1, 2, 3]
    ws = result.workspace
    assert all((ws / f"{n}.txt").exists() for n in "abc")


def test_long_code_block_keeps_scrolling_while_it_is_generated(scenario: Runner) -> None:
    """画面（40 行）を超えるコードブロック（120 行）の途中の行が完走より前に出る。"""
    seen: dict[str, float] = {}

    def during(index: int, driver: ReplDriver, fake: FakeOllama) -> None:
        base = tr.turn_ends(driver.transcript.plain)
        seen["at"] = driver.wait_for(r"line-0090", timeout=20)
        plain = driver.transcript.plain
        assert tr.turn_ends(plain) == base, "line-0090 が出たときにはターンが終わっていた"
        assert "line-0115" not in plain

    result = scenario(
        "long_code_block.jsonl", turns=("長いコードを本文に書いて",), plan=False, during=during
    )
    assert result.report.failures == [], result.report.failures
    assert seen["at"] < result.requests[0]["done_at"]
    text = result.transcript.text
    assert not any(line.strip() in ("...", "…") for line in result.transcript.lines)
    for n in (1, 37, 90, 120):
        assert f"<!-- line-{n:04d} -->" in text, n
    assert "以上です。" in text


def test_arg_panel_follows_each_write_file_in_a_multi_call_reply(scenario: Runner) -> None:
    """write_file 5 本を 1 応答で書く間、生成中パネルが 5 本目まで切り替わる。"""
    result = scenario("multi_write_stream.jsonl", turns=("五つの CSS を作って",), plan=False)
    assert result.report.failures == [], result.report.failures
    plain = result.transcript.plain
    assert re.search(r"write_file — 生成中 5 件目 f5\.css（\d+ 行）", plain), "5 本目の見出しが無い"
    assert re.search(r"write_file — 生成中 2 件目 f2\.css", plain)
    assert "✓ f4.css（40 行）" in plain
    assert "/*f5-l030*/" in plain
    ws = result.workspace
    for n in range(1, 6):
        assert (ws / f"f{n}.css").exists(), n


def test_write_file_of_a_later_part_is_rejected_and_the_step_recovers(scenario: Runner) -> None:
    """途中の部分の write_file は差し戻され、append_file に切り替えて完走する。"""
    result = scenario("step_overwrite_rejected.jsonl", turns=(REQUEST,))
    assert result.report.problems == [], result.report.problems
    flat = tr.flat(result.transcript.plain)
    assert tr.flat("✗ a.html — write_file rejected: would delete 5 lines of a.html") in flat
    assert tr.flat("a.html already has 5 lines (45 bytes) from this conversation") in flat
    assert tr.flat("(write_file + append_file)") in flat
    assert tr.flat("call append_file with the same content") in flat
    notes = [r["note"] for r in result.requests]
    assert "差し戻しを受けて append_file に切り替える" in notes, notes
    assert result.report.completed == [1, 2]
    text = (result.workspace / "a.html").read_text(encoding="utf-8")
    skeleton = "<!DOCTYPE html>\n<html>\n<head>\n</head>\n<body>\n"
    assert text == skeleton + "<style>\nbody { margin: 0; }\n</style>\n"


def test_second_silent_timeout_in_a_step_resumes_the_step(scenario: Runner) -> None:
    """段 1 で 2 回無音でも、ハーネスが段を再開して完走する（「続けて」を打たなくてよい）。"""
    result = scenario(
        "step_timeout_resume.jsonl",
        turns=(REQUEST,),
        settings={"provider": {"retry": {"request_timeout_s": 2, "extend_first_wait": False}}},
    )
    text = result.transcript.text
    flat = tr.flat(result.transcript.plain)
    assert "⚠ ターンが失敗しました" not in text
    assert tr.flat("— 2 秒応答がなかったので中断し、分けて出し直すよう依頼しました") in flat
    assert (
        tr.flat(
            "— 2 秒応答がなかったので中断し、段 1/2 を小さく分けて再開するよう依頼しました"
            "（この段で 2 回目・もう 1 度切れたら中止）"
        )
        in flat
    )
    assert len(result.report.failures) == 2, result.report.failures
    assert all("秒応答がなかったので中断し、" in line for line in result.report.failures)
    others = [p for p in result.report.problems if not p.startswith("failure: ")]
    assert others == [], others
    assert result.report.completed == [1, 2]
    notes = [r["note"] for r in result.requests]
    assert notes.count("段 1: 大きな書き込みを 1 回で出そうとして無音（1 回目の打ち切り）") == 1
    assert "1 回目の reminder の後も同じ大きさを繰り返す（2 回目の打ち切り）" in notes
    assert "段の再開の文を受けて小さく書く" in notes
    assert (result.workspace / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (result.workspace / "b.txt").read_text(encoding="utf-8") == "b\n"
    heads = [r["head"] for r in result.requests if r["role"] == "user"]
    assert any(
        "Your previous response was cancelled: nothing arrived for 2 seconds" in h for h in heads
    )
    assert any(
        "cancelled again" in h and "2 cancelled responses in a row in step 1 of 2" in h
        for h in heads
    )
    assert any("under ~60 lines" in h for h in heads), heads


LIST_REQUEST = "このディレクトリに何があるか、一覧で教えてください。"


def test_list_directory_runs_in_plan_mode_and_ls_is_a_variant(
    scenario: Runner, tmp_path: Path
) -> None:
    """list_directory は plan のまま実行され、ls は表記ゆれとして同じツールが走る。"""
    ws = tmp_path / "ws"
    (ws / "src").mkdir(parents=True)
    (ws / "src" / "a.py").write_text("print('a')\n", encoding="utf-8")
    (ws / "README.md").write_text("# demo\n", encoding="utf-8")
    (ws / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (ws / "node_modules" / "x").mkdir(parents=True)
    (ws / "node_modules" / "x" / "index.js").write_text("x\n", encoding="utf-8")
    result = scenario("list_directory_plan.jsonl", turns=(LIST_REQUEST,), plan=True)
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    tools = [r for r in result.requests if r["role"] == "tool"]
    assert [t["tool"] for t in tools] == ["list_directory", "ls"]
    first = tools[0]["head"]
    assert "src/" in first and "README.md" in first and "node_modules" not in first
    assert "plan mode is active" not in first and "unknown tool" not in first
    assert "a.py" in tools[1]["head"] and "ran list_directory" in tools[1]["head"]
    assert "一覧" in text and "✗" not in text
    assert "README.md と src/" in text


def test_turn_end_line_shows_the_token_breakdown(scenario: Runner) -> None:
    """ターン末尾に 出力・入力（回数）・文脈/しきい値 の内訳が 1 行出て、``tok`` は出ない。"""
    result = scenario(
        "token_summary.jsonl",
        turns=("このターンのトークンを数えてください。",),
        plan=False,
        settings={"compact_token_threshold": 50000, "compaction": {"auto_context_fit": False}},
    )
    text = result.transcript.text
    assert result.report.failures == [], result.report.failures
    assert "数えました。" in text
    assert "— 出力 678 · 入力 12.3K（1 回） · 文脈 13K/50K" in text
    assert " tok" not in text and "~" not in text
    assert tr.turn_ends(result.transcript.plain) == 1


def test_the_window_is_shown_and_rechecked_after_the_first_reply(scenario: Runner) -> None:
    """起動時は窓の内訳を出さない。"""
    result = scenario(
        "token_summary.jsonl",
        turns=("こんにちは",),
        plan=False,
        loaded_window=8192,
        cols=400,
    )
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    assert "文脈の窓 131,072 トークン" not in text
    assert "モデルはまだ読み込まれていません。" not in text
    assert "読み込んだモデルの窓は 8,192 トークンでした（起動時の見込みは 131,072）" in text
    assert "文脈の窓 8,192 トークン（実効 num_ctx（ロード済み））" in text
    assert "（6,553 から履歴を畳む）" in text
    assert "OLLAMA_CONTEXT_LENGTH=16384" in text
    assert re.search(r"文脈 [\d.]+K?/6\.55K", text), text
    assert text.index("読み込んだモデルの窓は 8,192 トークンでした") > text.index("— 出力 ")
    assert tr.turn_ends(result.transcript.plain) == 1


def test_a_modelfile_window_is_not_rechecked(scenario: Runner) -> None:
    """Modelfile の num_ctx はサーバの既定より優先されるので、未ロードでも測り直しを予告しない。"""
    result = scenario("token_summary.jsonl", turns=("こんにちは",), plan=False, cols=400)
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    assert "文脈の窓 131,072 トークン" not in text
    assert "モデルはまだ読み込まれていません。" not in text
    assert "読み込んだモデルの窓は" not in text


def test_a_call_still_broken_after_the_hint_is_rewritten(scenario: Runner, tmp_path: Path) -> None:
    """解析の 500 が誘導付きの再送の後も続いたら、ターンを落とさず書き直しを依頼して終える。"""
    result = scenario("tool_call_rewrite.jsonl", turns=("a.txt を作って",), plan=False)
    assert result.report.failures == [], result.report.failures
    text = result.transcript.text
    assert "— モデルの出力が途中で壊れたので、もう一度生成します" in text
    assert "— モデルの出力が繰り返し壊れたので、書き直すよう依頼しました" in text
    assert "a.txt を作りました。" in text
    assert (tmp_path / "ws" / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert tr.turn_ends(result.transcript.plain) == 1


@pytest.mark.parametrize(
    ("flavor", "server"),
    [("ollama", "Ollama"), ("lmstudio", "LM Studio"), ("llamacpp", None)],
)
def test_reloading_model_is_shown_in_a_later_turn(
    scenario: Runner, flavor: str, server: str | None
) -> None:
    """2 ターン目の前にモデルが降りていたら、そのターンの待ちでもロード中を出す。"""
    seen: dict[str, bool] = {}

    def during(index: int, driver: ReplDriver, fake: FakeOllama) -> None:
        if index != 1:
            return
        start = len(driver.transcript.plain)
        driver.wait_for(r"二つ目です。|をロード中…", timeout=20)
        driver.wait_for(r"二つ目です。", timeout=20)
        later = driver.transcript.plain[start:]
        seen["loading"] = f"{FAKE_MODEL} をロード中…" in later
        seen["before_reply"] = later.find("をロード中…") < later.find("二つ目です。")

    result = scenario(
        "model_reload.jsonl",
        turns=("一つ目を答えて", "二つ目を答えて"),
        plan=False,
        during=during,
        flavor=flavor,
    )
    assert result.report.failures == [], result.report.failures
    assert seen == {"loading": True, "before_reply": True}
    assert tr.turn_ends(result.transcript.plain) == 2
    if server is None:
        assert "を降ろしました" not in result.transcript.text
    else:
        assert f"{server} から {FAKE_MODEL} を降ろしました" in result.transcript.text


def test_lmstudio_window_is_read_and_the_model_is_unloaded(scenario: Runner) -> None:
    """LM Studio は起動時に判定し、読み込んだ窓に合わせ、終了時に読み込ませたモデルを降ろす。"""
    result = scenario(
        "token_summary.jsonl",
        turns=("こんにちは",),
        plan=False,
        loaded_window=8192,
        cols=400,
        flavor="lmstudio",
    )
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    assert "読み込んだモデルの窓は 8,192 トークンでした（起動時の見込みは 131,072）" in text
    assert "文脈の窓 8,192 トークン（ロード済みのモデルの窓（LM Studio））" in text
    assert f"`lms load {FAKE_MODEL} --context-length 16384`" in text
    assert "OLLAMA_CONTEXT_LENGTH" not in text
    assert re.search(r"文脈 [\d.]+K?/6\.55K", text), text
    assert f"LM Studio から {FAKE_MODEL} を降ろしました" in text
    assert result.unloads == [FAKE_MODEL]


def test_lmstudio_overflow_is_resent_once_with_the_loaded_window(scenario: Runner) -> None:
    """窓あふれの ``event: error`` を受けたら、読み込んだ窓に合わせて 1 回だけ送り直す。"""
    result = scenario(
        "lmstudio_overflow.jsonl",
        turns=("窓あふれを起こして",),
        plan=False,
        loaded_window=8192,
        cols=400,
        flavor="lmstudio",
    )
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    assert "モデルの窓より長い要求として断りました。窓 8,192 に合わせて送り直します" in text
    assert "送り直しを受けました。" in text
    assert [r["note"] for r in result.requests] == ["窓あふれ", "catch-all"]
    assert tr.turn_ends(result.transcript.plain) == 1


def test_llamacpp_overflow_is_resent_once_with_the_counted_size(
    scenario: Runner, tmp_path: Path
) -> None:
    """llama.cpp の窓あふれの 400 には、サーバが数えた量でツールの結果を切って 1 回だけ送り直す。"""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    lines = "".join(f"line {i:04d} of a large file\n" for i in range(1500))
    (ws / "big.txt").write_text(lines, encoding="utf-8")
    result = scenario(
        "llamacpp_overflow.jsonl",
        turns=("窓あふれを起こして",),
        plan=False,
        loaded_window=16384,
        cols=400,
        flavor="llamacpp",
    )
    text = result.transcript.plain
    assert result.report.failures == [], result.report.failures
    assert "OLLAMA_CONTEXT_LENGTH" not in text
    assert "モデルの窓より長い要求として断りました。窓 16,384 に合わせて送り直します" in text
    assert text.count("だけを渡しました") == 2
    assert "送り直しを受けました。" in text and "切り詰めました" not in text
    notes = [r["note"] for r in result.requests]
    assert notes == ["大きなファイルを読む", "窓あふれ", "catch-all"]
    assert tr.turn_ends(text) == 1
    assert "を降ろしました" not in text and result.unloads == []


@pytest.mark.parametrize("answer", ["y", "n"])
def test_first_open_asks_for_trust(scenario: Runner, tmp_path: Path, answer: str) -> None:
    """信頼の記録が無い作業ディレクトリでは、バナーの前に確認が出る。"""
    result = scenario(
        "trust_first_open.jsonl",
        turns=("こんにちは",),
        plan=False,
        settings={"provider": {"sampling": {"reasoning_effort": "high"}}},
        commands=(("/effort", r"reasoning effort: \w+"),),
        trust_answer=answer,
    )
    text = tr.flat(result.transcript.text)
    assert "このディレクトリを初めて開きます" in text
    record = tmp_path / "home" / ".satsuma" / "trusted.json"
    if answer == "y":
        assert "信頼しました" in text and tr.flat("reasoning effort: high") in text
        roots = json.loads(record.read_text(encoding="utf-8"))["roots"]
        assert roots == [str((tmp_path / "ws").resolve())]
    else:
        assert "未信頼:" in text and tr.flat("reasoning effort: default") in text
        assert not record.exists()
    assert "了解しました。" in text
    assert result.report.failures == [], result.report.failures


@pytest.mark.parametrize(("answer", "saved"), [("2", "en"), ("", "ja")])
def test_first_start_asks_for_the_screen_language(
    scenario: Runner, tmp_path: Path, answer: str, saved: str
) -> None:
    """ユーザーの設定に画面の言語が無ければ、信頼の確認より前に選んでもらい、設定に記録する。"""
    result = scenario(
        "trust_first_open.jsonl",
        turns=("こんにちは",),
        plan=False,
        trust_answer="y",
        screen_answer=answer,
    )
    text = tr.flat(result.transcript.text)
    assert tr.flat("画面の言語を選んでください / Choose the screen language") in text
    assert tr.flat("1) 日本語") in text and tr.flat("2) English") in text
    settings = tmp_path / "home" / ".satsuma" / "settings.json"
    assert json.loads(settings.read_text(encoding="utf-8")) == {"language": {"ui": saved}}
    if saved == "en":
        assert tr.flat("Screen language: English (saved as language.ui in") in text
        assert tr.flat("Opening this directory for the first time") in text
        assert tr.flat("/help for commands, /exit or Ctrl-D to quit") in text
        assert tr.flat("— output 1 · input") in text
        assert "このディレクトリを初めて開きます" not in text and "— 出力" not in text
    else:
        assert tr.flat("画面の言語: 日本語（") in text
        assert "このディレクトリを初めて開きます" in text
        assert tr.flat("— 出力 1 · 入力") in text
    assert "了解しました。" in text
    assert result.report.failures == [], result.report.failures


def test_language_command_switches_the_screen(scenario: Runner, tmp_path: Path) -> None:
    """``/language en`` は画面だけを英語にして設定に記録し、日本語の発話の後も英語のまま。"""
    result = scenario(
        "trust_first_open.jsonl",
        turns=("こんにちは",),
        plan=False,
        commands=(("/language en", r"Screen language → English"),),
    )
    text = tr.flat(result.transcript.text)
    assert tr.flat("Screen language → English (saved as language.ui in") in text
    assert tr.flat("— output 1 · input") in text and "— 出力" not in text
    settings = tmp_path / "home" / ".satsuma" / "settings.json"
    assert json.loads(settings.read_text(encoding="utf-8")) == {"language": {"ui": "en"}}
    assert "Reply in Japanese." in result.requests[0]["head"]
    assert "了解しました。" in text
    assert result.report.failures == [], result.report.failures


def test_the_first_screens_follow_the_environment_language(scenario: Runner) -> None:
    """最初の発話より前の画面（信頼の確認・バナー・作業場所）は、環境の言語で出る。"""
    result = scenario(
        "trust_first_open.jsonl",
        turns=("こんにちは",),
        plan=False,
        trust_answer="y",
        locale="en_US.UTF-8",
    )
    text = tr.flat(result.transcript.text)
    assert tr.flat("Opening this directory for the first time") in text
    assert "Trusted:" in text and tr.flat("(recorded in") in text
    assert tr.flat("/help for commands, /exit or Ctrl-D to quit") in text
    assert tr.flat("Working directory:") in text
    assert "このディレクトリを初めて開きます" not in text and "作業場所" not in text
    assert result.report.failures == [], result.report.failures
