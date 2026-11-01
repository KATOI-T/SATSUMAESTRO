"""ツールを呼ばずに本文だけで閉じるターンの差し戻し（引数だけの JSON・編集 0 回の完了報告）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.reminders import (
    args_only_reply_note,
    empty_final_note,
    no_change_report_note,
)
from satsumaestro.engine.reply_checks import (
    BareToolArgs,
    claims_completion,
    detect_bare_tool_args,
    mentioned_workspace_paths,
    no_change_report,
    parse_whole_json,
    strip_code_fence,
)
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.engine.turn import DenyPrompter
from satsumaestro.evals.agents import SatsumaAgent
from satsumaestro.evals.models import EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.allowlist import PermissionResolver
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry
from satsumaestro.util.errors import ToolCallingUnsupportedError
from tests.fixtures.mock_provider import SequencedMockProvider

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"

GLOB_ARGS = '{"path": ".", "pattern": "**/*.js", "limit": 1000, "exclude": []}'
LIST_ARGS = '{"path": ".", "limit": 100}'


def _runtime(
    tmp_path: Path,
    *,
    config: RuntimeConfig | None = None,
    plan: bool = False,
    mode: PermissionMode = PermissionMode.WORKSPACE_WRITE,
) -> tuple[ConversationRuntime, list[list[AssistantEvent]], SequencedMockProvider]:
    """code モード（既定）の runtime と、後から差し替えられる台本と、プロバイダを返す。"""
    config = config or RuntimeConfig()
    config.guidance.step_budget = False
    script: list[list[AssistantEvent]] = [[MessageStop()]]
    provider = SequencedMockProvider(script)
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=provider,  # type: ignore[arg-type]
        tools=default_registry(),
        policy=PermissionResolver(default_policy(mode), []),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=config,
    )
    if plan:
        rt.set_mode(main="plan")
        (tmp_path / ".satsuma" / "plans").mkdir(parents=True)
    return rt, script, provider


def _reminders(rt: ConversationRuntime) -> list[str]:
    """モデルへ差し込んだ ``<system-reminder>`` の本文（ターン内の注入だけ）。"""
    return [
        b.text
        for m in rt.session.messages[1:]
        if m.role == "user"
        for b in m.blocks
        if isinstance(b, TextBlock) and "<system-reminder>" in b.text
    ]


def _text(text: str) -> list[AssistantEvent]:
    return [TextDelta(text=text), MessageStop()]


def _write(id: str, path: str) -> ToolUse:
    return ToolUse(id=id, name="write_file", input={"path": path, "content": f"{path[0]}\n"})


def test_strip_code_fence_variants() -> None:
    body = '{"a": 1}'
    assert strip_code_fence(f"```json\n{body}\n```") == body
    assert strip_code_fence(f"```\n{body}\n```") == body
    assert strip_code_fence(f"```JSON {body}```") == body
    assert strip_code_fence(body) == body


def test_parse_whole_json_requires_the_whole_body() -> None:
    assert parse_whole_json(' {"a": 1} \n') == {"a": 1}
    assert parse_whole_json('```json\n{"a": "}"}\n```') == {"a": "}"}
    assert parse_whole_json('{"a": 1} and then prose') is None
    assert parse_whole_json('prose then {"a": 1}') is None
    assert parse_whole_json("[1, 2]") is None
    assert parse_whole_json("{broken") is None


def _schemas() -> list[dict[str, object]]:
    return default_registry().json_schemas()


def test_bare_args_match_one_tool_by_schema_keys() -> None:
    got = detect_bare_tool_args(GLOB_ARGS, _schemas())
    assert got == BareToolArgs(keys=("path", "pattern", "limit", "exclude"), tools=("glob_search",))


def test_bare_args_with_two_candidates_prefer_the_closest_and_drop_mutating_tools() -> None:
    ro = {"read_file", "glob_search", "grep_search", "list_directory"}
    got = detect_bare_tool_args(LIST_ARGS, _schemas(), read_only=lambda n: n in ro)
    assert got is not None and got.tools == ("list_directory", "read_file")
    got = detect_bare_tool_args(
        '{"path": "a.txt", "content": "x"}', _schemas(), read_only=lambda n: n in ro
    )
    assert got is not None and got.tools == ("append_file", "write_file")
    got = detect_bare_tool_args('{"path": "x"}', _schemas(), read_only=lambda n: n in ro)
    assert got is not None and "delete_path" not in got.tools


def test_bare_args_negatives() -> None:
    schemas = _schemas()
    assert detect_bare_tool_args("{}", schemas) is None
    assert detect_bare_tool_args('{"foo": 1}', schemas) is None
    assert detect_bare_tool_args(f"{GLOB_ARGS}\nThen I would read the files.", schemas) is None
    assert detect_bare_tool_args('{"name": "glob_search", "arguments": {}}', schemas) is None
    assert detect_bare_tool_args('{"tool_calls": [{"name": "bash"}]}', schemas) is None
    assert detect_bare_tool_args("Here is my answer.", schemas) is None
    assert detect_bare_tool_args('{"path": ".", "include": "*.py"}', schemas) is None


async def test_bare_glob_args_are_pushed_back_once_and_the_tool_runs(tmp_path: Path) -> None:
    (tmp_path / "a.js").write_text("x\n", encoding="utf-8")
    rt, script, provider = _runtime(tmp_path)
    script[:] = [
        _text(GLOB_ARGS),
        [ToolUse(id="g1", name="glob_search", input={"pattern": "**/*.js"}), MessageStop()],
        _text("a.js が 1 つあります。"),
    ]
    summary = await rt.run_turn("JS ファイルを探して")
    notes = _reminders(rt)
    assert len(notes) == 1
    expected = args_only_reply_note(("path", "pattern", "limit", "exclude"), ("glob_search",))
    assert expected in notes[0]
    assert "`glob_search`" in notes[0] and "`path`, `pattern`, `limit`, `exclude`" in notes[0]
    assert summary.final_text == "a.js が 1 つあります。"
    assert (summary.args_only_replies, summary.reminders, summary.tool_calls) == (1, 1, 1)
    assert len(provider.requests) == 3


async def test_two_candidates_are_listed(tmp_path: Path) -> None:
    rt, script, _ = _runtime(tmp_path)
    script[:] = [_text(LIST_ARGS), _text("ファイルはありません。")]
    summary = await rt.run_turn("何がある？")
    notes = _reminders(rt)
    assert len(notes) == 1 and "one of these tools: `list_directory`, `read_file`" in notes[0]
    assert "delete_path" not in notes[0]
    assert summary.args_only_replies == 1


async def test_named_call_json_still_raises_unsupported(tmp_path: Path) -> None:
    rt, script, _ = _runtime(tmp_path)
    script[:] = [_text('{"name": "glob_search", "arguments": {"pattern": "*"}}')]
    with pytest.raises(ToolCallingUnsupportedError):
        await rt.run_turn("探して")


async def test_bare_args_twice_is_not_looped(tmp_path: Path) -> None:
    rt, script, _ = _runtime(tmp_path)
    script[:] = [_text(GLOB_ARGS), _text(LIST_ARGS)]
    summary = await rt.run_turn("探して")
    assert len(_reminders(rt)) == 1
    assert summary.final_text == LIST_ARGS
    assert (summary.args_only_replies, summary.reminders) == (2, 1)


async def test_without_budget_it_counts_but_does_not_push_back(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.max_iterations = 1
    rt, script, _ = _runtime(tmp_path, config=config)
    script[:] = [_text(GLOB_ARGS), _text("never")]
    summary = await rt.run_turn("探して")
    assert _reminders(rt) == []
    assert summary.final_text == GLOB_ARGS
    assert (summary.args_only_replies, summary.reminders) == (1, 0)


async def test_disabled_by_config(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.guidance.args_only_retry = False
    rt, script, _ = _runtime(tmp_path, config=config)
    script[:] = [_text(GLOB_ARGS), _text("never")]
    summary = await rt.run_turn("探して")
    assert _reminders(rt) == [] and summary.final_text == GLOB_ARGS
    assert summary.args_only_replies == 1


async def test_fenced_json_is_detected_and_plain_answers_are_not(tmp_path: Path) -> None:
    rt, script, _ = _runtime(tmp_path)
    script[:] = [_text(f"```json\n{GLOB_ARGS}\n```"), _text("答えです。")]
    summary = await rt.run_turn("探して")
    assert len(_reminders(rt)) == 1 and summary.args_only_replies == 1
    for text in ("答えです。", '{"answer": 42}', "{}", f"{GLOB_ARGS}\n上の条件で探します。"):
        rt, script, _ = _runtime(tmp_path)
        script[:] = [_text(text)]
        summary = await rt.run_turn("質問です")
        assert _reminders(rt) == [] and summary.final_text == text
        assert summary.args_only_replies == 0


async def test_in_plan_mode_it_fires_before_the_exit_nudge(tmp_path: Path) -> None:
    rt, script, _ = _runtime(tmp_path, plan=True)
    plan_file = rt.plan_file_relpath()
    args = json.dumps({"path": plan_file, "content": "# plan\n"})
    script[:] = [_text(f"```json\n{args}\n```"), _text("調べた結果、変更は要りません。")]
    summary = await rt.run_turn("調べて")
    notes = _reminders(rt)
    assert len(notes) == 1 and "`write_file`" in notes[0]
    assert "exit_plan_mode" not in notes[0]
    assert (summary.args_only_replies, summary.plan_exit_nudges) == (1, 0)


async def test_latch_is_reset_when_a_step_advances(tmp_path: Path) -> None:
    """段階実行では段ごとに 1 回。"""
    rt, script, _ = _runtime(tmp_path)
    rt.session.plan_steps = ["1. a を書く", "2. b を書く"]
    rt.session.plan_step_index = 1
    script[:] = [
        _text(GLOB_ARGS),
        [_write("w1", "a.txt"), MessageStop()],
        _text("段 1 完了"),
        _text(GLOB_ARGS),
        [_write("w2", "b.txt"), MessageStop()],
        _text("すべて完了"),
    ]
    summary = await rt.run_turn("続けて")
    assert summary.plan_steps_run == 2
    pushed = [n for n in _reminders(rt) if "looks like the arguments" in n]
    assert len(pushed) == 2 and summary.args_only_replies == 2
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()


def test_code_mode_empty_final_text_is_neutral() -> None:
    note = empty_final_note()
    assert "do it now with tool calls" in note and "without calling any tools" not in note
    assert "state your final answer as plain message text" in note


async def test_counter_reaches_the_report_json(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    provider = SequencedMockProvider([_text(GLOB_ARGS), _text("JS ファイルはありません。")])
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    task = EvalTask.model_validate(
        {
            "id": "js",
            "prompt": "JS ファイルを探して",
            "fixture": str(fixture),
            "permission_mode": "workspace-write",
            "checks": {"contains_all": ["ありません"]},
        }
    )
    report = await EvalRunner(agent).run([task])
    result = report.results[0]
    assert result.passed and result.args_only_replies == 1
    dumped = json.loads(report.model_dump_json())
    assert dumped["results"][0]["args_only_replies"] == 1
    assert EvalReport.model_validate(dumped).results[0].args_only_replies == 1
    assert TaskResult.model_validate({"task_id": "t", "passed": True}).args_only_replies == 0


def test_ab_summary_has_the_row(tmp_path: Path) -> None:
    before = tmp_path / "b.json"
    before.write_text(
        json.dumps({"results": [{"task_id": "t0", "passed": True}]}), encoding="utf-8"
    )
    after = tmp_path / "a.json"
    after.write_text(
        json.dumps(
            {
                "agent": "satsuma",
                "results": [
                    {"task_id": "t0", "passed": True, "args_only_replies": 2},
                    {"task_id": "t1", "passed": False, "args_only_replies": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "x", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 引数だけの JSON の本文 | 0.0 | 3.0 | +3.0 |" in out


def test_mentioned_workspace_paths_filters_non_paths() -> None:
    text = (
        "Created `common/RankingQuestion.js` and updated `design.js` (see `design.js:12`). "
        "Not paths: `foo`, `https://x.y/z`, `/abs/path.js`, `../up.js`, `src/**/*.js`, "
        "`a b.js`, `--flag`, `~/home.js`, `./common/Question.js`"
    )
    assert mentioned_workspace_paths(text) == [
        "common/RankingQuestion.js",
        "design.js",
        "common/Question.js",
    ]


def test_claims_completion_positive_and_negative() -> None:
    for text in (
        "Implemented full RankingQuestion support.",
        "### Key Updates\n1. **Created `common/RankingQuestion.js`**",
        "- Added a button",
        "`common/OpenQuestion.js` を作成しました。",
        "SurveyDesign を更新した。",
    ):
        assert claims_completion(text), text
    for text in (
        "The class is already implemented in `common/Question.js`.",
        "`common/Question.js` は実装済みです。",
        "このファイルは選択肢を追加しています。",
        "`total()` は小計に税を掛けて返します。",
        "Question created the fieldset lazily.",
    ):
        assert not claims_completion(text), text


def test_no_change_report_verdicts(tmp_path: Path) -> None:
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "Question.js").write_text("x\n", encoding="utf-8")
    common = dict(cwd=tmp_path, written=(), since=float("inf"))
    got = no_change_report("Created `common/RankingQuestion.js`.", **common)
    assert got is not None and got.missing == ("common/RankingQuestion.js",)
    assert no_change_report("`common/RankingQuestion.js` を見てください。", **common) is None
    assert no_change_report("`src/foo.js` はありません。", **common) is None
    assert no_change_report("`tests/test_x.py` を足すとよいでしょう。", **common) is None
    assert no_change_report("`runtime.py` の説明です。", **common) is None
    assert no_change_report("`common/Question.js` は fieldset を作ります。", **common) is None
    got = no_change_report("`common/Question.js` を更新しました。", **common)
    assert got is not None and got.missing == ()
    written = {(tmp_path / "common" / "Question.js").resolve()}
    assert (
        no_change_report(
            "`common/Question.js` を更新しました。", cwd=tmp_path, written=written, since=0.0
        )
        is None
    )
    mtime = (tmp_path / "common" / "Question.js").stat().st_mtime
    assert (
        no_change_report(
            "`common/Question.js` を更新しました。", cwd=tmp_path, written=(), since=mtime - 1
        )
        is None
    )
    assert no_change_report("Implemented everything.", **common) is None
    assert no_change_report("Created `../outside/x.js`.", **common) is None


REPORT = "Implemented full RankingQuestion support.\n\n1. Created `common/RankingQuestion.js`"


def _read(id: str, path: str) -> ToolUse:
    return ToolUse(id=id, name="read_file", input={"path": path})


def _pushbacks(rt: ConversationRuntime) -> list[str]:
    """編集 0 回の完了報告への差し戻しだけ（言語修復などの他の注入は数えない）。"""
    return [n for n in _reminders(rt) if "changed nothing" in n]


async def test_report_without_any_write_is_pushed_back_and_the_file_gets_written(
    tmp_path: Path,
) -> None:
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "Question.js").write_text("x\n", encoding="utf-8")
    rt, script, _ = _runtime(tmp_path)
    script[:] = [
        [_read("r1", "common/Question.js"), MessageStop()],
        _text(REPORT),
        [_write("w1", "common/RankingQuestion.js"), MessageStop()],
        _text("`common/RankingQuestion.js` を作成しました。"),
    ]
    summary = await rt.run_turn("RankingQuestion を作って")
    notes = _pushbacks(rt)
    assert len(notes) == 1 and len(_reminders(rt)) == 1
    expected = no_change_report_note(("common/RankingQuestion.js",), ("common/RankingQuestion.js",))
    assert expected in notes[0] and "does not exist" in notes[0]
    assert (tmp_path / "common" / "RankingQuestion.js").exists()
    assert summary.final_text.endswith("作成しました。")
    assert (summary.no_change_reports, summary.reminders, summary.tool_calls) == (1, 1, 2)


async def test_existing_file_with_a_completion_claim_is_pushed_back_once(tmp_path: Path) -> None:
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "OpenQuestion.js").write_text("x\n", encoding="utf-8")
    rt, script, _ = _runtime(tmp_path)
    claim = "Implemented OpenQuestion support:\n1. `common/OpenQuestion.js` renders a textarea"
    script[:] = [
        [_read("r1", "common/OpenQuestion.js"), MessageStop()],
        _text(claim),
        _text("変更は要りません。`common/OpenQuestion.js` は既に textarea を描いています。"),
    ]
    summary = await rt.run_turn("OpenQuestion に name を付けて")
    assert len(_pushbacks(rt)) == 1 and "does not exist" not in _pushbacks(rt)[0]
    assert summary.final_text.startswith("変更は要りません。")
    assert summary.no_change_reports == 1


async def test_report_after_a_real_write_is_not_pushed_back(tmp_path: Path) -> None:
    rt, script, _ = _runtime(tmp_path)
    script[:] = [
        [_write("w1", "common/RankingQuestion.js"), MessageStop()],
        _text(REPORT),
    ]
    summary = await rt.run_turn("RankingQuestion を作って")
    assert _pushbacks(rt) == [] and summary.no_change_reports == 0


async def test_report_after_a_denied_write_is_not_pushed_back(tmp_path: Path) -> None:
    """書こうとして止められた形は POLICY_DENY の持ち場（「ツールで書け」と言うと矛盾する）。"""
    rt, script, _ = _runtime(tmp_path, mode=PermissionMode.READ_ONLY)
    script[:] = [
        [_write("w1", "common/RankingQuestion.js"), MessageStop()],
        _text(REPORT),
    ]
    summary = await rt.run_turn("RankingQuestion を作って", DenyPrompter())
    assert not (tmp_path / "common" / "RankingQuestion.js").exists()
    assert _pushbacks(rt) == [] and summary.no_change_reports == 0


async def test_qa_answers_are_not_pushed_back(tmp_path: Path) -> None:
    (tmp_path / "satsumaestro" / "engine").mkdir(parents=True)
    (tmp_path / "satsumaestro" / "engine" / "runtime.py").write_text("x\n", encoding="utf-8")
    rt, script, _ = _runtime(tmp_path)
    answer = (
        "`runtime.py` の `run_turn` がループの本体です。`satsumaestro/engine/runtime.py` は "
        "ツール結果を `session.messages` に積みます。"
    )
    script[:] = [[_read("r1", "satsumaestro/engine/runtime.py"), MessageStop()], _text(answer)]
    summary = await rt.run_turn("ループはどこ？")
    assert _reminders(rt) == [] and summary.final_text == answer
    assert summary.no_change_reports == 0


async def test_answers_about_missing_paths_are_not_pushed_back(tmp_path: Path) -> None:
    """無いパスに触れるだけ（完了の語なし）の回答・提案は差し戻さない。"""
    for answer in ("`src/foo.js` はありません。", "`tests/test_x.py` を足すとよいでしょう。"):
        rt, script, _ = _runtime(tmp_path)
        script[:] = [
            [ToolUse(id="g1", name="glob_search", input={"pattern": "src/foo.js"}), MessageStop()],
            _text(answer),
        ]
        summary = await rt.run_turn("`src/foo.js` はある？")
        assert _pushbacks(rt) == [] and summary.no_change_reports == 0


async def test_plan_mode_report_is_not_pushed_back(tmp_path: Path) -> None:
    rt, script, _ = _runtime(tmp_path, plan=True)
    script[:] = [_text(REPORT), _text("調査は終わりました。")]
    summary = await rt.run_turn("調べて")
    assert _pushbacks(rt) == [] and summary.no_change_reports == 0


async def test_disabled_by_config_counts_only(tmp_path: Path) -> None:
    config = RuntimeConfig()
    config.guidance.no_change_report_retry = False
    rt, script, _ = _runtime(tmp_path, config=config)
    script[:] = [_text(REPORT), _text("never")]
    summary = await rt.run_turn("RankingQuestion を作って")
    assert _pushbacks(rt) == [] and summary.no_change_reports == 1
    assert summary.tool_calls == 0


def test_ab_summary_has_the_no_change_row(tmp_path: Path) -> None:
    before = tmp_path / "b.json"
    before.write_text(
        json.dumps({"results": [{"task_id": "t0", "passed": True}]}), encoding="utf-8"
    )
    after = tmp_path / "a.json"
    after.write_text(
        json.dumps(
            {
                "agent": "satsuma",
                "results": [{"task_id": "t0", "passed": False, "no_change_reports": 1}],
            }
        ),
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "x", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 編集 0 回の完了報告 | 0.0 | 1.0 | +1.0 |" in out
