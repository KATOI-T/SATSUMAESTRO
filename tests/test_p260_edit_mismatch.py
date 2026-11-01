"""edit_file の不一致を、次の 1 回で直せる材料つきで返すテスト。"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console

from satsumaestro.cli.render import change_preview
from satsumaestro.config.models import HookConfig, RuntimeConfig, ToolsConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.reminders import EDIT_FILE_STREAK_HINT, ReminderState
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.evals.agents import SatsumaAgent
from satsumaestro.evals.defects import detect_defects
from satsumaestro.evals.models import EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import ToolResultBlock, ToolUseBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, edit_match
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.edit_file import EditFile
from satsumaestro.tools.builtins.read_file import ReadFile
from satsumaestro.tools.read_ledger import FLAG_STALE, ReadLedger
from satsumaestro.tools.registry import describe_validation_error
from tests.fixtures.mock_provider import SequencedMockProvider

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "eval_ab_summary.py"
_PREVIEW_JS = (Path(__file__).parent / "fixtures" / "issue260_preview.js").read_text(
    encoding="utf-8"
)

SOURCE = (
    "class Survey {\n"
    "  constructor(form) {\n"
    "    this.form = form;\n"
    "    this.questions = [];\n"
    "  }\n"
    "\n"
    "  add(question) {\n"
    "    this.questions.push(question);\n"
    "    this.render();\n"
    "  }\n"
    "}\n"
)


def _ctx(tmp_path: Path) -> ToolContext:
    jail = WorkspaceJail(tmp_path)
    return ToolContext(workspace_root=jail.root, jail=jail, config=RuntimeConfig())


def test_not_found_points_at_the_closest_lines_with_numbers() -> None:
    old = "  add(question) {\n    this.questions.append(question);\n    this.render();\n  }\n"
    match = edit_match.closest_match(SOURCE, old)
    assert match is not None and (match.start, match.end) == (7, 10)
    assert not match.whitespace_only and 0.8 < match.similarity < 1.0
    msg = edit_match.not_found_message("s.js", SOURCE, old, "x")
    lines = msg.splitlines()
    assert lines[0] == "old string not found in s.js."
    assert lines[1].startswith("The closest text in the file is at lines 7-10 (")
    assert lines[2:6] == [
        "   7 |   add(question) {",
        "   8 |     this.questions.push(question);",
        "   9 |     this.render();",
        "  10 |   }",
    ]
    assert "verbatim" in lines[-1]


def test_whitespace_only_difference_names_the_indentation() -> None:
    old = "add(question) {\n  this.questions.push(question);\n"
    match = edit_match.closest_match(SOURCE, old)
    assert match is not None and match.whitespace_only and (match.start, match.end) == (7, 8)
    msg = edit_match.not_found_message("s.js", SOURCE, old, "x")
    assert "whitespace or indentation differs" in msg
    assert (
        "First difference: line 7 is indented with 2 spaces in the file, "
        "but with no indentation in your `old`." in msg
    )


def test_first_difference_names_the_line_and_never_says_100_percent() -> None:
    text = "".join(f"  // line {i} of a long comment block here\n" for i in range(1, 30))
    text += "  /**\n   * @param {string} title - Title for the survey.\n   */\n"
    old = "  /**\n   @param {string} title - Title for the survey.\n   */\n"
    msg = edit_match.not_found_message("d.js", text, old, "x")
    assert "(99% similar)" in msg or "% similar)" in msg and "100%" not in msg
    assert "First difference at line 31:" in msg
    assert "  file:    * @param {string} title - Title for the survey." in msg
    assert "  old:     @param {string} title - Title for the survey." in msg


def test_first_difference_reports_skipped_extra_and_edge_blank_lines() -> None:
    window = SOURCE.splitlines()[6:10]
    skipped = edit_match.first_difference(window, "  add(question) {\n    this.render();\n  }\n", 7)
    assert skipped == (
        "First difference: your `old` skips line 8 of the file:     this.questions.push(question);"
    )
    extra = edit_match.first_difference(
        window, "\n".join([*window[:2], "    log();", *window[2:]]), 7
    )
    assert extra == (
        "First difference: your `old` has a line that is not in the file before line 9:     log();"
    )
    tail = edit_match.not_found_message(
        "s.js", SOURCE, "  add(question) {\n    this.questions.push(question);\n\n", "x"
    )
    assert "your `old` ends with 1 empty line after line 8, but line 9 of the file is:" in tail


def test_applied_already_and_nothing_similar() -> None:
    new = "    this.questions.push(question);\n"
    assert "applied already" in edit_match.not_found_message(
        "s.js", SOURCE, "    this.items.push(question);\n", new
    )
    msg = edit_match.not_found_message("s.js", SOURCE, "// ... ... ... ...", "x")
    assert "No similar text was found" in msg and "read_file s.js" in msg
    assert edit_match.closest_match("", "x") is None
    assert edit_match.closest_match(SOURCE, "\n\n") is None


def test_long_candidates_are_capped() -> None:
    text = "".join(f"line {i} = value_{i};\n" for i in range(1, 61))
    old = "".join(f"line {i} = value_{i}!;\n" for i in range(10, 40))
    msg = edit_match.not_found_message("big.js", text, old, "x")
    numbered = [ln for ln in msg.splitlines() if " | " in ln]
    assert len(numbered) == edit_match.MAX_SHOWN_LINES
    assert "more lines)" in msg


def test_real_webbench_preview_js_points_at_the_line_the_model_wrote_next() -> None:
    old = "import RatingQuestion from './RatingQuestion.js';"
    match = edit_match.closest_match(_PREVIEW_JS, old)
    assert match is not None and (match.start, match.end) == (7, 7)
    msg = edit_match.not_found_message("preview.js", _PREVIEW_JS, old, "x")
    assert "7 | import RatingQuestion from './common/RatingQuestion.js';" in msg


def test_search_stays_fast_on_a_long_old_in_a_long_file() -> None:
    text = "".join(
        f"  method{i}(arg) {{\n    return this.value{i} + arg * {i};\n  }}\n" for i in range(300)
    )
    old = "".join(
        f"  method{i}(arg) {{\n    return this.value{i} - arg * {i};\n  }}\n"
        for i in range(120, 160)
    )
    started = time.perf_counter()
    match = edit_match.closest_match(text, old)
    assert time.perf_counter() - started < 1.0
    assert match is not None and match.start == 120 * 3 + 1


def test_not_unique_lists_each_match_line() -> None:
    text = "a = 1\nfoo()\nb = 2\nfoo()\nc = 3\nfoo()\n"
    assert edit_match.match_lines(text, "foo()") == [2, 4, 6]
    msg = edit_match.not_unique_message("m.py", text, "foo()", 3)
    assert msg.splitlines()[0] == (
        "old string is not unique in m.py (3 matches, starting at lines 2, 4, 6):"
    )
    assert "  2 | foo()" in msg and "replace_all=true" in msg


async def test_edit_file_returns_the_hint_and_flag_without_writing(tmp_path: Path) -> None:
    (tmp_path / "s.js").write_text(SOURCE, encoding="utf-8")
    ctx = _ctx(tmp_path)
    miss = await EditFile().run(
        EditFile.Input(path="s.js", old="this.questions.append(question);", new="x"), ctx
    )
    assert miss.is_error and miss.flags == [edit_match.FLAG_MISMATCH]
    assert "8 |     this.questions.push(question);" in miss.output
    dup = await EditFile().run(EditFile.Input(path="s.js", old="this.", new="self."), ctx)
    assert dup.is_error and dup.flags == [edit_match.FLAG_MISMATCH]
    assert "starting at lines 3, 4, 8, 9" in dup.output
    assert (tmp_path / "s.js").read_text(encoding="utf-8") == SOURCE
    ok = await EditFile().run(
        EditFile.Input(path="s.js", old="this.questions.push", new="this.questions.unshift"), ctx
    )
    assert not ok.is_error and ok.flags == []


INDENTED = (
    "class Q {\n"
    "  render() {\n"
    "      legend.addEventListener('click', () => {\n"
    "        const on = legend.getAttribute('contenteditable') === 'true';\n"
    "        legend.setAttribute('contenteditable', String(!on));\n"
    "      });\n"
    "  }\n"
    "}\n"
)
DEEP_OLD = (
    "        legend.addEventListener('click', () => {\n"
    "          const on = legend.getAttribute('contenteditable') === 'true';\n"
)
DEEP_NEW = (
    "        legend.addEventListener('click', (e) => {\n"
    "          e.preventDefault();\n"
    "          const on = legend.getAttribute('contenteditable') === 'true';\n"
)


def test_reindent_shifts_old_and_new_to_the_files_indentation() -> None:
    r = edit_match.reindent_edit(INDENTED, DEEP_OLD, DEEP_NEW)
    assert r is not None and (r.shift, r.start, r.end) == (-2, 3, 4)
    assert r.old == DEEP_OLD.replace("        ", "      ", 1).replace("          ", "        ", 1)
    assert r.new.splitlines()[1] == "        e.preventDefault();"
    shallow = edit_match.reindent_edit(INDENTED, DEEP_OLD.replace("        ", "    "), "    x();\n")
    assert shallow is not None and shallow.shift == 2 and shallow.new == "      x();\n"
    assert "4 spaces" not in edit_match.reindent_note(r) and "2 spaces shallower" in (
        edit_match.reindent_note(r)
    )


def test_reindent_refuses_anything_but_a_single_uniform_space_shift() -> None:
    uneven = "        legend.addEventListener('click', () => {\n      const on = 1;\n"
    assert edit_match.reindent_edit(INDENTED, uneven, "x") is None
    inner = DEEP_OLD.replace("() =>", "()  =>")
    assert edit_match.reindent_edit(INDENTED, inner, "x") is None
    twice = "a {\n  go();\n}\nb {\n  go();\n}\n"
    assert edit_match.reindent_edit(twice, "    go();\n", "    stop();\n") is None
    assert edit_match.reindent_edit(INDENTED, DEEP_OLD.replace("        ", "\t", 1), "x") is None
    assert edit_match.reindent_edit(INDENTED, DEEP_OLD, "\tx();") is None
    assert edit_match.reindent_edit(INDENTED, DEEP_OLD, "x();\n") is None


async def test_edit_file_applies_a_reindented_edit_and_says_so(tmp_path: Path) -> None:
    (tmp_path / "q.js").write_text(INDENTED, encoding="utf-8")
    res = await EditFile().run(
        EditFile.Input(path="q.js", old=DEEP_OLD, new=DEEP_NEW), _ctx(tmp_path)
    )
    assert not res.is_error and res.flags == [edit_match.FLAG_REINDENT]
    assert res.output.startswith("replaced 1 occurrence(s) in q.js\nnote: `old` did not match")
    lines = (tmp_path / "q.js").read_text(encoding="utf-8").splitlines()
    assert lines[2] == "      legend.addEventListener('click', (e) => {"
    assert lines[3] == "        e.preventDefault();"
    assert lines[5] == "        legend.setAttribute('contenteditable', String(!on));"


async def test_edit_file_reindent_can_be_turned_off(tmp_path: Path) -> None:
    (tmp_path / "q.js").write_text(INDENTED, encoding="utf-8")
    jail = WorkspaceJail(tmp_path)
    ctx = ToolContext(
        workspace_root=jail.root,
        jail=jail,
        config=RuntimeConfig(tools=ToolsConfig(edit_reindent=False)),
    )
    res = await EditFile().run(EditFile.Input(path="q.js", old=DEEP_OLD, new=DEEP_NEW), ctx)
    assert res.is_error and res.flags == [edit_match.FLAG_MISMATCH]
    assert "indented with 6 spaces in the file, but with 8 spaces in your `old`" in res.output
    assert (tmp_path / "q.js").read_text(encoding="utf-8") == INDENTED


def test_change_preview_shows_the_reindented_diff(tmp_path: Path) -> None:
    (tmp_path / "q.js").write_text(INDENTED, encoding="utf-8")
    call = ToolUse(
        id="1", name="edit_file", input={"path": "q.js", "old": DEEP_OLD, "new": DEEP_NEW}
    )
    buf = io.StringIO()
    Console(file=buf, width=200, color_system=None).print(change_preview(call, tmp_path))
    out = buf.getvalue()
    assert "e.preventDefault();" in out and "4" in out


def test_failure_streak_on_edit_file_adds_the_reread_hint() -> None:
    state = ReminderState()
    session = new_session(Path("/tmp"), "m")

    def fail(tool: str, n: int) -> list[str]:
        calls = [ToolUseBlock(id=str(n), name=tool, input={})]
        results = [ToolResultBlock(tool_use_id=str(n), tool_name=tool, output="x", is_error=True)]
        return state.collect(calls, results, session)

    assert fail("edit_file", 1) == []
    notes = fail("edit_file", 2)
    assert len(notes) == 1 and notes[0].endswith(EDIT_FILE_STREAK_HINT)
    fail("bash", 3)
    assert EDIT_FILE_STREAK_HINT not in " ".join(fail("bash", 4))


def test_validation_error_names_the_missing_field_and_the_expected_ones() -> None:
    try:
        EditFile.Input.model_validate({"old": "a", "new": "b", "replace_all": True})
    except ValidationError as e:
        msg = describe_validation_error("edit_file", EditFile.Input, e)
    assert msg.splitlines() == [
        "invalid input for tool edit_file: missing required field `path`.",
        "Required fields: path, old, new. Optional: replace_all.",
    ]
    try:
        EditFile.Input.model_validate({"path": "p", "old": "a", "new": "b", "replace_all": "?"})
    except ValidationError as e:
        msg = describe_validation_error("edit_file", EditFile.Input, e)
    assert msg.splitlines()[1].startswith("field `replace_all`: Input should be a valid boolean")
    assert "errors.pydantic.dev" not in msg and "input_value" not in msg


def _runtime(scripts: list[list[AssistantEvent]], tmp_path: Path) -> ConversationRuntime:
    return ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(scripts),  # type: ignore[arg-type]
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )


def _edit(old: str, new: str, *, id: str) -> ToolUse:
    return ToolUse(id=id, name="edit_file", input={"path": "s.js", "old": old, "new": new})


async def test_runtime_counts_mismatches_and_the_model_can_follow_the_hint(tmp_path: Path) -> None:
    (tmp_path / "s.js").write_text(SOURCE, encoding="utf-8")
    rt = _runtime(
        [
            [_edit("this.questions.append(question);", "x", id="e1"), MessageStop()],
            [_edit("this.", "self.", id="e2"), MessageStop()],
            [_edit("    this.questions.push(question);", "    this.list.push(q);", id="e3")]
            + [MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("s.js を直して")
    assert (summary.edit_mismatches, summary.edit_reindents) == (2, 0)
    assert "this.list.push(q);" in (tmp_path / "s.js").read_text(encoding="utf-8")


async def test_metrics_reach_the_report_json_and_the_defect_record(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "s.js").write_text(SOURCE, encoding="utf-8")
    misses: list[list[AssistantEvent]] = [
        [_edit(f"nothing like this {i}", "x", id=f"e{i}"), MessageStop()] for i in range(3)
    ]
    provider = SequencedMockProvider([*misses, [TextDelta(text="s.js failed"), MessageStop()]])
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    task = EvalTask.model_validate(
        {
            "id": "edit",
            "prompt": "s.js を直して",
            "fixture": str(fixture),
            "permission_mode": "workspace-write",
            "checks": {"contains_all": ["s.js"]},
        }
    )
    report = await EvalRunner(agent).run([task])
    result = report.results[0]
    assert result.edit_mismatches == 3
    dumped = json.loads(report.model_dump_json())
    assert EvalReport.model_validate(dumped).results[0].edit_mismatches == 3
    defects = detect_defects(result, "satsuma")
    assert [d.kind for d in defects] == ["edit_file mismatch"]
    assert detect_defects(result.model_copy(update={"edit_mismatches": 2}), "satsuma") == []
    assert TaskResult.model_validate({"task_id": "t", "passed": True}).edit_mismatches == 0


async def test_runtime_counts_reindented_edits(tmp_path: Path) -> None:
    (tmp_path / "q.js").write_text(INDENTED, encoding="utf-8")
    rt = _runtime(
        [
            [
                ToolUse(
                    id="r1",
                    name="edit_file",
                    input={"path": "q.js", "old": DEEP_OLD, "new": DEEP_NEW},
                ),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("q.js を直して")
    assert (summary.edit_mismatches, summary.edit_reindents) == (0, 1)


def test_ab_summary_folds_the_count(tmp_path: Path) -> None:
    before = tmp_path / "b.json"
    before.write_text(
        json.dumps(
            {
                "results": [
                    {"task_id": "t0", "passed": True, "edit_mismatches": 4, "edit_reindents": 1}
                ]
            }
        ),
        encoding="utf-8",
    )
    after = tmp_path / "a.json"
    after.write_text(
        json.dumps({"results": [{"task_id": "t0", "passed": True, "edit_mismatches": 1}]}),
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "mismatch", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| edit_file の不一致 | 4.0 | 1.0 | -3.0 |" in out
    assert "| 字下げを合わせて適用 | 1.0 | 0.0 | -1.0 |" in out


def test_tool_description_is_unchanged() -> None:
    assert EditFile.spec.description == "Replace an exact string in a file within the workspace."
    fields = EditFile.Input.model_fields
    assert fields["old"].description == (
        "Exact text to replace. Must match the file byte for byte, including indentation."
    )


async def test_stale_note_after_own_edits_and_when_unread(tmp_path: Path) -> None:
    (tmp_path / "s.js").write_text(SOURCE, encoding="utf-8")
    jail = WorkspaceJail(tmp_path)
    ctx = ToolContext(
        workspace_root=jail.root, jail=jail, config=RuntimeConfig(), reads=ReadLedger()
    )
    miss = EditFile.Input(path="s.js", old="this.questions.append(question);", new="x")
    unread = await EditFile().run(miss, ctx)
    assert unread.flags == [edit_match.FLAG_MISMATCH, FLAG_STALE]
    assert "no read_file of s.js is recorded in this session" in unread.output
    await ReadFile().run(ReadFile.Input(path="s.js"), ctx)
    fresh = await EditFile().run(miss, ctx)
    assert fresh.flags == [edit_match.FLAG_MISMATCH] and "note:" not in fresh.output
    await EditFile().run(
        EditFile.Input(path="s.js", old="this.render();", new="this.render(true);"), ctx
    )
    after_own = await EditFile().run(miss, ctx)
    assert FLAG_STALE in after_own.flags
    assert "your own 1 edit since then modified it" in after_own.output
    (tmp_path / "s.js").write_text(SOURCE + "// touched\n", encoding="utf-8")
    other = await EditFile().run(miss, ctx)
    assert "was modified since you last read it (not only by your own edits)" in other.output
    await ReadFile().run(ReadFile.Input(path="s.js", offset=1, limit=3), ctx)
    reread = await EditFile().run(miss, ctx)
    assert FLAG_STALE not in reread.flags


def test_ledger_is_injected_by_the_runtime_and_optional_in_tools(tmp_path: Path) -> None:
    rt = _runtime([[TextDelta(text="ok"), MessageStop()]], tmp_path)
    assert isinstance(rt._ctx.reads, ReadLedger)
    assert _ctx(tmp_path).reads is None


async def test_runtime_counts_stale_notes(tmp_path: Path) -> None:
    (tmp_path / "s.js").write_text(SOURCE, encoding="utf-8")
    rt = _runtime(
        [
            [_edit("this.questions.append(question);", "x", id="e1"), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("s.js を直して")
    assert (summary.edit_mismatches, summary.edit_stale_notes) == (1, 1)
