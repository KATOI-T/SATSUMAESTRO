"""編集直後の構文検査（``tools/syntax_check.py``）。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig, ToolsConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse, Usage
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
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, syntax_check
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.append_file import AppendFile
from satsumaestro.tools.builtins.edit_file import EditFile
from satsumaestro.tools.builtins.write_file import WriteFile
from tests.fixtures.mock_provider import SequencedMockProvider

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"
needs_node = pytest.mark.skipif(
    syntax_check.node_path() is None, reason="node が無い環境では JavaScript は検査しない"
)

VALID_PY = "def f(x):\n    return x + 1\n"
BROKEN_PY = "def f(x)\n    return x + 1\n"

TASK19_OLD = "  constructor(title, name, preview = false, options = []) {\n"
TASK19_NEW = (
    "+   * @param {boolean} [isShuffleMode=false] - If true, shuffle options in preview mode.\n"
    "   */\n"
    "-  constructor(title, name, preview = false, options = []) {\n"
    "+  constructor(title, name, preview = false, options = [], isShuffleMode = false) {\n"
)
SINGLE_SELECTION_JS = (
    "import Question from './Question.js';\n\n"
    "export default class SingleSelectionQuestion extends Question {\n"
    "  /**\n   * @param {string} title - Question title.\n   */\n"
    + TASK19_OLD
    + "    super(title, name, preview);\n    this.options = options;\n  }\n}\n"
)
SURVEY_DESIGN_JS = (
    "export default class SurveyDesign {\n"
    "  /**\n"
    "   * @param {string} formSelector - CSS selector for the target form element.\n"
    "   */\n"
    "  constructor(formSelector) {\n    this.form = document.querySelector(formSelector);\n  }\n}\n"
)


def _ctx(tmp_path: Path, **tools: bool) -> ToolContext:
    jail = WorkspaceJail(tmp_path)
    return ToolContext(
        workspace_root=jail.root, jail=jail, config=RuntimeConfig(tools=ToolsConfig(**tools))
    )


def test_python_and_json_checks_report_line_and_message() -> None:
    assert syntax_check.check_syntax(Path("a.py"), VALID_PY) == syntax_check.SyntaxVerdict(ok=True)
    broken = syntax_check.check_syntax(Path("a.py"), BROKEN_PY)
    assert broken is not None and not broken.ok
    assert broken.line == 1 and broken.message.startswith("SyntaxError")
    assert "line 1: SyntaxError" in broken.describe()
    assert syntax_check.check_syntax(
        Path("p.json"), '{"a": [1, 2]}\n'
    ) == syntax_check.SyntaxVerdict(ok=True)
    bad_json = syntax_check.check_syntax(Path("p.json"), '{\n  "a": 1,\n}\n')
    assert bad_json is not None and not bad_json.ok and bad_json.line in (2, 3)
    assert bad_json.message.startswith("JSON: ")
    assert syntax_check.check_syntax(Path("index.html"), "<div>") is None
    assert syntax_check.check_syntax(Path("README.md"), "- broken {{") is None
    assert syntax_check.language_of(Path("x.MJS")) == "javascript"


@needs_node
def test_javascript_check_uses_node(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = syntax_check.check_syntax(Path("a.js"), SURVEY_DESIGN_JS)
    assert ok == syntax_check.SyntaxVerdict(ok=True)
    broken = syntax_check.check_syntax(Path("a.js"), SURVEY_DESIGN_JS.replace("  /**\n", "\n"))
    assert broken is not None and not broken.ok
    assert broken.line == 3 and broken.message.startswith("SyntaxError")
    diffy = syntax_check.check_syntax(
        Path("a.js"), SINGLE_SELECTION_JS.replace(TASK19_OLD, TASK19_NEW)
    )
    assert diffy is not None and not diffy.ok and "Unexpected token" in diffy.message
    monkeypatch.setattr(syntax_check, "node_path", lambda: None)
    assert syntax_check.check_syntax(Path("a.js"), "let = ;") is None


@needs_node
def test_javascript_js_is_checked_as_cjs_then_esm(monkeypatch: pytest.MonkeyPatch) -> None:
    suffixes: list[str] = []
    real = syntax_check._node_check

    def spy(node: str, text: str, suffix: str) -> syntax_check.SyntaxVerdict | None:
        suffixes.append(suffix)
        return real(node, text, suffix)

    monkeypatch.setattr(syntax_check, "_node_check", spy)
    cjs = "const a = require('a');\nmodule.exports = { a };\n"
    assert syntax_check.check_syntax(Path("a.js"), cjs) == syntax_check.SyntaxVerdict(ok=True)
    assert suffixes == [".cjs"]
    suffixes.clear()
    broken_esm = SURVEY_DESIGN_JS.replace("  /**\n", "\n")
    verdict = syntax_check.check_syntax(Path("a.js"), broken_esm)
    assert suffixes == [".cjs", ".mjs"]
    assert verdict is not None and not verdict.ok and verdict.line == 3
    broken_cjs = syntax_check.check_syntax(Path("a.js"), "const a = 1;\nmodule.exports = {\n")
    assert broken_cjs is not None and not broken_cjs.ok and broken_cjs.line is not None
    suffixes.clear()
    assert syntax_check.check_syntax(Path("a.mjs"), "export {};\n") == syntax_check.SyntaxVerdict(
        ok=True
    )
    assert suffixes == [".mjs"]


def test_check_edit_reads_before_only_when_after_is_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real = syntax_check.check_syntax

    def spy(path: Path, text: str) -> syntax_check.SyntaxVerdict | None:
        calls.append(text)
        return real(path, text)

    monkeypatch.setattr(syntax_check, "check_syntax", spy)
    fine = syntax_check.check_edit(Path("a.py"), BROKEN_PY, VALID_PY)
    assert fine.before is None and fine.after is not None and fine.after.ok
    assert calls == [VALID_PY]
    broke = syntax_check.check_edit(Path("a.py"), VALID_PY, BROKEN_PY)
    assert broke.after_broken and broke.before_ok and not broke.before_known_broken
    still = syntax_check.check_edit(Path("a.py"), BROKEN_PY, BROKEN_PY)
    assert still.after_broken and still.before_known_broken
    new = syntax_check.check_edit(Path("a.py"), None, BROKEN_PY)
    assert new.after_broken and not new.before_ok and not new.before_known_broken


def test_looks_like_patch_and_its_exclusions() -> None:
    assert "paired" in (syntax_check.looks_like_patch(TASK19_NEW) or "")
    assert "conflict" in (
        syntax_check.looks_like_patch("<<<<<<< HEAD\nx\n=======\ny\n>>>>>>> b\n") or ""
    )
    assert "headers" in (syntax_check.looks_like_patch("@@ -1,3 +1,4 @@\n x\n+y\n") or "")
    assert "headers" in (syntax_check.looks_like_patch("--- a/x.js\n+++ b/x.js\n") or "")
    assert syntax_check.looks_like_patch("- first\n- second\n") is None
    assert syntax_check.looks_like_patch('const s = "a"\n+ "b"\n+ "c";\n') is None
    assert syntax_check.looks_like_patch("x = [\n-1,\n-2,\n]\n") is None
    assert syntax_check.looks_like_patch("Title\n=======\n") is None
    assert syntax_check.looks_like_patch("") is None
    js = Path("a.js")
    assert syntax_check.patch_rejection(js, TASK19_NEW) is not None
    assert syntax_check.patch_rejection(Path("README.md"), TASK19_NEW) is None
    assert syntax_check.patch_rejection(Path("fix.diff"), TASK19_NEW) is None
    assert syntax_check.patch_rejection(js, TASK19_NEW, old="-  a()\n+  b()\n") is None


async def test_edit_file_rejects_an_edit_that_breaks_a_valid_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(VALID_PY, encoding="utf-8")
    res = await EditFile().run(
        EditFile.Input(path="a.py", old="return x + 1", new="return (x + 1"), _ctx(tmp_path)
    )
    assert res.is_error and res.flags == [syntax_check.FLAG_REJECT]
    assert res.output.startswith(
        "edit rejected: it would break the syntax of a.py (line 2: SyntaxError"
    )
    assert "left unchanged" in res.output and "do not paste a diff" in res.output
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == VALID_PY


async def test_edit_file_warn_mode_applies_the_edit_and_flags_it(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(VALID_PY, encoding="utf-8")
    res = await EditFile().run(
        EditFile.Input(path="a.py", old="return x + 1", new="return (x + 1"),
        _ctx(tmp_path, syntax_reject_edits=False),
    )
    assert not res.is_error and res.flags == [syntax_check.FLAG_WARNING]
    assert res.output.startswith("replaced 1 occurrence(s) in a.py\nwarning: a.py no longer parses")
    assert "(x + 1" in (tmp_path / "a.py").read_text(encoding="utf-8")


async def test_edit_file_leaves_an_already_broken_file_alone(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(BROKEN_PY, encoding="utf-8")
    ctx = _ctx(tmp_path)
    still = await EditFile().run(EditFile.Input(path="a.py", old="x + 1", new="x + 2"), ctx)
    assert not still.is_error and still.flags == []
    assert (
        still.output == "replaced 1 occurrence(s) in a.py\n"
        "note: a.py still does not parse (line 1: SyntaxError: expected ':')."
    )
    fixed = await EditFile().run(
        EditFile.Input(path="a.py", old="def f(x)\n", new="def f(x):\n"), ctx
    )
    assert fixed.output == "replaced 1 occurrence(s) in a.py" and fixed.flags == []
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "def f(x):\n    return x + 2\n"


async def test_edit_file_check_can_be_turned_off(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(VALID_PY, encoding="utf-8")
    res = await EditFile().run(
        EditFile.Input(path="a.py", old="return x + 1", new="return (x + 1"),
        _ctx(tmp_path, syntax_check=False),
    )
    assert res.output == "replaced 1 occurrence(s) in a.py" and res.flags == []


async def test_edit_file_rejects_patch_shaped_new_without_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(syntax_check, "node_path", lambda: None)
    path = tmp_path / "common" / "SingleSelectionQuestion.js"
    path.parent.mkdir()
    path.write_text(SINGLE_SELECTION_JS, encoding="utf-8")
    res = await EditFile().run(
        EditFile.Input(path="common/SingleSelectionQuestion.js", old=TASK19_OLD, new=TASK19_NEW),
        _ctx(tmp_path),
    )
    assert res.is_error and res.flags == [syntax_check.FLAG_REJECT]
    assert res.output.startswith(
        "edit rejected: `new` looks like a patch (it has paired '-' and '+' lines"
    )
    assert "not a diff" in res.output
    assert path.read_text(encoding="utf-8") == SINGLE_SELECTION_JS


@needs_node
async def test_edit_file_replays_task20_and_passes_a_sound_js_edit(tmp_path: Path) -> None:
    path = tmp_path / "common" / "SurveyDesign.js"
    path.parent.mkdir()
    path.write_text(SURVEY_DESIGN_JS, encoding="utf-8")
    ctx = _ctx(tmp_path)
    res = await EditFile().run(
        EditFile.Input(path="common/SurveyDesign.js", old="  /**\n", new="\n"), ctx
    )
    assert res.is_error and "line 3: SyntaxError" in res.output
    assert path.read_text(encoding="utf-8") == SURVEY_DESIGN_JS
    ok = await EditFile().run(
        EditFile.Input(
            path="common/SurveyDesign.js",
            old="this.form = document.querySelector(formSelector);",
            new="this.form = document.querySelector(formSelector);\n    this.questions = [];",
        ),
        ctx,
    )
    assert ok.output == "replaced 1 occurrence(s) in common/SurveyDesign.js" and ok.flags == []


async def test_edit_file_json_respects_the_before_state(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        '{\n  "name": "a",\n  "version": "1"\n}\n', encoding="utf-8"
    )
    (tmp_path / "tsconfig.json").write_text(
        '{\n  // comment\n  "strict": true\n}\n', encoding="utf-8"
    )
    ctx = _ctx(tmp_path)
    res = await EditFile().run(
        EditFile.Input(path="package.json", old='"version": "1"', new='"version": "1",'), ctx
    )
    assert res.is_error and res.output.startswith("edit rejected: it would break the syntax")
    assert ": JSON: " in res.output
    res = await EditFile().run(
        EditFile.Input(path="tsconfig.json", old='"strict": true', new='"strict": false'), ctx
    )
    assert not res.is_error and "still does not parse" in res.output


async def test_write_file_notes_unparsable_content_and_flags_only_a_broken_valid_file(
    tmp_path: Path,
) -> None:
    ctx = _ctx(tmp_path)
    first = await WriteFile().run(WriteFile.Input(path="a.py", content=BROKEN_PY), ctx)
    assert not first.is_error and first.flags == []
    assert first.output.startswith(
        f"wrote {len(BROKEN_PY)} bytes to a.py\nwarning: a.py does not parse (line 1"
    )
    assert "continue with append_file" in first.output
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == BROKEN_PY
    await WriteFile().run(WriteFile.Input(path="a.py", content=VALID_PY), ctx)
    broke = await WriteFile().run(WriteFile.Input(path="a.py", content=BROKEN_PY), ctx)
    assert broke.flags == [syntax_check.FLAG_WARNING]
    plain = await WriteFile().run(WriteFile.Input(path="a.py", content=VALID_PY), ctx)
    replaced = f" (replaced 2 lines, {len(BROKEN_PY.encode())} bytes)"
    assert plain.output == f"wrote {len(VALID_PY)} bytes to a.py{replaced}" and plain.flags == []


async def test_append_file_notes_pieces_until_the_file_is_complete(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    head = await WriteFile().run(WriteFile.Input(path="m.py", content="def f(x):\n"), ctx)
    assert "does not parse" in head.output and head.flags == []
    mid = await AppendFile().run(AppendFile.Input(path="m.py", content="    y = (x +\n"), ctx)
    assert mid.output.startswith(
        "appended 13 bytes to m.py (now 23 bytes)\nnote: m.py does not parse yet"
    )
    assert mid.flags == []
    done = await AppendFile().run(
        AppendFile.Input(path="m.py", content="         1)\n    return y\n"), ctx
    )
    assert done.output == "appended 25 bytes to m.py (now 48 bytes)" and done.flags == []
    bad = await AppendFile().run(AppendFile.Input(path="m.py", content="def g(:\n"), ctx)
    assert bad.flags == [syntax_check.FLAG_WARNING] and "no longer parses" in bad.output
    assert (tmp_path / "m.py").read_text(encoding="utf-8").endswith("def g(:\n")


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


def _edit(path: str, old: str, new: str, *, id: str = "e1") -> ToolUse:
    return ToolUse(id=id, name="edit_file", input={"path": path, "old": old, "new": new})


async def test_runtime_counts_rejects_and_warnings_into_the_summary(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(VALID_PY, encoding="utf-8")
    rt = _runtime(
        [
            [_edit("a.py", "return x + 1", "return (x + 1"), Usage(output_tokens=5), MessageStop()],
            [
                ToolUse(id="w1", name="write_file", input={"path": "a.py", "content": BROKEN_PY}),
                MessageStop(),
            ],
            [TextDelta(text="done"), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("a.py を直して")
    assert (summary.syntax_rejects, summary.syntax_warnings) == (1, 1)
    blocks = [b for m in rt.session.messages if m.role == "tool" for b in m.blocks]
    results = [b for b in blocks if getattr(b, "type", "") == "tool_result"]
    assert results[0].is_error and "edit rejected" in results[0].output
    assert not results[1].is_error and "warning: a.py does not parse" in results[1].output
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == BROKEN_PY


async def test_metrics_reach_the_report_json_and_the_defect_record(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "a.py").write_text(VALID_PY, encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [_edit("a.py", "return x + 1", "return (x + 1"), MessageStop()],
            [TextDelta(text="a.py は直せませんでした"), MessageStop()],
        ]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    task = EvalTask.model_validate(
        {
            "id": "edit",
            "prompt": "a.py を直して",
            "fixture": str(fixture),
            "permission_mode": "workspace-write",
            "checks": {"contains_all": ["a.py"]},
        }
    )
    report = await EvalRunner(agent).run([task])
    result = report.results[0]
    assert (result.syntax_rejects, result.syntax_warnings) == (1, 0)
    dumped = json.loads(report.model_dump_json())
    assert dumped["results"][0]["syntax_rejects"] == 1
    assert EvalReport.model_validate(dumped).results[0].syntax_rejects == 1
    defects = detect_defects(result, "satsuma")
    assert [d.kind for d in defects] == ["syntax-breaking edit"]
    assert "(sent back: 1, broken writes: 0)" in defects[0].detail


def test_old_report_without_the_keys_still_loads() -> None:
    old = TaskResult.model_validate({"task_id": "t", "passed": True})
    assert (old.syntax_rejects, old.syntax_warnings) == (0, 0)
    assert detect_defects(old, "satsuma") == []


def test_ab_summary_folds_the_counts(tmp_path: Path) -> None:
    before = tmp_path / "b.json"
    before.write_text(
        json.dumps(
            {
                "agent": "satsuma",
                "results": [
                    {"task_id": "t0", "passed": True, "syntax_rejects": 2, "syntax_warnings": 1},
                    {"task_id": "t1", "passed": True, "syntax_rejects": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    after = tmp_path / "a.json"
    after.write_text(json.dumps({"results": [{"task_id": "t0", "passed": True}]}), encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "syntax", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 構文を壊す編集の差し戻し | 3.0 | 0.0 | -3.0 |" in out
    assert "| 通っていたファイルを壊した書き込み | 1.0 | 0.0 | -1.0 |" in out


def test_defaults_and_catalog_sentence() -> None:
    cfg = RuntimeConfig().tools
    assert cfg.syntax_check is True and cfg.syntax_reject_edits is True
    assert EditFile.spec.description == "Replace an exact string in a file within the workspace."
    assert EditFile.Input.model_fields["new"].description == "Replacement text."
