"""新しい派生クラスを書いたとき、基底クラスを直接作る未変更の行を添える。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.config.models import HookConfig, RuntimeConfig, ToolsConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, related_code
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.edit_file import EditFile
from satsumaestro.tools.builtins.write_file import WriteFile
from satsumaestro.tools.read_ledger import ReadLedger
from tests.fixtures.mock_provider import SequencedMockProvider

QUESTION = (
    "export default class Question {\n"
    "  constructor(title, name, preview = false) {\n"
    "    this.title = title;\n"
    "  }\n"
    "}\n"
)
PREVIEW = (
    "import SurveyPreview from './common/SurveyPreview.js';\n"
    "import Question from './common/Question.js';\n"
    "const data = JSON.parse(localStorage.data || '{}');\n"
    "const preview = new SurveyPreview('form', data.title);\n"
    "for (const q of data.questions || []) {\n"
    "  const question = new Question(q.title, q.name, true);\n"
    "  preview.appendQuestion(question);\n"
    "}\n"
)
SUBCLASS = (
    "import Question from './Question.js';\n"
    "export default class SingleSelectionQuestion extends Question {\n"
    "  constructor(title, name, preview, options) {\n"
    "    super(title, name, preview);\n"
    "  }\n"
    "}\n"
)


def _project(root: Path) -> None:
    (root / "common").mkdir()
    (root / "common" / "Question.js").write_text(QUESTION, encoding="utf-8")
    (root / "preview.js").write_text(PREVIEW, encoding="utf-8")
    nm = root / "node_modules" / "lib"
    nm.mkdir(parents=True)
    (nm / "x.js").write_text("const q = new Question('a', 'b');\n", encoding="utf-8")


def _ctx(root: Path, **tools: bool) -> ToolContext:
    jail = WorkspaceJail(root)
    return ToolContext(
        workspace_root=jail.root,
        jail=jail,
        config=RuntimeConfig(tools=ToolsConfig(**tools)),
        reads=ReadLedger(),
    )


def test_new_subclasses_only_returns_pairs_that_appeared() -> None:
    assert related_code.new_subclasses(None, SUBCLASS, ".js") == [
        ("SingleSelectionQuestion", "Question")
    ]
    edited = SUBCLASS.replace("super(title, name, preview);", "super(title, name, true);")
    assert related_code.new_subclasses(SUBCLASS, edited, ".js") == []
    py = "class Rating(Question):\n    pass\n\nclass Meta(Base, metaclass=M):\n    pass\n"
    assert related_code.new_subclasses("", py, ".py") == [("Rating", "Question"), ("Meta", "Base")]
    assert related_code.new_subclasses(None, "class A(object):\n    pass\n", ".py") == []
    assert related_code.new_subclasses(None, SUBCLASS, ".css") == []


def test_construction_sites_skip_written_files_and_node_modules(tmp_path: Path) -> None:
    _project(tmp_path)
    root = tmp_path.resolve()
    sites = related_code.construction_sites(root, "Question", set())
    assert sites is not None
    assert [(s.rel, s.line) for s in sites] == [("preview.js", 6)]
    assert sites[0].text == "const question = new Question(q.title, q.name, true);"
    assert related_code.construction_sites(root, "Question", {root / "preview.js"}) == []


def test_bases_defined_outside_the_workspace_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "a.js").write_text("throw new Error('x');\n", encoding="utf-8")
    assert related_code.construction_sites(tmp_path.resolve(), "Error", set()) == []


def test_large_projects_are_not_scanned(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _project(tmp_path)
    monkeypatch.setattr(related_code, "MAX_SCAN_FILES", 1)
    assert related_code.construction_sites(tmp_path.resolve(), "Question", set()) is None


async def test_write_file_points_at_the_preview_line(tmp_path: Path) -> None:
    _project(tmp_path)
    ctx = _ctx(tmp_path)
    path = "common/SingleSelectionQuestion.js"
    res = await WriteFile().run(WriteFile.Input(path=path, content=SUBCLASS), ctx)
    assert res.flags == [related_code.FLAG_RELATED]
    assert "note: SingleSelectionQuestion extends Question." in res.output
    assert "preview.js:6  const question = new Question(q.title, q.name, true);" in res.output
    again = SUBCLASS + "// more\n"
    res2 = await WriteFile().run(WriteFile.Input(path=path, content=again), ctx)
    assert res2.flags == [] and "note:" not in res2.output


async def test_first_edit_in_a_new_session_also_points_at_it(tmp_path: Path) -> None:
    """再試行は新しい会話。"""
    _project(tmp_path)
    (tmp_path / "common" / "SingleSelectionQuestion.js").write_text(SUBCLASS, encoding="utf-8")
    ctx = _ctx(tmp_path)
    inp = EditFile.Input(
        path="common/SingleSelectionQuestion.js",
        old="super(title, name, preview);",
        new="super(title, name, preview);\n    this.options = options;",
    )
    res = await EditFile().run(inp, ctx)
    assert "preview.js:6" in res.output
    inp2 = EditFile.Input(
        path="common/SingleSelectionQuestion.js", old="this.options = options;", new="this.o = 1;"
    )
    assert "note:" not in (await EditFile().run(inp2, ctx)).output


async def test_edited_construction_sites_are_not_shown(tmp_path: Path) -> None:
    _project(tmp_path)
    ctx = _ctx(tmp_path)
    fixed = PREVIEW.replace("new Question(", "Question.fromJSON(")
    await WriteFile().run(WriteFile.Input(path="preview.js", content=fixed), ctx)
    await WriteFile().run(WriteFile.Input(path="preview.js", content=PREVIEW), ctx)
    res = await WriteFile().run(
        WriteFile.Input(path="common/SingleSelectionQuestion.js", content=SUBCLASS), ctx
    )
    assert "note:" not in res.output


async def test_disabled_by_config(tmp_path: Path) -> None:
    _project(tmp_path)
    ctx = _ctx(tmp_path, related_code_notes=False)
    res = await WriteFile().run(
        WriteFile.Input(path="common/SingleSelectionQuestion.js", content=SUBCLASS), ctx
    )
    assert res.flags == [] and "note:" not in res.output


async def test_runtime_counts_related_code_notes(tmp_path: Path) -> None:
    _project(tmp_path)
    write = ToolUse(
        id="w1",
        name="write_file",
        input={"path": "common/SingleSelectionQuestion.js", "content": SUBCLASS},
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(  # type: ignore[arg-type]
            [[write, MessageStop()], [TextDelta(text="done"), MessageStop()]]
        ),
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    summary = await rt.run_turn("単一選択の質問を足して")
    assert summary.related_code_notes == 1
