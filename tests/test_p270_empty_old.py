"""edit_file が空の old を受け付け、replace_all でファイルを壊していた。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.config.models import HookConfig, RuntimeConfig
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, edit_match
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.edit_file import EditFile
from tests.fixtures.mock_provider import SequencedMockProvider

FILES = {
    "style.css": "body {\n  color: red;\n}\n",
    "notes.md": "# Title\n\nSome text.\n",
    "page.html": "<main>\n  <p>hi</p>\n</main>\n",
    "app.js": "const a = 1;\nexport default a;\n",
}
NEW = "  // Apply required attribute to hidden input based on config\n"


def _ctx(root: Path) -> ToolContext:
    jail = WorkspaceJail(root)
    return ToolContext(workspace_root=jail.root, jail=jail, config=RuntimeConfig())


@pytest.mark.parametrize("name", sorted(FILES))
@pytest.mark.parametrize("replace_all", [False, True])
async def test_empty_old_never_changes_the_file(
    tmp_path: Path, name: str, replace_all: bool
) -> None:
    path = tmp_path / name
    path.write_text(FILES[name], encoding="utf-8")
    inp = EditFile.Input(path=name, old="", new=NEW, replace_all=replace_all)
    res = await EditFile().run(inp, _ctx(tmp_path))
    assert res.is_error
    assert path.read_text(encoding="utf-8") == FILES[name]
    assert res.flags == [edit_match.FLAG_MISMATCH]
    assert res.output.startswith(f"old is empty, so {name} was not changed")
    assert "To insert text" in res.output and "not unique" not in res.output


async def test_empty_old_on_a_missing_file_gets_the_same_answer(tmp_path: Path) -> None:
    res = await EditFile().run(EditFile.Input(path="nope.css", old="", new=NEW), _ctx(tmp_path))
    assert res.is_error and res.output.startswith("old is empty")
    assert not (tmp_path / "nope.css").exists()


async def test_whitespace_old_is_still_a_normal_edit(tmp_path: Path) -> None:
    """空白だけの old は正当な編集になり得る（字下げの一括置換など）。"""
    (tmp_path / "a.md").write_text("a\tb\tc\n", encoding="utf-8")
    inp = EditFile.Input(path="a.md", old="\t", new="  ", replace_all=True)
    res = await EditFile().run(inp, _ctx(tmp_path))
    assert not res.is_error and (tmp_path / "a.md").read_text(encoding="utf-8") == "a  b  c\n"


async def test_runtime_counts_it_as_a_mismatch(tmp_path: Path) -> None:
    (tmp_path / "style.css").write_text(FILES["style.css"], encoding="utf-8")
    call = ToolUse(
        id="e1",
        name="edit_file",
        input={"path": "style.css", "old": "", "new": NEW, "replace_all": True},
    )
    rt = ConversationRuntime(
        session=new_session(tmp_path, "mock-model"),
        provider=SequencedMockProvider(  # type: ignore[arg-type]
            [[call, MessageStop()], [TextDelta(text="done"), MessageStop()]]
        ),
        tools=default_registry(),
        policy=default_policy(PermissionMode.WORKSPACE_WRITE),
        prompt_builder=SystemPromptBuilder(),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), tmp_path),
        config=RuntimeConfig(),
    )
    summary = await rt.run_turn("コメントを足して")
    assert summary.edit_mismatches == 1
    assert (tmp_path / "style.css").read_text(encoding="utf-8") == FILES["style.css"]
