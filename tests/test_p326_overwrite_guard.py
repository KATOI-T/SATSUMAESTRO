"""この会話で書いたファイルを途中の部分で置き換える write_file の差し戻し。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from satsumaestro.config.models import HookConfig, RuntimeConfig, ToolsConfig
from satsumaestro.engine.events import AssistantEvent, MessageStop, TextDelta, ToolUse
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.evals.agents import SatsumaAgent
from satsumaestro.evals.models import EvalReport, EvalTask, TaskResult
from satsumaestro.evals.runner import EvalRunner
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import default_policy
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, overwrite_guard
from satsumaestro.tools.base import ToolContext
from satsumaestro.tools.builtins.append_file import AppendFile
from satsumaestro.tools.builtins.read_file import ReadFile
from satsumaestro.tools.builtins.write_file import WriteFile
from satsumaestro.tools.read_ledger import ReadLedger
from tests.fixtures.mock_provider import SequencedMockProvider

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_ab_summary.py"

SKELETON = '<!DOCTYPE html>\n<html lang="ja">\n<head>\n    <meta charset="UTF-8">\n'
PIECES = (
    '    <meta name="viewport" content="width=device-width">\n',
    "</head>\n",
    "<body>\n",
)
CSS = (
    "    <style>\n"
    + "".join(f"        .c{i} {{ color: #{i:03x}; }}\n" for i in range(236))
    + "    </style>\n"
)
INDEX = "dashboards/index.html"


def _ctx(root: Path, **tools: bool) -> ToolContext:
    """本番と同じく ReadLedger を注入する（conftest の tool_ctx は注入しないので門は無効）。"""
    return ToolContext(
        workspace_root=root,
        jail=WorkspaceJail(root),
        config=RuntimeConfig(tools=ToolsConfig(**tools)),
        reads=ReadLedger(),
    )


async def _write(ctx: ToolContext, path: str, content: str):  # type: ignore[no-untyped-def]
    return await WriteFile().run(WriteFile.Input(path=path, content=content), ctx)


async def _append(ctx: ToolContext, path: str, content: str):  # type: ignore[no-untyped-def]
    return await AppendFile().run(AppendFile.Input(path=path, content=content), ctx)


def test_fragment_rules() -> None:
    doc = "<!DOCTYPE html>\n<html>\n<body>old</body>\n</html>\n"
    assert overwrite_guard.is_fragment(doc, "    <style>\nbody {}\n    </style>\n")
    assert overwrite_guard.is_fragment(doc, "<body>\n  <main></main>\n</body>\n")
    assert not overwrite_guard.is_fragment(
        doc, "<!DOCTYPE html>\n<html>\n<body>new</body>\n</html>\n"
    )
    assert not overwrite_guard.is_fragment(doc, "<!DOCTYPE html>\n<html>\n<head>\n")
    assert not overwrite_guard.is_fragment("a\nb\n", "a\nb\nc\n")
    assert not overwrite_guard.is_fragment("a\nb\n", "x\na\nb\n")
    assert not overwrite_guard.is_fragment("", "x\n")
    assert not overwrite_guard.is_fragment("\n  \n", "x\n")
    assert not overwrite_guard.is_fragment(
        "\n\n<!DOCTYPE html>\n<p>a</p>\n", "  \n<!DOCTYPE html>\n<p>b</p>\n"
    )
    assert overwrite_guard.is_fragment("\n<!DOCTYPE html>\n<p>a</p>\n", "\n<p>b</p>\n")
    assert overwrite_guard.first_line("") == ""
    assert overwrite_guard.first_line("\n  x  \ny\n") == "x"


def test_messages_name_sizes_tools_and_the_first_line() -> None:
    current = SKELETON + "".join(PIECES)
    tools = ["write_file", "append_file", "append_file"]
    msg = overwrite_guard.reject_message(INDEX, current, tools)
    first, second, rest = msg.split("\n", 2)
    assert first == f"write_file rejected: would delete 7 lines of {INDEX}"
    assert len(first) <= 80
    assert second == (
        f"{INDEX} already has 7 lines ({len(current)} bytes) from this conversation "
        "(write_file + 2 append_file)."
    )
    assert "(`<!DOCTYPE html>`)" in rest and f"{INDEX} was left unchanged" in rest
    assert "call append_file with the same content" in rest
    assert (
        "use edit_file" in rest and "call write_file again with the same path and the same" in rest
    )
    assert "from this conversation.\n" in overwrite_guard.reject_message("a.txt", "a\n", [])
    assert overwrite_guard.replaced_note("a\nb\n") == " (replaced 2 lines, 4 bytes)"
    assert overwrite_guard.replaced_note("あ\n") == " (replaced 1 lines, 2 bytes)"
    assert overwrite_guard.replaced_note("") == ""


async def test_replays_the_incident_and_rejects_the_css_part(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert not (await _write(ctx, INDEX, SKELETON)).is_error
    for piece in PIECES:
        assert not (await _append(ctx, INDEX, piece)).is_error
    current = SKELETON + "".join(PIECES)
    await ReadFile().run(ReadFile.Input(path=INDEX), ctx)
    res = await _write(ctx, INDEX, CSS)
    assert res.is_error and res.flags == [overwrite_guard.FLAG_REJECT]
    assert res.output.startswith(f"write_file rejected: would delete 7 lines of {INDEX}\n")
    assert f"{INDEX} already has 7 lines (" in res.output
    assert "(write_file + 3 append_file)" in res.output
    assert (tmp_path / INDEX).read_text(encoding="utf-8") == current
    res = await _append(ctx, INDEX, CSS)
    assert not res.is_error and res.flags == []
    assert (tmp_path / INDEX).read_text(encoding="utf-8") == current + CSS
    assert len((tmp_path / INDEX).read_text(encoding="utf-8").splitlines()) == 7 + 238
    res = await _write(ctx, INDEX, "<body>\n<main></main>\n</body>\n")
    assert res.is_error and res.flags == [overwrite_guard.FLAG_REJECT]
    assert "already has 245 lines" in res.output


async def test_second_write_file_to_the_same_path_is_written(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    await _write(ctx, "a.md", "# Title\n\nbody\n")
    await _append(ctx, "a.md", "more\n")
    before = "# Title\n\nbody\nmore\n"
    section = "## Section\n\ntext\n"
    rejected = await _write(ctx, "a.md", section)
    assert rejected.is_error and rejected.flags == [overwrite_guard.FLAG_REJECT]
    assert ctx.reads is not None
    assert ctx.reads.overwrite_pending == {tmp_path / "a.md": "## Section"}
    written = await _write(ctx, "a.md", section)
    assert not written.is_error and written.flags == [overwrite_guard.FLAG_OVERRIDE]
    assert (
        written.output
        == f"wrote {len(section)} bytes to a.md (replaced 4 lines, {len(before)} bytes)"
    )
    assert (tmp_path / "a.md").read_text(encoding="utf-8") == section
    assert ctx.reads.overwrite_pending == {} and ctx.reads.overwrite_rejections == {}
    third = await _write(ctx, "a.md", "- item\n")
    assert third.is_error and third.flags == [overwrite_guard.FLAG_REJECT]


async def test_a_repeat_with_different_content_is_rejected_again_until_the_limit(
    tmp_path: Path,
) -> None:
    """実機（qwen）: 違う内容で繰り返した 1 回（既存の CSS が消えた）を止める。"""
    ctx = _ctx(tmp_path)
    await _write(ctx, "styles.css", "/* CSS Variables */\n:root { --a: 1; }\n")
    await _append(ctx, "styles.css", ".card { color: red; }\n")
    first = await _write(ctx, "styles.css", "/* Dashboard Styles - Modern Design */\n.x {}\n")
    assert first.is_error and first.flags == [overwrite_guard.FLAG_REJECT]
    assert "different content is checked again" in first.output
    second = await _write(ctx, "styles.css", "/* Modern Dashboard Styles */\n.y {}\n")
    assert second.is_error and second.flags == [overwrite_guard.FLAG_REJECT]
    assert (tmp_path / "styles.css").read_text(encoding="utf-8") == (
        "/* CSS Variables */\n:root { --a: 1; }\n.card { color: red; }\n"
    )
    assert ctx.reads is not None
    assert ctx.reads.overwrite_rejections == {tmp_path / "styles.css": 2}
    third = await _write(ctx, "styles.css", "/* Yet another */\n.z {}\n")
    assert not third.is_error and third.flags == [overwrite_guard.FLAG_OVERRIDE]
    assert (tmp_path / "styles.css").read_text(encoding="utf-8") == "/* Yet another */\n.z {}\n"
    assert ctx.reads.overwrite_rejections == {}
    await _append(ctx, "styles.css", ".w {}\n")
    rewrite = await _write(ctx, "styles.css", "/* Yet another */\n.z { top: 0; }\n")
    assert not rewrite.is_error and rewrite.flags == []


async def test_rewrites_from_the_top_and_foreign_files_pass_with_a_note(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    v1 = "<!DOCTYPE html>\n<p>a</p>\n"
    first = await _write(ctx, "page.html", v1)
    assert first.output == f"wrote {len(v1)} bytes to page.html" and first.flags == []
    v2 = "<!DOCTYPE html>\n<p>b</p>\n<p>c</p>\n"
    again = await _write(ctx, "page.html", v2)
    assert not again.is_error and again.flags == []
    assert again.output == f"wrote {len(v2)} bytes to page.html (replaced 2 lines, {len(v1)} bytes)"
    grown = await _write(ctx, "page.html", "<!DOCTYPE html>\n<p>b</p>\n<p>c</p>\n<p>d</p>\n")
    assert not grown.is_error and grown.flags == []
    theirs = "/* theirs */\nbody {}\n"
    (tmp_path / "theirs.css").write_text(theirs, encoding="utf-8")
    foreign = await _write(ctx, "theirs.css", ".card {}\n")
    assert not foreign.is_error and foreign.flags == []
    assert foreign.output == f"wrote 9 bytes to theirs.css (replaced 2 lines, {len(theirs)} bytes)"
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    assert (await _write(ctx, "empty.txt", "x\n")).output == "wrote 2 bytes to empty.txt"


async def test_plan_file_is_never_guarded(tmp_path: Path) -> None:
    plan = ".satsuma/plans/0123456789abcdef0123456789abcdef.md"
    ctx = ToolContext(
        workspace_root=tmp_path,
        jail=WorkspaceJail(tmp_path, visible_state=[tmp_path / plan]),
        config=RuntimeConfig(),
        reads=ReadLedger(),
    )
    assert not (await _write(ctx, plan, "# 計画 A\n\n## Steps\n1. a (~5 lines)\n")).is_error
    rewritten = await _write(ctx, plan, "## Steps\n1. a (~5 lines)\n2. b (~5 lines)\n")
    assert not rewritten.is_error and rewritten.flags == []
    assert (await _write(ctx, plan, "# 計画 B\n")).flags == []


async def test_guard_can_be_turned_off(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, overwrite_guard=False)
    await _write(ctx, INDEX, SKELETON)
    await _append(ctx, INDEX, PIECES[0])
    res = await _write(ctx, INDEX, CSS)
    assert not res.is_error and res.flags == []
    assert res.output.startswith(f"wrote {len(CSS)} bytes to {INDEX} (replaced 5 lines, ")
    assert (tmp_path / INDEX).read_text(encoding="utf-8") == CSS


async def test_without_a_ledger_nothing_changes(tmp_path: Path) -> None:
    ctx = ToolContext(workspace_root=tmp_path, jail=WorkspaceJail(tmp_path), config=RuntimeConfig())
    await _write(ctx, "a.txt", "a\nb\n")
    res = await _write(ctx, "a.txt", "c\n")
    assert not res.is_error and res.output == "wrote 2 bytes to a.txt (replaced 2 lines, 4 bytes)"


async def test_read_file_past_the_end_reports_the_line_count(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    lines = "".join(f"row {i}\n" for i in range(1, 227))
    (tmp_path / "f.txt").write_text(lines, encoding="utf-8")
    past = await ReadFile().run(ReadFile.Input(path="f.txt", offset=330, limit=20), ctx)
    assert not past.is_error
    assert (
        past.output
        == "read_file: f.txt has 226 lines; nothing to read from line 330. Use offset 1-226."
    )
    just_past = await ReadFile().run(ReadFile.Input(path="f.txt", offset=227), ctx)
    assert just_past.output.startswith(
        "read_file: f.txt has 226 lines; nothing to read from line 227."
    )
    tail = await ReadFile().run(ReadFile.Input(path="f.txt", offset=226, limit=20), ctx)
    assert tail.output == "row 226\n"
    head = await ReadFile().run(ReadFile.Input(path="f.txt", offset=1, limit=2), ctx)
    assert head.output == "row 1\nrow 2\n"
    (tmp_path / "e.txt").write_text("", encoding="utf-8")
    assert (await ReadFile().run(ReadFile.Input(path="e.txt", offset=5), ctx)).output == (
        "read_file: e.txt is empty (0 lines)."
    )
    assert (await ReadFile().run(ReadFile.Input(path="e.txt", limit=3), ctx)).output == ""


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


def _call(id: str, name: str, path: str, content: str) -> ToolUse:
    return ToolUse(id=id, name=name, input={"path": path, "content": content})


_INCIDENT_SCRIPTS: list[list[AssistantEvent]] = [
    [_call("w1", "write_file", "a.html", "<!DOCTYPE html>\n<html>\n"), MessageStop()],
    [_call("a1", "append_file", "a.html", "<head></head>\n"), MessageStop()],
    [_call("w2", "write_file", "a.html", "<body>\n</body>\n"), MessageStop()],
    [_call("a2", "append_file", "a.html", "<body>\n</body>\n"), MessageStop()],
    [TextDelta(text="done"), MessageStop()],
]


async def test_runtime_counts_rejects_into_the_summary(tmp_path: Path) -> None:
    rt = _runtime(_INCIDENT_SCRIPTS, tmp_path)
    summary = await rt.run_turn("a.html を作って")
    assert (summary.overwrite_rejects, summary.overwrite_overrides) == (1, 0)
    blocks = [b for m in rt.session.messages if m.role == "tool" for b in m.blocks]
    results = [b for b in blocks if getattr(b, "type", "") == "tool_result"]
    assert [r.is_error for r in results] == [False, False, True, False]
    rejected = next(r for r in results if r.is_error)
    assert rejected.output.startswith("write_file rejected: would delete 3 lines of a.html\n")
    assert (tmp_path / "a.html").read_text(encoding="utf-8") == (
        "<!DOCTYPE html>\n<html>\n<head></head>\n<body>\n</body>\n"
    )


async def test_override_is_counted(tmp_path: Path) -> None:
    rt = _runtime(
        [
            [_call("w1", "write_file", "a.html", "<!DOCTYPE html>\n"), MessageStop()],
            [_call("a1", "append_file", "a.html", "<html>\n"), MessageStop()],
            [_call("w2", "write_file", "a.html", "<body>\n"), MessageStop()],
            [_call("w3", "write_file", "a.html", "<body>\n"), MessageStop()],
            [TextDelta(text="done"), MessageStop()],
        ],
        tmp_path,
    )
    summary = await rt.run_turn("a.html を作って")
    assert (summary.overwrite_rejects, summary.overwrite_overrides) == (1, 1)
    assert (tmp_path / "a.html").read_text(encoding="utf-8") == "<body>\n"


async def test_metrics_reach_the_report_json_and_the_tracer_keys(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    provider = SequencedMockProvider(
        [
            [_call("w1", "write_file", "a.html", "<!DOCTYPE html>\n"), MessageStop()],
            [_call("a1", "append_file", "a.html", "<html>\n"), MessageStop()],
            [_call("w2", "write_file", "a.html", "<body>\n"), MessageStop()],
            [TextDelta(text="a.html を作りました"), MessageStop()],
        ]
    )
    agent = SatsumaAgent(provider, RuntimeConfig(), "mock-model")  # type: ignore[arg-type]
    task = EvalTask.model_validate(
        {
            "id": "html",
            "prompt": "a.html を作って",
            "fixture": str(fixture),
            "permission_mode": "workspace-write",
            "checks": {"contains_all": ["a.html"]},
        }
    )
    report = await EvalRunner(agent).run([task])
    result = report.results[0]
    assert (result.overwrite_rejects, result.overwrite_overrides) == (1, 0)
    dumped = json.loads(report.model_dump_json())
    assert dumped["results"][0]["overwrite_rejects"] == 1
    assert EvalReport.model_validate(dumped).results[0].overwrite_rejects == 1
    chunks = [(c.tool, c.ok) for c in result.write_chunks]
    assert chunks == [("write_file", True), ("append_file", True), ("write_file", False)]


def test_old_report_without_the_keys_still_loads() -> None:
    old = TaskResult.model_validate({"task_id": "t", "passed": True})
    assert (old.overwrite_rejects, old.overwrite_overrides) == (0, 0)


def test_ab_summary_folds_the_counts(tmp_path: Path) -> None:
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
                    {
                        "task_id": "t0",
                        "passed": True,
                        "overwrite_rejects": 2,
                        "overwrite_overrides": 1,
                    },
                    {"task_id": "t1", "passed": True, "overwrite_rejects": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    out = subprocess.run(
        [sys.executable, str(_SCRIPT), "--title", "ow", "--before", str(before)]
        + ["--after", str(after)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "| 途中の部分での置き換えの差し戻し | 0.0 | 3.0 | +3.0 |" in out
    assert "| 差し戻し後の繰り返しで書いた | 0.0 | 1.0 | +1.0 |" in out


def test_defaults() -> None:
    assert ToolsConfig().overwrite_guard is True
    assert overwrite_guard.FLAG_REJECT == "overwrite-reject"
    assert overwrite_guard.FLAG_OVERRIDE == "overwrite-override"
