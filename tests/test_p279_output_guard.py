"""頼まれていない成果物の削除と、エージェントの状態ディレクトリの削除を止める。"""

from __future__ import annotations

from pathlib import Path

import pytest

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.turn import PermissionOutcome
from satsumaestro.security.deletion_guard import (
    PROHIBITED_DELETION_PREFIX,
    bash_targets,
    check_bash,
    check_path,
)
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.output_guard import (
    UNREQUESTED_DELETION_PREFIX,
    SessionOutputs,
    requests_deletion,
)
from satsumaestro.session.models import ToolResultBlock
from tests.test_p145_plan_exit_nudge import _runtime


class _Capture:
    """確認カードに回った呼び出しと警告を記録して、決めた答えを返すユーザー。"""

    def __init__(self, decision: str = "allow") -> None:
        self.asked: list[tuple[str, str | None]] = []
        self._decision = decision

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.asked.append((call.name, warning))
        return PermissionOutcome(decision=self._decision, reason="denied by user")  # type: ignore[arg-type]


def _write(call_id: str, path: str, content: str = "<html>\n</html>\n") -> list[object]:
    return [
        ToolUse(id=call_id, name="write_file", input={"path": path, "content": content}),
        MessageStop(),
    ]


def _bash(call_id: str, command: str) -> list[object]:
    return [ToolUse(id=call_id, name="bash", input={"command": command}), MessageStop()]


def _delete(call_id: str, path: str) -> list[object]:
    return [
        ToolUse(id=call_id, name="delete_path", input={"path": path, "recursive": True}),
        MessageStop(),
    ]


def _outputs(rt: object) -> dict[str, str]:
    return {
        b.tool_use_id: b.output
        for m in rt.session.messages  # type: ignore[attr-defined]
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }


@pytest.mark.parametrize(
    "text",
    [
        "作成したファイルを削除してください",
        "dashboards を消して",
        "片付けておいて",
        "初期状態に戻して",
        "please delete the files",
        "clean up the build output",
        "rm -rf dist してよい",
    ],
)
def test_deletion_requests(text: str) -> None:
    assert requests_deletion([text])


@pytest.mark.parametrize(
    "text",
    ["続けて", "最先端のダッシュボードのデザインをhtmlとcssで作成して", "はい", "テストを実行して"],
)
def test_not_deletion_requests(text: str) -> None:
    assert not requests_deletion([text])


def test_affected_outputs_include_files_under_a_deleted_directory(tmp_path: Path) -> None:
    outputs = SessionOutputs()
    page = tmp_path / "dashboards" / "index.html"
    outputs.add(page)
    assert outputs.affected([tmp_path / "dashboards"]) == [page]
    assert outputs.affected([page]) == [page]
    assert outputs.affected([tmp_path / "other"]) == []
    outputs.forget([tmp_path / "dashboards"])
    assert len(outputs) == 0


async def test_unrequested_rm_of_session_output_is_blocked_without_a_card(tmp_path: Path) -> None:
    """実機の再現: 作った後の「続けて」で rm -rf。"""
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        _write("w1", "dashboards/index.html"),
        [TextDelta(text="作成しました。"), MessageStop()],
    ]
    user = _Capture()
    await rt.run_turn("ダッシュボードを作成して", user)  # type: ignore[arg-type]
    script += [
        _bash("r1", f"cd {tmp_path} && rm -rf dashboards"),
        [TextDelta(text="確認しました。"), MessageStop()],
    ]
    summary = await rt.run_turn("続けて", user)  # type: ignore[arg-type]

    assert user.asked == []
    output = _outputs(rt)["r1"]
    assert output.startswith(f"{UNREQUESTED_DELETION_PREFIX} dashboards/index.html")
    assert "Keep them" in output
    assert summary.output_delete_blocks == 1
    assert (tmp_path / "dashboards" / "index.html").exists()


async def test_blocked_even_when_the_mode_would_not_ask(tmp_path: Path) -> None:
    """danger-full-access でカードが出ない設定でも、頼まれていなければ止める。"""
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code", sub="danger-full-access")
    script[:] = [
        _write("w1", "report.md", "# report\n"),
        _delete("d1", "report.md"),
        [TextDelta(text="ok"), MessageStop()],
    ]
    summary = await rt.run_turn("レポートを書いて", _Capture())  # type: ignore[arg-type]
    assert _outputs(rt)["d1"].startswith(UNREQUESTED_DELETION_PREFIX)
    assert summary.output_delete_blocks == 1
    assert (tmp_path / "report.md").exists()


async def test_requested_deletion_goes_to_the_card_with_a_note(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [_write("w1", "dashboards/index.html"), [TextDelta(text="ok"), MessageStop()]]
    user = _Capture(decision="allow")
    await rt.run_turn("ダッシュボードを作成して", user)  # type: ignore[arg-type]
    script += [_delete("d1", "dashboards"), [TextDelta(text="削除しました"), MessageStop()]]
    summary = await rt.run_turn("作成したファイルを削除してください", user)  # type: ignore[arg-type]

    assert user.asked == [
        ("delete_path", "このセッションで作成したファイルを消します: dashboards/index.html（2 行）")
    ]
    assert summary.output_delete_blocks == 0
    assert not (tmp_path / "dashboards").exists()


async def test_yes_after_a_deletion_question_still_counts(tmp_path: Path) -> None:
    """「削除してよいですか」→「はい」の往復は、1 つ前の発言の依頼で通す。"""
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [_write("w1", "a.txt", "a\n"), [TextDelta(text="ok"), MessageStop()]]
    user = _Capture(decision="allow")
    await rt.run_turn("a.txt を作って", user)  # type: ignore[arg-type]
    script += [[TextDelta(text="a.txt を削除してよいですか？"), MessageStop()]]
    await rt.run_turn("a.txt を消して", user)  # type: ignore[arg-type]
    script += [_bash("r1", "rm a.txt"), [TextDelta(text="削除しました"), MessageStop()]]
    await rt.run_turn("はい", user)  # type: ignore[arg-type]
    assert [name for name, _ in user.asked] == ["bash"]
    assert not (tmp_path / "a.txt").exists()


async def test_files_that_existed_before_the_session_are_not_guarded(tmp_path: Path) -> None:
    """成果物だけが対象。"""
    (tmp_path / "old.txt").write_text("old\n")
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    script[:] = [
        _write("w1", "old.txt", "new\n"),
        _bash("r1", "rm old.txt"),
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture(decision="deny")
    summary = await rt.run_turn("続けて", user)  # type: ignore[arg-type]
    assert user.asked == [("bash", None)]
    assert summary.output_delete_blocks == 0


def test_state_dir_deletion_is_prohibited(tmp_path: Path) -> None:
    for command in (
        "rm -rf .satsuma",
        "rm -rf .satsuma/plans/plan.md",
        f"cd {tmp_path} && rm -rf .satsuma/sessions",
        "find .satsuma -delete",
    ):
        reason = check_bash(command, tmp_path)
        assert reason is not None and reason.startswith(PROHIBITED_DELETION_PREFIX), command
        assert ".satsuma/" in reason
    for path in (".satsuma", ".satsuma/plans/plan.md", "./.satsuma/"):
        assert check_path(path, tmp_path) is not None, path


def test_state_dir_glob_is_prohibited(tmp_path: Path) -> None:
    (tmp_path / ".satsuma").mkdir()
    assert check_bash("rm -rf .*", tmp_path) is not None


def test_config_home_is_prohibited(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home" / ".satsuma"
    workspace = tmp_path / "work"
    workspace.mkdir()
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(home))
    assert check_bash(f"rm -rf {home}/logs", workspace) is not None
    assert check_path(str(home), workspace) is not None


def test_workspace_inside_config_home_is_still_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SATSUMA_CONFIG_HOME を作業場所に向けても、普通のファイルは消せる（.satsuma だけを守る）。"""
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path))
    workspace = tmp_path / "work"
    workspace.mkdir()
    assert check_bash("rm -rf build", workspace) is None
    assert check_bash("rm -rf .satsuma", workspace) is not None


def test_similar_names_are_not_state_dirs(tmp_path: Path) -> None:
    assert check_bash("rm -rf .satsumadb satsuma", tmp_path) is None


def test_bash_targets(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    assert bash_targets("rm -rf a b && ls", tmp_path) == [root / "a", root / "b"]
    assert bash_targets(f"cd {tmp_path}/x && rm y", tmp_path) == [root / "x" / "y"]
    assert bash_targets("ls -la", tmp_path) == []


async def test_state_dir_is_refused_before_the_card(tmp_path: Path) -> None:
    """実機の再現: 「作成したファイルを削除して」で delete_path(".satsuma")。"""
    rt, script = _runtime(tmp_path)
    rt.set_mode(main="code")
    (tmp_path / ".satsuma").mkdir(exist_ok=True)
    script[:] = [
        _delete("d1", ".satsuma"),
        _bash("r1", "rm -rf .satsuma"),
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture(decision="allow")
    await rt.run_turn("作成したファイルを削除してください", user)  # type: ignore[arg-type]
    assert user.asked == []
    outputs = _outputs(rt)
    assert outputs["d1"].startswith(PROHIBITED_DELETION_PREFIX)
    assert outputs["r1"].startswith(PROHIBITED_DELETION_PREFIX)
    assert (tmp_path / ".satsuma").is_dir()
