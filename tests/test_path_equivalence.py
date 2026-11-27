"""別の書き方で同じファイルを指しても、名前とパスの規則の判定が変わらないこと。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.engine.turn import PermissionOutcome
from satsumaestro.prompt.builder import SystemPromptBuilder
from satsumaestro.security import PermissionMode, classify, deletion_guard, state_write_guard
from satsumaestro.security.allowlist import (
    MemoryLayer,
    PermissionResolver,
    SessionAllowlist,
    rule_to_line,
    signature_for,
)
from satsumaestro.security.jail import WorkspaceJail
from satsumaestro.security.policy import RuleMatch, rule_input
from satsumaestro.security.policy_markdown import parse_policy_markdown
from satsumaestro.security.state_write_guard import PROHIBITED_STATE_WRITE_PREFIX
from satsumaestro.session.models import ToolResultBlock
from satsumaestro.util.errors import StateDirError
from satsumaestro.util.paths import in_state_dir, names_component, rule_form
from tests.test_p145_plan_exit_nudge import _runtime

RO = PermissionMode.READ_ONLY
WW = PermissionMode.WORKSPACE_WRITE
DFA = PermissionMode.DANGER_FULL_ACCESS


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("SATSUMA_CONFIG_HOME", raising=False)
    root = (tmp_path / "proj").resolve()
    for name in (".git", ".satsuma", "docs", "secrets", "config"):
        (root / name).mkdir(parents=True)
    (root / ".git" / "config").write_text("[core]\n")
    (root / ".satsuma" / "policy.md").write_text("mode: workspace-write\n")
    (root / "docs" / "a.md").write_text("a\n")
    (root / "secrets" / "key.txt").write_text("k\n")
    (root / ".env").write_text("K=1\n")
    return root


# ── 名前の同値: .git と状態ディレクトリ ──
@pytest.mark.parametrize(
    "path",
    [".git", ".git/config", ".GIT/config", ".Git/hooks/pre-commit", "vendor/lib/.GIT/config"],
)
def test_git_directory_is_found_in_any_letter_case(path: str) -> None:
    assert names_component(Path(path), ".git")


@pytest.mark.parametrize(
    "path", [".github/workflows/ci.yml", ".gitignore", "src/git/config", ".git2/config"]
)
def test_similar_names_are_not_the_git_directory(path: str) -> None:
    assert not names_component(Path(path), ".git")


def test_state_directory_is_found_in_any_letter_case() -> None:
    assert in_state_dir(Path(".SATSUMA/policy.md"))
    assert in_state_dir(Path("sub/.Satsuma/sessions/x.jsonl"))
    # 大小を区別しない FS は ſ（長い s）も s と同じに読む。lower() では拾えない。
    assert in_state_dir(Path(".ſatsuma/policy.md"))
    assert not in_state_dir(Path(".satsumadb/x"))


def test_an_alias_is_found_through_the_filesystem(ws: Path) -> None:
    """名前が似ていなくても、同じ実体を指す書き方（ここではリンク）は当たる。"""
    (ws / "g").symlink_to(".git")
    assert not names_component(Path("g/config"), ".git")
    assert names_component(Path("g/config"), ".git", ws)
    assert not names_component(Path("docs/a.md"), ".git", ws)


@pytest.mark.parametrize(
    "path", [".GIT/config", ".Git/hooks/pre-commit", "./.GIT/config", "docs/../.GIT/config"]
)
def test_writing_under_git_in_another_case_needs_full_access(path: str) -> None:
    for tool in ("write_file", "edit_file", "append_file"):
        assert classify(tool, {"path": path}, WW) is DFA, tool
    assert classify("notebook_edit", {"notebook_path": ".GIT/x.ipynb"}, WW) is DFA


def test_similar_names_stay_ordinary_writes(ws: Path) -> None:
    for path in (".github/workflows/ci.yml", ".gitignore", "docs/a.md"):
        assert classify("write_file", {"path": path}, WW, ws) is WW, path


def test_a_link_to_git_needs_full_access(ws: Path) -> None:
    (ws / "g").symlink_to(".git")
    assert classify("write_file", {"path": "g/config"}, WW, ws) is DFA
    assert classify("edit_file", {"path": str(ws / "g" / "config")}, WW, ws) is DFA


def test_a_git_link_to_a_directory_inside_needs_full_access(tmp_path: Path) -> None:
    """``.git`` が中の別のディレクトリへのリンクでも、その中への書き込みは確認に回る。"""
    root = (tmp_path / "proj").resolve()
    (root / "metadata" / "repository").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / ".git").symlink_to("metadata/repository")
    (root / "g").symlink_to(".git")
    for path in ("metadata/repository/config", "g/config", str(root / "metadata/repository")):
        assert classify("write_file", {"path": path}, WW, root) is DFA, path
    for path in ("metadata/notes.md", "docs/a.md"):
        assert classify("write_file", {"path": path}, WW, root) is WW, path


def test_a_git_file_that_points_inside_needs_full_access(tmp_path: Path) -> None:
    """``gitdir:`` のファイル（``git init --separate-git-dir``）が指す中のディレクトリも同じ。"""
    root = (tmp_path / "proj").resolve()
    (root / "meta" / "repo" / "hooks").mkdir(parents=True)
    (root / ".git").write_text(f"gitdir: {root / 'meta' / 'repo'}\n")
    assert classify("write_file", {"path": "meta/repo/hooks/pre-commit"}, WW, root) is DFA
    (root / ".git").write_text("gitdir: meta/repo\n")
    assert classify("edit_file", {"path": "meta/repo/config"}, WW, root) is DFA
    assert classify("write_file", {"path": "meta/notes.md"}, WW, root) is WW
    (root / ".git").write_text("not a pointer\n")
    assert classify("write_file", {"path": "meta/repo/config"}, WW, root) is WW


def test_state_directory_in_another_case_is_hidden(ws: Path) -> None:
    jail = WorkspaceJail(ws)
    assert jail.hides(".SATSUMA/policy.md")
    with pytest.raises(StateDirError):
        jail.resolve(".SATSUMA/policy.md")
    assert not jail.hides("docs/a.md")


@pytest.mark.parametrize(
    "command",
    [
        "echo 'mode: allow' > .SATSUMA/policy.md",
        "sed -i '' s/write/allow/ .Satsuma/policy.md",
        "cp docs/a.md .SATSUMA/settings.local.json",
    ],
)
def test_bash_writes_into_state_in_another_case_are_prohibited(ws: Path, command: str) -> None:
    assert state_write_guard.check_bash(command, ws) is not None


def test_write_tools_into_state_in_another_case_are_prohibited(ws: Path) -> None:
    for name in ("write_file", "edit_file", "append_file", "notebook_edit"):
        assert state_write_guard.check_tool(name, ".SATSUMA/policy.md", ws) is not None, name
    assert state_write_guard.check_tool("write_file", "docs/a.md", ws) is None


def test_deleting_state_in_another_case_is_prohibited(ws: Path) -> None:
    assert deletion_guard.check_path(".SATSUMA", ws) is not None
    assert deletion_guard.check_bash("rm -rf .SATSUMA/sessions", ws) is not None
    assert deletion_guard.check_path("docs/a.md", ws) is None


def test_user_config_home_in_another_case_is_protected(
    ws: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = (tmp_path / "home").resolve()
    (home / ".satsuma").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    assert deletion_guard.check_bash("rm -rf ~/.SATSUMA", ws) is not None
    assert state_write_guard.check_bash("echo x > ~/.SATSUMA/settings.json", ws) is not None
    assert deletion_guard.check_bash("rm -rf ~/.satsuma-old", ws) is None


class _Capture:
    """確認カードに回った呼び出しを記録して、決めた答えを返すユーザー。"""

    def __init__(self, decision: str = "allow") -> None:
        self.asked: list[str] = []
        self._decision = decision

    def decide(
        self, call: ToolUse, required: PermissionMode, warning: str | None = None
    ) -> PermissionOutcome:
        self.asked.append(call.name)
        return PermissionOutcome(decision=self._decision, reason="denied by user")  # type: ignore[arg-type]


def _outputs(rt: object) -> dict[str, ToolResultBlock]:
    return {
        b.tool_use_id: b
        for m in rt.session.messages  # type: ignore[attr-defined]
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }


async def test_policy_cannot_be_rewritten_through_another_spelling(tmp_path: Path) -> None:
    """カードが出ない danger-full-access でも、カードの前で断ってポリシーは変わらない。"""
    rt, script = _runtime(tmp_path)  # .satsuma/plans はここで作られる
    policy = tmp_path / ".satsuma" / "policy.md"
    policy.write_text("mode: workspace-write\n")
    rt.set_mode(main="code", sub=DFA)
    script[:] = [
        [
            ToolUse(
                id="w1",
                name="write_file",
                input={"path": ".SATSUMA/policy.md", "content": "mode: allow\n"},
            ),
            MessageStop(),
        ],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture()
    summary = await rt.run_turn("設定を変えて", user)  # type: ignore[arg-type]
    assert user.asked == []
    assert _outputs(rt)["w1"].output.startswith(PROHIBITED_STATE_WRITE_PREFIX)
    assert summary.state_write_blocks == 1
    assert policy.read_text() == "mode: workspace-write\n"


async def test_git_config_in_another_case_goes_to_the_card(tmp_path: Path) -> None:
    """workspace-write では書き込みは確認なしで通るが、.git の下は別の書き方でも確認に回る。"""
    rt, script = _runtime(tmp_path)
    (tmp_path / ".git").mkdir()
    rt.set_mode(main="code", sub=WW)
    script[:] = [
        [
            ToolUse(
                id="w1",
                name="write_file",
                input={"path": ".GIT/config", "content": "[diff]\n\texternal = /tmp/x\n"},
            ),
            MessageStop(),
        ],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture(decision="deny")
    await rt.run_turn("設定を変えて", user)  # type: ignore[arg-type]
    assert user.asked == ["write_file"]
    assert not (tmp_path / ".git" / "config").exists()


# ── policy のパスの規則 ──
_POLICY = """---
mode: workspace-write
---
## Deny
- * when path glob:secrets/**

## Ask
- write_file when path glob:config/**

## Allow
- read_file when path glob:docs/**
"""


@pytest.mark.parametrize(
    ("written", "form"),
    [
        ("secrets/key.txt", "secrets/key.txt"),
        ("./secrets/key.txt", "secrets/key.txt"),
        ("docs/../secrets/key.txt", "secrets/key.txt"),
        ("docs/../.env", ".env"),
        ("docs/", "docs"),
        (".", "."),
    ],
)
def test_rule_form_is_the_resolved_path_from_the_root(ws: Path, written: str, form: str) -> None:
    assert rule_form(written, ws) == form


def test_rule_form_of_other_spellings(ws: Path) -> None:
    assert rule_form(str(ws / "secrets" / "key.txt"), ws) == "secrets/key.txt"
    # 外は絶対パスのまま（相対にすると、中のパスと見分けがつかなくなる）。
    assert rule_form("../other.txt", ws) == (ws.parent / "other.txt").as_posix()
    (ws / "pub").symlink_to("secrets")
    assert rule_form("pub/key.txt", ws) == "secrets/key.txt"
    # 解決できないパスは、書かれたまま。
    assert rule_form("a\x00b", ws) == "a\x00b"


def test_rule_form_of_the_root_written_in_another_case(ws: Path) -> None:
    """根を別の大小で書いた絶対パスも、同じ実体なら中のパスとして照らす（MCP のツールなど）。"""
    other = ws.parent / "other"
    other.mkdir()
    assert rule_form(str(other / "x.txt"), ws) == (other / "x.txt").as_posix()
    upper = ws.parent / ws.name.upper()
    if not upper.exists():
        pytest.skip("this file system tells letter case apart")
    assert rule_form(str(upper / "secrets" / "key.txt"), ws) == "secrets/key.txt"
    assert rule_form(str(upper), ws) == "."
    policy = parse_policy_markdown(_POLICY).policy
    inp = rule_input({"path": str(upper / "secrets" / "key.txt")}, ws)
    assert policy.authorize("mcp__files__read", inp, WW).decision == "deny"


def test_rule_input_keeps_the_written_path(ws: Path) -> None:
    inp = rule_input({"path": "./docs/a.md", "content": "x"}, ws)
    assert inp == {"path": "docs/a.md", "content": "x"}
    assert inp.written == {"path": "./docs/a.md"}
    nb = rule_input({"notebook_path": "docs/../nb.ipynb"}, ws)
    assert nb["notebook_path"] == "nb.ipynb"
    # パスの無い入力・文字列でないパスは、そのまま。
    assert rule_input({"command": "ls ./docs"}, ws) == {"command": "ls ./docs"}
    assert rule_input({"path": 3}, ws) == {"path": 3}


@pytest.mark.parametrize(
    "path",
    [
        "secrets/key.txt",
        "./secrets/key.txt",
        "docs/../secrets/key.txt",
        "SECRETS/key.txt",
        "Secrets/KEY.txt",
    ],
)
def test_deny_rule_holds_for_every_spelling(ws: Path, path: str) -> None:
    policy = parse_policy_markdown(_POLICY).policy
    for tool in ("read_file", "write_file", "edit_file"):
        outcome = policy.authorize(tool, rule_input({"path": path}, ws), WW)
        assert outcome.decision == "deny", (tool, path)


def test_deny_rule_holds_for_an_absolute_path_and_a_link(ws: Path) -> None:
    policy = parse_policy_markdown(_POLICY).policy
    absolute = str(ws / "secrets" / "key.txt")
    assert policy.authorize("read_file", rule_input({"path": absolute}, ws), WW).decision == "deny"
    (ws / "pub").symlink_to("secrets")
    linked = rule_input({"path": "pub/key.txt"}, ws)
    assert policy.authorize("read_file", linked, WW).decision == "deny"
    nb = rule_input({"notebook_path": "./secrets/nb.ipynb"}, ws)
    assert policy.authorize("notebook_edit", nb, WW).decision == "deny"


def test_deny_rule_still_matches_the_written_name(ws: Path) -> None:
    """リンクの名前で書いた拒否は、リンクの先に解決した後も効く（書かれた形でも照らす）。"""
    (ws / "pub").symlink_to("docs")
    policy = parse_policy_markdown(
        "---\nmode: workspace-write\n---\n## Deny\n- * when path glob:pub/**\n"
    ).policy
    assert policy.authorize("read_file", rule_input({"path": "pub/a.md"}, ws), RO).decision == (
        "deny"
    )


@pytest.mark.parametrize("path", ["config/app.toml", "./config/app.toml", "CONFIG/app.toml"])
def test_ask_rule_holds_for_every_spelling(ws: Path, path: str) -> None:
    policy = parse_policy_markdown(_POLICY).policy
    assert policy.authorize("write_file", rule_input({"path": path}, ws), WW).decision == "ask"


@pytest.mark.parametrize("path", ["docs/a.md", "./docs/a.md", "config/../docs/a.md"])
def test_allow_rule_matches_other_spellings_of_the_same_file(ws: Path, path: str) -> None:
    policy = parse_policy_markdown(_POLICY).policy
    outcome = policy.explicit("read_file", rule_input({"path": path}, ws))
    assert outcome is not None and outcome.decision == "allow" and outcome.scoped


def test_a_path_written_through_a_denied_directory_stays_denied(ws: Path) -> None:
    """解決した先が許可の中でも、拒否のディレクトリを通る書き方は拒否のまま（これまでと同じ）。"""
    policy = parse_policy_markdown(_POLICY).policy
    outcome = policy.explicit("read_file", rule_input({"path": "secrets/../docs/a.md"}, ws))
    assert outcome is not None and outcome.decision == "deny"


@pytest.mark.parametrize("path", ["docs/../.env", "docs/../config/app.toml", "DOCS/a.md"])
def test_allow_rule_does_not_reach_past_its_directory(ws: Path, path: str) -> None:
    """``docs/**`` の許可は、``docs`` を通って外へ出るパスに当たらない。大小の違いも当てない。"""
    policy = parse_policy_markdown(_POLICY).policy
    assert policy.explicit("read_file", rule_input({"path": path}, ws)) is None


def test_plain_input_is_matched_as_written() -> None:
    """解決していない入力（根を渡さない呼び方）は、これまでどおり書かれた文字列で照らす。"""
    rule = RuleMatch(tool="read_file", field="path", glob="docs/**")
    assert rule.matches("read_file", {"path": "docs/a.md"})
    assert not rule.matches("read_file", {"path": "./docs/a.md"})
    assert not rule.matches("read_file", {"path": "DOCS/a.md"})
    assert rule.matches("read_file", {"path": "DOCS/a.md"}, strict=True)


def test_command_rules_keep_their_letter_case(ws: Path) -> None:
    rule = RuleMatch(tool="bash", field="command", glob="git push*")
    assert rule.matches("bash", rule_input({"command": "git push"}, ws), strict=True)
    assert not rule.matches("bash", rule_input({"command": "GIT PUSH"}, ws), strict=True)


def test_always_allow_is_recorded_in_the_resolved_form(ws: Path) -> None:
    """``a`` の記録は解決した形で残り、同じファイルの別の書き方にだけ効く。"""
    memory = MemoryLayer(ws / ".satsuma" / "memory.md", writable=True)
    resolver = PermissionResolver(
        parse_policy_markdown("---\nmode: read-only\n---\n").policy, [memory], SessionAllowlist()
    )
    resolver.record("write_file", rule_input({"path": "./docs/a.md"}, ws))
    assert rule_to_line(signature_for("write_file", rule_input({"path": "./docs/a.md"}, ws))) == (
        r"write_file when path re: \Adocs/a\.md\Z"
    )
    assert r"- write_file when path re: \Adocs/a\.md\Z" in memory.path.read_text()

    fresh = PermissionResolver(
        parse_policy_markdown("---\nmode: read-only\n---\n").policy,
        [MemoryLayer(memory.path, writable=True)],
        SessionAllowlist(),
    )
    for path in ("docs/a.md", str(ws / "docs" / "a.md"), "secrets/../docs/a.md"):
        assert fresh.authorize("write_file", rule_input({"path": path}, ws), WW).decision == (
            "allow"
        ), path
    for path in ("docs/b.md", "docs/a.md/../../.env", "DOCS/a.md"):
        assert fresh.authorize("write_file", rule_input({"path": path}, ws), WW).decision == (
            "ask"
        ), path


async def test_deny_rule_stops_another_spelling_in_the_runtime(tmp_path: Path) -> None:
    """runtime が policy に渡す入力は解決した形。``./secrets/…`` も拒否の規則で止まる。"""
    rt, script = _runtime(tmp_path)
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "key.txt").write_text("k\n")
    rt.policy = PermissionResolver(parse_policy_markdown(_POLICY).policy, [])
    rt.set_mode(main="code", sub=WW)
    script[:] = [
        [
            ToolUse(id="r1", name="read_file", input={"path": "./secrets/key.txt"}),
            ToolUse(id="r2", name="read_file", input={"path": "docs/../secrets/key.txt"}),
            MessageStop(),
        ],
        [TextDelta(text="ok"), MessageStop()],
    ]
    user = _Capture()
    await rt.run_turn("鍵を読んで", user)  # type: ignore[arg-type]
    assert user.asked == []
    for call in ("r1", "r2"):
        block = _outputs(rt)[call]
        assert block.is_error and "deny rule" in block.output, call


# ── system の git 節が起こす git ──
async def test_git_section_does_not_run_commands_named_in_the_repository_config(
    tmp_path: Path,
) -> None:
    """``.git/config`` が指す diff・変換・監視のコマンドを、git 節の採取で起こさない。"""
    repo = tmp_path / "repo"
    marks = tmp_path / "marks"
    repo.mkdir()
    marks.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True)

    def command(name: str, body: str) -> str:
        script = tmp_path / f"{name}.sh"
        script.write_text(f"#!/bin/sh\ntouch '{marks / name}'\n{body}\n")
        script.chmod(0o755)
        return str(script)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "a.txt").write_text("a\n")
    (repo / ".gitattributes").write_text("*.txt diff=tc\n")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    (repo / "a.txt").write_text("a\nb\n")
    git("config", "diff.external", command("external-diff", "exit 0"))
    git("config", "diff.tc.textconv", command("textconv", 'cat "$1"'))
    git("config", "core.fsmonitor", command("fsmonitor", "exit 1"))

    turn = await SystemPromptBuilder().snapshot(repo)

    assert sorted(p.name for p in marks.iterdir()) == []
    assert " M a.txt" in turn.git_state
    assert "a.txt +1 -0" in turn.git_state
