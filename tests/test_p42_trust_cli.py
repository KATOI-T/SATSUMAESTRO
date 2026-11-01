"""起動時の信頼の判断と CLI（--trust・satsumaestro trust・doctor・ワーカーの子）。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

import satsumaestro.cli.app as cli_app
from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.cli.trust_gate import (
    TrustBasis,
    TrustQuit,
    decide_trust,
    question_lines,
    untrusted_notice,
)
from satsumaestro.config.models import RuntimeConfig
from satsumaestro.orchestration.worker import WorkerState
from satsumaestro.security.trust import TrustStore
from satsumaestro.skill.resolver import list_skills
from satsumaestro.tools.builtins.skill import Skill, UntrustedSkill
from tests.test_p33_worker_supervisor import _supervisor
from tests.test_p153_missing_model import _ollama, _patch_provider, _route_http

runner = CliRunner()
DEFAULT_MAX = RuntimeConfig().max_iterations


@pytest.fixture
def store() -> TrustStore:
    return TrustStore()


def _repo(root: Path) -> Path:
    """設定・policy.md・memory.md・.env を同梱したリポジトリ。"""
    state = root / ".satsuma"
    state.mkdir(parents=True)
    (root / ".git").mkdir()
    (state / "settings.json").write_text(
        json.dumps({"max_iterations": 3, "hooks": {"pre_tool_use": ["echo x"]}}), encoding="utf-8"
    )
    (state / "policy.md").write_text("## Allow\n- bash\n", encoding="utf-8")
    (state / "memory.md").write_text("## Allow\n- write_file\n", encoding="utf-8")
    (root / ".env").write_text("SATSUMA_TRUST_PROBE=loaded\n", encoding="utf-8")
    return root


def test_the_flag_trusts_without_recording(tmp_path: Path, store: TrustStore) -> None:
    decision = decide_trust(tmp_path, trust_flag=True, ask=None)
    assert decision.basis is TrustBasis.FLAG and decision.trusted
    assert store.roots() == []


def test_a_recorded_root_is_trusted_without_asking(tmp_path: Path, store: TrustStore) -> None:
    store.trust(tmp_path)
    decision = decide_trust(tmp_path, trust_flag=False, ask=lambda root: pytest.fail("asked"))
    assert decision.basis is TrustBasis.RECORDED and decision.trusted
    assert decide_trust(tmp_path, trust_flag=True, ask=None).basis is TrustBasis.RECORDED


def test_non_interactive_reads_nothing_and_records_nothing(
    tmp_path: Path, store: TrustStore
) -> None:
    decision = decide_trust(tmp_path, trust_flag=False, ask=None)
    assert decision.basis is TrustBasis.UNTRUSTED and not decision.trusted
    assert store.roots() == []


@pytest.mark.parametrize(
    ("answer", "basis", "recorded"),
    [("y", TrustBasis.ANSWERED, True), ("n", TrustBasis.DECLINED, False)],
)
def test_the_answer_decides_and_y_records_the_git_root(
    tmp_path: Path, store: TrustStore, answer: str, basis: TrustBasis, recorded: bool
) -> None:
    repo = _repo(tmp_path / "repo")
    asked: list[Path] = []

    def ask(root: Path) -> str:
        asked.append(root)
        return answer

    decision = decide_trust(repo / ".satsuma", trust_flag=False, ask=ask)
    assert asked == [repo.resolve()]
    assert decision.basis is basis
    assert store.roots() == ([repo.resolve()] if recorded else [])


def test_q_quits(tmp_path: Path) -> None:
    with pytest.raises(TrustQuit):
        decide_trust(tmp_path, trust_flag=False, ask=lambda root: "q")


def test_trusting_part_of_a_repository_does_not_trust_its_root(
    tmp_path: Path, store: TrustStore
) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "sub").mkdir()
    store.trust(repo / "sub")
    decision = decide_trust(repo / "sub", trust_flag=False, ask=None)
    assert not decision.trusted


def test_the_texts_name_the_root_and_avoid_the_card_markers(tmp_path: Path) -> None:
    text = "\n".join(line.plain for line in question_lines(tmp_path))
    assert str(tmp_path) in text and "[y]" in text and "[q]" in text
    assert "許可しますか？" not in text and "プランを承認して" not in text
    notice = untrusted_notice(tmp_path)
    assert "\n" not in notice and f"satsumaestro trust --cwd {tmp_path}" in notice


def _answers(monkeypatch: pytest.MonkeyPatch, *answers: str | BaseException) -> list[str]:
    prompts: list[str] = []
    queue = list(answers)

    def fake_input(prompt: str, **kwargs: object) -> str:
        prompts.append(prompt)
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(cli_app._console, "input", fake_input)
    return prompts


def test_the_prompt_repeats_until_y_n_or_q(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store: TrustStore,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = _answers(monkeypatch, "maybe", "Y")
    assert cli_app._decide_trust(tmp_path, trust=False, interactive=True) is True
    assert prompts == ["(y/n/q): ", "(y/n/q): "]
    assert store.roots() == [tmp_path.resolve()]
    out = capsys.readouterr().out
    assert "このディレクトリを初めて開きます" in out and "信頼しました" in out


@pytest.mark.parametrize(
    ("answer", "trusted"),
    [("n", False), (EOFError(), False)],
)
def test_n_and_eof_continue_untrusted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    store: TrustStore,
    answer: str | BaseException,
    trusted: bool,
) -> None:
    _answers(monkeypatch, answer)
    assert cli_app._decide_trust(tmp_path, trust=False, interactive=True) is trusted
    assert store.roots() == []


@pytest.mark.parametrize("answer", ["q", KeyboardInterrupt()])
def test_q_and_ctrl_c_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answer: str | BaseException
) -> None:
    _answers(monkeypatch, answer)
    with pytest.raises(typer.Exit):
        cli_app._decide_trust(tmp_path, trust=False, interactive=True)


async def _assemble(
    monkeypatch: pytest.MonkeyPatch, cwd: Path, *, trusted: bool
) -> cli_app.ConversationRuntime:
    _patch_provider(monkeypatch, [])
    _route_http(monkeypatch, _ollama())
    monkeypatch.setenv("SATSUMA_TRUST_PROBE", "_")
    monkeypatch.delenv("SATSUMA_TRUST_PROBE")
    rt, _, _ = await cli_app._assemble_runtime(
        "gpt-oss:120b", cwd, interactive=False, trusted=trusted
    )
    return rt


async def test_untrusted_runtime_ignores_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _repo(tmp_path / "repo")
    rt = await _assemble(monkeypatch, repo, trusted=False)
    await rt.mcp.aclose()

    assert rt.config.max_iterations == DEFAULT_MAX
    assert rt.config.hooks.pre_tool_use == []
    assert "SATSUMA_TRUST_PROBE" not in os.environ
    resolver = rt.policy
    assert isinstance(resolver, cli_app.PermissionResolver)
    assert resolver.base.explicit("bash", {"command": "ls"}) is None
    assert all(layer.path != repo / ".satsuma" / "memory.md" for layer in resolver.layers)


async def test_trusted_runtime_reads_the_repository(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _repo(tmp_path / "repo")
    rt = await _assemble(monkeypatch, repo, trusted=True)
    await rt.mcp.aclose()

    assert rt.config.max_iterations == 3
    assert rt.config.hooks.pre_tool_use == ["echo x"]
    assert os.environ["SATSUMA_TRUST_PROBE"] == "loaded"
    resolver = rt.policy
    assert isinstance(resolver, cli_app.PermissionResolver)
    allowed = resolver.base.explicit("bash", {"command": "ls"})
    assert allowed is not None and allowed.decision == "allow"
    assert resolver.layers[0].path == repo / ".satsuma" / "memory.md"


def _config_show(cwd: Path, *extra: str) -> tuple[dict[str, object], str]:
    result = runner.invoke(cli_app.app, ["config", "show", "--cwd", str(cwd), *extra])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout), result.stderr


def test_config_show_follows_the_decision(tmp_path: Path, store: TrustStore) -> None:
    repo = _repo(tmp_path / "repo")

    shown, err = _config_show(repo)
    assert shown["max_iterations"] == DEFAULT_MAX
    assert f"satsumaestro trust --cwd {repo.resolve()}" in err

    shown, err = _config_show(repo, "--trust")
    assert shown["max_iterations"] == 3 and err == ""
    assert store.roots() == []


def test_trust_add_list_remove(tmp_path: Path, store: TrustStore) -> None:
    repo = _repo(tmp_path / "repo")
    (repo / "pkg").mkdir()

    added = runner.invoke(cli_app.app, ["trust", "--cwd", str(repo / "pkg")])
    assert added.exit_code == 0 and str(repo.resolve()) in added.stdout
    assert store.roots() == [repo.resolve()]
    assert _config_show(repo / "pkg")[0]["max_iterations"] == 3

    listed = runner.invoke(cli_app.app, ["trust", "list"])
    assert listed.stdout.splitlines() == [str(repo.resolve())]

    removed = runner.invoke(cli_app.app, ["trust", "remove", "--cwd", str(repo)])
    assert removed.exit_code == 0 and store.roots() == []
    again = runner.invoke(cli_app.app, ["trust", "remove", "--cwd", str(repo)])
    assert again.exit_code == 1 and "記録にありません" in again.stderr
    assert runner.invoke(cli_app.app, ["trust", "list"]).stdout.strip() == (
        "（信頼したディレクトリはありません）"
    )


def test_a_trusted_directory_outside_git_ignores_its_parents(
    tmp_path: Path, store: TrustStore
) -> None:
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text('{"max_iterations": 1}', encoding="utf-8")
    cwd = tmp_path / "x"
    cwd.mkdir()
    store.trust(cwd)
    assert _config_show(cwd)[0]["max_iterations"] == DEFAULT_MAX


def test_doctor_shows_the_trust_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    repo = _repo(tmp_path / "repo")

    untrusted = runner.invoke(cli_app.app, ["doctor", "--cwd", str(repo)])
    assert untrusted.exit_code == 0, untrusted.output
    assert "未信頼" in untrusted.stdout and "ユーザー設定と環境変数だけ" in untrusted.stdout

    flagged = runner.invoke(cli_app.app, ["doctor", "--cwd", str(repo), "--trust"])
    assert flagged.exit_code == 0, flagged.output
    assert "--trust" in flagged.stdout and "読み込み済み" in flagged.stdout
    assert "未信頼" not in flagged.stdout


def test_worker_children_inherit_the_cleared_trust(tmp_path: Path) -> None:
    sup = _supervisor(tmp_path)
    inside = sup.create()
    assert inside.state is WorkerState.READY
    assert sup._command_for(inside)[-1] == "--trust"

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    gated = sup.create(str(outside))
    assert gated.state is WorkerState.AWAITING_TRUST
    assert "--trust" not in sup._command_for(gated)


def _skill(directory: Path, name: str) -> None:
    folder = directory / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: {name}\n---\nsteps\n", encoding="utf-8")


async def test_untrusted_skips_skills_in_the_state_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _repo(tmp_path / "repo")
    _skill(repo / ".satsuma" / "skills", "deploy")
    _skill(repo / ".claude" / "skills", "review")
    assert [m.name for m in list_skills(repo, state_dir=False)] == ["review"]
    assert [m.name for m in list_skills(repo)] == ["deploy", "review"]

    untrusted = await _assemble(monkeypatch, repo, trusted=False)
    await untrusted.mcp.aclose()
    assert untrusted.tools.get("skill") is UntrustedSkill
    ctx = untrusted._ctx
    missing = await UntrustedSkill().run(UntrustedSkill.Input(skill="deploy"), ctx)
    assert missing.is_error and "known skills: review" in missing.output
    found = await UntrustedSkill().run(UntrustedSkill.Input(skill="review"), ctx)
    assert not found.is_error

    trusted = await _assemble(monkeypatch, repo, trusted=True)
    await trusted.mcp.aclose()
    assert trusted.tools.get("skill") is Skill


async def test_doctor_in_the_repl_follows_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(
        cli_app, "_doctor", lambda cwd, *, probe=False, trust=False: seen.append(trust)
    )
    rt = SimpleNamespace(cwd=Path("."))
    for trusted in (True, False):
        await SlashDispatcher(rt, Console(), trusted=trusted).dispatch(  # type: ignore[arg-type]
            parse_slash("/doctor")  # type: ignore[arg-type]
        )
    assert seen == [True, False]


def test_the_question_is_coloured_like_the_language_question(tmp_path: Path) -> None:
    heading, explanation, caution, choices = question_lines(tmp_path)

    assert heading.style == "bold cyan"
    assert not explanation.style
    assert caution.style == "yellow"
    keys = [choices.plain[span.start : span.end] for span in choices.spans if span.style == "cyan"]
    assert keys == ["[y]", "[n]", "[q]"]
