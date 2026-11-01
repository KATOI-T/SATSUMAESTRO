"""Aider Polyglot を、SATSUMAESTRO 以外のハーネス（aider・OpenCode）でも回す（polyglot.py）。"""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_official_run import _SOLVED, _no_toolchains, exercise, pg  # noqa: F401


def _fake_aider_python(tmp_path: Path, seen: Path) -> Path:
    """aider の Python の代わり。"""
    bindir = tmp_path / "aider-tool" / "bin"
    bindir.mkdir(parents=True)
    fake = bindir / "python"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"seen = pathlib.Path({str(seen)!r})\n"
        "calls = json.loads(seen.read_text()) if seen.exists() else []\n"
        "calls.append(sys.argv[1:])\n"
        "seen.write_text(json.dumps(calls))\n"
        "if len(calls) == 2:\n"
        f"    pathlib.Path('two_fer.py').write_text({_SOLVED!r})\n"
        "print('Applied edit to two_fer.py')\n"
        "print(json.dumps({'iterations': 2, 'usage': {'input_tokens': 50, 'output_tokens': 5},"
        " 'stop_reason': 'end_turn'}))\n",
        encoding="utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    (bindir / "aider").symlink_to(fake)
    return bindir / "aider"


def _arg(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


@pytest.mark.usefixtures("exercise")
def test_aider_runs_like_the_official_benchmark_and_resumes_its_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = tmp_path / "seen.json"
    monkeypatch.setattr(pg.wb, "AIDER_BIN", str(_fake_aider_python(tmp_path, seen)))
    monkeypatch.setattr(pg, "loaded_context", lambda model: 131072)
    monkeypatch.setattr(pg.wb, "ollama_env", lambda: {"OLLAMA_HOST": "http://h:11434/v1"})
    task = {"id": "python/two-fer", "prompt": "Say hi.\n"}
    result, attempts = pg.solve(
        task, "gpt-oss:120b", tmp_path / "out", agent="aider", effort="high"
    )
    assert [a.passed for a in attempts] == [False, True] and result.passed
    first, second = json.loads(seen.read_text())
    assert first[:2] == ["-I", str(pg.AIDER_RUNNER)]
    assert _arg(first, "--model") == "ollama_chat/gpt-oss:120b"
    assert _arg(first, "--edit-format") == "diff" and _arg(first, "--num-ctx") == "131072"
    ignored = [first[i + 1] for i, a in enumerate(first) if a == "--ignore"]
    assert "two_fer_test.py" in ignored
    assert first[-1] == "two_fer.py"
    assert "--restore" not in first and "--restore" in second
    assert _arg(first, "--reasoning-effort") == "high" == _arg(second, "--reasoning-effort")
    assert _arg(first, "--history") == _arg(second, "--history")
    assert Path(_arg(first, "--message-file")).read_text().startswith("Say hi.\n")
    retry = Path(_arg(second, "--message-file")).read_text()
    assert "See the testing errors above." in retry
    assert "Say hi." not in retry
    assert result.iterations == 4 and result.usage.input_tokens == 100


@pytest.mark.usefixtures("exercise")
def test_opencode_continues_the_same_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions: list[str] = []

    efforts: list[str] = []

    def fake_opencode(
        prompt: str,
        workdir: Path,
        model: str,
        timeout_s: float,
        log: Path,
        session: str = "",
        effort: str = "",
    ) -> tuple[dict[str, object], str, float]:
        sessions.append(session)
        efforts.append(effort)
        if len(sessions) == 2:
            (workdir / "two_fer.py").write_text(_SOLVED)
        payload = {"session_id": "ses_1", "iterations": 4, "tool_calls": 3}
        return payload, "", 1.0

    monkeypatch.setattr(pg.wb, "run_opencode", fake_opencode)
    task = {"id": "python/two-fer", "prompt": "Say hi.\n"}
    result, _ = pg.solve(task, "m", tmp_path / "out", agent="opencode", effort="high")
    assert result.passed and sessions == ["", "ses_1"] and efforts == ["high", "high"]
    assert result.tool_calls == 6


@pytest.mark.usefixtures("exercise")
def test_a_write_outside_the_workdir_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.setattr(pg, "REPO", repo)

    def escaping_satsuma(
        prompt: str, workdir: Path, model: str, timeout_s: float, log: Path, resume: str = ""
    ) -> tuple[dict[str, object], str, float]:
        (repo / "two_fer.py").write_text(_SOLVED)
        return {}, "", 1.0

    monkeypatch.setattr(pg.wb, "run_satsuma", escaping_satsuma)
    with pytest.raises(SystemExit, match="two_fer.py"):
        pg.solve({"id": "python/two-fer", "prompt": "x"}, "m", tmp_path / "out")


def test_official_test_commands_for_cpp_go_and_java() -> None:
    assert pg.test_command("go") == ["go", "test", "./..."]
    assert pg.test_command("java") == ["./gradlew", "test"]
    cpp = pg.test_command("cpp")
    assert cpp[:2] == ["bash", "-c"] and "cmake -DEXERCISM_RUN_ALL_TESTS=1" in cpp[2]
    with pytest.raises(SystemExit):
        pg.test_command("cobol")


def test_java_disabled_tests_are_turned_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    test = "src/test/java/ATest.java"
    (src / test).parent.mkdir(parents=True)
    (src / test).write_text('@Test\n@Disabled("Remove to run test")\npublic void b() {}\n')
    ex = pg.Exercise("java", "a", src, ["src/main/java/A.java"], [test], [])
    monkeypatch.setattr(pg, "test_command", lambda lang: ["true"])
    workdir = tmp_path / "w"
    workdir.mkdir()
    assert pg.run_tests(ex, workdir)[0]
    assert (workdir / test).read_text() == "@Test\npublic void b() {}\n"


def test_toolchains_prefer_the_users_java_and_turn_off_the_gradle_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.undo()
    jdk = tmp_path / "jdk"
    (jdk / "bin").mkdir(parents=True)
    monkeypatch.setattr(pg, "JDK_HOME", jdk)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.delenv("GRADLE_OPTS", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    pg.use_toolchains()
    assert os.environ["JAVA_HOME"] == str(jdk)
    assert os.environ["PATH"] == f"{jdk / 'bin'}{os.pathsep}/usr/bin"
    assert os.environ["GRADLE_OPTS"] == "-Dorg.gradle.daemon=false"
    monkeypatch.setenv("JAVA_HOME", "/mine")
    pg.use_toolchains()
    assert os.environ["JAVA_HOME"] == "/mine"


def test_loaded_context_loads_the_model_once_when_it_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[str] = []

    def fake_urlopen(req: object, timeout: float) -> io.BytesIO:
        url = req if isinstance(req, str) else req.full_url  # type: ignore[attr-defined]
        requests.append(url.rsplit("/", 1)[-1])
        loaded = "generate" in requests
        models = [{"name": "m", "context_length": 65536}] if loaded else []
        return io.BytesIO(json.dumps({"models": models}).encode())

    monkeypatch.setattr(pg.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(pg.wb, "ollama_env", lambda: {"OLLAMA_HOST": "http://h:11434/v1"})
    pg.loaded_context.cache_clear()
    try:
        assert pg.loaded_context("m") == 65536
        assert pg.loaded_context("m") == 65536
    finally:
        pg.loaded_context.cache_clear()
    assert requests == ["ps", "generate", "ps"]


def _exercise(
    tmp_path: Path, lang: str, files: dict[str, list[str]], meta: dict[str, str]
) -> pg.Exercise:
    src = tmp_path / lang / "exercises" / "practice" / "x"
    (src / ".meta").mkdir(parents=True)
    (src / ".meta" / "config.json").write_text(json.dumps({"files": files}))
    for rel, text in meta.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    pg.POLYGLOT_DIR, saved = tmp_path, pg.POLYGLOT_DIR
    try:
        return pg.load_exercise(f"{lang}/x")
    finally:
        pg.POLYGLOT_DIR = saved


def test_reference_files_go_where_the_solution_lives(tmp_path: Path) -> None:
    ref = ".meta/src/reference/java/"
    java = _exercise(
        tmp_path,
        "java",
        {
            "solution": ["src/main/java/Game.java"],
            "example": [ref + "Game.java", ref + "Frame.java"],
        },
        {ref + "Game.java": "g", ref + "Frame.java": "f"},
    )
    work = tmp_path / "w-java"
    pg.place_reference(java, work)
    assert (work / "src/main/java/Game.java").read_text() == "g"
    assert (work / "src/main/java/Frame.java").read_text() == "f"

    cpp = _exercise(
        tmp_path,
        "cpp",
        {"solution": ["tree.cpp", "tree.h"], "example": [".meta/example.h"]},
        {".meta/example.h": "h"},
    )
    work = tmp_path / "w-cpp"
    pg.place_reference(cpp, work)
    assert (work / "tree.h").read_text() == "h" and not (work / "tree.cpp").exists()

    rust = _exercise(
        tmp_path,
        "rust",
        {"solution": ["src/lib.rs", "Cargo.toml"], "example": [".meta/example.rs"]},
        {".meta/example.rs": "r", ".meta/Cargo-example.toml": "[dependencies]\n"},
    )
    work = tmp_path / "w-rust"
    pg.place_reference(rust, work)
    assert (work / "src/lib.rs").read_text() == "r"
    assert (work / "Cargo.toml").read_text() == "[dependencies]\n"


def test_build_outputs_are_removed_after_the_exercise(tmp_path: Path) -> None:
    for rel in ("target/debug/x", "build/y", "node_modules/z", "src/keep.rs"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    pg.remove_build_outputs(tmp_path)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()) == [
        "src/keep.rs"
    ]


@pytest.mark.usefixtures("exercise")
def test_satsuma_gets_the_effort_through_the_copied_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, object]] = []

    def fake_satsuma(
        prompt: str, workdir: Path, model: str, timeout_s: float, log: Path, resume: str = ""
    ) -> tuple[dict[str, object], str, float]:
        local = workdir / ".satsuma" / "settings.local.json"
        seen.append(json.loads(local.read_text()))
        (workdir / "two_fer.py").write_text(_SOLVED)
        return {"session_id": "s"}, "", 1.0

    monkeypatch.setattr(pg.wb, "run_satsuma", fake_satsuma)
    pg.solve({"id": "python/two-fer", "prompt": "x"}, "m", tmp_path / "out", effort="high")
    assert seen[0]["provider"]["sampling"]["reasoning_effort"] == "high"  # type: ignore[index]


def test_relative_places_reach_the_harnesses_as_absolute_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """結果の置き場所を相対パスで渡しても、作業ディレクトリで起動するハーネスが見つけられる。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pg.wb, "ollama_env", lambda: {"OLLAMA_HOST": "http://h:11434"})
    env = pg.wb.opencode_env(Path("out/logs/opencode-home"), "m")
    assert Path(env["XDG_CONFIG_HOME"]).is_absolute()
    assert (Path(env["XDG_CONFIG_HOME"]) / "opencode" / "opencode.json").is_file()
    seen = tmp_path / "seen.json"
    monkeypatch.setattr(pg.wb, "AIDER_BIN", str(_fake_aider_python(tmp_path, seen)))
    monkeypatch.setattr(pg, "loaded_context", lambda model: 131072)
    work = tmp_path / "work"
    work.mkdir()
    ex = pg.Exercise("python", "x", work, ["a.py"], [], [])
    pg.run_aider("x", ex, work, "m", Path("out/logs/x-1.log"), restore=False)
    argv = json.loads(seen.read_text())[0]
    assert Path(_arg(argv, "--message-file")).is_absolute()
    assert Path(_arg(argv, "--history")).is_absolute()
