"""Aider Polyglot の題を、ハーネス（SATSUMAESTRO・aider・OpenCode）に解かせて採点する。"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import eval_web_bench as wb  # noqa: E402

from satsumaestro.evals.models import TaskResult  # noqa: E402
from satsumaestro.session.models import TokenUsage  # noqa: E402

AIDER_REV = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
POLYGLOT_DIR = REPO / ".satsuma" / "benchmarks" / "aider-polyglot"
WORK_ROOT = REPO / ".satsuma" / "benchmarks" / "polyglot-work"
AIDER_RUNNER = Path(__file__).resolve().parent / "aider_runner.py"
AGENTS = ("satsuma", "aider", "opencode")
JDK_HOME = Path("/opt/homebrew/opt/openjdk@21")
CPP_TEST = """set -e
[ ! -d "build" ] && mkdir build
cd build
cmake -DEXERCISM_RUN_ALL_TESTS=1 -G "Unix Makefiles" ..
make
"""

INSTRUCTIONS_ADDENDUM = """
####

Use the above instructions to modify the supplied files: {file_list}
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.
"""  # noqa: E501
TEST_FAILURES = """
####

See the testing errors above.
The tests are correct, don't try and change them.
Fix the code in {file_list} to resolve the errors.
"""
TRIES = 2
TEST_TIMEOUT_S = 3 * 60
ATTEMPT_TIMEOUT_S = 20 * 60
NOT_SOLUTION = frozenset({"CMakeLists.txt", "Cargo.toml"})
GIT_EXCLUDES = (
    ".satsuma/",
    "node_modules/",
    "target/",
    "build/",
    ".gradle/",
    "__pycache__/",
    ".pytest_cache/",
    ".aider*",
)
_XTEST = re.compile(r"\bxtest\(")
_DISABLED = re.compile(r"@Disabled\([^)]*\)\s*\n")
_TIMING = re.compile(r"\bin \d+\.\d+s\b")


def test_command(lang: str) -> list[str]:
    """公式の命令。"""
    if lang == "python":
        return [sys.executable, "-m", "pytest", "-c", os.devnull, "--rootdir", "."]
    if lang == "rust":
        return ["cargo", "test", "--", "--include-ignored"]
    if lang == "javascript":
        return ["npm", "run", "test"]
    if lang == "go":
        return ["go", "test", "./..."]
    if lang == "java":
        return ["./gradlew", "test"]
    if lang == "cpp":
        return ["bash", "-c", CPP_TEST]
    raise SystemExit(f"{lang} の採点の命令がありません")


def use_toolchains() -> None:
    """採点とハーネスが使う道具を通す（子プロセスは ``os.environ`` を引き継ぐ）。"""
    if "JAVA_HOME" not in os.environ and JDK_HOME.is_dir():
        os.environ["JAVA_HOME"] = str(JDK_HOME)
        os.environ["PATH"] = f"{JDK_HOME / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("GRADLE_OPTS", "-Dorg.gradle.daemon=false")


@dataclass
class Exercise:
    lang: str
    name: str
    src: Path
    solution: list[str]
    tests: list[str]
    examples: list[str]

    @property
    def file_list(self) -> str:
        return " ".join(Path(f).name for f in self.solution)


def load_exercise(task_id: str) -> Exercise:
    lang, name = task_id.split("/", 1)
    src = POLYGLOT_DIR / lang / "exercises" / "practice" / name
    files = json.loads((src / ".meta" / "config.json").read_text(encoding="utf-8"))["files"]
    tests = list(files.get("test", []))
    examples = list(files.get("example", []))
    skip = NOT_SOLUTION | set(tests) | set(examples)
    solution = [
        f
        for f in files.get("solution", [])
        if f not in skip and not f.startswith((".meta/", ".docs/"))
    ]
    return Exercise(lang, name, src, solution, tests, examples)


def first_prompt(prompt: str, ex: Exercise) -> str:
    return prompt + INSTRUCTIONS_ADDENDUM.format(file_list=ex.file_list)


def retry_prompt(test_output: str, ex: Exercise) -> str:
    return test_output + TEST_FAILURES.format(file_list=ex.file_list)


def prepare_workdir(ex: Exercise, workdir: Path, effort: str = "") -> None:
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(ex.src, workdir, ignore=shutil.ignore_patterns(".meta", ".docs"))
    wb.run_git(workdir, "init", "-q")
    (workdir / ".git" / "info" / "exclude").write_text(
        "\n".join(GIT_EXCLUDES) + "\n", encoding="utf-8"
    )
    wb.copy_settings(workdir, effort)
    wb.commit_all(workdir, "exercise")


def place_reference(ex: Exercise, workdir: Path) -> None:
    """公式の解答の見本を、解答のファイルの位置に置く（``--dry-run reference``）。"""
    if not ex.examples or not ex.solution:
        raise SystemExit(f"{ex.lang}/{ex.name}: 見本か解答のファイルがありません")
    for example in ex.examples:
        name = Path(example).name
        suffix = Path(example).suffix
        same_suffix = [f for f in ex.solution if Path(f).suffix == suffix]
        target = next((f for f in ex.solution if Path(f).name == name), None)
        single = len(ex.examples) == 1 or name.startswith("example")
        if target is None and single and len(same_suffix) == 1:
            target = same_suffix[0]
        dst = workdir / target if target else workdir / Path(ex.solution[0]).parent / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ex.src / example, dst)
    cargo = ex.src / ".meta" / "Cargo-example.toml"
    if cargo.is_file():
        shutil.copy(cargo, workdir / "Cargo.toml")


def ensure_js_env() -> None:
    """JavaScript の採点の依存を ``WORK_ROOT/node_modules`` に入れる（無いときだけ・数分）。"""
    work_root = WORK_ROOT
    if (work_root / "node_modules" / "jest").is_dir():
        return
    work_root.mkdir(parents=True, exist_ok=True)
    sample = next((POLYGLOT_DIR / "javascript" / "exercises" / "practice").iterdir())
    dev = json.loads((sample / "package.json").read_text(encoding="utf-8"))["devDependencies"]
    (work_root / "package.json").write_text(
        json.dumps({"private": True, "devDependencies": dev}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"JavaScript の採点の依存を入れる: {work_root}（npm install）", flush=True)
    subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"],
        cwd=work_root,
        check=True,
        capture_output=True,
    )


def cleanup_test_output(output: str, workdir: Path) -> str:
    """公式と同じく、所要秒と作業ディレクトリの絶対パスを消す（試行ごとに文が揺れないように）。"""
    return _TIMING.sub("", output).replace(str(workdir), workdir.name)


def remove_build_outputs(workdir: Path) -> None:
    """公式と同じく、1 題が終わったらビルドの置き場所を消す（全題でもディスクが膨らまない）。"""
    for rel in ("target/debug", "build", "node_modules"):
        shutil.rmtree(workdir / rel, ignore_errors=True)


def run_tests(ex: Exercise, workdir: Path) -> tuple[bool, str, float]:
    """テストのファイルを元に戻して採点する。"""
    for f in ex.tests:
        dst = workdir / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ex.src / f, dst)
        if ex.lang == "javascript":
            dst.write_text(_XTEST.sub("test(", dst.read_text(encoding="utf-8")), encoding="utf-8")
        if ex.lang == "java":
            dst.write_text(_DISABLED.sub("", dst.read_text(encoding="utf-8")), encoding="utf-8")
    started = time.monotonic()
    try:
        proc = wb.run_group(
            test_command(ex.lang),
            TEST_TIMEOUT_S,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return False, "Tests timed out!", time.monotonic() - started
    output = cleanup_test_output(proc.stdout, workdir)
    return proc.returncode == 0, output, time.monotonic() - started


@dataclass
class Attempt:
    attempt: int
    passed: bool
    agent_s: float
    test_s: float
    iterations: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    stop_reason: str
    agent_error: str
    test_output_head: str
    session: str


@functools.cache
def loaded_context(model: str) -> int:
    """サーバがそのモデルを読み込んでいる窓。"""
    base = wb.ollama_api_base(wb.ollama_env())

    def context() -> int:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=10) as resp:  # noqa: S310
            models = json.loads(resp.read()).get("models", [])
        return next((int(m.get("context_length") or 0) for m in models if m["name"] == model), 0)

    if not context():
        body = json.dumps({"model": model, "prompt": "", "stream": False}).encode()
        req = urllib.request.Request(
            f"{base}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310
            resp.read()
    window = context()
    if not window:
        raise SystemExit(f"{model} の窓を測れません（{base}/api/ps に出てこない）")
    return window


def run_aider(
    prompt: str,
    ex: Exercise,
    workdir: Path,
    model: str,
    log: Path,
    restore: bool,
    effort: str = "",
) -> tuple[dict[str, object], str, float]:
    """aider を公式の benchmark と同じ作りで 1 回走らせる（``aider_runner.py``）。"""
    env = wb.ollama_env()
    env["OLLAMA_API_BASE"] = wb.ollama_api_base(env)
    log = log.resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    message = log.with_suffix(".message.md")
    message.write_text(prompt, encoding="utf-8")
    cmd = [
        str(Path(wb.AIDER_BIN).resolve().parent / "python"),
        "-I",
        str(AIDER_RUNNER),
        "--model",
        f"ollama_chat/{model}",
        "--edit-format",
        wb.AIDER_EDIT_FORMAT,
        "--num-ctx",
        str(loaded_context(model)),
        "--message-file",
        str(message),
        "--history",
        str(log.parent / f"{ex.lang}-{ex.name}.aider.chat.md"),
    ]
    if restore:
        cmd.append("--restore")
    if effort:
        cmd += ["--reasoning-effort", effort]
    for f in sorted(NOT_SOLUTION | set(ex.tests) | set(ex.examples)):
        cmd += ["--ignore", f]
    cmd += ex.solution
    proc, elapsed = wb._run_logged(cmd, workdir, env, ATTEMPT_TIMEOUT_S, log)
    if proc is None:
        return {}, f"agent timeout ({ATTEMPT_TIMEOUT_S:.0f}s)", elapsed
    lines = proc.stdout.strip().splitlines()
    try:
        payload: dict[str, object] = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        payload = {}
    error = ""
    if proc.returncode != 0:
        error = f"exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-600:]}"
    elif not wb._as_int(payload.get("iterations")):
        error = "no LLM response"
    return payload, error, elapsed


def run_harness(
    agent: str,
    prompt: str,
    ex: Exercise,
    workdir: Path,
    model: str,
    log: Path,
    session: str,
    effort: str = "",
) -> tuple[dict[str, object], str, float]:
    """1 回の試行。"""
    if agent == "satsuma":
        return wb.run_satsuma(prompt, workdir, model, ATTEMPT_TIMEOUT_S, log, resume=session)
    if agent == "aider":
        return run_aider(prompt, ex, workdir, model, log, restore=bool(session), effort=effort)
    if agent == "opencode":
        return wb.run_opencode(
            prompt, workdir, model, ATTEMPT_TIMEOUT_S, log, session=session, effort=effort
        )
    raise SystemExit(f"知らないハーネス: {agent}")


def solve(
    task: dict[str, object],
    model: str,
    out: Path,
    dry_run: str = "",
    agent: str = "satsuma",
    effort: str = "",
) -> tuple[TaskResult, list[Attempt]]:
    """1 題を解かせて採点する。"""
    task_id = str(task["id"])
    ex = load_exercise(task_id)
    use_toolchains()
    if ex.lang == "javascript":
        ensure_js_env()
    workdir = WORK_ROOT / out.name / ex.lang / ex.name
    prepare_workdir(ex, workdir, effort)
    if dry_run == "reference":
        place_reference(ex, workdir)
    logs = out / "logs"
    prompt = first_prompt(str(task["prompt"]), ex)
    session = ""
    attempts: list[Attempt] = []
    for n in range(1, TRIES + 1):
        payload: dict[str, object] = {}
        agent_error, agent_s = "", 0.0
        if not dry_run:
            log = logs / f"{ex.lang}-{ex.name}-{n}.log"
            before = wb.outside_changes(REPO)
            payload, agent_error, agent_s = run_harness(
                agent, prompt, ex, workdir, model, log, session, effort
            )
            wb.check_no_escape(before, wb.outside_changes(REPO), log)
            new_session = "aider" if agent == "aider" and not agent_error else ""
            session = str(payload.get("session_id") or new_session or session)
        passed, output, test_s = run_tests(ex, workdir)
        wb.commit_all(workdir, f"attempt {n}: {'pass' if passed else 'FAIL'}")
        usage = wb._usage(payload)
        attempts.append(
            Attempt(
                attempt=n,
                passed=passed,
                agent_s=round(agent_s, 1),
                test_s=round(test_s, 1),
                iterations=wb._as_int(payload.get("iterations")),
                tool_calls=wb._as_int(payload.get("tool_calls")),
                input_tokens=wb._as_int(usage.get("input_tokens")),
                output_tokens=wb._as_int(usage.get("output_tokens")),
                stop_reason=str(payload.get("stop_reason", "")),
                agent_error=agent_error,
                test_output_head=output[:600],
                session=session,
            )
        )
        if passed or dry_run:
            break
        tail = retry_prompt(output[: wb.ERROR_HEAD_CHARS], ex)
        prompt = tail if session else f"{prompt}\n{tail}"
    remove_build_outputs(workdir)
    return to_result(task_id, ex.lang, attempts), attempts


def to_result(task_id: str, lang: str, attempts: list[Attempt]) -> TaskResult:
    passed = any(a.passed for a in attempts)
    errors = [f"attempt {a.attempt}: {a.agent_error}" for a in attempts if a.agent_error]
    return TaskResult(
        task_id=task_id,
        passed=passed,
        category=lang,
        failures=[] if passed else [f"attempt {attempts[-1].attempt}: tests failed"],
        error="; ".join(errors)[:2000] if errors and not passed else "",
        iterations=sum(a.iterations for a in attempts),
        tool_calls=sum(a.tool_calls for a in attempts),
        usage=TokenUsage(
            input_tokens=sum(a.input_tokens for a in attempts),
            output_tokens=sum(a.output_tokens for a in attempts),
        ),
        duration_ms=int(sum(a.agent_s + a.test_s for a in attempts) * 1000),
    )


def attempts_json(attempts: list[Attempt]) -> str:
    return json.dumps([asdict(a) for a in attempts], ensure_ascii=False, indent=2)
