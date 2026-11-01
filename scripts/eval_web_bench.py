#!/usr/bin/env python3
"""Web-Bench の 1 プロジェクトを SATSUMAESTRO で通す実践評価。"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from satsumaestro.evals.models import EvalReport, TaskResult  # noqa: E402
from satsumaestro.session.models import TokenUsage  # noqa: E402
from satsumaestro.util.clock import now_ms  # noqa: E402

BENCH = REPO / ".satsuma" / "benchmarks" / "web-bench"
RUN_ROOT = REPO / ".satsuma" / "benchmarks" / "web-bench-run"
WORK_ROOT = REPO / ".satsuma" / "benchmarks" / "web-bench-work"

PREFACE = (
    "You are working inside the project directory (the current working directory). "
    "Implement the task below by creating or editing the project files in place "
    "(HTML / CSS / JavaScript). Do not ask questions and do not only describe the change: "
    "make the files.\n\n"
)
ERROR_HEAD_CHARS = 16000
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_PROGRESS_RE = re.compile(r"^\[\d+/\d+\]")
DEFAULT_ATTEMPT_TIMEOUT_S = 20 * 60
TEST_TIMEOUT_S = 10 * 60
TEST_INFRA_MARKS = ("browserType.launch", "EADDRINUSE")

GIT_IDENT = ["-c", "user.name=web-bench", "-c", "user.email=web-bench@example.invalid"]
GIT_EXCLUDES = (".satsuma/", "node_modules/", ".aider*")
SETTINGS_SKIP_KEYS = frozenset({"mcp_servers"})
SETTINGS_FILES = ("settings.json", "settings.local.json")

AIDER_BIN = shutil.which("aider") or str(Path.home() / ".local" / "bin" / "aider")
OPENCODE_BIN = shutil.which("opencode") or "/opt/homebrew/bin/opencode"
OPENCODE_DENY_SHELL = (
    "rm",
    "rm *",
    "rmdir *",
    "unlink *",
    "shred *",
    "truncate *",
    "find * -delete*",
    "git clean*",
    "git rm *",
    "sudo *",
)
AIDER_EDIT_FORMAT = "diff"
_NOT_SOURCE_DIRS = frozenset({"node_modules", ".git", ".satsuma"})
_TOKENS_RE = re.compile(r"Tokens: (\S+) sent.*?, (\S+) received\.")
_REFLECTIONS_RE = re.compile(r"Only \d+ reflections allowed, stopping\.")


@dataclass
class Attempt:
    task_id: str
    attempt: int
    passed: bool
    agent_s: float
    test_s: float
    iterations: int
    input_tokens: int
    output_tokens: int
    stop_reason: str
    agent_error: str
    test_error_head: str
    no_changes: bool = False
    test_infra_error: str = ""
    session: str = ""
    permission_denied: int = 0


@dataclass
class TaskRecord:
    task_id: str
    level: str
    passed: bool
    attempts: list[Attempt] = field(default_factory=list)
    undetermined: bool = False


@dataclass(frozen=True)
class GradeOutcome:
    """1 試行ぶんの採点の結果。"""

    passed: bool
    error: str
    seconds: float
    infra: str = ""


def load_tasks(project: str) -> list[dict[str, str]]:
    path = BENCH / "projects" / project / "tasks.jsonl"
    tasks = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not tasks or tasks[0]["id"] != "init":
        raise SystemExit(f"tasks.jsonl の先頭が init ではありません: {path}")
    return tasks


def task_index(task_id: str) -> int:
    """test.sh の引数（0 = init.spec のみ、N = init ＋ task-1 … task-N）。"""
    return 0 if task_id == "init" else int(task_id.removeprefix("task-"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def ollama_env() -> dict[str, str]:
    """作業ディレクトリには .env が無いので、リポジトリの .env から OLLAMA_* を持ち込む。"""
    env = dict(os.environ)
    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OLLAMA_") and "=" in line and not line.startswith("#"):
                key, _, value = line.partition("=")
                if value.strip():
                    env[key.strip()] = value.strip()
    return env


def check_ollama(env: dict[str, str]) -> str | None:
    """``OLLAMA_HOST`` のサーバが応答するか（しなければ理由）。"""
    host = (env.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    base = host[: -len("/v1")] if host.endswith("/v1") else host
    if "://" not in base:
        base = "http://" + base
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=5) as resp:  # noqa: S310
            resp.read(1)
    except (OSError, ValueError) as e:
        return f"{base} に接続できません（{e}）"
    return None


def run_group(
    cmd: list[str], timeout_s: float, **popen: object
) -> subprocess.CompletedProcess[str]:
    """子プロセスを新しいプロセスグループで回し、時間切れならグループごと止める。"""
    with subprocess.Popen(cmd, start_new_session=True, **popen) as proc:  # type: ignore[call-overload]
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                out, err = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = "", ""
            raise subprocess.TimeoutExpired(cmd, timeout_s, output=out, stderr=err) from None
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _run_logged(
    cmd: list[str], workdir: Path, env: dict[str, str], timeout_s: float, log: Path
) -> tuple[subprocess.CompletedProcess[str] | None, float]:
    """子プロセスを回し、標準出力・標準エラーをそのまま ``log`` に残す（timeout なら None）。"""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        proc = run_group(
            cmd,
            timeout_s,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        log.write_text(f"$ {' '.join(cmd[:8])} …\n[timeout {timeout_s:.0f}s]\n{out}\n{err}")
        return None, time.monotonic() - started
    elapsed = time.monotonic() - started
    log.write_text(
        f"$ {' '.join(cmd[:8])} …\n[exit {proc.returncode} in {elapsed:.0f}s]\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
        encoding="utf-8",
    )
    return proc, elapsed


def cli_text(text: str) -> str:
    """コマンドの引数として渡す文。"""
    return "\n" + text if text.startswith("-") else text


def run_satsuma(
    prompt: str, workdir: Path, model: str, timeout_s: float, log: Path, resume: str = ""
) -> tuple[dict[str, object], str, float]:
    """SATSUMAESTRO を非対話で 1 回走らせ、JSON の結果・エラー文・所要秒を返す。"""
    cmd = [
        "uv",
        "run",
        "--project",
        str(REPO),
        "satsumaestro",
        "prompt",
        cli_text(prompt),
        "--model",
        model,
        "--mode",
        "code:danger-full-access",
        "--cwd",
        str(workdir),
        "--output-format",
        "json",
        "--trust",
    ]
    if resume:
        cmd += ["--resume", resume]
    started = time.time()
    proc, elapsed = _run_logged(cmd, workdir, ollama_env(), timeout_s, log)
    if proc is None:
        return {}, f"agent timeout ({timeout_s:.0f}s)", elapsed
    payload: dict[str, object] = {}
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    error = ""
    if proc.returncode != 0:
        error = f"exit {proc.returncode}: {proc.stderr.strip()[-600:]}"
        if not payload:
            payload = recover_from_session(workdir, started, proc.stderr)
    elif not payload:
        error = "no JSON result on stdout"
    return payload, error, elapsed


def recover_from_session(workdir: Path, since: float, stderr: str) -> dict[str, object]:
    """この試行のセッション記録から往復とトークンを数える（``since`` 以降に更新されたもの）。"""
    latest = latest_session(workdir, since)
    if latest is None:
        return {}
    iterations = input_tokens = output_tokens = 0
    for line in latest.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "message" or rec.get("role") != "assistant":
            continue
        iterations += 1
        usage = rec.get("usage") or {}
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
    stop = "max_iterations" if "max_iterations" in stderr else "error"
    return {
        "iterations": iterations,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "stop_reason": stop,
        "recovered_from_session": True,
    }


def source_files(workdir: Path) -> list[str]:
    """aider に載せるファイル（作業ディレクトリ相対）。"""
    files: list[str] = []
    for path in sorted(workdir.rglob("*")):
        rel = path.relative_to(workdir)
        if not path.is_file():
            continue
        if any(part in _NOT_SOURCE_DIRS or part.startswith(".aider") for part in rel.parts):
            continue
        files.append(str(rel))
    return files


def _tokens(text: str) -> int:
    """aider の丸め表記（``850`` / ``1.2k`` / ``12k``）を整数に。"""
    return int(float(text[:-1]) * 1000) if text.endswith("k") else int(float(text))


def ollama_api_base(env: dict[str, str]) -> str:
    """litellm（aider）用。"""
    host = env.get("OLLAMA_HOST", "http://localhost:11434")
    return host.rstrip("/").removesuffix("/v1")


def run_aider(
    prompt: str, workdir: Path, model: str, timeout_s: float, log: Path
) -> tuple[dict[str, object], str, float]:
    """aider を非対話（``--message``）で 1 回走らせ、SATSUMAESTRO と同じ形の結果を返す。"""
    env = ollama_env()
    env["OLLAMA_API_BASE"] = ollama_api_base(env)
    hist = log.with_suffix("")
    cmd = [
        AIDER_BIN,
        "--model",
        f"ollama_chat/{model}",
        "--edit-format",
        AIDER_EDIT_FORMAT,
        "--message",
        cli_text(prompt),
        "--yes-always",
        "--no-auto-commits",
        "--no-dirty-commits",
        "--no-attribute-author",
        "--no-attribute-committer",
        "--no-gitignore",
        "--no-show-model-warnings",
        "--no-check-update",
        "--analytics-disable",
        "--no-pretty",
        "--no-stream",
        "--no-fancy-input",
        "--no-detect-urls",
        "--no-suggest-shell-commands",
        "--no-restore-chat-history",
        "--chat-history-file",
        f"{hist}.chat.md",
        "--input-history-file",
        f"{hist}.input.txt",
        "--llm-history-file",
        f"{hist}.llm.txt",
    ]
    for rel in source_files(workdir):
        cmd += ["--file", rel]
    proc, elapsed = _run_logged(cmd, workdir, env, timeout_s, log)
    if proc is None:
        return {}, f"agent timeout ({timeout_s:.0f}s)", elapsed
    text = proc.stdout + "\n" + proc.stderr
    reports = _TOKENS_RE.findall(text)
    sent, received = reports[-1] if reports else ("0", "0")
    stop_reason = "end_turn"
    if _REFLECTIONS_RE.search(text):
        stop_reason = "max_reflections"
    payload: dict[str, object] = {
        "iterations": len(reports),
        "usage": {"input_tokens": _tokens(sent), "output_tokens": _tokens(received)},
        "stop_reason": stop_reason,
    }
    error = ""
    if proc.returncode != 0:
        error = f"exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-600:]}"
    elif not reports:
        error = "no LLM response (no 'Tokens:' line)"
    return payload, error, elapsed


def opencode_config(model: str, env: dict[str, str], effort: str = "") -> dict[str, object]:
    """OpenCode の設定。"""
    entry: dict[str, object] = {"name": model}
    if effort:
        entry["options"] = {"reasoningEffort": effort}
    shell: dict[str, str] = {"*": "allow"}
    shell.update(dict.fromkeys(OPENCODE_DENY_SHELL, "deny"))
    return {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "provider": {
            "ollama": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Ollama",
                "options": {"baseURL": ollama_api_base(env) + "/v1"},
                "models": {model: entry},
            }
        },
        "permission": {"edit": "allow", "bash": shell, "external_directory": "deny"},
    }


def opencode_env(home: Path, model: str, effort: str = "") -> dict[str, str]:
    """``home`` の下を OpenCode の設定・記録の置き場所にした環境（設定を書いてから返す）。"""
    home = home.resolve()
    env = ollama_env()
    config = home / "config" / "opencode" / "opencode.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(opencode_config(model, env, effort), indent=2), encoding="utf-8")
    for kind in ("config", "data", "state", "cache"):
        env[f"XDG_{kind.upper()}_HOME"] = str(home / kind)
    return env


def parse_opencode(stdout: str) -> tuple[dict[str, object], str]:
    """``opencode run --format json`` の出力（1 行 1 イベント）を、SATSUMAESTRO と同じ形にする。"""
    steps = input_tokens = output_tokens = tool_calls = denied = 0
    reason = session = ""
    errors: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = event.get("part") or {}
        kind = event.get("type")
        session = session or str(event.get("sessionID") or "")
        if kind == "step_finish":
            steps += 1
            tokens = part.get("tokens") or {}
            input_tokens += _as_int(tokens.get("input"))
            output_tokens += _as_int(tokens.get("output")) + _as_int(tokens.get("reasoning"))
            reason = str(part.get("reason") or reason)
        elif kind == "tool_use":
            tool_calls += 1
            state = part.get("state") or {}
            if "Permission denied" in str(state.get("error") or ""):
                denied += 1
        elif kind == "error":
            errors.append(json.dumps(event.get("error") or event, ensure_ascii=False)[:300])
    payload: dict[str, object] = {
        "iterations": steps,
        "tool_calls": tool_calls,
        "permission_denied": denied,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "stop_reason": reason or "error",
        "session_id": session,
    }
    error = "; ".join(errors)
    if not error and steps == 0:
        error = "no LLM response (no step_finish event)"
    return payload, error


def opencode_replies(export: dict[str, object], since_ms: float) -> dict[str, object] | None:
    """``opencode session export`` の会話から、``since_ms`` 以後の応答の数・トークン・止まり方。"""
    messages = export.get("messages")
    if not isinstance(messages, list):
        return None
    replies = [
        m
        for m in messages
        if isinstance(m, dict)
        and m.get("type") == "assistant"
        and _as_int((m.get("time") or {}).get("created")) >= since_ms
    ]
    if not replies:
        return None
    tokens = [m.get("tokens") or {} for m in replies]
    return {
        "iterations": len(replies),
        "usage": {
            "input_tokens": sum(_as_int(t.get("input")) for t in tokens),
            "output_tokens": sum(
                _as_int(t.get("output")) + _as_int(t.get("reasoning")) for t in tokens
            ),
        },
        "stop_reason": str(replies[-1].get("finish") or "error"),
    }


def export_opencode(
    session: str, workdir: Path, env: dict[str, str], log: Path
) -> dict[str, object] | None:
    """OpenCode の会話を書き出す（記録は ``log`` の隣の ``.export.json``。できなければ None）。"""
    cmd = [OPENCODE_BIN, "session", "export", session, "--standalone"]
    try:
        proc = run_group(
            cmd,
            120,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except ValueError:
        return None
    log.with_suffix(".export.json").write_text(proc.stdout, encoding="utf-8")
    return data if isinstance(data, dict) else None


def run_opencode(
    prompt: str,
    workdir: Path,
    model: str,
    timeout_s: float,
    log: Path,
    session: str = "",
    effort: str = "",
) -> tuple[dict[str, object], str, float]:
    """OpenCode を非対話で 1 回走らせ、SATSUMAESTRO と同じ形の結果を返す。"""
    env = opencode_env(log.parent / "opencode-home", model, effort)
    env["PWD"] = str(workdir)
    cmd = [
        OPENCODE_BIN,
        "run",
        "--standalone",
        "--auto",
        "--format",
        "json",
        "--model",
        f"ollama/{model}",
    ]
    if session:
        cmd += ["--session", session]
    cmd.append(cli_text(prompt))
    started_ms = time.time() * 1000
    proc, elapsed = _run_logged(cmd, workdir, env, timeout_s, log)
    if proc is None:
        return {}, f"agent timeout ({timeout_s:.0f}s)", elapsed
    payload, error = parse_opencode(proc.stdout)
    exported = (
        export_opencode(str(payload["session_id"]), workdir, env, log)
        if payload["session_id"]
        else None
    )
    replies = opencode_replies(exported, started_ms) if exported else None
    if replies:
        payload.update(replies)
        if error.startswith("no LLM response"):
            error = ""
    if proc.returncode != 0:
        error = f"exit {proc.returncode}: {(proc.stderr or error).strip()[-600:]}"
    return payload, error, elapsed


def outside_changes(repo: Path) -> frozenset[str]:
    """親のリポジトリの作業ツリーで、変更・追加として見えているパス（gitignore の分は見えない）。"""
    out = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return frozenset(line[3:] for line in out.splitlines() if line.strip())


def check_no_escape(before: frozenset[str], after: frozenset[str], log: Path) -> None:
    """試行の間に親のリポジトリへ増えた変更があれば、それ以上回さずに止める。"""
    escaped = sorted(after - before)
    if escaped:
        raise SystemExit(
            "エージェントが作業ディレクトリの外（親のリポジトリ）に書き込みました。評価を止めます: "
            f"{', '.join(escaped[:10])}（ログ: {log}）"
        )


def run_agent(
    agent: str, prompt: str, workdir: Path, model: str, timeout_s: float, log: Path
) -> tuple[dict[str, object], str, float]:
    if agent == "aider":
        return run_aider(prompt, workdir, model, timeout_s, log)
    if agent == "opencode":
        return run_opencode(prompt, workdir, model, timeout_s, log)
    return run_satsuma(prompt, workdir, model, timeout_s, log)


def ensure_run_env(project: str) -> None:
    """採点のテスト環境（``RUN_ROOT/<project>``）が無ければ作る（GPU は使わない）。"""
    run_dir = RUN_ROOT / project
    if (run_dir / "node_modules").is_dir():
        return
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(BENCH / "projects" / project, run_dir)
    pkg_path = run_dir / "package.json"
    pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
    for section in ("dependencies", "devDependencies"):
        for name, spec in pkg.get(section, {}).items():
            if name.startswith("@web-bench/") and spec.startswith("workspace:"):
                lib = BENCH / "libraries" / name.removeprefix("@web-bench/")
                pkg[section][name] = "file:" + os.path.relpath(lib, run_dir)
    pkg_path.write_text(json.dumps(pkg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"テスト環境を作る: {run_dir}（npm install）", flush=True)
    subprocess.run(
        ["npm", "install", "--no-audit", "--no-fund"], cwd=run_dir, check=True, capture_output=True
    )


def ensure_test_env(project: str) -> str:
    """採点の前にテスト環境を点検し、壊れていれば直す。"""
    run_dir = RUN_ROOT / project
    if not (run_dir / "node_modules" / "@playwright" / "test").is_dir():
        raise SystemExit(
            f"テスト環境に @playwright/test がありません（{run_dir} で npm install し直す）"
        )
    util = run_dir / "node_modules" / "@web-bench" / "test-util"
    if util.is_dir() and not util.is_symlink():
        return ""
    if util.is_symlink() or util.exists():
        util.unlink()
    shutil.copytree(BENCH / "libraries" / "test-util", util)
    return f"テスト環境を修復: {util.relative_to(run_dir)} を実体コピーに戻した"


def run_tests(project: str, workdir: Path, index: int) -> GradeOutcome:
    """公式と同じ ``npm run test -- <index>``（累積）を、作業ディレクトリに向けて回す。"""
    run_dir = RUN_ROOT / project
    repaired = ensure_test_env(project)
    if repaired:
        print(f"  ! {repaired}", flush=True)
    env = dict(os.environ)
    env.update(
        {
            "EVAL_PROJECT_ROOT": str(workdir),
            "EVAL_PROJECT_PORT": str(free_port()),
            "IS_EVAL_PRODUCTION": "true",
            "EVAL": "true",
            "FORCE_COLOR": "0",
        }
    )
    started = time.monotonic()
    try:
        proc = run_group(
            ["npm", "run", "test", "--", str(index)],
            TEST_TIMEOUT_S,
            cwd=run_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return GradeOutcome(
            False,
            "Test execution timed out. Check for infinite loops or high code complexity.",
            time.monotonic() - started,
        )
    elapsed = time.monotonic() - started
    if proc.returncode == 0:
        return GradeOutcome(True, "", elapsed)
    return GradeOutcome(
        False,
        error_context(proc.stdout, proc.stderr, run_dir, workdir),
        elapsed,
        infra=infra_failure_mark(proc.stdout + proc.stderr),
    )


def infra_failure_mark(output: str) -> str:
    """採点の出力に、テスト環境の失敗の印があればそれを返す（無ければ空）。"""
    return next((mark for mark in TEST_INFRA_MARKS if mark in output), "")


def grade_attempt(run: Callable[[], GradeOutcome]) -> GradeOutcome:
    """採点する。"""
    first = run()
    if not first.infra:
        return first
    print(f"  ! テスト環境の失敗（{first.infra}）。試行に数えずにテストだけ回し直す", flush=True)
    again = run()
    return GradeOutcome(again.passed, again.error, first.seconds + again.seconds, again.infra)


def has_changes(workdir: Path) -> bool:
    """作業ディレクトリに、前のコミットからの変更があるか（GIT_EXCLUDES は数えない）。"""
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=workdir, capture_output=True, text=True, check=True
    ).stdout
    return bool(out.strip())


def latest_session(workdir: Path, since: float) -> Path | None:
    """``since`` 以降に更新された、いちばん新しいセッション記録。"""
    sessions = [
        p
        for p in (workdir / ".satsuma" / "sessions").glob("*/*.jsonl")
        if p.stat().st_mtime >= since
    ]
    return max(sessions, key=lambda p: p.stat().st_mtime) if sessions else None


def error_context(stdout: str, stderr: str, run_dir: Path, workdir: Path) -> str:
    """次の試行へ渡す失敗出力。"""
    text = stdout if stdout.strip() else stderr
    text = _ANSI_RE.sub("", text)
    text = text.replace(str(run_dir), "").replace(str(workdir), "src")
    lines = [
        ln
        for ln in text.splitlines()
        if ln.strip()
        and not _PROGRESS_RE.match(ln)
        and not ln.startswith("<")
        and "node_modules" not in ln
    ]
    return "\n".join(lines)[:ERROR_HEAD_CHARS]


def _as_int(value: object) -> int:
    """JSON の数値（無ければ 0）を int に。"""
    return int(value) if isinstance(value, int | float) else 0


def _usage(payload: dict[str, object]) -> dict[str, object]:
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else {}


def official_pass_counts(records: list[TaskRecord], retry: int) -> list[tuple[int, int]]:
    """公式と同じ pass@1 … pass@retry。"""
    tasks = [r for r in records if r.task_id != "init"]
    total = len(tasks)
    counts: list[tuple[int, int]] = []
    for n in range(1, retry + 1):
        first_n = [[a.passed for a in r.attempts][:n] for r in tasks]
        index = next((i for i, res in enumerate(first_n) if not any(res)), -1)
        counts.append((total if index < 0 else index, total))
    return counts


def task_result(record: TaskRecord) -> TaskResult:
    """課題 1 つの試行をまとめた結果（評価のレポートと同じ形）。"""
    failures: list[str] = []
    if record.undetermined:
        failures = [f"判定不能: テスト環境の失敗（{record.attempts[-1].test_infra_error}）"]
    elif not record.passed:
        failures = [f"attempt {len(record.attempts)}: {record.attempts[0].test_error_head[:200]}"]
    errors = [f"attempt {a.attempt}: {a.agent_error}" for a in record.attempts if a.agent_error]
    return TaskResult(
        task_id=record.task_id,
        passed=record.passed,
        category="HOW/system-design",
        failures=failures,
        error="; ".join(errors)[:ERROR_HEAD_CHARS] if errors and not record.passed else "",
        iterations=sum(a.iterations for a in record.attempts),
        usage=TokenUsage(
            input_tokens=sum(a.input_tokens for a in record.attempts),
            output_tokens=sum(a.output_tokens for a in record.attempts),
        ),
        duration_ms=int(sum(a.agent_s + a.test_s for a in record.attempts) * 1000),
    )


def run_notes(records: list[TaskRecord]) -> list[str]:
    """公式指標の横に添える注記（判定不能の課題・変更の無い試行）。"""
    notes: list[str] = []
    undetermined = [r.task_id for r in records if r.undetermined]
    if undetermined:
        notes.append(
            f"判定不能 {len(undetermined)} 課題（テスト環境の失敗。公式指標では通っていない扱い）: "
            + ", ".join(undetermined)
        )
    idle = [a for r in records for a in r.attempts if a.no_changes]
    if idle:
        failed = sum(1 for a in idle if not a.passed)
        notes.append(f"変更の無い試行 {len(idle)} 回（うち不合格 {failed} 回）")
    return notes


def build_prompt(description: str, error: str) -> str:
    prompt = f"{PREFACE}# Task\n{description}\n"
    if error:
        prompt += f"\n# Error Context\n{error}\n"
    return prompt


def run_git(workdir: Path, *args: str) -> None:
    """作業ディレクトリで git を 1 本回す（失敗は例外。出力は捨てる）。"""
    subprocess.run(
        ["git", *GIT_IDENT, *args], cwd=workdir, check=True, capture_output=True, text=True
    )


def commit_all(workdir: Path, message: str) -> None:
    """作業ディレクトリの変更を全部コミットする（変更が無くても空コミット）。"""
    run_git(workdir, "add", "-A")
    run_git(workdir, "commit", "-q", "--allow-empty", "-m", message)


def copy_settings(workdir: Path, effort: str = "") -> None:
    """リポジトリの ``.satsuma/settings*.json`` を作業ディレクトリの ``.satsuma/`` へ写す。"""
    state = workdir / ".satsuma"
    state.mkdir(exist_ok=True)
    for name in SETTINGS_FILES:
        src = REPO / ".satsuma" / name
        if not src.is_file():
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        data = {k: v for k, v in data.items() if k not in SETTINGS_SKIP_KEYS}
        (state / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if effort:
        local = state / "settings.local.json"
        data = json.loads(local.read_text(encoding="utf-8")) if local.is_file() else {}
        data.setdefault("provider", {}).setdefault("sampling", {})["reasoning_effort"] = effort
        local.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_workdir(project: str, run_id: str) -> Path:
    """``src-init`` を写し、git リポジトリにして、リポジトリの設定を写す（冒頭の「隔離」）。"""
    workdir = WORK_ROOT / project / run_id / "src"
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(BENCH / "projects" / project / "src-init", workdir)
    run_git(workdir, "init", "-q")
    (workdir / ".git" / "info" / "exclude").write_text(
        "\n".join(GIT_EXCLUDES) + "\n", encoding="utf-8"
    )
    copy_settings(workdir)
    commit_all(workdir, "src-init")
    return workdir


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--project", default="survey")
    ap.add_argument(
        "--agent",
        choices=["satsuma", "aider", "opencode"],
        default="satsuma",
        help=(
            "課題を解くハーネス（既定 satsuma。aider は uv tool install aider-chat、"
            "opencode は brew install opencode）"
        ),
    )
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--start", default="init", help="この課題から（既定 init）")
    ap.add_argument("--end", default="", help="この課題まで（既定 最後）")
    ap.add_argument(
        "--retry", type=int, default=2, help="1 課題の試行回数（公式既定 2 = 1 回だけ再試行）"
    )
    ap.add_argument("--attempt-timeout", type=float, default=DEFAULT_ATTEMPT_TIMEOUT_S)
    ap.add_argument("--dry-run", choices=["reference", "empty"], default="")
    ap.add_argument("--stop-on-fail", action="store_true", help="課題が落ちたら以降を回さない")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not (BENCH / "projects" / args.project).is_dir():
        raise SystemExit(
            f"Web-Bench のプロジェクトが見つかりません: {BENCH / 'projects' / args.project}"
        )
    ensure_run_env(args.project)

    tasks = load_tasks(args.project)
    ids = [t["id"] for t in tasks]
    lo = ids.index(args.start)
    hi = ids.index(args.end) if args.end else len(ids) - 1
    tasks = tasks[lo : hi + 1]

    if args.agent == "aider" and not args.dry_run and not Path(AIDER_BIN).exists():
        raise SystemExit(f"aider が見つかりません: {AIDER_BIN}（uv tool install aider-chat）")
    if args.agent == "opencode" and not args.dry_run and not Path(OPENCODE_BIN).exists():
        raise SystemExit(f"OpenCode が見つかりません: {OPENCODE_BIN}（brew install opencode）")
    if not args.dry_run:
        env = ollama_env()
        problem = check_ollama(env)
        if problem is not None:
            raise SystemExit(f"Ollama に接続できません: {problem}（.env の OLLAMA_HOST を確認）")
        print(f"ollama={env.get('OLLAMA_HOST')}", flush=True)

    run_id = time.strftime("%Y%m%dT%H%M%S")
    workdir = prepare_workdir(args.project, run_id)
    log_dir = workdir.parent / "logs"
    if args.dry_run == "reference":
        shutil.copytree(BENCH / "projects" / args.project / "src", workdir, dirs_exist_ok=True)
        commit_all(workdir, "reference solution")
    print(
        f"project={args.project} agent={args.agent} model={args.model} "
        f"tasks={tasks[0]['id']}..{tasks[-1]['id']} workdir={workdir}",
        flush=True,
    )
    settings = {
        name: json.loads((workdir / ".satsuma" / name).read_text(encoding="utf-8"))
        for name in SETTINGS_FILES
        if (workdir / ".satsuma" / name).is_file()
    }
    print(f"settings={json.dumps(settings, ensure_ascii=False)}", flush=True)

    records: list[TaskRecord] = []
    results: list[TaskResult] = []
    started_ms = now_ms()
    for task in tasks:
        task_id = str(task["id"])
        record = TaskRecord(task_id=task_id, level=str(task.get("level", "")), passed=False)
        error = ""
        for attempt in range(1, args.retry + 1):
            payload: dict[str, object] = {}
            agent_error = ""
            agent_s = 0.0
            began = time.time()
            if not args.dry_run:
                log = log_dir / f"{task_id}-{attempt}.log"
                before = outside_changes(REPO)
                payload, agent_error, agent_s = run_agent(
                    args.agent,
                    build_prompt(str(task["description"]), error),
                    workdir,
                    args.model,
                    args.attempt_timeout,
                    log,
                )
                check_no_escape(before, outside_changes(REPO), log)
            no_changes = not args.dry_run and not has_changes(workdir)
            session = latest_session(workdir, began) if args.agent == "satsuma" else None
            outcome = grade_attempt(
                functools.partial(run_tests, args.project, workdir, task_index(task_id))
            )
            usage = _usage(payload)
            record.attempts.append(
                Attempt(
                    task_id=task_id,
                    attempt=attempt,
                    passed=outcome.passed,
                    agent_s=round(agent_s, 1),
                    test_s=round(outcome.seconds, 1),
                    iterations=_as_int(payload.get("iterations")),
                    input_tokens=_as_int(usage.get("input_tokens")),
                    output_tokens=_as_int(usage.get("output_tokens")),
                    stop_reason=str(payload.get("stop_reason", "")),
                    agent_error=agent_error,
                    test_error_head=outcome.error[:600],
                    no_changes=no_changes,
                    test_infra_error=outcome.infra,
                    session=str(session.relative_to(workdir)) if session else "",
                    permission_denied=_as_int(payload.get("permission_denied")),
                )
            )
            flag = "pass" if outcome.passed else ("ENV" if outcome.infra else "FAIL")
            commit_all(workdir, f"{task_id} attempt {attempt}: {flag}")
            print(
                f"{task_id:8} attempt {attempt}: {flag:4} agent {agent_s:6.0f}s "
                f"test {outcome.seconds:4.0f}s iters {record.attempts[-1].iterations} "
                f"out_tok {record.attempts[-1].output_tokens}"
                f"{' no_changes' if no_changes else ''} "
                f"{agent_error[:80]}",
                flush=True,
            )
            if outcome.passed:
                record.passed = True
                break
            if outcome.infra:
                record.undetermined = True
                print(f"  ! {task_id} は判定不能（テスト環境の失敗）", flush=True)
                break
            error = outcome.error or agent_error
            if args.dry_run:
                break
        records.append(record)
        results.append(task_result(record))
        if args.stop_on_fail and not record.passed and not record.undetermined:
            print("stop-on-fail: 以降の課題は回しません", flush=True)
            break

    report = EvalReport(
        agent=args.agent,
        model=args.model if not args.dry_run else f"dry-run:{args.dry_run}",
        started_at_ms=started_ms,
        suite=f"web-bench/{args.project}",
        results=results,
    )
    passed_count = sum(1 for r in results if r.passed)
    total_s = sum(r.duration_ms for r in results) / 1000
    official = " ・ ".join(
        f"pass@{n} {c}/{t} ({c / t * 100:.0f}%)" if t else f"pass@{n} -"
        for n, (c, t) in enumerate(official_pass_counts(records, args.retry), start=1)
    )
    print(f"\n公式指標: {official}")
    print(
        f"合格した課題 {passed_count}/{len(results)}（init 込み・再試行込み）"
        f" ・ 所要 {total_s:.0f} 秒 ・ workdir {workdir}"
    )
    for line in run_notes(records):
        print(line)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        detail = out.with_name(out.stem + ".attempts.json")
        detail.write_text(
            json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"レポート: {out}（試行の詳細: {detail}）")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
