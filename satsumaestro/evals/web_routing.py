"""Web Routing 評価（web_search を使うかの判断の精度）。"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import tomllib
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from satsumaestro.config.models import HookConfig, RuntimeConfig, WebFetchConfig, WebSearchConfig
from satsumaestro.engine.runtime import ConversationRuntime
from satsumaestro.evals.models import EvalReport, TaskResult
from satsumaestro.hooks.runner import HookRunner
from satsumaestro.i18n.text import ui
from satsumaestro.prompt.builder import SystemPromptBuilder, format_tool_catalog
from satsumaestro.prompt.instructions import instruction_tokens
from satsumaestro.providers.factory import build_provider
from satsumaestro.providers.registry import expand_model_name
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.security.policy import default_policy
from satsumaestro.session.models import TokenUsage, ToolUseBlock
from satsumaestro.session.store import JSONLSessionStore, new_session
from satsumaestro.tools import default_registry, unavailable_tools
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import ConfigError

_log = logging.getLogger(__name__)

SUITE_NAME = "web_routing"
PATTERNS = ("must", "skip", "augment")


@dataclass
class RoutingTask:
    id: str
    pattern: str
    prompt: str
    contains_any: list[str]
    search_keyword: str = ""
    page_html: str = ""


@dataclass
class RunOutcome:
    """1 回の実行の生記録。"""

    task_id: str
    pattern: str
    searched: bool
    tool_calls: int
    checks_ok: bool
    misrouted: bool
    duration_ms: int
    final_head: str = ""
    error: str = ""
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def success(self) -> bool:
        """成功条件（実行エラー無し ＋ 経路 ＋ 回答）。"""
        if self.error:
            return False
        if self.pattern == "must":
            return self.searched and self.checks_ok
        if self.pattern == "skip":
            return self.tool_calls == 0 and self.checks_ok
        return self.checks_ok

    @property
    def failures(self) -> list[str]:
        found: list[str] = []
        if self.pattern == "must" and not self.searched:
            found.append("misrouted: web_search was not called")
        if self.pattern == "skip" and self.tool_calls != 0:
            found.append(f"misrouted: {self.tool_calls} tool call(s) for a no-tool task")
        if not self.checks_ok:
            found.append("checks: answer does not contain any expected token")
        return found


def load_routing_tasks(suite: Path) -> list[RoutingTask]:
    tasks: list[RoutingTask] = []
    for path in sorted(suite.glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigError(
                ui("evals.web_routing.cannot_read_the_web_routing", path=path, error=e)
            ) from e
        pattern = str(data.get("pattern", ""))
        if pattern not in PATTERNS:
            raise ConfigError(
                ui("evals.web_routing.invalid_pattern_in_the_web", path=path, pattern=pattern)
            )
        tasks.append(
            RoutingTask(
                id=str(data["id"]),
                pattern=pattern,
                prompt=str(data["prompt"]),
                contains_any=[str(x) for x in data.get("checks", {}).get("contains_any", [])],
                search_keyword=str(data.get("search_keyword", "")),
                page_html=str(data.get("page_html", "")),
            )
        )
    if not tasks:
        raise ConfigError(ui("evals.web_routing.no_web_routing_tasks_toml", suite=suite))
    return tasks


class _MockHandler(BaseHTTPRequestHandler):
    tasks: dict[str, RoutingTask] = {}
    port: int = 0

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/search":
            q = urllib.parse.parse_qs(parsed.query).get("q", [""])[0].lower()
            results = [
                {
                    "title": f"{t.id} reference",
                    "url": f"http://127.0.0.1:{self.port}/page/{keyword}",
                }
                for keyword, t in self.tasks.items()
                if keyword and keyword in q
            ]
            self._reply(json.dumps({"results": results}).encode(), "application/json")
            return
        if parsed.path.startswith("/page/"):
            task = self.tasks.get(parsed.path.removeprefix("/page/"))
            if task is not None:
                self._reply(task.page_html.encode(), "text/html; charset=utf-8")
                return
        self.send_response(404)
        self.end_headers()

    def _reply(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return


def start_mock_server(tasks: list[RoutingTask]) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    _MockHandler.tasks = {t.search_keyword: t for t in tasks if t.search_keyword}
    _MockHandler.port = port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, port


def _tools_used(runtime: ConversationRuntime) -> list[str]:
    return [
        block.name
        for msg in runtime.session.messages
        for block in msg.blocks
        if isinstance(block, ToolUseBlock)
    ]


async def run_once(task: RoutingTask, model: str, port: int, workspace: Path) -> RunOutcome:
    cfg = RuntimeConfig(
        web_search=WebSearchConfig(provider="generic", base_url=f"http://127.0.0.1:{port}/search"),
        web_fetch=WebFetchConfig(enabled=True, allow_private_hosts=True, confirm="never"),
    )
    tools = default_registry(core_override=cfg.tools.core, unavailable=unavailable_tools(cfg))
    resolved = expand_model_name(model)
    provider = build_provider(cfg, resolved, os.environ)
    runtime = ConversationRuntime(
        session=new_session(workspace, resolved),
        provider=provider,
        tools=tools,
        policy=default_policy(PermissionMode.READ_ONLY),
        prompt_builder=SystemPromptBuilder(
            instruction_tokens=instruction_tokens(cfg.compaction.context_window),
            tool_catalog=format_tool_catalog(
                tools.deferred_catalog(),
                run_on_first_call=cfg.tools.deferred_run_on_first_call,
            ),
        ),
        store=JSONLSessionStore(),
        hooks=HookRunner(HookConfig(), workspace),
        config=cfg,
    )
    start = time.monotonic()
    error = ""
    iterations = 0
    usage = TokenUsage()
    try:
        summary = await runtime.run_turn(task.prompt)
        final = summary.final_text or summary.last_text
        tool_calls = summary.tool_calls
        iterations = summary.iterations
        usage = summary.usage
    except Exception as e:
        final = ""
        tool_calls = -1
        error = f"{type(e).__name__}: {e}"
    duration_ms = int((time.monotonic() - start) * 1000)
    used = _tools_used(runtime)
    searched = "web_search" in used
    normalized = re.sub(r"[\s   ]+", " ", final)
    checks_ok = (not task.contains_any) or any(m in normalized for m in task.contains_any)
    if task.pattern == "must":
        misrouted = not searched
    elif task.pattern == "skip":
        misrouted = tool_calls != 0
    else:
        misrouted = False
    return RunOutcome(
        task_id=task.id,
        pattern=task.pattern,
        searched=searched,
        tool_calls=tool_calls,
        checks_ok=checks_ok,
        misrouted=misrouted,
        duration_ms=duration_ms,
        final_head=final[:120].replace("\n", " "),
        error=error,
        tools_used=used,
        iterations=iterations,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def to_task_result(outcome: RunOutcome) -> TaskResult:
    """生記録を共通レポートの 1 行へ（``passed`` は成功条件）。"""
    return TaskResult(
        task_id=outcome.task_id,
        passed=outcome.success,
        category="HOW/api-framework-support",
        failures=[] if outcome.success else outcome.failures,
        error=outcome.error,
        final_text=outcome.final_head,
        iterations=max(outcome.iterations, 0),
        tool_calls=max(outcome.tool_calls, 0),
        usage=TokenUsage(input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens),
        duration_ms=outcome.duration_ms,
    )


def summarize(outcomes: list[RunOutcome]) -> str:
    """検索率・誤ルーティング率・回答正解率を個別に出す。"""
    total = len(outcomes)
    if not total:
        return ui("evals.web_routing.web_routing_no_records")
    misroutes = sum(1 for o in outcomes if o.misrouted)
    check_fails = sum(1 for o in outcomes if not o.checks_ok)
    errors = sum(1 for o in outcomes if o.error)
    line = ui(
        "evals.web_routing.misrouted_wrong_answers_run_errors",
        misroutes=misroutes,
        total=total,
        check_fails=check_fails,
        errors=errors,
    )
    aug = [o for o in outcomes if o.pattern == "augment"]
    if aug:
        searched = f"{sum(o.searched for o in aug)}/{len(aug)}"
        line += ui("evals.web_routing.augment_search_rate_recorded_only", searched=searched)
    return line


async def run_web_routing(
    suite: Path, model: str, *, only: set[str] | None = None
) -> tuple[EvalReport, list[RunOutcome]]:
    """スイートを 1 回走らせ、共通レポートと生記録を返す（反復は呼び出し側の ``--runs``）。"""
    tasks = load_routing_tasks(suite)
    if only is not None:
        unknown = sorted(only - {t.id for t in tasks})
        if unknown:
            ids = ", ".join(unknown)
            raise ConfigError(ui("evals.web_routing.task_ids_not_in_web", ids=ids))
        tasks = [t for t in tasks if t.id in only]
    resolved = expand_model_name(model)
    server, port = start_mock_server(tasks)
    outcomes: list[RunOutcome] = []
    started = now_ms()
    try:
        with tempfile.TemporaryDirectory(prefix="web-routing-") as tmp:
            for task in tasks:
                outcome = await run_once(task, model, port, Path(tmp))
                outcomes.append(outcome)
                flag = (
                    "MISROUTE"
                    if outcome.misrouted
                    else ("ok" if outcome.checks_ok else "CHECK_FAIL")
                )
                _log.info(
                    "%-7s %-16s %-10s searched=%s tools=%s (%dms) %s",
                    task.pattern,
                    task.id,
                    flag,
                    outcome.searched,
                    outcome.tool_calls,
                    outcome.duration_ms,
                    outcome.error,
                )
    finally:
        server.shutdown()
    report = EvalReport(
        agent="satsuma",
        model=resolved,
        started_at_ms=started,
        suite=SUITE_NAME,
        results=[to_task_result(o) for o in outcomes],
    )
    return report, outcomes


def write_outcomes(outcomes: list[RunOutcome], out: Path) -> None:
    """生記録の JSON（``RunOutcome`` のリスト）を書く。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([o.__dict__ for o in outcomes], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
