"""評価中の異常を不具合記録として残す。"""

from __future__ import annotations

import datetime
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from satsumaestro.evals.models import EvalTask, TaskResult
from satsumaestro.util.paths import AGENT_DIRNAME

REPEAT_THRESHOLD = 5
EDIT_MISMATCH_DEFECT = 3
UNKNOWN_TOOL_DEFECT = 3
_FILED, _KINDS, _UNFILED = "filed", "kinds", "no"
_EXCERPT_CHARS = 1500
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._+-]+")


@dataclass(frozen=True)
class Defect:
    """記録の対象になった異常 1 件（種別と、何が起きたかの 1 行）。"""

    kind: str
    detail: str


def language_leaked(result: TaskResult) -> bool:
    """修復後も期待言語と違うまま終わったか。"""
    if not result.language_expected:
        return False
    return result.language_detected not in (result.language_expected, "", "und")


def _used_up_iterations(result: TaskResult) -> bool:
    """例外にならずに、往復の上限まで使って不合格で終えたか。"""
    return (
        not result.passed
        and not result.error
        and result.iteration_limit > 0
        and result.plan_steps_run == 0
        and result.iterations >= result.iteration_limit
    )


def _count(n: int, noun: str) -> str:
    """``1 time`` / ``3 times``（記録の英語の文の数え方）。"""
    return f"{n} {noun}" + ("" if n == 1 else "s")


def detect_defects(result: TaskResult, agent_name: str) -> list[Defect]:
    """1 タスクの結果から「実行例外」と「経路の異常」を拾う。"""
    found: list[Defect] = []
    if result.error:
        found.append(Defect("run exception", result.error))
    if _used_up_iterations(result):
        found.append(
            Defect(
                "iteration limit",
                f"failed after using up the limit of {result.iteration_limit} iterations "
                f"({_count(result.tool_calls, 'tool call')})",
            )
        )
    if "+plan" in agent_name and not result.error and result.plan_exits == 0:
        found.append(
            Defect(
                "plan path not taken",
                "started in plan mode but ended without ever going through exit_plan_mode "
                "(the approval card) (plan_exits=0)",
            )
        )
    if result.timeout_recoveries:
        resumed = (
            f", {_count(result.timeout_resumes, 'step resume')}" if result.timeout_resumes else ""
        )
        found.append(
            Defect(
                "timeout recovery",
                "asked for a smaller resend after a read timeout "
                f"({_count(result.timeout_recoveries, 'time')}{resumed})",
            )
        )
    if result.repetition_aborts:
        found.append(
            Defect(
                "repetition abort",
                "stopped a repeating generation mid-stream and sent it back "
                f"({_count(result.repetition_aborts, 'time')})",
            )
        )
    if result.syntax_rejects or result.syntax_warnings:
        found.append(
            Defect(
                "syntax-breaking edit",
                "failed the syntax check right after an edit "
                f"(sent back: {result.syntax_rejects}, broken writes: {result.syntax_warnings})",
            )
        )
    if result.edit_mismatches >= EDIT_MISMATCH_DEFECT:
        found.append(
            Defect(
                "edit_file mismatch",
                "the old text of edit_file did not match "
                f"{_count(result.edit_mismatches, 'time')} "
                "(no match or several matches)",
            )
        )
    if result.unknown_tool_calls >= UNKNOWN_TOOL_DEFECT:
        found.append(
            Defect(
                "unknown tool name",
                f"called unknown tool names {_count(result.unknown_tool_calls, 'time')} "
                "(candidates were returned and nothing ran)",
            )
        )
    broken = [s for s in result.plan_step_records if s.over_budget or s.timeouts]
    if broken:
        found.append(
            Defect(
                "step over budget",
                "; ".join(
                    f"step {s.index}: estimate {s.estimated_lines or 0} lines / largest write "
                    f"{s.max_write_lines} lines / limit {s.budget_lines} lines / "
                    f"timeouts {s.timeouts}"
                    for s in broken
                ),
            )
        )
    if result.plan_budget_forced:
        found.append(
            Defect(
                "plan send-back limit",
                f"still over budget after {_count(result.plan_budget_sendbacks, 'send-back')}, "
                "so the "
                "approval card was shown over budget "
                f"(verdicts: {' -> '.join(result.plan_budget_verdicts)})",
            )
        )
    if result.empty_final:
        found.append(
            Defect(
                "empty final reply",
                "the final reply was empty (graded with the fallback text)",
            )
        )
    if result.args_only_replies:
        found.append(
            Defect(
                "arguments-only JSON reply",
                "replied with argument JSON without a tool name "
                f"({_count(result.args_only_replies, 'time')})",
            )
        )
    if result.no_change_reports:
        found.append(
            Defect(
                "completion report without edits",
                "reported the work as done without changing any file "
                f"({_count(result.no_change_reports, 'time')})",
            )
        )
    if result.tool_call_repairs or result.empty_tool_args:
        found.append(
            Defect(
                "broken tool_call",
                f"repair resends: {result.tool_call_repairs}, fallbacks to {{}} arguments: "
                f"{result.empty_tool_args}",
            )
        )
    if language_leaked(result):
        found.append(
            Defect(
                "language mismatch left",
                f"expected {result.language_expected}, but the final reply is in "
                f"{result.language_detected}",
            )
        )
    return found


def save_defect_record(
    defects_dir: Path,
    task: EvalTask,
    result: TaskResult,
    defects: list[Defect],
    *,
    agent_name: str,
    model: str,
    workspace: Path,
    now: datetime.datetime | None = None,
) -> Path:
    """不具合記録を書き、ワークスペースのセッション記録と計画ファイルを隣へ写す。"""
    now = now or datetime.datetime.now().astimezone()
    stem = _unique_stem(
        defects_dir,
        f"{now:%Y%m%d-%H%M%S}-{_safe(task.id)}-{_safe(agent_name)}",
    )
    attachments = defects_dir / stem
    copied = _copy_records(workspace, attachments)
    record = defects_dir / f"{stem}.md"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        _render(task, result, defects, agent_name, model, now, copied), encoding="utf-8"
    )
    return record


def _copy_records(workspace: Path, dest: Path) -> list[str]:
    """セッション JSONL（``.satsuma/sessions/**``）と計画ファイル（``.satsuma/plans/*``）を写す。"""
    state = workspace / AGENT_DIRNAME
    sources = [
        *sorted((state / "sessions").rglob("*.jsonl")),
        *sorted((state / "plans").glob("*")),
    ]
    copied: list[str] = []
    for src in sources:
        if not src.is_file():
            continue
        kind = "sessions" if src.suffix == ".jsonl" else "plans"
        target = dest / kind / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        copied.append(f"{dest.name}/{kind}/{src.name}")
    return copied


def _render(
    task: EvalTask,
    result: TaskResult,
    defects: list[Defect],
    agent_name: str,
    model: str,
    now: datetime.datetime,
    copied: list[str],
) -> str:
    kinds = ", ".join(d.kind for d in defects)
    lines = [
        "---",
        f"{_FILED}: {_UNFILED}",
        f"{_KINDS}: {kinds}",
        f"task: {task.id}",
        f"agent: {agent_name}",
        "---",
        "",
        f"# Defect record from an eval: {task.id} ({agent_name}) — {kinds}",
        "",
        "This record is a candidate. A person decides whether to file it "
        f"(after filing, replace `{_FILED}: {_UNFILED}` at the top with the issue number).",
        "",
        "## What happened",
        "",
        f"- When: {now:%Y-%m-%d %H:%M:%S %z}",
        f"- Agent / model: {agent_name} / {model or '-'}",
        f"- Task: {task.id}" + (f" ({task.category})" if task.category else ""),
        f"- Permission: {task.permission_mode}",
        f"- Took: {result.duration_ms / 1000:.1f} s / iterations {result.iterations} / "
        f"tool calls {result.tool_calls}",
        "- Request:",
        "",
        *(f"  > {line}" if line else "  >" for line in task.prompt.strip().splitlines()),
        "",
        "## Findings",
        "",
        *(f"- **{d.kind}**: {d.detail}" for d in defects),
        "- Grade: "
        + (
            "passed"
            if result.passed
            else "failed" + (f" ({' / '.join(result.failures)})" if result.failures else "")
        ),
        "",
    ]
    text = result.final_text.strip()
    if text:
        long = len(text) > _EXCERPT_CHARS
        excerpt = text[:_EXCERPT_CHARS] + "\n… (truncated)" if long else text
        lines += ["Final reply" + (" (excerpt)" if long else "") + ":", ""]
        lines += ["```text", excerpt, "```", ""]
    lines += ["## Records", ""]
    if copied:
        lines += [f"- `{path}`" for path in copied]
    else:
        lines.append("- none (this agent leaves no session records in the workspace)")
    lines.append("")
    return "\n".join(lines)


def _safe(name: str) -> str:
    return _UNSAFE_RE.sub("_", name).strip("_") or "agent"


def _unique_stem(defects_dir: Path, stem: str) -> str:
    """同じ秒・同じタスクで複数回書いても上書きしない。"""
    candidate, n = stem, 2
    while (defects_dir / f"{candidate}.md").exists() or (defects_dir / candidate).exists():
        candidate, n = f"{stem}-{n}", n + 1
    return candidate


@dataclass(frozen=True)
class RepeatedDefect:
    """同じ種別の未起票の記録がしきい値以上たまったもの（評価の完了後に知らせる）。"""

    kind: str
    count: int
    records: list[Path]


def repeated_defects(defects_dir: Path, threshold: int = REPEAT_THRESHOLD) -> list[RepeatedDefect]:
    """``defects_dir`` の未起票の記録を種別ごとに数え、``threshold`` 件以上の種別を多い順に返す。"""
    by_kind: dict[str, list[Path]] = {}
    for record in sorted(defects_dir.glob("*.md")):
        for kind in _unfiled_kinds(_read_header(record)):
            by_kind.setdefault(kind, []).append(record)
    found = [
        RepeatedDefect(kind, len(paths), paths)
        for kind, paths in by_kind.items()
        if len(paths) >= threshold
    ]
    return sorted(found, key=lambda r: (-r.count, r.kind))


def _unfiled_kinds(header: dict[str, str]) -> list[str]:
    """未起票の記録の種別。"""
    if header.get(_FILED) != _UNFILED:
        return []
    kinds = (name.strip() for name in header.get(_KINDS, "").split(","))
    return [kind for kind in kinds if kind]


def _read_header(record: Path) -> dict[str, str]:
    """記録の先頭の ``---`` で囲んだ ``キー: 値`` を読む（無ければ空）。"""
    try:
        lines = record.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    header: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return header
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip()] = value.strip()
    return {}
