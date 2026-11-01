#!/usr/bin/env python3
"""公式の評価の題（``evals/official/<参照>/ai.toml``）を選んで、SATSUMAESTRO で回す。"""

from __future__ import annotations

import argparse
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_web_bench as wb  # noqa: E402
import polyglot  # noqa: E402

from satsumaestro.evals.models import EvalReport, TaskResult  # noqa: E402
from satsumaestro.util.clock import now_ms  # noqa: E402

OFFICIAL = REPO / "evals" / "official"
RESULTS = REPO / ".satsuma" / "evals" / "official"

Task = dict[str, object]
Runner = Callable[[list[Task], str, Path, str, str, str], list[TaskResult]]
AGENTS = ("satsuma", "aider", "opencode")


def load_catalog(source: str) -> dict[str, object]:
    path = OFFICIAL / source / "ai.toml"
    if not path.is_file():
        raise SystemExit(f"題の目録がありません: {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def select(
    catalog: dict[str, object],
    *,
    smoke: bool = False,
    priority: str = "high",
    ids: list[str] | None = None,
    limit: int = 0,
) -> list[Task]:
    tasks = catalog["tasks"]
    assert isinstance(tasks, list)
    by_id: dict[str, Task] = {str(t["id"]): t for t in tasks}
    if ids or smoke:
        smoke_ids = catalog.get("smoke", [])
        assert isinstance(smoke_ids, list)
        want = ids or [str(i) for i in smoke_ids]
        if not want:
            raise SystemExit("この参照には smoke がありません（--priority か --ids で選ぶ）")
        missing = [i for i in want if i not in by_id]
        if missing:
            raise SystemExit(f"目録に無い題: {', '.join(missing)}")
        chosen = [by_id[i] for i in want]
    else:
        chosen = [t for t in tasks if priority == "all" or t["priority"] == priority]
    return chosen[:limit] if limit else chosen


def default_out(
    source: str, model: str, label: str, dry_run: str, agent: str = "satsuma", effort: str = ""
) -> Path:
    who = f"dry-{dry_run}" if dry_run else model.replace(":", "_").replace("/", "_")
    if agent != "satsuma":
        who = f"{agent}-{who}"
    if effort and not dry_run:
        who = f"{who}-{effort}"
    return RESULTS / source / f"{who}-{label}"


def task_file(out: Path, task_id: str) -> Path:
    return out / "tasks" / (task_id.replace("/", "__") + ".json")


def run_polyglot(
    tasks: list[Task], model: str, out: Path, dry_run: str, agent: str = "satsuma", effort: str = ""
) -> list[TaskResult]:
    results: list[TaskResult] = []
    for i, task in enumerate(tasks, 1):
        path = task_file(out, str(task["id"]))
        if path.is_file():
            results.append(TaskResult.model_validate_json(path.read_text(encoding="utf-8")))
            continue
        result, attempts = polyglot.solve(task, model, out, dry_run, agent, effort)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        path.with_suffix(".attempts.json").write_text(
            polyglot.attempts_json(attempts), encoding="utf-8"
        )
        first = "pass" if attempts[0].passed else ("pass@2" if result.passed else "FAIL")
        print(
            f"[{i}/{len(tasks)}] {task['id']:32} {first:6} {result.duration_ms / 1000:5.0f}s",
            flush=True,
        )
        results.append(result)
    return results


RUNNERS: dict[str, Runner] = {
    "aider-polyglot": run_polyglot,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", help="参照（evals/official の下のフォルダの名前）")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--smoke", action="store_true", help="ai.toml の smoke の題")
    pick.add_argument("--priority", choices=["high", "low", "all"], default="high")
    pick.add_argument("--ids", default="", help="題の ID（カンマ区切り）")
    ap.add_argument("--limit", type=int, default=0, help="先頭の N 題だけ（0 = 全部）")
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--agent", choices=AGENTS, default="satsuma", help="課題を解くハーネス")
    ap.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="",
        help="推論の深さ（どのハーネスにも同じ値を渡す。省略するとどれにも渡さない）",
    )
    ap.add_argument("--dry-run", choices=["reference", "empty"], default="")
    ap.add_argument("--out", default="", help="結果の置き場所（既定はモデルと選び方から決める）")
    args = ap.parse_args()

    runner = RUNNERS.get(args.source)
    if runner is None:
        raise SystemExit(f"{args.source} はまだ回せません（回せる参照: {', '.join(RUNNERS)}）")
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    tasks = select(
        load_catalog(args.source),
        smoke=args.smoke,
        priority=args.priority,
        ids=ids,
        limit=args.limit,
    )
    label = "smoke" if args.smoke else ("ids" if ids else args.priority)
    out = (
        Path(args.out)
        if args.out
        else default_out(
            args.source, args.model, label, args.dry_run, args.agent, args.reasoning_effort
        )
    ).resolve()
    if not args.dry_run:
        problem = wb.check_ollama(wb.ollama_env())
        if problem is not None:
            raise SystemExit(f"Ollama に接続できません: {problem}（.env の OLLAMA_HOST を確認）")
    effort = args.reasoning_effort or "（指定なし）"
    print(
        f"{args.source}: {len(tasks)} 題 agent={args.agent} model={args.model} "
        f"reasoning_effort={effort} out={out}",
        flush=True,
    )

    started_ms = now_ms()
    began = time.monotonic()
    results = runner(tasks, args.model, out, args.dry_run, args.agent, args.reasoning_effort)
    report = EvalReport(
        agent=args.agent,
        model=args.model if not args.dry_run else f"dry-run:{args.dry_run}",
        started_at_ms=started_ms,
        suite=f"official/{args.source}",
        results=results,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"\n{args.source}: 合格 {report.passed}/{report.total}"
        f"（選んだ題 {len(tasks)}）・この回の所要 {time.monotonic() - began:.0f} 秒"
        f" → {out / 'report.json'}"
    )
    for r in results:
        print(f"  {'pass' if r.passed else 'FAIL':4} {r.task_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
