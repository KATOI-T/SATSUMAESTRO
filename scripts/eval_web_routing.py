"""Web Routing 評価の互換エントリポイント（本体は ``satsumaestro/evals/web_routing.py``）。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satsumaestro.evals.web_routing import (  # noqa: E402
    RunOutcome,
    run_web_routing,
    summarize,
    write_outcomes,
)

REPO = Path(__file__).resolve().parent.parent


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--suite", default=str(REPO / "evals" / "web_routing"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.runs < 1:
        ap.error("--runs は 1 以上")
    outcomes: list[RunOutcome] = []
    for run in range(1, args.runs + 1):
        _, batch = await run_web_routing(Path(args.suite), args.model)
        for o in batch:
            flag = "MISROUTE" if o.misrouted else ("ok" if o.checks_ok else "CHECK_FAIL")
            print(
                f"run{run} {o.pattern:7} {o.task_id:16} {flag:10} "
                f"searched={o.searched} tools={o.tool_calls} ({o.duration_ms}ms) {o.error}"
            )
        outcomes.extend(batch)
    print("\n" + summarize(outcomes))
    if args.out:
        write_outcomes(outcomes, Path(args.out))
        print(f"レポート: {args.out}")
    return 0 if all(o.success for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
