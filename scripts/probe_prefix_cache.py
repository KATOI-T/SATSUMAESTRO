#!/usr/bin/env python3
"""Ollama のプレフィックスキャッシュ再利用を prefill 時間で直接観測する。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

_SYSTEM_FILLER_CHARS = 24_000
_TOOL_RESULT_CHARS = 2_000


@dataclass(frozen=True)
class TurnStat:
    turn: int
    prompt_eval_count: int
    prompt_eval_ms: float
    total_duration_ms: float


def _system(base: str, *, salt: str) -> str:
    """system プロンプト。"""
    return f"{base}\n\n# Git\n$ git status --short\n M {salt}\n"


async def _one_turn(
    client: httpx.AsyncClient, host: str, model: str, messages: list[dict[str, Any]], system: str
) -> TurnStat:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": False,
        "options": {"num_predict": 1, "temperature": 0},
        "keep_alive": "30m",
    }
    r = await client.post(f"{host}/api/chat", json=payload)
    r.raise_for_status()
    body = r.json()
    if "prompt_eval_count" not in body:
        raise SystemExit(
            f"prompt_eval_count が返っていない（/api/chat か確認）: {json.dumps(body)[:200]}"
        )
    return TurnStat(
        turn=len(messages),
        prompt_eval_count=int(body["prompt_eval_count"]),
        prompt_eval_ms=float(body.get("prompt_eval_duration", 0)) / 1e6,
        total_duration_ms=float(body.get("total_duration", 0)) / 1e6,
    )


async def _series(
    client: httpx.AsyncClient, host: str, model: str, turns: int, *, mutate_system: bool
) -> list[TurnStat]:
    """1 系列を回す。"""
    base = "You are a coding assistant.\n" + ("x" * _SYSTEM_FILLER_CHARS)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "はじめまして。"}]
    stats: list[TurnStat] = []
    for i in range(turns):
        salt = f"file_{i}.py" if mutate_system else "file_0.py"
        stats.append(await _one_turn(client, host, model, messages, _system(base, salt=salt)))
        messages.append({"role": "assistant", "content": f"ステップ {i} を実行します。"})
        messages.append({"role": "user", "content": "y" * _TOOL_RESULT_CHARS})
    return stats


def _report(label: str, stats: list[TurnStat]) -> float:
    print(f"\n── {label} ──")
    print(f"{'turn':>4}  {'prompt tokens':>14}  {'prefill (ms)':>13}  {'total (ms)':>11}")
    for i, s in enumerate(stats):
        print(
            f"{i:>4}  {s.prompt_eval_count:>14,}  "
            f"{s.prompt_eval_ms:>13,.0f}  {s.total_duration_ms:>11,.0f}"
        )
    after_first = sum(s.prompt_eval_ms for s in stats[1:])
    print(f"{'2 回目以降 prefill 合計':>26}: {after_first:,.0f} ms")
    return after_first


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--turns", type=int, default=5)
    args = ap.parse_args()

    timeout = httpx.Timeout(600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await _one_turn(
            client, args.host, args.model, [{"role": "user", "content": "ping"}], "warmup"
        )
        stable = await _series(client, args.host, args.model, args.turns, mutate_system=False)
        mutating = await _series(client, args.host, args.model, args.turns, mutate_system=True)

    stable_total = _report("stable: system 固定（対策後の状態）", stable)
    mutating_total = _report("mutating: 毎ターン system が変わる（対応前の状態）", mutating)

    print("\n── まとめ ──")
    if stable_total:
        print(f"再 prefill 時間: mutating / stable = {mutating_total / stable_total:.1f} 倍")
    print(f"system 固定で削れた prefill 時間: {mutating_total - stable_total:,.0f} ms")


if __name__ == "__main__":
    asyncio.run(main())
