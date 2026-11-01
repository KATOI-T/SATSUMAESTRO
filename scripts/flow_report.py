#!/usr/bin/env python3
"""開発フロー計測 v0（おまけツール / ドッグフーディング）。"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from collections import Counter

REWORK_RE = re.compile(r"修正|解消|削除|統一|直し|直す|fix|バグ|エラー|やり直")
REVIEW_RE = re.compile(r"リンク|snake_case|ツール名|指摘|レビュー")
PHASE_RULES = [
    ("設計/ドキュメント", re.compile(r"設計|基本設計|ポリシー|名称|配置|色|ドキュメント", re.I)),
    ("CI/ツール整備", re.compile(r"\bci\b|フック|make|lint|ruff|mypy|pytest|hook", re.I)),
    ("実装", re.compile(r"実装|環境変数|satsumaestro|機能|追加|feat", re.I)),
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False).stdout


def files_of(commit: str) -> list[str]:
    out = git("show", "--stat", "--pretty=", commit).strip().splitlines()
    paths = []
    for ln in out:
        m = re.match(r"\s*(\S+)\s+\|", ln)
        if m:
            paths.append(m.group(1))
    return paths


def classify(subject: str, paths: list[str]) -> tuple[str, bool, bool]:
    docs_only = bool(paths) and all(
        p.startswith("docs/") or p == "policy.md" or p.lower().endswith("readme.md") for p in paths
    )
    phase = "その他"
    for name, rule in PHASE_RULES:
        if rule.search(subject):
            phase = name
            break
    if phase == "その他" and docs_only:
        phase = "設計/ドキュメント"
    return phase, bool(REWORK_RE.search(subject)), bool(REVIEW_RE.search(subject))


def human_gap(minutes: float) -> str:
    if minutes >= 120:
        return f"◀ {int(minutes)}分の空き（夜間/長期中断の可能性）"
    if minutes >= 30:
        return f"◀ {int(minutes)}分の空き（フェーズ移行 or 滞留）"
    return ""


def main() -> int:
    rng = sys.argv[1] if len(sys.argv) > 1 else None
    log_args = ["log", "--reverse", "--pretty=%at|%h|%s"]
    if rng:
        log_args.append(rng)
    raw = git(*log_args).strip().splitlines()
    if not raw:
        print("コミットが見つかりません（範囲指定を確認してください）。")
        return 1

    rows = []
    prev_t = None
    for ln in raw:
        at, h, subj = ln.split("|", 2)
        t = dt.datetime.fromtimestamp(int(at))
        gap = (t - prev_t).total_seconds() / 60 if prev_t else 0.0
        phase, rework, review = classify(subj, files_of(h))
        rows.append(dict(t=t, h=h, subj=subj, gap=gap, phase=phase, rework=rework, review=review))
        prev_t = t

    line = "=" * 78
    print(line)
    print("SATSUMAESTRO 開発フロー v0 レポート（git履歴のみ / 分類はヒューリスティック）")
    print(line)
    print(
        f"期間: {rows[0]['t']:%Y-%m-%d %H:%M} 〜 {rows[-1]['t']:%Y-%m-%d %H:%M}"
        f"  総コミット: {len(rows)}"
    )
    print()
    print("― タイムライン（●=コミット, ≥30分の空きに印）―")
    for r in rows:
        flags = []
        if r["rework"]:
            flags.append("手戻り")
        if r["review"]:
            flags.append("レビュー起因")
        tag = f"  [{'/'.join(flags)}]" if flags else ""
        gap = human_gap(r["gap"])
        gap = f"  {gap}" if gap else ""
        print(f" {r['t']:%m-%d %H:%M} ● {r['phase']:<12} {r['subj'][:32]:<32}{tag}{gap}")

    ph = Counter(r["phase"] for r in rows)
    rework_n = sum(r["rework"] for r in rows)
    review_n = sum(r["review"] for r in rows)
    active = sorted(r["gap"] for r in rows[1:] if 0 < r["gap"] < 120)
    max_gap = max((r["gap"] for r in rows[1:]), default=0.0)
    print()
    print("― サマリ ―")
    print(" フェーズ別: " + ", ".join(f"{k}={v}" for k, v in ph.most_common()))
    print(f" 手戻りコミット率: {rework_n}/{len(rows)} = {100 * rework_n // len(rows)}%")
    print(f" うちレビュー起因: {review_n} 件（設計レビュー前倒しで潰せた候補）")
    if active:
        med = active[len(active) // 2]
        print(f" 実働ギャップ中央値: 約{med:.0f}分  最大空き: {int(max_gap)}分")

    print()
    print("― 自動所見 ―")
    rate = 100 * rework_n // len(rows)
    if rate >= 40:
        worst = Counter(r["phase"] for r in rows if r["rework"]).most_common(1)
        where = worst[0][0] if worst else "-"
        print(f" ・手戻り率{rate}%と高め。手戻りが最も集中するのは「{where}」フェーズ。")
    else:
        print(f" ・手戻り率{rate}%。大きな手戻りテールは見られない。")
    if review_n:
        rv = [r["subj"][:24] for r in rows if r["review"]]
        print(f" ・レビュー起因の後追い修正 {review_n}件（{', '.join(rv)}）は、")
        print("   設計/実装レビューを前倒しすれば発生前に潰せた種類。")
    if max_gap >= 120:
        print(" ・最大の空きは夜間/長期中断の可能性が高く、純粋な滞留とは区別が必要。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
