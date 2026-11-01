#!/usr/bin/env python3
"""評価レポートの A/B 集計。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


@dataclass(frozen=True)
class RunStats:
    """1 レポート（= 1 回の eval 実行）の集計値。"""

    path: str
    total: int
    passed: int
    duration_s: float
    iterations: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    errors: int
    empty_final: int
    tool_call_repairs: int
    empty_tool_args: int
    compacted: int
    reminders: int
    language_mismatch: int
    language_repaired: int
    prompt_prefix_breaks: int
    timeout_recoveries: int
    timeout_resumes: int
    tool_call_rewrites: int
    repetition_aborts: int
    syntax_rejects: int
    syntax_warnings: int
    edit_mismatches: int
    edit_reindents: int
    edit_stale_notes: int
    related_code_notes: int
    overwrite_rejects: int
    overwrite_overrides: int
    step_write_rejects: int
    step_write_overrides: int
    plan_exit_not_active: int
    plan_exit_step_done: int
    mkdir_skips: int
    output_delete_blocks: int
    state_write_blocks: int
    secret_read_checks: int
    first_fetch_asks: int
    lsp_start_asks: int
    tool_result_cuts: int
    written_bodies_shrunk: int
    omitted_copies: int
    presend_compactions: int
    context_truncations: int
    prefill_waits: int
    unknown_tool_calls: int
    args_only_replies: int
    no_change_reports: int
    tool_name_fixes: int
    deferred_autoloads: int
    plan_steps_run: int
    plan_exits: int
    plan_exit_nudges: int
    write_calls: int
    write_chars_median: float
    write_chars_max: int
    tokens_per_s: float
    plan_steps_over_budget: int
    plan_budget_sendbacks: int
    tasks: dict[str, tuple[bool, float]]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def language_drift_rate(self) -> float:
        """修復前に期待言語を外していた割合（これが下がることが狙い）。"""
        if not self.total:
            return 0.0
        return (self.language_mismatch + self.language_repaired) / self.total

    @property
    def prefix_breaks_per_iteration(self) -> float:
        """反復 1 回あたりの接頭辞破壊回数。"""
        return self.prompt_prefix_breaks / self.iterations if self.iterations else 0.0

    @property
    def language_leak_rate(self) -> float:
        """修復後も期待言語と違うまま終わった割合（ユーザーが目にする英語化率）。"""
        return self.language_mismatch / self.total if self.total else 0.0


def _empty_final(r: dict[str, object]) -> bool:
    """本文空の終了か。"""
    if "empty_final" in r:
        return bool(r.get("empty_final"))
    return not str(r.get("final_text", "")).strip() and not r.get("error")


def _language_mismatch(r: dict[str, object]) -> bool:
    """修復後も期待言語と違うまま終わったか。"""
    expected = str(r.get("language_expected", ""))
    detected = str(r.get("language_detected", ""))
    return bool(expected) and detected not in (expected, "", "und")


def load_run(path: Path) -> RunStats:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    tasks: dict[str, tuple[bool, float]] = {}
    for r in results:
        tasks[str(r.get("task_id", "?"))] = (bool(r.get("passed")), r.get("duration_ms", 0) / 1000)
    chunk_chars = [int(c.get("chars", 0)) for r in results for c in r.get("write_chunks", [])]
    rates = [float(r["tokens_per_s"]) for r in results if r.get("tokens_per_s")]
    return RunStats(
        path=str(path),
        total=len(results),
        passed=sum(1 for r in results if r.get("passed")),
        duration_s=sum(r.get("duration_ms", 0) for r in results) / 1000,
        iterations=sum(int(r.get("iterations", 0)) for r in results),
        tool_calls=sum(int(r.get("tool_calls", 0)) for r in results),
        input_tokens=sum(int(r.get("usage", {}).get("input_tokens", 0)) for r in results),
        output_tokens=sum(int(r.get("usage", {}).get("output_tokens", 0)) for r in results),
        errors=sum(1 for r in results if r.get("error")),
        empty_final=sum(1 for r in results if _empty_final(r)),
        tool_call_repairs=sum(int(r.get("tool_call_repairs", 0)) for r in results),
        empty_tool_args=sum(int(r.get("empty_tool_args", 0)) for r in results),
        compacted=sum(1 for r in results if r.get("compacted")),
        reminders=sum(int(r.get("reminders", 0)) for r in results),
        language_mismatch=sum(1 for r in results if _language_mismatch(r)),
        language_repaired=sum(1 for r in results if r.get("language_repaired")),
        prompt_prefix_breaks=sum(int(r.get("prompt_prefix_breaks", 0)) for r in results),
        timeout_recoveries=sum(int(r.get("timeout_recoveries", 0)) for r in results),
        timeout_resumes=sum(int(r.get("timeout_resumes", 0)) for r in results),
        tool_call_rewrites=sum(int(r.get("tool_call_rewrites", 0)) for r in results),
        repetition_aborts=sum(int(r.get("repetition_aborts", 0)) for r in results),
        syntax_rejects=sum(int(r.get("syntax_rejects", 0)) for r in results),
        syntax_warnings=sum(int(r.get("syntax_warnings", 0)) for r in results),
        edit_mismatches=sum(int(r.get("edit_mismatches", 0)) for r in results),
        edit_reindents=sum(int(r.get("edit_reindents", 0)) for r in results),
        edit_stale_notes=sum(int(r.get("edit_stale_notes", 0)) for r in results),
        related_code_notes=sum(int(r.get("related_code_notes", 0)) for r in results),
        overwrite_rejects=sum(int(r.get("overwrite_rejects", 0)) for r in results),
        overwrite_overrides=sum(int(r.get("overwrite_overrides", 0)) for r in results),
        step_write_rejects=sum(int(r.get("step_write_rejects", 0)) for r in results),
        step_write_overrides=sum(int(r.get("step_write_overrides", 0)) for r in results),
        plan_exit_not_active=sum(int(r.get("plan_exit_not_active", 0)) for r in results),
        plan_exit_step_done=sum(int(r.get("plan_exit_step_done", 0)) for r in results),
        mkdir_skips=sum(int(r.get("mkdir_skips", 0)) for r in results),
        output_delete_blocks=sum(int(r.get("output_delete_blocks", 0)) for r in results),
        state_write_blocks=sum(int(r.get("state_write_blocks", 0)) for r in results),
        secret_read_checks=sum(int(r.get("secret_read_checks", 0)) for r in results),
        first_fetch_asks=sum(int(r.get("first_fetch_asks", 0)) for r in results),
        lsp_start_asks=sum(int(r.get("lsp_start_asks", 0)) for r in results),
        tool_result_cuts=sum(int(r.get("tool_result_cuts", 0)) for r in results),
        written_bodies_shrunk=sum(int(r.get("written_bodies_shrunk", 0)) for r in results),
        omitted_copies=sum(int(r.get("omitted_copies", 0)) for r in results),
        presend_compactions=sum(int(r.get("presend_compactions", 0)) for r in results),
        context_truncations=sum(int(r.get("context_truncations", 0)) for r in results),
        prefill_waits=sum(int(r.get("prefill_waits", 0)) for r in results),
        unknown_tool_calls=sum(int(r.get("unknown_tool_calls", 0)) for r in results),
        args_only_replies=sum(int(r.get("args_only_replies", 0)) for r in results),
        no_change_reports=sum(int(r.get("no_change_reports", 0)) for r in results),
        tool_name_fixes=sum(int(r.get("tool_name_fixes", 0)) for r in results),
        deferred_autoloads=sum(int(r.get("deferred_autoloads", 0)) for r in results),
        plan_steps_run=sum(int(r.get("plan_steps_run", 0)) for r in results),
        plan_exits=sum(int(r.get("plan_exits", 0)) for r in results),
        plan_exit_nudges=sum(int(r.get("plan_exit_nudges", 0)) for r in results),
        write_calls=len(chunk_chars),
        write_chars_median=median(chunk_chars) if chunk_chars else 0.0,
        write_chars_max=max(chunk_chars, default=0),
        tokens_per_s=mean(rates) if rates else 0.0,
        plan_steps_over_budget=sum(
            1
            for r in results
            for s in r.get("plan_step_records", [])
            if int(s.get("budget_lines", 0))
            and int(s.get("max_write_lines", 0)) > int(s.get("budget_lines", 0))
        ),
        plan_budget_sendbacks=sum(int(r.get("plan_budget_sendbacks", 0)) for r in results),
        tasks=tasks,
    )


_METRICS: list[tuple[str, str, str]] = [
    ("合格率", "pass_rate", "pct"),
    ("合格数", "passed", "f1"),
    ("所要時間合計 (s)", "duration_s", "f1"),
    ("反復合計", "iterations", "f1"),
    ("ツール呼び出し合計", "tool_calls", "f1"),
    ("入力トークン合計", "input_tokens", "int"),
    ("出力トークン合計", "output_tokens", "int"),
    ("エラー件数", "errors", "f1"),
    ("本文空の終了 (empty_final)", "empty_final", "f1"),
    ("tool_call 修復回数", "tool_call_repairs", "f1"),
    ("引数 {} fallback 回数", "empty_tool_args", "f1"),
    ("圧縮発火タスク数", "compacted", "f1"),
    ("reminder 注入件数", "reminders", "f1"),
    ("言語ドリフト率 (修復前)", "language_drift_rate", "pct"),
    ("言語不一致率 (修復後に残存)", "language_leak_rate", "pct"),
    ("言語修復の発火タスク数", "language_repaired", "f1"),
    ("接頭辞破壊 / 反復", "prefix_breaks_per_iteration", "f2"),
    ("接頭辞破壊 合計", "prompt_prefix_breaks", "f1"),
    ("タイムアウト復帰", "timeout_recoveries", "f1"),
    ("打ち切り 2 回目の段の再開", "timeout_resumes", "f1"),
    ("壊れた呼び出しの書き直し", "tool_call_rewrites", "f1"),
    ("反復の打ち切り", "repetition_aborts", "f1"),
    ("構文を壊す編集の差し戻し", "syntax_rejects", "f1"),
    ("通っていたファイルを壊した書き込み", "syntax_warnings", "f1"),
    ("edit_file の不一致", "edit_mismatches", "f1"),
    ("字下げを合わせて適用", "edit_reindents", "f1"),
    ("不一致に読み直しを添えた", "edit_stale_notes", "f1"),
    ("派生クラスに基底クラスの生成箇所を添えた", "related_code_notes", "f1"),
    ("途中の部分での置き換えの差し戻し", "overwrite_rejects", "f1"),
    ("差し戻し後の繰り返しで書いた", "overwrite_overrides", "f1"),
    ("段の上限を超えた書き込みの差し戻し", "step_write_rejects", "f1"),
    ("段の上限の差し戻しの後で書いた", "step_write_overrides", "f1"),
    ("plan でないときの exit_plan_mode", "plan_exit_not_active", "f1"),
    ("段の完了として扱った exit_plan_mode", "plan_exit_step_done", "f1"),
    ("書く前の mkdir を「要らない」と返した", "mkdir_skips", "f1"),
    ("頼まれていない成果物の削除を止めた", "output_delete_blocks", "f1"),
    ("状態ディレクトリへの書き込みを拒否した", "state_write_blocks", "f1"),
    ("秘密ファイルの読み取りを確認に回した", "secret_read_checks", "f1"),
    ("最初の web_fetch を確認に回した", "first_fetch_asks", "f1"),
    ("言語サーバの起動を確認に回した", "lsp_start_asks", "f1"),
    ("大きすぎるツールの結果を切った", "tool_result_cuts", "f1"),
    ("会話の中で書き込みの本文を縮めた", "written_bodies_shrunk", "f1"),
    ("縮めた書き込みの形を真似た", "omitted_copies", "f1"),
    ("送る前に履歴を畳んだ", "presend_compactions", "f1"),
    ("サーバが要求を切り詰めた", "context_truncations", "f1"),
    ("最初の応答までの待ちを延ばした", "prefill_waits", "f1"),
    ("存在しないツール名", "unknown_tool_calls", "f1"),
    ("引数だけの JSON の本文", "args_only_replies", "f1"),
    ("編集 0 回の完了報告", "no_change_reports", "f1"),
    ("ツール名の表記ゆれを読み替え", "tool_name_fixes", "f1"),
    ("遅延ツールの自動読み込み", "deferred_autoloads", "f1"),
    ("段階実行の段数", "plan_steps_run", "f1"),
    ("計画→承認の回数 (plan_exits)", "plan_exits", "f1"),
    ("承認へ差し戻した回数", "plan_exit_nudges", "f1"),
    ("書き込み回数", "write_calls", "f1"),
    ("書き込み 1 回の文字数 中央値", "write_chars_median", "int"),
    ("書き込み 1 回の文字数 最大", "write_chars_max", "int"),
    ("生成速度の下限 (tok/s)", "tokens_per_s", "f1"),
    ("段の予算超過 (段数)", "plan_steps_over_budget", "f1"),
    ("計画の差し戻し (予算超過)", "plan_budget_sendbacks", "f1"),
]


def _fmt(value: float, kind: str, *, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    if kind == "pct":
        return f"{sign}{value * 100:.1f}%"
    if kind == "int":
        return f"{sign}{value:.0f}"
    if kind == "f2":
        return f"{sign}{value:.2f}"
    return f"{sign}{value:.1f}"


def _avg(runs: list[RunStats], attr: str) -> float:
    return mean(float(getattr(r, attr)) for r in runs)


def _marks(runs: list[RunStats], task: str) -> str:
    return "".join(("○" if r.tasks[task][0] else "×") if task in r.tasks else "-" for r in runs)


def _avg_secs(runs: list[RunStats], task: str) -> str:
    secs = [r.tasks[task][1] for r in runs if task in r.tasks]
    return f"{mean(secs):.1f}" if secs else "-"


def render(
    title: str,
    before: list[RunStats],
    after: list[RunStats],
    label_before: str,
    label_after: str,
) -> str:
    if len(before) == len(after):
        n = f"{len(before)} 回平均"
    else:
        n = f"{len(before)}/{len(after)} 回平均"
    total = before[0].total
    lines = [f"### `{title}`（{n} / {total} 題）", ""]
    lines += [f"| 指標 | {label_before} | {label_after} | 差 |", "|---|---:|---:|---:|"]
    for name, attr, kind in _METRICS:
        b, a = _avg(before, attr), _avg(after, attr)
        diff = _fmt(a - b, kind, signed=True)
        lines.append(f"| {name} | {_fmt(b, kind)} | {_fmt(a, kind)} | {diff} |")
    lines.append("")

    def each(label: str, runs: list[RunStats]) -> str:
        return "; ".join(
            f"{label}#{i} {r.passed}/{r.total} {r.duration_s:.0f}s"
            + (f" empty={r.empty_final}" if r.empty_final else "")
            for i, r in enumerate(runs, 1)
        )

    lines.append(f"各回: {each('before', before)} / {each('after', after)}")
    lines.append("")
    lines += [
        f"| タスク | {label_before} 合否 | {label_after} 合否 "
        f"| {label_before} 秒 | {label_after} 秒 |",
        "|---|:-:|:-:|---:|---:|",
    ]
    for task in before[0].tasks:
        lines.append(
            f"| {task} | {_marks(before, task)} | {_marks(after, task)} "
            f"| {_avg_secs(before, task)} | {_avg_secs(after, task)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="評価レポートの before/after 集計")
    parser.add_argument("--title", required=True, help="見出し（スイート名など）")
    parser.add_argument("--before", nargs="+", required=True, type=Path, help="before レポート")
    parser.add_argument("--after", nargs="+", required=True, type=Path, help="after レポート")
    parser.add_argument("--label-before", default="before")
    parser.add_argument("--label-after", default="after")
    args = parser.parse_args(argv)

    before = [load_run(p) for p in args.before]
    after = [load_run(p) for p in args.after]
    if not before or not after:
        print("error: before / after のレポートが空です", file=sys.stderr)
        return 2
    totals = {r.total for r in before + after}
    if len(totals) != 1:
        print(f"warning: タスク数が揃っていません: {sorted(totals)}", file=sys.stderr)
    print(render(args.title, before, after, args.label_before, args.label_after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
