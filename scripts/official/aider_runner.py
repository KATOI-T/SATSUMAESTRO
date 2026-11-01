"""aider を、公式の benchmark（aider の ``benchmark/benchmark.py``）と同じ作りで 1 回走らせる。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="aider のモデル名（ollama_chat/<モデル>）")
    ap.add_argument("--edit-format", default="diff")
    ap.add_argument("--num-ctx", type=int, default=0, help="Ollama の窓（0 = aider に任せる）")
    ap.add_argument("--reasoning-effort", default="")
    ap.add_argument("--message-file", required=True, help="渡す依頼の文")
    ap.add_argument("--history", required=True, help="会話の記録（Markdown）")
    ap.add_argument("--restore", action="store_true", help="会話の記録から続ける")
    ap.add_argument("--ignore", action="append", default=[], help="言及を無視するファイル")
    ap.add_argument("files", nargs="+", help="直させるファイル（作業ディレクトリからの相対）")
    args = ap.parse_args()

    io = InputOutput(pretty=False, yes=True, chat_history_file=args.history)
    model = Model(args.model)
    extra = dict(model.extra_params or {})
    if args.reasoning_effort:
        extra["reasoning_effort"] = args.reasoning_effort
    if args.num_ctx:
        extra["num_ctx"] = args.num_ctx
    model.extra_params = extra
    coder = Coder.create(
        model,
        args.edit_format,
        io,
        fnames=args.files,
        use_git=False,
        stream=False,
        cache_prompts=True,
        suggest_shell_commands=False,
        ignore_mentions=set(args.ignore),
        restore_chat_history=args.restore,
    )
    coder.get_file_mentions = lambda _content: set()

    calls = 0
    send = coder.send

    def counted_send(*a: Any, **k: Any) -> Iterator[Any]:
        nonlocal calls
        calls += 1
        return send(*a, **k)  # type: ignore[no-any-return]

    coder.send = counted_send
    message = Path(args.message_file).read_text(encoding="utf-8")
    coder.run(with_message=message, preproc=False)
    print(
        json.dumps(
            {
                "iterations": calls,
                "usage": {
                    "input_tokens": coder.total_tokens_sent,
                    "output_tokens": coder.total_tokens_received,
                },
                "malformed_responses": coder.num_malformed_responses,
                "exhausted_context_windows": coder.num_exhausted_context_windows,
                "stop_reason": "max_reflections"
                if coder.num_reflections >= coder.max_reflections
                else "end_turn",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
