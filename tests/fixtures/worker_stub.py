"""Worker 子 CLI のスタブ。"""

from __future__ import annotations

import json
import os
import sys
import time

_MODE: str | None = None


def _mode() -> str:
    global _MODE
    if _MODE is not None:
        return _MODE
    mode_file = os.environ.get("STUB_MODE_FILE", "")
    if mode_file:
        try:
            with open(mode_file, encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
            _MODE = lines[0] if lines else "ok"
            if len(lines) > 1:
                with open(mode_file, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines[1:]) + "\n")
        except OSError:
            _MODE = "ok"
        return _MODE
    _MODE = os.environ.get("STUB_MODE", "ok")
    return _MODE


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _ready(cwd: str, receipt: object) -> None:
    _emit(
        {
            "event": "worker.ready",
            "protocol": 1,
            "session_id": "sess-stub",
            "cwd": cwd,
            "pid": os.getpid(),
            "receipt": receipt,
            "emitted_at_ms": 0,
        }
    )


def _result(stop_reason: str, tokens: int) -> None:
    _emit(
        {
            "event": "worker.result",
            "final_text": f"stub done ({_mode()})",
            "stop_reason": stop_reason,
            "iterations": 1,
            "usage": {"input_tokens": 3, "output_tokens": tokens},
            "session_id": "sess-stub",
            "emitted_at_ms": 0,
        }
    )


def main() -> int:
    mode = _mode()
    raw_receipt = os.environ.get("SATSUMA_WORKER_RECEIPT") or "null"
    try:
        receipt = json.loads(raw_receipt)
    except json.JSONDecodeError:
        receipt = None

    if mode == "exit_early":
        return 0
    if mode == "garbage_first_line":
        print("zsh: command not found: Implement", flush=True)
        return 127
    if mode == "hang_before_ready":
        time.sleep(60)
        return 0

    cwd = os.getcwd()
    if mode == "wrong_cwd":
        cwd = "/tmp/definitely-the-wrong-target"
    if mode == "wrong_receipt":
        receipt = {"repo": "someone-elses-repo"}

    _ready(cwd, receipt)

    if mode == "hang_after_ready":
        time.sleep(60)
        return 0
    if mode == "finish_unknown_zero_tokens":
        _result("unknown", 0)
        return 0
    _result("end_turn", 7)
    return 0


if __name__ == "__main__":
    sys.exit(main())
