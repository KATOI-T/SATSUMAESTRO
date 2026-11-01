"""テスト用スタブ LSP サーバ。"""

from __future__ import annotations

import json
import os
import sys


def read_message() -> dict | None:
    content_length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        stripped = line.strip()
        if not stripped:
            break
        if stripped.lower().startswith(b"content-length:"):
            content_length = int(stripped.split(b":", 1)[1])
    if content_length is None:
        return None
    return json.loads(sys.stdin.buffer.read(content_length))


def send(payload: dict) -> None:
    body = json.dumps(payload).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()


def reply(msg_id: object, result: object) -> None:
    send({"jsonrpc": "2.0", "id": msg_id, "result": result})


def main() -> int:
    mode = os.environ.get("LSP_STUB_MODE", "ok")
    while True:
        msg = read_message()
        if msg is None:
            return 0
        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params") or {}
        if method == "initialize":
            if mode == "hang_init":
                continue
            reply(msg_id, {"capabilities": {"hoverProvider": True}})
        elif method == "textDocument/didOpen":
            if mode != "silent_diagnostics":
                uri = params["textDocument"]["uri"]
                send(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {
                            "uri": uri,
                            "diagnostics": [
                                {
                                    "range": {
                                        "start": {"line": 1, "character": 4},
                                        "end": {"line": 1, "character": 9},
                                    },
                                    "severity": 1,
                                    "source": "stub",
                                    "message": "stub type error",
                                }
                            ],
                        },
                    }
                )
        elif method == "textDocument/hover":
            reply(msg_id, {"contents": {"kind": "markdown", "value": "```py\ndef f() -> int\n```"}})
        elif method == "textDocument/definition":
            reply(
                msg_id,
                [
                    {
                        "uri": params["textDocument"]["uri"],
                        "range": {
                            "start": {"line": 0, "character": 4},
                            "end": {"line": 0, "character": 5},
                        },
                    }
                ],
            )
        elif method == "textDocument/references":
            uri = params["textDocument"]["uri"]
            reply(
                msg_id,
                [
                    {
                        "uri": uri,
                        "range": {
                            "start": {"line": 0, "character": 4},
                            "end": {"line": 0, "character": 5},
                        },
                    },
                    {
                        "uri": uri,
                        "range": {
                            "start": {"line": 3, "character": 8},
                            "end": {"line": 3, "character": 9},
                        },
                    },
                ],
            )
        elif method == "textDocument/completion":
            reply(
                msg_id,
                {
                    "isIncomplete": False,
                    "items": [
                        {"label": "frobnicate", "kind": 3, "detail": "def frobnicate() -> None"},
                        {"label": "value", "kind": 6},
                    ],
                },
            )
        elif method == "textDocument/documentSymbol":
            reply(
                msg_id,
                [
                    {
                        "name": "Greeter",
                        "kind": 5,
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 5, "character": 0},
                        },
                        "selectionRange": {
                            "start": {"line": 0, "character": 6},
                            "end": {"line": 0, "character": 13},
                        },
                        "children": [
                            {
                                "name": "greet",
                                "kind": 6,
                                "range": {
                                    "start": {"line": 1, "character": 0},
                                    "end": {"line": 2, "character": 0},
                                },
                                "selectionRange": {
                                    "start": {"line": 1, "character": 8},
                                    "end": {"line": 1, "character": 13},
                                },
                            }
                        ],
                    }
                ],
            )
        elif method == "workspace/symbol":
            reply(
                msg_id,
                [
                    {
                        "name": "Greeter",
                        "kind": 5,
                        "location": {
                            "uri": "file:///stub/greeter.py",
                            "range": {
                                "start": {"line": 0, "character": 6},
                                "end": {"line": 0, "character": 13},
                            },
                        },
                    }
                ],
            )
        elif method == "textDocument/formatting":
            reply(
                msg_id,
                [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 3},
                        },
                        "newText": "x = 1",
                    }
                ],
            )
        elif method == "shutdown":
            reply(msg_id, None)
        elif method == "exit":
            return 0
        elif msg_id is not None:
            reply(msg_id, None)


if __name__ == "__main__":
    raise SystemExit(main())
