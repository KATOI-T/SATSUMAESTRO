"""OpenAI 互換の SSE を返す擬似 Ollama（``flavor`` で LM Studio・llama.cpp のネイティブ API）。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
FAKE_MODEL = "gpt-oss:120b"
NO_RULE = "[fake] no rule matched"
_PLAN_FILE = re.compile(r"\.satsuma/plans/[0-9a-f]{32}\.md")
_HEAD = 200


@dataclass
class Rule:
    index: int
    when: dict[str, Any]
    reply: dict[str, Any]
    note: str = ""
    times: int | None = None
    hits: int = 0

    def matches(self, role: str, tool: str | None, text: str) -> bool:
        if self.times is not None and self.hits >= self.times:
            return False
        w = self.when
        if w.get("role", "any") not in ("any", role):
            return False
        if "tool" in w and w["tool"] != tool:
            return False
        if any(s not in text for s in _as_list(w.get("contains"))):
            return False
        if any(s in text for s in _as_list(w.get("not_contains"))):
            return False
        any_of = _as_list(w.get("any_of"))
        if any_of and not any(s in text for s in any_of):
            return False
        return not ("regex" in w and re.search(w["regex"], text) is None)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in cast(list[object], value)]


def load_script(path: Path) -> list[Rule]:
    rules: list[Rule] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        rules.append(
            Rule(
                index=lineno,
                when=obj.get("when") or {},
                reply=obj.get("reply") or {},
                note=str(obj.get("note", "")),
                times=obj.get("times"),
            )
        )
    if not rules:
        raise ValueError(f"台本が空です: {path}")
    return rules


def text_of(message: dict[str, Any]) -> str:
    """OpenAI 形式のメッセージ本文（文字列か ``[{"type": "text", "text": …}]``）を文字列にする。"""
    content = message.get("content")
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content or "")


def last_tool_name(messages: list[dict[str, Any]]) -> str | None:
    """直近が tool 結果なら、その呼び出しのツール名（直前の assistant の tool_calls から引く）。"""
    if not messages or messages[-1].get("role") != "tool":
        return None
    call_id = messages[-1].get("tool_call_id")
    for message in reversed(messages[:-1]):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("id") == call_id:
                return str(call.get("function", {}).get("name", ""))
    return None


def find_plan_file(body: dict[str, Any]) -> str | None:
    for message in body.get("messages", []):
        m = _PLAN_FILE.search(text_of(message))
        if m is not None:
            return m.group(0)
    return None


def substitute(value: Any, variables: dict[str, str]) -> Any:
    """``reply`` の文字列の葉すべてに変数を当てる（``str.replace``）。"""
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [substitute(v, variables) for v in cast(list[Any], value)]
    if isinstance(value, dict):
        return {k: substitute(v, variables) for k, v in cast(dict[str, Any], value).items()}
    return value


class Script:
    """台本と要求ログ。"""

    def __init__(self, rules: list[Rule], *, workspace: Path | None = None) -> None:
        self.rules = rules
        self.workspace = workspace
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def decide(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.respond(body)[0]

    def respond(self, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """応答と、その要求ログの ``seq``（送り終えたら :meth:`mark_done` に渡す）。"""
        messages = cast(list[dict[str, Any]], body.get("messages", []))
        last = messages[-1] if messages else {}
        role = str(last.get("role", ""))
        tool = last_tool_name(messages)
        text = text_of(last)
        variables = {"plan_file": find_plan_file(body) or "{plan_file}"}
        if self.workspace is not None:
            variables["workspace"] = str(self.workspace)
        with self._lock:
            rule = next((r for r in self.rules if r.matches(role, tool, text)), None)
            if rule is not None:
                rule.hits += 1
                reply = cast(dict[str, Any], substitute(rule.reply, variables))
            else:
                reply = {"text": f"{NO_RULE}: role={role} tool={tool} head={text[:80]!r}"}
            seq = len(self.requests) + 1
            self.requests.append(
                {
                    "seq": seq,
                    "model": body.get("model"),
                    "role": role,
                    "tool": tool,
                    "head": text[:_HEAD],
                    "rule": rule.index if rule is not None else None,
                    "note": rule.note if rule is not None else NO_RULE,
                    "reply": _summarize(reply),
                }
            )
        return reply, seq

    def mark_done(self, seq: int) -> None:
        """応答の最後のチャンクを書き終えた時刻（``time.monotonic``）を要求ログに残す。"""
        with self._lock:
            self.requests[seq - 1]["done_at"] = time.monotonic()


def _summarize(reply: dict[str, Any]) -> str:
    calls = [c.get("name") for c in reply.get("tool_calls") or []]
    text = str(reply.get("text", ""))[:60]
    return f"tool_calls={calls}" if calls else f"text={text!r}"


def _chunk(model: str, delta: dict[str, Any], *, finish: str | None = None) -> bytes:
    body = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n".encode()


def _usage_chunk(model: str, completion_tokens: int, prompt_tokens: int = 10) -> bytes:
    body = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(body)}\n\n".encode()


def _usage_of(
    reply: dict[str, Any], text: str, reasoning: str, calls: int, prompt_default: int
) -> tuple[int, int]:
    """``(prompt_tokens, completion_tokens)``。"""
    usage = cast(dict[str, Any] | None, reply.get("usage")) or {}
    prompt = int(usage.get("prompt_tokens", prompt_default))
    completion = int(
        usage.get("completion_tokens", max(1, (len(text) + len(reasoning)) // 4 + 20 * calls))
    )
    return prompt, completion


def _stream_pacing(reply: dict[str, Any]) -> tuple[int, float]:
    """``reply.stream`` の ``(chunk_chars, delay_s)``。"""
    stream = cast(dict[str, Any] | None, reply.get("stream"))
    if not stream:
        return 0, 0.0
    return int(stream.get("chunk_chars", 0)), float(stream.get("delay_ms", 0)) / 1000


def _pieces(s: str, chunk_chars: int, default: tuple[str, ...]) -> list[str]:
    """``chunk_chars`` 字ずつに割る。"""
    if chunk_chars <= 0:
        return [p for p in default if p]
    return [s[i : i + chunk_chars] for i in range(0, len(s), chunk_chars)]


def sse_chunks(reply: dict[str, Any], model: str, prompt_tokens: int = 10) -> Iterator[bytes]:
    """``reply`` を provider が読む形の SSE チャンク列にする。"""
    chunk_chars, delay = _stream_pacing(reply)
    stall = float(reply.get("stall_ms", 0)) / 1000
    if stall:
        time.sleep(stall)

    def paced(chunk: bytes) -> bytes:
        if delay:
            time.sleep(delay)
        return chunk

    reasoning = str(reply.get("reasoning", ""))
    if reasoning:
        yield _chunk(model, {"role": "assistant", "reasoning": reasoning})
    text = str(reply.get("text", ""))
    if text:
        half = max(1, len(text) // 2)
        for n, piece in enumerate(_pieces(text, chunk_chars, (text[:half], text[half:]))):
            delta = {"role": "assistant", "content": piece} if n == 0 else {"content": piece}
            yield paced(_chunk(model, delta))
    calls = cast(list[dict[str, Any]], reply.get("tool_calls") or [])
    for i, call in enumerate(calls):
        arguments = json.dumps(call.get("args", {}), ensure_ascii=False)
        call_id = f"call_{time.time_ns()}_{i}"
        default = (arguments, "") if i or len(arguments) < 4 else (arguments[:3], arguments[3:])
        for n, piece in enumerate(_pieces(arguments, chunk_chars, default)):
            if n == 0:
                delta: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": i,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": call["name"], "arguments": piece},
                        }
                    ],
                }
            else:
                delta = {"tool_calls": [{"index": i, "function": {"arguments": piece}}]}
            yield paced(_chunk(model, delta))
    finish = reply.get("finish") or ("tool_calls" if calls else "stop")
    if finish == "cut":
        return
    yield _chunk(model, {}, finish=str(finish))
    prompt, completion = _usage_of(reply, text, reasoning, len(calls), prompt_tokens)
    yield _usage_chunk(model, completion, prompt)
    yield b"data: [DONE]\n\n"


def _llamacpp_overflow(window: int, overflow: dict[str, Any]) -> dict[str, Any]:
    """llama-server が窓より長い要求に返す 400 の ``error``。"""
    tokens = int(overflow.get("n_prompt_tokens", window + 1))
    return {
        "code": 400,
        "message": (
            f"request ({tokens} tokens) exceeds the available context size ({window} tokens), "
            "try increasing it"
        ),
        "type": "exceed_context_size_error",
        "n_prompt_tokens": tokens,
        "n_ctx": window,
    }


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        script: Script,
        model: str,
        window: int,
        loaded_window: int,
        flavor: str,
        unloads: list[str],
    ) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.script = script
        self.model = model
        self.window = window
        self.loaded_window = loaded_window
        self.flavor = flavor
        self.loaded = False
        self.unloads = unloads


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        return

    @property
    def _srv(self) -> _Server:
        return cast(_Server, self.server)

    def _json(self, obj: object, status: int = 200) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _lmstudio_models(self) -> dict[str, Any]:
        srv = self._srv
        instance = {
            "id": srv.model,
            "config": {"context_length": srv.loaded_window, "parallel": 4},
            "remaining_ttl_seconds": 3600,
        }
        entry = {
            "type": "llm",
            "key": srv.model,
            "max_context_length": srv.window,
            "loaded_instances": [instance] if srv.loaded else [],
        }
        return {"models": [entry]}

    def _llamacpp_props(self) -> dict[str, Any]:
        srv = self._srv
        return {
            "default_generation_settings": {"params": {}, "n_ctx": srv.loaded_window},
            "total_slots": 4,
            "model_alias": srv.model,
            "model_path": f"/models/{srv.model}.gguf",
            "is_sleeping": not srv.loaded,
        }

    def do_GET(self) -> None:  # noqa: N802
        model = self._srv.model
        if self._srv.flavor == "llamacpp" and not self.path.startswith("/v1/"):
            if self.path.startswith("/props"):
                self._json(self._llamacpp_props())
            else:
                self._json({"error": {"code": 404, "type": "not_found_error"}}, 404)
        elif self._srv.flavor == "lmstudio" and self.path.startswith("/api/"):
            if self.path.startswith("/api/v1/models"):
                self._json(self._lmstudio_models())
            else:
                self._json({"error": "not found"}, 404)
        elif self.path.startswith("/api/tags"):
            self._json({"models": [{"name": model, "model": model}]})
        elif self.path.startswith("/api/ps"):
            entry = {"name": model, "model": model, "context_length": self._srv.loaded_window}
            self._json({"models": [entry] if self._srv.loaded else []})
        elif self.path.startswith("/v1/models"):
            self._json({"data": [{"id": model}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        if self._srv.flavor == "lmstudio" and self.path.startswith("/api/"):
            self._lmstudio_post(raw)
            return
        if self.path.startswith("/api/show"):
            asked = str(json.loads(raw or b"{}").get("model", ""))
            if asked.casefold() != self._srv.model.casefold():
                self._json({"error": f"model '{asked}' not found"}, 404)
                return
            shown: dict[str, Any] = {
                "model_info": {
                    "general.architecture": "gptoss",
                    "gptoss.context_length": self._srv.window,
                },
            }
            if self._srv.loaded_window == self._srv.window:
                shown["parameters"] = f"num_ctx {self._srv.window}"
            self._json(shown)
            return
        if self.path.startswith("/api/generate"):
            asked = json.loads(raw or b"{}")
            if asked.get("keep_alive") == 0:
                self._srv.unloads.append(str(asked.get("model", "")))
            self._json({})
            return
        if not self.path.startswith("/v1/chat/completions"):
            self._json({"error": "not found"}, 404)
            return
        body = cast(dict[str, Any], json.loads(raw or b"{}"))
        reply, seq = self._srv.script.respond(body)
        error = cast(dict[str, Any] | None, reply.get("http_error"))
        if error:
            self._srv.loaded = True
            message = str(error.get("message", ""))
            self._json({"error": {"message": message, "type": "api_error"}}, int(error["status"]))
            self._srv.script.mark_done(seq)
            return
        if "stream_error" in reply:
            self._srv.loaded = True
            self._stream_error(str(reply["stream_error"]))
            self._srv.script.mark_done(seq)
            return
        overflow = cast(dict[str, Any] | None, reply.get("overflow"))
        if overflow:
            self._srv.loaded = True
            self._json({"error": _llamacpp_overflow(self._srv.loaded_window, overflow)}, 400)
            self._srv.script.mark_done(seq)
            return
        load = float(reply.get("load_ms", 0)) / 1000
        if load:
            self._srv.loaded = False
            time.sleep(load)
        self._srv.loaded = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in sse_chunks(reply, self._srv.model, max(len(raw) // 4, 10)):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self._srv.script.mark_done(seq)

    def _lmstudio_post(self, raw: bytes) -> None:
        if not self.path.startswith("/api/v1/models/unload"):
            self._json({"error": "not found"}, 404)
            return
        instance = str(json.loads(raw or b"{}").get("instance_id", ""))
        if not self._srv.loaded or instance != self._srv.model:
            self._json({"error": {"code": "model_not_found"}}, 404)
            return
        self._srv.loaded = False
        self._srv.unloads.append(instance)
        self._json({"instance_id": instance})

    def _stream_error(self, message: str) -> None:
        """LM Studio と同じく、200 の SSE で ``event: error`` を返す。"""
        data = json.dumps({"error": {"message": message}, "message": message})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(f"event: error\ndata: {data}\n\n".encode())
        self.wfile.flush()


class FakeOllama:
    """台本で応答する擬似 Ollama。"""

    def __init__(
        self,
        script: Path,
        *,
        model: str = FAKE_MODEL,
        workspace: Path | None = None,
        window: int = 131072,
        loaded_window: int | None = None,
        flavor: str = "ollama",
    ) -> None:
        self.script = Script(load_script(script), workspace=workspace)
        self.model = model
        self.window = window
        self.loaded_window = loaded_window or window
        self.flavor = flavor
        self._unloads: list[str] = []
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    def bind(self) -> int:
        """port を決める（listen まで。接続は backlog に溜まり、``serve()`` で応答が始まる）。"""
        if self._server is None:
            self._server = _Server(
                self.script,
                self.model,
                self.window,
                self.loaded_window,
                self.flavor,
                self._unloads,
            )
        return self.port

    def serve(self) -> None:
        """応答のスレッドを始める。"""
        assert self._server is not None, "bind() before serve()"
        if self._thread is None:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()

    def start(self) -> int:
        port = self.bind()
        self.serve()
        return port

    def stop(self) -> None:
        if self._server is not None:
            if self._thread is not None:
                self._server.shutdown()
                self._thread.join(timeout=5)
                self._thread = None
            self._server.server_close()
            self._server = None

    @property
    def port(self) -> int:
        assert self._server is not None, "start() before port"
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        root = f"http://127.0.0.1:{self.port}"
        return root if self.flavor == "ollama" else f"{root}/v1"

    @property
    def base_url_env(self) -> str:
        return "OLLAMA_HOST" if self.flavor == "ollama" else "OPENAI_BASE_URL"

    @property
    def unloads(self) -> list[str]:
        """降ろされたモデル（LM Studio の unload と Ollama の ``keep_alive: 0``）。"""
        return list(self._unloads)

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.script.requests

    def write_log(self, path: Path) -> None:
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.requests),
            encoding="utf-8",
        )

    def __enter__(self) -> FakeOllama:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="台本で応答する擬似 Ollama（OpenAI 互換 SSE）")
    ap.add_argument("--script", type=Path, required=True, help="JSONL の台本")
    ap.add_argument("--model", default=FAKE_MODEL)
    ap.add_argument("--workspace", type=Path, default=None, help="{workspace} の値")
    ap.add_argument("--log", type=Path, default=None, help="要求ログ（JSONL）の出力先")
    args = ap.parse_args(argv)
    fake = FakeOllama(args.script, model=args.model, workspace=args.workspace)
    port = fake.start()
    print(f"port={port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        fake.stop()
        if args.log is not None:
            fake.write_log(args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
