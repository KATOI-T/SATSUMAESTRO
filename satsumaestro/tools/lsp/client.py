"""LSP JSON-RPC クライアント。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from satsumaestro.security.secret_env import strip_secret_env
from satsumaestro.util.errors import ToolError
from satsumaestro.util.proc import terminate

_log = logging.getLogger(__name__)

_MAX_FRAME_BYTES = 32 * 1024 * 1024


def encode_message(payload: dict[str, Any]) -> bytes:
    """JSON-RPC メッセージを Content-Length フレームへ包む。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body


async def read_message(stream: asyncio.StreamReader) -> dict[str, Any] | None:
    """1 フレーム読み取る。"""
    content_length: int | None = None
    while True:
        line = await stream.readline()
        if not line:
            return None
        stripped = line.strip()
        if not stripped:
            break
        if stripped.lower().startswith(b"content-length:"):
            try:
                content_length = int(stripped.split(b":", 1)[1])
            except ValueError as e:
                raise ToolError(f"lsp: invalid Content-Length header: {line!r}") from e
    if content_length is None or not 0 <= content_length <= _MAX_FRAME_BYTES:
        raise ToolError(f"lsp: missing or oversized Content-Length: {content_length}")
    body = await stream.readexactly(content_length)
    try:
        parsed = json.loads(body)
    except ValueError as e:
        raise ToolError(f"lsp: invalid JSON frame: {e}") from e
    if not isinstance(parsed, dict):
        raise ToolError("lsp: frame is not a JSON object")
    return parsed


class LspClient:
    """1 言語サーバ 1 クライアント（ターン跨ぎで再利用）。"""

    def __init__(
        self,
        command: list[str],
        root: Path,
        *,
        timeout_s: float = 15.0,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.command = command
        self.root = root
        self.timeout_s = timeout_s
        self.env: dict[str, str] = dict(env) if env is not None else strip_secret_env(os.environ)
        self.capabilities: dict[str, Any] = {}
        self.diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._diag_events: dict[str, asyncio.Event] = {}
        self._open_uris: set[str] = set()
        self._next_id = 0
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=self.root,
            env=self.env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        result = await self.request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.root.as_uri(),
                "capabilities": {},
                "workspaceFolders": [{"uri": self.root.as_uri(), "name": self.root.name}],
            },
        )
        if isinstance(result, dict):
            caps = result.get("capabilities")
            if isinstance(caps, dict):
                self.capabilities = caps
        await self.notify("initialized", {})

    async def shutdown(self) -> None:
        """``shutdown``→``exit`` の礼儀を尽くしてから pgroup ごと回収する。"""
        if self._proc is None:
            return
        try:
            await asyncio.wait_for(self.request("shutdown", None), timeout=3.0)
            await self.notify("exit", {})
        except Exception:
            _log.debug("lsp shutdown handshake failed; killing", exc_info=True)
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        await terminate(self._proc)
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    async def request(self, method: str, params: object) -> object:
        """id 付き要求を送り、result を返す。"""
        self._next_id += 1
        msg_id = self._next_id
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = fut
        await self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        try:
            reply = await asyncio.wait_for(fut, timeout=self.timeout_s)
        except TimeoutError as e:
            self._pending.pop(msg_id, None)
            raise ToolError(f"lsp: {method} timed out after {self.timeout_s}s") from e
        if "error" in reply:
            err = reply["error"]
            raise ToolError(f"lsp: {method} failed: {err.get('message', err)}")
        return reply.get("result")

    async def notify(self, method: str, params: object) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, payload: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            raise ToolError("lsp: server process is not running")
        async with self._write_lock:
            proc.stdin.write(encode_message(payload))
            await proc.stdin.drain()

    async def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        try:
            while True:
                msg = await read_message(proc.stdout)
                if msg is None:
                    break
                await self._route(msg)
        except (ToolError, asyncio.IncompleteReadError):
            _log.debug("lsp reader stopped on malformed stream", exc_info=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.debug("lsp reader crashed", exc_info=True)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_result({"error": {"message": "server exited"}})

    async def _route(self, msg: dict[str, Any]) -> None:
        if "id" in msg and ("result" in msg or "error" in msg):
            fut = self._pending.pop(msg["id"], None)
            if fut is not None and not fut.done():
                fut.set_result(msg)
            return
        method = msg.get("method", "")
        if method == "textDocument/publishDiagnostics":
            params = msg.get("params") or {}
            uri = str(params.get("uri", ""))
            diags = params.get("diagnostics")
            self.diagnostics[uri] = diags if isinstance(diags, list) else []
            event = self._diag_events.get(uri)
            if event is not None:
                event.set()
            return
        if "id" in msg:
            result: object = None
            if method == "workspace/configuration":
                items = (msg.get("params") or {}).get("items") or []
                result = [None] * len(items)
            await self._send({"jsonrpc": "2.0", "id": msg["id"], "result": result})

    async def open_document(self, path: Path, language_id: str) -> str:
        """ファイルを（再）オープンして uri を返す。"""
        uri = path.as_uri()
        text = path.read_text(encoding="utf-8", errors="replace")
        if uri in self._open_uris:
            await self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
            self._open_uris.discard(uri)
        self.diagnostics.pop(uri, None)
        self._diag_events[uri] = asyncio.Event()
        await self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 1,
                    "text": text,
                }
            },
        )
        self._open_uris.add(uri)
        return uri

    async def wait_diagnostics(self, uri: str, timeout_s: float) -> list[dict[str, Any]]:
        """didOpen 後の ``publishDiagnostics`` を待つ（届かなければ空 = 問題なし扱い）。"""
        event = self._diag_events.get(uri)
        if event is not None:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_s)
            except TimeoutError:
                _log.debug("lsp: no diagnostics for %s within %ss", uri, timeout_s)
        return self.diagnostics.get(uri, [])
