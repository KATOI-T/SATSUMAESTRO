"""LSP レジストリ: 言語→クライアントの遅延起動・dispatch・整形。"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from satsumaestro.config.models import LspConfig, SandboxConfig
from satsumaestro.security.sandbox import child_environment
from satsumaestro.tools.lsp.client import LspClient
from satsumaestro.tools.lsp.models import (
    COMPLETION_KIND_NAMES,
    SEVERITY_NAMES,
    SYMBOL_KIND_NAMES,
    LspRequest,
)
from satsumaestro.tools.lsp.servers import (
    EXTENSION_LANGUAGES,
    language_for_path,
    resolve_server_command,
)
from satsumaestro.util.errors import ToolError

_log = logging.getLogger(__name__)

_MAX_ITEMS = 100
_DIAGNOSTICS_WAIT_S = 10.0


def _uri_to_path(uri: str) -> str:
    parsed = urlparse(uri)
    return unquote(parsed.path) if parsed.scheme == "file" else uri


_POSITION_METHODS: dict[LspRequest, str] = {
    LspRequest.HOVER: "textDocument/hover",
    LspRequest.DEFINITION: "textDocument/definition",
    LspRequest.REFERENCES: "textDocument/references",
    LspRequest.COMPLETION: "textDocument/completion",
}
_PATH_OPTIONAL = frozenset({LspRequest.DIAGNOSTICS, LspRequest.SYMBOLS})


class LanguageServers:
    """言語ごとに :class:`LspClient` を遅延起動・キャッシュする。"""

    def __init__(
        self,
        root: Path,
        config: LspConfig,
        *,
        sandbox: SandboxConfig | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self._env = child_environment(sandbox, env)
        self._clients: dict[str, LspClient | None] = {}

    async def _client_for(self, path: Path) -> tuple[LspClient, str]:
        """path の言語のクライアントを（必要なら起動して）返す。"""
        language = language_for_path(path)
        if language is None:
            handled = " ".join(EXTENSION_LANGUAGES)
            raise ToolError(
                f"lsp has no language server for {path.suffix or path.name} files "
                f"(it handles: {handled})"
            )
        if language in self._clients:
            client = self._clients[language]
            if client is None:
                raise ToolError(self._missing_message(language))
            return client, language
        command = resolve_server_command(language, self.config.servers)
        if command is None:
            self._clients[language] = None
            raise ToolError(self._missing_message(language))
        client = LspClient(
            command, self.root, timeout_s=self.config.request_timeout_s, env=self._env
        )
        try:
            await client.start()
        except Exception as e:
            await client.shutdown()
            self._clients[language] = None
            _log.warning("lsp server for %s failed to start: %s", language, e)
            raise ToolError(f"lsp: failed to start {language} server ({command[0]}): {e}") from e
        self._clients[language] = client
        return client, language

    def _missing_message(self, language: str) -> str:
        return (
            f"lsp: no language server available for {language}. Install one "
            f"(see doctor output) or configure lsp.servers.{language}."
        )

    async def aclose(self) -> None:
        """全クライアントを shutdown する（AsyncExitStack / _shutdown が呼ぶ）。"""
        for client in self._clients.values():
            if client is not None:
                await client.shutdown()
        self._clients.clear()

    async def dispatch(
        self,
        action: LspRequest,
        *,
        path: Path | None,
        line: int | None,
        character: int | None,
        query: str | None,
    ) -> str:
        if path is None and action not in _PATH_OPTIONAL:
            raise ToolError(f"lsp {action.value} needs path (a file in the workspace)")
        method = _POSITION_METHODS.get(action)
        if method is not None and (line is None or character is None):
            raise ToolError(f"lsp {action.value} needs line and character (both 0-based)")
        if action is LspRequest.DIAGNOSTICS:
            return await self._diagnostics(path)
        if action is LspRequest.SYMBOLS:
            return await self._symbols(path, query)
        assert path is not None
        if action is LspRequest.FORMAT:
            return await self._format(path)
        assert method is not None and line is not None and character is not None
        client, _ = await self._client_for(path)
        uri = await client.open_document(path, language_for_path(path) or "plaintext")
        params: dict[str, Any] = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }
        if action is LspRequest.REFERENCES:
            params["context"] = {"includeDeclaration": True}
        result = await client.request(method, params)
        if action is LspRequest.HOVER:
            return self._format_hover(result)
        if action is LspRequest.COMPLETION:
            return self._format_completion(result)
        kind = "definition" if action is LspRequest.DEFINITION else "reference"
        return self._format_locations(kind, result)

    async def _diagnostics(self, path: Path | None) -> str:
        if path is None:
            items: list[str] = []
            for client in self._clients.values():
                if client is None:
                    continue
                for uri, diags in sorted(client.diagnostics.items()):
                    items.extend(self._diagnostic_lines(uri, diags))
            if not items:
                return "no cached diagnostics (query a specific path first)"
            return self._capped(f"{len(items)} diagnostics (cached)", items)
        client, _ = await self._client_for(path)
        uri = await client.open_document(path, language_for_path(path) or "plaintext")
        diags = await client.wait_diagnostics(uri, _DIAGNOSTICS_WAIT_S)
        lines = self._diagnostic_lines(uri, diags)
        if not lines:
            return f"no diagnostics for {self._rel(path)}"
        return self._capped(f"{len(lines)} diagnostics", lines)

    def _diagnostic_lines(self, uri: str, diags: list[dict[str, Any]]) -> list[str]:
        lines = []
        for d in diags:
            start = (d.get("range") or {}).get("start") or {}
            severity = SEVERITY_NAMES.get(d.get("severity") or 0, "info")
            source = d.get("source")
            suffix = f" [{source}]" if source else ""
            lines.append(
                f"{self._rel_uri(uri)}:{start.get('line', 0)}:{start.get('character', 0)} "
                f"{severity} {d.get('message', '').strip()}{suffix}"
            )
        return lines

    async def _symbols(self, path: Path | None, query: str | None) -> str:
        if path is None and not query:
            raise ToolError(
                "lsp symbols needs path (the symbols of that file) or query (a workspace search)"
            )
        if path is not None:
            client, _ = await self._client_for(path)
            uri = await client.open_document(path, language_for_path(path) or "plaintext")
            result = await client.request(
                "textDocument/documentSymbol", {"textDocument": {"uri": uri}}
            )
            lines = self._symbol_lines(result, default_uri=uri)
            header = f"{len(lines)} symbols in {self._rel(path)}"
        else:
            running = next((c for c in self._clients.values() if c is not None), None)
            if running is None:
                raise ToolError(
                    "lsp symbols with only a query needs a language server that is already "
                    "running; ask about a file first"
                )
            result = await running.request("workspace/symbol", {"query": query})
            lines = self._symbol_lines(result, default_uri="")
            header = f"{len(lines)} symbols matching {query!r}"
        if not lines:
            return "no symbols found"
        return self._capped(header, lines)

    def _symbol_lines(self, result: object, *, default_uri: str, indent: int = 0) -> list[str]:
        """DocumentSymbol[]（階層）/ SymbolInformation[]（平坦）の両形式を写像する。"""
        lines: list[str] = []
        if not isinstance(result, list):
            return lines
        for item in result:
            if not isinstance(item, dict):
                continue
            kind = SYMBOL_KIND_NAMES.get(item.get("kind") or 0, "symbol")
            name = item.get("name", "?")
            if "location" in item:
                loc = item.get("location") or {}
                start = ((loc.get("range") or {}).get("start")) or {}
                where = f"{self._rel_uri(str(loc.get('uri', default_uri)))}"
            else:
                start = ((item.get("selectionRange") or item.get("range") or {}).get("start")) or {}
                where = self._rel_uri(default_uri)
            prefix = "  " * indent
            lines.append(
                f"{prefix}{kind} {name} {where}:{start.get('line', 0)}:{start.get('character', 0)}"
            )
            children = item.get("children")
            if isinstance(children, list) and children:
                lines.extend(
                    self._symbol_lines(children, default_uri=default_uri, indent=indent + 1)
                )
        return lines

    async def _format(self, path: Path) -> str:
        client, _ = await self._client_for(path)
        uri = await client.open_document(path, language_for_path(path) or "plaintext")
        result = await client.request(
            "textDocument/formatting",
            {
                "textDocument": {"uri": uri},
                "options": {"tabSize": 4, "insertSpaces": True},
            },
        )
        if not isinstance(result, list) or not result:
            return f"already formatted: {self._rel(path)}"
        text = path.read_text(encoding="utf-8", errors="replace")
        new_text = _apply_edits(text, result)
        path.write_text(new_text, encoding="utf-8")
        return f"applied {len(result)} formatting edits to {self._rel(path)}"

    def _format_hover(self, result: object) -> str:
        if not isinstance(result, dict):
            return "no hover information"
        contents = result.get("contents")
        parts: list[str] = []

        def _one(item: object) -> None:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("value")
                if isinstance(value, str):
                    parts.append(value)

        if isinstance(contents, list):
            for item in contents:
                _one(item)
        else:
            _one(contents)
        text = "\n".join(p.strip() for p in parts if p.strip())
        return text or "no hover information"

    def _format_locations(self, label: str, result: object) -> str:
        items = result if isinstance(result, list) else [result] if result else []
        lines = []
        for loc in items:
            if not isinstance(loc, dict):
                continue
            uri = loc.get("uri") or loc.get("targetUri") or ""
            rng = loc.get("range") or loc.get("targetSelectionRange") or {}
            start = rng.get("start") or {}
            lines.append(
                f"{self._rel_uri(str(uri))}:{start.get('line', 0)}:{start.get('character', 0)}"
            )
        if not lines:
            return f"no {label}s found"
        return self._capped(f"{len(lines)} {label}(s)", lines)

    def _format_completion(self, result: object) -> str:
        items = (
            result.get("items")
            if isinstance(result, dict)
            else result
            if isinstance(result, list)
            else []
        )
        lines = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            kind = COMPLETION_KIND_NAMES.get(item.get("kind") or 0)
            detail = item.get("detail")
            line = item.get("label", "?")
            if kind:
                line += f" ({kind})"
            if isinstance(detail, str) and detail.strip():
                line += f" — {detail.strip()}"
            lines.append(line)
        if not lines:
            return "no completions"
        return self._capped(f"{len(lines)} completions", lines)

    def _capped(self, header: str, lines: list[str]) -> str:
        shown = lines[:_MAX_ITEMS]
        note = "" if len(lines) <= _MAX_ITEMS else f"\n… ({len(lines) - _MAX_ITEMS} more truncated)"
        return header + "\n" + "\n".join(shown) + note

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _rel_uri(self, uri: str) -> str:
        return self._rel(Path(_uri_to_path(uri))) if uri else "?"


def _apply_edits(text: str, edits: list[dict[str, Any]]) -> str:
    """TextEdit 群を末尾から適用する（オフセットずれ防止）。"""
    lines = text.split("\n")
    offsets = [0]
    for line in lines[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)

    def to_offset(pos: dict[str, Any]) -> int:
        line_no = min(int(pos.get("line", 0)), len(lines) - 1)
        return offsets[line_no] + int(pos.get("character", 0))

    ordered = sorted(
        (e for e in edits if isinstance(e, dict)),
        key=lambda e: to_offset((e.get("range") or {}).get("start") or {}),
        reverse=True,
    )
    for edit in ordered:
        rng = edit.get("range") or {}
        start = to_offset(rng.get("start") or {})
        end = to_offset(rng.get("end") or {})
        text = text[:start] + str(edit.get("newText", "")) + text[end:]
    return text
