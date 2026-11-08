"""llama.cpp の llama-server のネイティブ API（``/props``。router mode は ``/models``）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from satsumaestro.providers.detect import LocalKind, ModelContextInfo

_log = logging.getLogger(__name__)

_UP = ("loaded", "sleeping")


def _positive(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


@dataclass(frozen=True)
class LlamaCppProps:
    """``/props`` から読んだサーバの形（router mode か・1 スロットの窓・スロット数・ツール）。"""

    router: bool
    n_ctx: int | None
    slots: int | None
    sleeping: bool
    model: str
    tools: bool | None = None


@dataclass(frozen=True)
class _Model:
    id: str
    aliases: tuple[str, ...]
    status: str
    n_ctx: int | None
    n_ctx_train: int | None

    def matches(self, name: str) -> bool:
        """``name`` がこのモデルを指すか（router と同じく id か別名に完全に一致）。"""
        wanted = name.strip()
        return wanted == self.id or wanted in self.aliases


def _props(data: dict[str, Any]) -> LlamaCppProps:
    settings = data.get("default_generation_settings") or {}
    caps = data.get("chat_template_caps")
    tools = caps.get("supports_tools") if isinstance(caps, dict) else None
    return LlamaCppProps(
        router=data.get("role") == "router",
        n_ctx=_positive(settings.get("n_ctx")),
        slots=_positive(data.get("total_slots")),
        sleeping=data.get("is_sleeping") is True,
        model=str(data.get("model_alias") or ""),
        tools=tools if isinstance(tools, bool) else None,
    )


def _model(entry: dict[str, Any]) -> _Model:
    status = entry.get("status") or {}
    meta = entry.get("meta") or {}
    return _Model(
        id=str(entry.get("id") or ""),
        aliases=tuple(str(a) for a in entry.get("aliases") or []),
        status=str(status.get("value") or "") if isinstance(status, dict) else "",
        n_ctx=_positive(meta.get("n_ctx")),
        n_ctx_train=_positive(entry.get("context_length")) or _positive(meta.get("n_ctx_train")),
    )


_FIRST_SEEN: dict[tuple[str, str], bool] = {}
_USED: dict[tuple[str, str], LlamaCppServer] = {}


def reset_llamacpp_state() -> None:
    """記録を空にする（テスト用）。"""
    _FIRST_SEEN.clear()
    _USED.clear()


@dataclass(frozen=True)
class LlamaCppServer:
    """llama-server（``root`` は ``/v1`` を除いたルート。``token`` は ``--api-key`` の鍵）。"""

    root: str
    token: str | None = field(default=None, repr=False)

    @property
    def kind(self) -> LocalKind:
        return "llamacpp"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def _get(self, http: httpx.AsyncClient, path: str) -> dict[str, Any]:
        resp = await http.get(f"{self.root}{path}", headers=self.headers())
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"{path} did not return an object")
        return data

    async def _read(
        self, client: httpx.AsyncClient | None, timeout_s: float
    ) -> tuple[LlamaCppProps, list[_Model]] | None:
        """``/props`` と、router mode なら ``/models`` の一覧。"""
        owned = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
        try:
            props = _props(await self._get(http, "/props"))
            if not props.router:
                return props, []
            data = (await self._get(http, "/models")).get("data") or []
            return props, [_model(e) for e in data if isinstance(e, dict)]
        except (httpx.HTTPError, ValueError) as exc:
            _log.debug("llama.cpp query failed: %s", exc)
            return None
        finally:
            if owned:
                await http.aclose()

    def _find(self, models: list[_Model], model: str) -> _Model | None:
        entry = next((m for m in models if m.matches(model)), None)
        if entry is not None:
            _FIRST_SEEN.setdefault((self.root, entry.id), entry.status in _UP)
        return entry

    async def reachable(
        self, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> str | None:
        """``/props`` に届くか（読み込み中の 503 も届いたとみなす）。届かなければその理由。"""
        owned = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
        try:
            resp = await http.get(f"{self.root}/props", headers=self.headers())
            if resp.status_code != 503:
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            if owned:
                await http.aclose()
        return None

    async def has_model(
        self, model: str, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> bool | None:
        """router mode の一覧にモデルがあるか（1 つのモデルのサーバは名前を見ないので真）。"""
        read = await self._read(client, timeout_s)
        if read is None:
            return None
        props, models = read
        return True if not props.router else self._find(models, model) is not None

    async def list_models(
        self, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> list[str] | None:
        """モデルの名前（1 つのモデルを出すサーバならその 1 件）。"""
        read = await self._read(client, timeout_s)
        if read is None:
            return None
        props, models = read
        if not props.router:
            return [props.model] if props.model else []
        return [m.id for m in models if m.status != "incomplete"]

    async def context_info(
        self, model: str, *, client: httpx.AsyncClient | None = None
    ) -> ModelContextInfo:
        """1 スロットの窓（router mode で読み込まれていなければ、モデルが学習した窓）。"""
        read = await self._read(client, 2.0)
        if read is None:
            return ModelContextInfo(server="llamacpp")
        props, models = read
        if not props.router:
            return ModelContextInfo(
                effective=props.n_ctx,
                source="llamacpp.slot" if props.n_ctx is not None else "none",
                loaded=not props.sleeping,
                ps_ok=True,
                server="llamacpp",
                slots=props.slots,
            )
        entry = self._find(models, model)
        if entry is None:
            return ModelContextInfo(ps_ok=True, server="llamacpp")
        if entry.status in _UP and entry.n_ctx is not None:
            return ModelContextInfo(
                effective=entry.n_ctx,
                native=entry.n_ctx_train,
                source="llamacpp.slot",
                loaded=entry.status == "loaded",
                ps_ok=True,
                server="llamacpp",
            )
        return ModelContextInfo(
            native=entry.n_ctx_train,
            source="llamacpp.train" if entry.n_ctx_train is not None else "none",
            loaded=False,
            ps_ok=True,
            server="llamacpp",
        )

    async def model_loaded(self, model: str) -> bool | None:
        """モデルが載っているか（眠っている・router で未ロードなら偽）。"""
        read = await self._read(None, 1.5)
        if read is None:
            return None
        props, models = read
        if not props.router:
            return not props.sleeping
        entry = self._find(models, model)
        return None if entry is None else entry.status == "loaded"

    async def props(self, *, client: httpx.AsyncClient | None = None) -> LlamaCppProps | None:
        """``/props`` の中身（読めなければ ``None``）。"""
        owned = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=2.0)
        try:
            return _props(await self._get(http, "/props"))
        except (httpx.HTTPError, ValueError) as exc:
            _log.debug("llama.cpp /props query failed: %s", exc)
            return None
        finally:
            if owned:
                await http.aclose()

    async def slots(self, *, client: httpx.AsyncClient | None = None) -> int | None:
        """同時に処理できる要求の数（1 つのモデルを出すサーバの ``total_slots``）。"""
        props = await self.props(client=client)
        return None if props is None or props.router else props.slots

    async def tools_supported(self, *, client: httpx.AsyncClient | None = None) -> bool | None:
        """1 つのモデルを出すサーバのチャットテンプレートがツールを受け付けるか。"""
        props = await self.props(client=client)
        return None if props is None or props.router else props.tools

    def note_used(self, model: str) -> None:
        """このプロセスがモデルを使った（応答を受け始めた）と記録する。"""
        _USED[(self.root, model)] = self

    async def unload(self, model: str, *, client: httpx.AsyncClient) -> bool:
        """router mode の ``/models/unload`` でモデルを降ろす。"""
        try:
            resp = await client.post(
                f"{self.root}/models/unload", json={"model": model}, headers=self.headers()
            )
        except httpx.HTTPError as exc:
            _log.debug("llama.cpp unload failed: %s", exc)
            return False
        if resp.status_code >= 400:
            _log.debug("llama.cpp refused the unload: %s %s", resp.status_code, resp.text[:200])
            return False
        return True


async def unload_models_loaded_here(
    *, root: str | None = None, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
) -> list[str]:
    """router mode で、このプロセスが使い、最初に見たときには載っていなかったモデルを降ろす。"""
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
    done: list[str] = []
    try:
        for (used_root, model), server in list(_USED.items()):
            if root is not None and used_root != root:
                continue
            read = await server._read(http, timeout_s)
            if read is None or not read[0].router:
                continue
            entry = next((m for m in read[1] if m.matches(model)), None)
            if entry is None or entry.status not in (*_UP, "loading"):
                continue
            if _FIRST_SEEN.get((server.root, entry.id)) is not False:
                continue
            if await server.unload(entry.id, client=http):
                done.append(entry.id)
                _USED.pop((used_root, model), None)
    finally:
        if owned:
            await http.aclose()
    return done
