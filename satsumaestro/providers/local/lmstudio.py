"""LM Studio のネイティブ API（``/api/v1``。古い版は ``/api/v0``）。"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from satsumaestro.providers.detect import LocalKind, ModelContextInfo

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Instance:
    id: str
    context_length: int | None
    ttl_s: int | None


@dataclass(frozen=True)
class _Model:
    key: str
    max_context: int | None
    instances: tuple[_Instance, ...]

    def matches(self, name: str) -> bool:
        """``name`` がこのモデルを指すか（key・インスタンスの id・key の最後の部分）。"""
        wanted = name.strip().casefold()
        names = {self.key.casefold(), self.key.rsplit("/", 1)[-1].casefold()}
        names.update(i.id.casefold() for i in self.instances)
        return wanted in names


def _positive(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _from_v1(entry: dict[str, Any]) -> _Model:
    instances = tuple(
        _Instance(
            id=str(inst.get("id") or entry.get("key") or ""),
            context_length=_positive((inst.get("config") or {}).get("context_length")),
            ttl_s=_positive(inst.get("remaining_ttl_seconds")),
        )
        for inst in entry.get("loaded_instances") or []
    )
    return _Model(
        str(entry.get("key") or ""), _positive(entry.get("max_context_length")), instances
    )


def _from_v0(entry: dict[str, Any]) -> _Model:
    key = str(entry.get("id") or "")
    loaded = entry.get("state") == "loaded"
    instances = (
        (
            _Instance(
                id=key, context_length=_positive(entry.get("loaded_context_length")), ttl_s=None
            ),
        )
        if loaded
        else ()
    )
    return _Model(key, _positive(entry.get("max_context_length")), instances)


_FIRST_SEEN: dict[tuple[str, str], bool] = {}
_USED: dict[tuple[str, str], LMStudioServer] = {}


def reset_lmstudio_state() -> None:
    """記録を空にする（テスト用）。"""
    _FIRST_SEEN.clear()
    _USED.clear()


@dataclass(frozen=True)
class LMStudioServer:
    """LM Studio のサーバ（``root`` は ``/v1`` を除いたルート。``token`` は API トークン）。"""

    root: str
    token: str | None = field(default=None, repr=False)

    @property
    def kind(self) -> LocalKind:
        return "lmstudio"

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def _models(self, http: httpx.AsyncClient) -> list[_Model]:
        """LLM の一覧（v1 が無い古い版は v0 で読む）。"""
        resp = await http.get(f"{self.root}/api/v1/models", headers=self.headers())
        if resp.status_code == 404:
            resp = await http.get(f"{self.root}/api/v0/models", headers=self.headers())
            resp.raise_for_status()
            data = resp.json().get("data") or []
            return [_from_v0(e) for e in data if e.get("type") in ("llm", "vlm")]
        resp.raise_for_status()
        return [_from_v1(e) for e in resp.json().get("models") or [] if e.get("type") == "llm"]

    async def _fetch(
        self, client: httpx.AsyncClient | None, timeout_s: float
    ) -> list[_Model] | None:
        owned = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
        try:
            return await self._models(http)
        except (httpx.HTTPError, ValueError) as exc:
            _log.debug("LM Studio model list query failed: %s", exc)
            return None
        finally:
            if owned:
                await http.aclose()

    def _find(self, models: list[_Model], model: str) -> _Model | None:
        entry = next((m for m in models if m.matches(model)), None)
        if entry is not None:
            _FIRST_SEEN.setdefault((self.root, entry.key), bool(entry.instances))
        return entry

    async def reachable(
        self, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> str | None:
        """``/api/v1/models`` に届くか。届かなければその理由。"""
        owned = client is None
        http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
        try:
            await self._models(http)
        except (httpx.HTTPError, ValueError) as exc:
            return f"{type(exc).__name__}: {exc}"
        finally:
            if owned:
                await http.aclose()
        return None

    async def has_model(
        self, model: str, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> bool | None:
        """一覧にモデルがあるか（読めなければ ``None``）。"""
        models = await self._fetch(client, timeout_s)
        if models is None:
            return None
        return self._find(models, model) is not None

    async def list_models(
        self, *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
    ) -> list[str] | None:
        """LLM の key。"""
        models = await self._fetch(client, timeout_s)
        return None if models is None else [m.key for m in models]

    async def context_info(
        self, model: str, *, client: httpx.AsyncClient | None = None
    ) -> ModelContextInfo:
        """載っていればそのインスタンスの窓、載っていなければモデルの最大の窓。"""
        models = await self._fetch(client, 2.0)
        if models is None:
            return ModelContextInfo(server="lmstudio")
        entry = self._find(models, model)
        if entry is None:
            return ModelContextInfo(ps_ok=True, server="lmstudio")
        instance = next(
            (i for i in entry.instances if i.id.casefold() == model.strip().casefold()),
            entry.instances[0] if entry.instances else None,
        )
        if instance is not None and instance.context_length is not None:
            return ModelContextInfo(
                effective=instance.context_length,
                native=entry.max_context,
                source="lmstudio.loaded",
                loaded=True,
                expires_at=_expiry(instance.ttl_s),
                ps_ok=True,
                server="lmstudio",
            )
        return ModelContextInfo(
            native=entry.max_context,
            source="lmstudio.max" if entry.max_context is not None else "none",
            loaded=instance is not None,
            ps_ok=True,
            server="lmstudio",
        )

    async def model_loaded(self, model: str) -> bool | None:
        """モデルのインスタンスが載っているか。"""
        models = await self._fetch(None, 1.5)
        entry = None if models is None else self._find(models, model)
        return None if entry is None else bool(entry.instances)

    def note_used(self, model: str) -> None:
        """このプロセスがモデルを使った（応答を受け始めた）と記録する。"""
        _USED[(self.root, model)] = self

    async def unload(self, instance_id: str, *, client: httpx.AsyncClient) -> bool:
        """``/api/v1/models/unload`` でインスタンスを降ろす。"""
        try:
            resp = await client.post(
                f"{self.root}/api/v1/models/unload",
                json={"instance_id": instance_id},
                headers=self.headers(),
            )
        except httpx.HTTPError as exc:
            _log.debug("LM Studio unload failed: %s", exc)
            return False
        if resp.status_code >= 400:
            _log.debug("LM Studio refused the unload: %s %s", resp.status_code, resp.text[:200])
            return False
        return True


def _expiry(ttl_s: int | None) -> str | None:
    if ttl_s is None:
        return None
    at = datetime.datetime.now().astimezone() + datetime.timedelta(seconds=ttl_s)
    return at.isoformat(timespec="seconds")


async def unload_models_loaded_here(
    *, client: httpx.AsyncClient | None = None, timeout_s: float = 3.0
) -> list[str]:
    """このプロセスが使い、最初に見たときには載っていなかったモデルを降ろし、その key を返す。"""
    owned = client is None
    http = client if client is not None else httpx.AsyncClient(timeout=timeout_s)
    done: list[str] = []
    try:
        for (_, model), server in list(_USED.items()):
            models = await server._fetch(http, timeout_s)
            entry = None if models is None else next((m for m in models if m.matches(model)), None)
            if entry is None or _FIRST_SEEN.get((server.root, entry.key)) is not False:
                continue
            results = [await server.unload(i.id, client=http) for i in entry.instances]
            if results and all(results):
                done.append(entry.key)
    finally:
        if owned:
            await http.aclose()
    return done
