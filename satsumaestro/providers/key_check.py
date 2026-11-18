"""クラウドの API の鍵を、トークンを使わない要求で確かめる。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from satsumaestro.providers.choice import CLOUD_KEYS
from satsumaestro.providers.failures import body_fields, failure_kind

_log = logging.getLogger(__name__)

_URLS = {
    "openai": "https://api.openai.com/v1/models",
    "anthropic": "https://api.anthropic.com/v1/models",
    "xai": "https://api.x.ai/v1/api-key",
}
_XAI_BLOCKS = ("api_key_blocked", "api_key_disabled", "team_blocked")
_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class KeyCheck:
    """確かめた結果（断られたら状態とプロバイダの文。止められた xAI の鍵は 200 と項目の名前）。"""

    choice: str
    rejected: bool
    status: int
    message: str = ""


def _headers(choice: str, key: str) -> dict[str, str]:
    if choice == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"}


async def check_key(client: httpx.AsyncClient, choice: str, key: str) -> KeyCheck | None:
    """鍵を 1 つ確かめる（届かない・断られたか読めないときは ``None``）。"""
    try:
        response = await client.get(_URLS[choice], headers=_headers(choice, key))
    except httpx.HTTPError as exc:
        _log.info("key check for %s did not finish: %s", choice, exc)
        return None
    status = response.status_code
    try:
        body = response.json()
    except ValueError:
        body = None
    if status == 200:
        blocked = [name for name in _XAI_BLOCKS if isinstance(body, dict) and body.get(name)]
        _log.info("key check for %s: %s", choice, ", ".join(blocked) or "accepted")
        return KeyCheck(choice, bool(blocked), status, ", ".join(blocked))
    codes, message = body_fields(body)
    if failure_kind(status, codes, message) != "key_rejected":
        _log.info("key check for %s answered %s", choice, status)
        return None
    _log.info("key check for %s: rejected (%s)", choice, status)
    return KeyCheck(choice, True, status, message.strip() or response.reason_phrase)


async def check_keys(
    env: Mapping[str, str], *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, KeyCheck]:
    """設定されたクラウドの鍵をまとめて確かめる（結果の分かった物だけを返す）。"""
    keys = {choice: env[name] for choice, name in CLOUD_KEYS.items() if env.get(name)}
    if not keys:
        return {}
    async with httpx.AsyncClient(timeout=_TIMEOUT_S, transport=transport) as client:
        found = await asyncio.gather(*(check_key(client, c, k) for c, k in keys.items()))
    return {check.choice: check for check in found if check is not None}


class KeyChecks:
    """REPL の起動時に裏で確かめた鍵（終わるまでは何も断られていない扱い）。"""

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport
        self._found: dict[str, KeyCheck] = {}

    async def run(self, env: Mapping[str, str]) -> None:
        self._found = await check_keys(env, transport=self._transport)

    def rejected(self) -> dict[str, KeyCheck]:
        """断られた鍵（種類ごと）。"""
        return {choice: check for choice, check in self._found.items() if check.rejected}


__all__ = ["KeyCheck", "KeyChecks", "check_key", "check_keys"]
