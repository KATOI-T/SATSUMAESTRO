"""prompt cache を失ったことの検知（ログに残すだけ）。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from satsumaestro.engine.events import Usage
from satsumaestro.providers.base import ApiRequest
from satsumaestro.session.models import ConversationMessage

_MSG_ADAPTER: TypeAdapter[ConversationMessage] = TypeAdapter(ConversationMessage)

_Part = tuple[str, int]


@dataclass(frozen=True)
class CacheLoss:
    """cache_read が前回より大きく減ったこと。"""

    before: int
    after: int
    changed: str | None

    @property
    def lost(self) -> int:
        return self.before - self.after

    @property
    def unexpected(self) -> bool:
        """リクエストの先頭は前回のままなのに失った（リクエストの側に理由が無い）。"""
        return self.changed is None


def _digest(payload: bytes) -> int:
    """8 バイトの blake2b を int にした安定ハッシュ。"""
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _tools_digest(tools: Sequence[dict[str, Any]]) -> int:
    """ツールの schema の一覧のダイジェスト（鍵の並びに依らないよう並べ替えて直列化する）。"""
    blob = json.dumps(
        list(tools), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return _digest(blob.encode())


def _parts(req: ApiRequest) -> list[_Part]:
    """リクエストを、プロバイダが先頭から読む順の部分に分ける。"""
    parts = [
        ("the model", _digest(req.model.encode())),
        ("the tool list", _tools_digest(req.tools)),
        ("the fixed part of system", _digest(req.system.encode())),
        (
            "the changing part of system (git state, TODOs and so on)",
            _digest(req.system_dynamic.encode()),
        ),
    ]
    parts += [
        (f"message {i} of the conversation", _digest(_MSG_ADAPTER.dump_json(m)))
        for i, m in enumerate(req.messages, 1)
    ]
    return parts


def _first_change(before: list[_Part], after: list[_Part]) -> str | None:
    """前回の部分のうち、今回の同じ位置に残っていない最初のものの名前。"""
    for i, part in enumerate(before):
        if i >= len(after) or after[i] != part:
            return part[0]
    return None


class PromptCacheMonitor:
    """1 往復ごとの ``Usage`` を見て、prompt cache を失ったことを知らせる。"""

    def __init__(self, min_drop: int = 2000) -> None:
        self._min_drop = min_drop
        self._last: tuple[list[_Part], int] | None = None

    def observe(self, req: ApiRequest, usage: Usage) -> CacheLoss | None:
        """1 往復分を見る。"""
        parts = _parts(req)
        last, self._last = self._last, (parts, usage.cache_read)
        if last is None:
            return None
        last_parts, last_read = last
        if last_read - usage.cache_read < self._min_drop:
            return None
        return CacheLoss(
            before=last_read, after=usage.cache_read, changed=_first_change(last_parts, parts)
        )
