"""待ちが続いたら、モデルが Ollama に載っているかを 1 回だけ確かめる。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Protocol

from satsumaestro.engine.context_fit import ollama_root_for
from satsumaestro.providers.detect import model_loaded

if TYPE_CHECKING:
    from satsumaestro.engine.runtime import ConversationRuntime

_log = logging.getLogger(__name__)

LOAD_PROBE_AFTER_S = 3.0
_POLL_S = 0.5

LoadProbe = Callable[[], Awaitable[str | None]]


class LoadWatchRenderer(Protocol):
    def waiting_since(self) -> float | None: ...

    def on_model_loading(self, model: str, since: float) -> None: ...


def ollama_load_probe(rt: ConversationRuntime, env: Mapping[str, str]) -> LoadProbe:
    """``rt`` がいま使っているモデルを、照会のたびに引き直して確かめる照会を作る。"""

    async def probe() -> str | None:
        model = rt.session.model
        root = ollama_root_for(rt.config, model, env)
        if root is None:
            return None
        return model if await model_loaded(root, model) is False else None

    return probe


async def watch_model_load(
    render: LoadWatchRenderer,
    probe: LoadProbe,
    *,
    after_s: float = LOAD_PROBE_AFTER_S,
    poll_s: float = _POLL_S,
) -> None:
    """ターンの間じゅう待ちの区間を見張り、長引いた区間で 1 回だけ照会する（cancel で終わる）。"""
    probed: float | None = None
    while True:
        since = render.waiting_since()
        if since is not None and since != probed and time.monotonic() - since >= after_s:
            probed = since
            try:
                model = await probe()
                if model is not None:
                    render.on_model_loading(model, since)
            except Exception as exc:  # noqa: BLE001
                _log.debug("could not check whether the model is loaded: %s", exc)
        await asyncio.sleep(poll_s)


@contextlib.asynccontextmanager
async def watching_model_load(
    render: LoadWatchRenderer, probe: LoadProbe | None
) -> AsyncIterator[None]:
    """この区間だけ :func:`watch_model_load` を走らせる（``probe`` が None なら何もしない）。"""
    if probe is None:
        yield
        return
    task = asyncio.create_task(watch_model_load(render, probe))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
