"""スラッシュコマンドの出力に、自動の色付けを入れない。"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from satsumaestro.cli.commands import SlashDispatcher, parse_slash
from satsumaestro.telemetry.log import TraceEntry, TraceLog
from tests.test_p153_missing_model import _dispatch_runtime, _ollama, _route_http

_TAGS = ["glm-4.7-flash:latest", "llama3.3:70b-instruct-q6_K", "qwen3.6:35b-a3b"]
_SGR = re.compile(r"\x1b\[([0-9;]*)m")


async def _colored(rt: Any, line: str, trace_log: TraceLog | None = None) -> str:
    """色の指定ごと書き出した出力（端末に出すときと同じ）。"""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor", width=200)
    await SlashDispatcher(rt, console, trace_log=trace_log).dispatch(parse_slash(line))  # type: ignore[arg-type]
    return buf.getvalue()


async def test_model_list_has_no_color(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _route_http(monkeypatch, _ollama(_TAGS))
    rt = _dispatch_runtime(tmp_path)
    rt.session.model = "qwen3.6:35b-a3b"

    out = await _colored(rt, "/model")

    assert "\x1b[" not in out
    assert "model: qwen3.6:35b-a3b" in out
    assert "    glm-4.7-flash:latest" in out and "  * qwen3.6:35b-a3b" in out


async def test_model_switch_line_is_only_dim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_http(monkeypatch, _ollama(_TAGS))
    rt = _dispatch_runtime(tmp_path)

    out = await _colored(rt, "/model llama3.3:70b-instruct-q6_K")

    assert "model → llama3.3:70b-instruct-q6_K" in _SGR.sub("", out)
    assert set(_SGR.findall(out)) == {"2", "0"}


async def test_missing_model_keeps_its_red_and_yellow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_http(monkeypatch, _ollama(_TAGS))
    rt = _dispatch_runtime(tmp_path)

    out = await _colored(rt, "/model glm-4.7-flash:lates")

    codes = set(_SGR.findall(out))
    assert {"31", "33"} <= codes
    assert not any(c.startswith("1;") for c in codes)


async def test_insights_numbers_have_no_color(tmp_path: Path) -> None:
    log = TraceLog()
    log.add(TraceEntry("s", 1, "turn.end", 0, {"iterations": 4}))
    rt = _dispatch_runtime(tmp_path)

    out = await _colored(rt, "/insights", trace_log=log)

    assert "\x1b[" not in out.split("\n", 1)[0]
    assert "ターン: 1" in out
