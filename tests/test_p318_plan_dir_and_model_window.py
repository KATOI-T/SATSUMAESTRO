"""計画ファイルの名前の写し間違いと、モデルを替えたときの窓の合わせ直し。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from satsumaestro.security.state_write_guard import (
    PROHIBITED_STATE_WRITE_GUIDANCE,
    PROHIBITED_STATE_WRITE_PREFIX,
    plan_dir_guidance,
)
from satsumaestro.session.models import ToolResultBlock, plan_file_relpath
from tests.test_p145_plan_exit_nudge import _runtime
from tests.test_p153_missing_model import _dispatch_runtime, _model, _route_http
from tests.test_p232_state_write_guard import _Capture

_PLAN = ".satsuma/plans/1a072581aba04a339cf4a2bea3d41c5c.md"


def _outputs(rt: object) -> dict[str, ToolResultBlock]:
    return {
        b.tool_use_id: b
        for m in rt.session.messages  # type: ignore[attr-defined]
        for b in m.blocks
        if isinstance(b, ToolResultBlock)
    }


@pytest.mark.parametrize(
    "path",
    [
        ".satsuma/plans/1a072581aba04a3c9cf4a2bea3d41c5c.md",
        "./.satsuma/plans/plan.md",
        ".satsuma/plans/../plans/other.md",
    ],
)
def test_a_write_into_the_plan_directory_gets_the_plan_file(tmp_path: Path, path: str) -> None:
    guidance = plan_dir_guidance("write_file", path, tmp_path, _PLAN)
    assert guidance is not None
    assert f"`{_PLAN}`" in guidance
    assert "path spelling" not in guidance


def test_an_absolute_path_into_the_plan_directory_is_recognized(tmp_path: Path) -> None:
    path = str(tmp_path / ".satsuma" / "plans" / "x.md")
    assert plan_dir_guidance("append_file", path, tmp_path, _PLAN) is not None


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("write_file", ".satsuma/settings.local.json"),
        ("write_file", ".satsuma/plans.md"),
        ("write_file", "notes/plans/x.md"),
        ("bash", ".satsuma/plans/x.md"),
        ("write_file", ""),
    ],
)
def test_other_writes_keep_the_usual_guidance(tmp_path: Path, name: str, path: str) -> None:
    assert plan_dir_guidance(name, path, tmp_path, _PLAN) is None


async def test_a_miscopied_plan_file_name_is_told_the_right_path(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    sid = rt.session.session_id
    wrong = plan_file_relpath(sid[:10] + ("c" if sid[10] != "c" else "d") + sid[11:])
    plan = plan_file_relpath(sid)
    script[:] = [
        [
            ToolUse(id="w1", name="write_file", input={"path": wrong, "content": "# Plan\n"}),
            MessageStop(),
        ],
        [
            ToolUse(id="w2", name="write_file", input={"path": plan, "content": "# Plan\n"}),
            MessageStop(),
        ],
        [TextDelta(text="ok"), MessageStop()],
    ]
    summary = await rt.run_turn("計画して", _Capture())  # type: ignore[arg-type]
    out = _outputs(rt)
    assert out["w1"].output.startswith(PROHIBITED_STATE_WRITE_PREFIX)
    assert f"Write the plan to `{plan}`" in out["w1"].output
    assert PROHIBITED_STATE_WRITE_GUIDANCE not in out["w1"].output
    assert out["w2"].is_error is False
    assert summary.state_write_blocks == 1
    assert not (tmp_path / wrong).exists()
    assert (tmp_path / plan).read_text() == "# Plan\n"


async def test_other_state_writes_keep_the_usual_guidance(tmp_path: Path) -> None:
    rt, script = _runtime(tmp_path)
    script[:] = [
        [
            ToolUse(
                id="w1",
                name="write_file",
                input={"path": ".satsuma/settings.local.json", "content": "{}"},
            ),
            MessageStop(),
        ],
        [TextDelta(text="ok"), MessageStop()],
    ]
    await rt.run_turn("設定して", _Capture())  # type: ignore[arg-type]
    assert PROHIBITED_STATE_WRITE_GUIDANCE in _outputs(rt)["w1"].output


def _ollama(windows: dict[str, int]) -> object:
    """``windows`` のモデルが読み込み済み（``/api/ps`` にその窓）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            names = [*windows, "gpt-oss:120b"]
            return httpx.Response(200, json={"models": [{"name": n, "model": n} for n in names]})
        if request.url.path == "/api/show":
            name = json.loads(request.content)["model"]
            return httpx.Response(200, json={"model": name, "model_info": {}})
        if request.url.path == "/api/ps":
            loaded = [{"name": n, "model": n, "context_length": w} for n, w in windows.items()]
            return httpx.Response(200, json={"models": loaded})
        return httpx.Response(404)

    return handler


async def test_switching_the_model_fits_the_new_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_http(monkeypatch, _ollama({"qwen3:14b-ctx8k": 8_192, "gpt-oss:20b": 131_072}))  # type: ignore[arg-type]
    rt = _dispatch_runtime(tmp_path)
    rt.config.compaction.context_window = 131_072
    rt.config.compact_token_threshold = 104_857
    rt.fit_base = (120_000, 0)

    out = await _model(rt, "/model qwen3:14b-ctx8k")
    assert rt.session.model == "qwen3:14b-ctx8k"
    assert (rt.config.compaction.context_window, rt.config.compact_token_threshold) == (
        8_192,
        6_553,
    )
    assert rt.prompt_builder._instruction_tokens == 1_024
    assert "文脈の窓 8,192 トークン（実効 num_ctx（ロード済み））" in out
    assert "OLLAMA_CONTEXT_LENGTH=16384" in out

    out = await _model(rt, "/model gpt-oss:20b")
    assert (rt.config.compaction.context_window, rt.config.compact_token_threshold) == (
        131_072,
        104_857,
    )
    assert rt.prompt_builder._instruction_tokens == 16_384
    assert "文脈の窓 131,072 トークン" in out
    assert "OLLAMA_CONTEXT_LENGTH" not in out


async def test_switching_without_a_start_value_leaves_the_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _route_http(monkeypatch, _ollama({"qwen3:14b-ctx8k": 8_192}))  # type: ignore[arg-type]
    rt = _dispatch_runtime(tmp_path)
    rt.config.compaction.context_window = 131_072
    out = await _model(rt, "/model qwen3:14b-ctx8k")
    assert rt.session.model == "qwen3:14b-ctx8k"
    assert rt.config.compaction.context_window == 131_072
    assert "文脈の窓" not in out
