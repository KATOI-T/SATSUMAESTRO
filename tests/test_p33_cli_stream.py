"""Worker 子 CLI プロトコルのテスト。"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from satsumaestro.cli import app as cli_app
from satsumaestro.engine.events import MessageStop, TextDelta, ToolUse
from tests.fixtures.mock_provider import MockProvider, SequencedMockProvider

runner = CliRunner()


def _patch_provider(monkeypatch, provider) -> None:
    monkeypatch.setattr(cli_app, "build_provider", lambda config, model, env: provider)


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SATSUMA_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SATSUMA_WORKER_RECEIPT", raising=False)


def _stream(monkeypatch, tmp_path: Path, provider, *extra: str):
    _patch_provider(monkeypatch, provider)
    return runner.invoke(
        cli_app.app,
        ["prompt", "q", "--output-format", "stream-json", "--cwd", str(tmp_path), *extra],
    )


def _events(result) -> list[dict]:
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def test_stream_first_line_ready_last_line_result(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    (tmp_path / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    provider = SequencedMockProvider(
        [
            [ToolUse(id="c1", name="read_file", input={"path": "main.py"}), MessageStop()],
            [TextDelta(text="answer"), MessageStop()],
        ]
    )
    result = _stream(monkeypatch, tmp_path, provider)
    assert result.exit_code == 0, result.output
    events = _events(result)
    ready, last = events[0], events[-1]
    assert ready["event"] == "worker.ready"
    assert ready["protocol"] == 1
    assert ready["session_id"] and isinstance(ready["pid"], int)
    assert Path(ready["cwd"]).resolve() == tmp_path.resolve()
    assert ready["receipt"] is None
    assert last["event"] == "worker.result"
    assert last["final_text"] == "answer"
    assert last["session_id"] == ready["session_id"]
    kinds = {e["kind"] for e in events if e["event"] == "worker.progress"}
    assert "tool_start" in kinds and "tool_end" in kinds and "iteration" in kinds


def test_stream_receipt_is_echoed_verbatim(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    receipt = {
        "repo": "SATSUMAESTRO",
        "task_kind": "bugfix",
        "source_surface": "worker",
        "objective_preview": "fix issue",
    }
    monkeypatch.setenv("SATSUMA_WORKER_RECEIPT", json.dumps(receipt))
    result = _stream(monkeypatch, tmp_path, MockProvider([TextDelta(text="ok"), MessageStop()]))
    assert result.exit_code == 0, result.output
    assert _events(result)[0]["receipt"] == receipt


def test_stream_broken_receipt_echoes_null(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("SATSUMA_WORKER_RECEIPT", "not-json{")
    result = _stream(monkeypatch, tmp_path, MockProvider([TextDelta(text="ok"), MessageStop()]))
    assert result.exit_code == 0, result.output
    assert _events(result)[0]["receipt"] is None


def test_stream_max_iterations_emits_worker_error(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    (tmp_path / ".satsuma").mkdir()
    (tmp_path / ".satsuma" / "settings.json").write_text('{"max_iterations": 2}', encoding="utf-8")
    (tmp_path / "f.txt").write_text("x", encoding="utf-8")
    provider = MockProvider(
        [ToolUse(id="t", name="read_file", input={"path": "f.txt"}), MessageStop()]
    )
    result = _stream(monkeypatch, tmp_path, provider)
    assert result.exit_code == 1
    events = _events(result)
    assert events[0]["event"] == "worker.ready"
    assert events[-1]["event"] == "worker.error"
    assert events[-1]["error_type"] == "MaxIterationsError"
    assert events[-1]["session_id"] == events[0]["session_id"]


def test_prompt_json_payload_includes_session_id(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    _patch_provider(monkeypatch, MockProvider([TextDelta(text="hi"), MessageStop()]))
    result = runner.invoke(
        cli_app.app,
        ["prompt", "q", "--output-format", "json", "--cwd", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["final_text"] == "hi"
    assert payload["session_id"]


def test_prompt_resume_continues_session(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    _patch_provider(monkeypatch, MockProvider([TextDelta(text="first"), MessageStop()]))
    r1 = runner.invoke(
        cli_app.app, ["prompt", "q1", "--output-format", "json", "--cwd", str(tmp_path)]
    )
    assert r1.exit_code == 0, r1.output
    sid = json.loads(r1.stdout)["session_id"]

    _patch_provider(monkeypatch, MockProvider([TextDelta(text="second"), MessageStop()]))
    r2 = runner.invoke(
        cli_app.app,
        ["prompt", "q2", "--output-format", "json", "--cwd", str(tmp_path), "--resume", sid],
    )
    assert r2.exit_code == 0, r2.output
    assert json.loads(r2.stdout)["session_id"] == sid


def test_prompt_resume_unknown_session_errors(monkeypatch, tmp_path: Path) -> None:
    _isolate(monkeypatch, tmp_path)
    _patch_provider(monkeypatch, MockProvider([TextDelta(text="x"), MessageStop()]))
    result = runner.invoke(
        cli_app.app,
        ["prompt", "q", "--cwd", str(tmp_path), "--resume", "deadbeef"],
    )
    assert result.exit_code == 1
    assert "セッションが見つかりません" in result.stderr
