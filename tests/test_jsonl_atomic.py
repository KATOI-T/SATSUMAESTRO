"""JSONL 往復・原子性テスト。"""

from __future__ import annotations

from pathlib import Path

from satsumaestro.util.jsonl import read, write_atomic


def test_write_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    lines = [{"role": "user", "text": "こんにちは"}, {"role": "assistant", "text": "hi"}]

    write_atomic(path, lines)

    assert read(path) == lines
    assert "こんにちは" in path.read_text(encoding="utf-8")


def test_write_atomic_overwrites_and_leaves_no_tmp(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    write_atomic(path, [{"n": 1}])
    write_atomic(path, [{"n": 2}, {"n": 3}])

    assert read(path) == [{"n": 2}, {"n": 3}]
    assert not (tmp_path / "session.jsonl.tmp").exists()


def test_write_atomic_creates_missing_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / ".satsuma" / "sessions" / "abc123" / "session.jsonl"
    write_atomic(path, [{"n": 1}])

    assert read(path) == [{"n": 1}]


def test_read_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")

    assert read(path) == [{"a": 1}, {"b": 2}]
