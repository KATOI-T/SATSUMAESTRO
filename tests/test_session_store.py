"""JSONLSessionStore の保存・読込・fork・prune テスト。"""

from __future__ import annotations

import os
from pathlib import Path

from satsumaestro.session.models import ConversationMessage, TextBlock
from satsumaestro.session.store import JSONLSessionStore, new_session


def test_save_load_roundtrip(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    s = new_session(tmp_path, "m")
    s.prompt_history.append("hi")
    s.messages.append(ConversationMessage(role="user", blocks=[TextBlock(text="x")]))
    store.save(s)
    loaded = store.load(store.path_for(tmp_path, s.session_id))
    assert loaded.session_id == s.session_id
    assert loaded.prompt_history == ["hi"]
    assert loaded.messages[0].blocks[0].text == "x"


def test_fork_gets_new_id_but_keeps_history(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    s = new_session(tmp_path, "m")
    s.messages.append(ConversationMessage(role="user", blocks=[TextBlock(text="x")]))
    f = store.fork(s)
    assert f.session_id != s.session_id
    assert f.messages[0].blocks[0].text == "x"


def test_prune_keeps_newest_by_mtime(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    ids: list[str] = []
    for _ in range(5):
        s = new_session(tmp_path, "m")
        store.save(s)
        ids.append(s.session_id)
    sdir = store.path_for(tmp_path, "_").parent
    for i, sid in enumerate(ids):
        os.utime(sdir / f"{sid}.jsonl", (1000 + i, 1000 + i))

    removed = store.prune(tmp_path, max_sessions=2)
    assert removed == 3
    remaining = {p.stem for p in sdir.glob("*.jsonl")}
    assert remaining == {ids[3], ids[4]}


def test_prune_noop_when_under_limit_or_disabled(tmp_path: Path) -> None:
    store = JSONLSessionStore()
    for _ in range(3):
        store.save(new_session(tmp_path, "m"))
    assert store.prune(tmp_path, max_sessions=10) == 0
    assert store.prune(tmp_path, max_sessions=0) == 0
    assert len(list((store.path_for(tmp_path, "_").parent).glob("*.jsonl"))) == 3
