"""セッションの JSONL 永続化・resume・fork。"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from satsumaestro.i18n.text import ui
from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.session.models import (
    ConversationMessage,
    ModeState,
    OmittedBody,
    Session,
    SessionCompaction,
    TodoItem,
)
from satsumaestro.session.redact import redact_rows, secret_values
from satsumaestro.util import jsonl
from satsumaestro.util.clock import now_ms
from satsumaestro.util.errors import SessionError
from satsumaestro.util.paths import AGENT_DIRNAME, ensure_state_gitignore

DEFAULT_MAX_SESSIONS = 10_000


def new_session(workspace_root: str | Path, model: str) -> Session:
    """新規セッションを生成する（session_id は UUID4）。"""
    ts = now_ms()
    return Session(
        session_id=uuid.uuid4().hex,
        created_at_ms=ts,
        updated_at_ms=ts,
        workspace_root=str(Path(workspace_root)),
        model=model,
    )


class JSONLSessionStore:
    """セッションの JSONL 保存・読込・fork。"""

    def path_for(self, root: Path, sid: str) -> Path:
        """ワークスペースを sha256 でハッシュ化したサブディレクトリ配下のパス。"""
        h = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
        return root / AGENT_DIRNAME / "sessions" / h / f"{sid}.jsonl"

    def save(self, s: Session) -> None:
        """セッションを行種別で JSONL へ原子的保存する（秘密は書く直前に伏せる）。"""
        s.updated_at_ms = now_ms()
        lines: list[dict[str, Any]] = [{"type": "session_meta", **_meta(s)}]
        lines += [{"type": "prompt_history", "text": t} for t in s.prompt_history]
        lines += [{"type": "message", **m.model_dump()} for m in s.messages]
        if s.compaction is not None:
            lines.append({"type": "compaction", **s.compaction.model_dump()})
        if s.todos:
            items = [t.model_dump(mode="json", by_alias=True) for t in s.todos]
            lines.append({"type": "todos", "items": items})
        lines.append({"type": "mode", **s.mode.model_dump(mode="json")})
        if s.plan_steps:
            row: dict[str, Any] = {
                "type": "plan_steps",
                "items": list(s.plan_steps),
                "index": s.plan_step_index,
            }
            if s.plan_step_estimates:
                row["estimates"] = list(s.plan_step_estimates)
            if s.plan_budget_lines:
                row["budget_lines"] = s.plan_budget_lines
            lines.append(row)
        if s.git_context is not None:
            lines.append(
                {
                    "type": "git_context",
                    "system": s.git_context,
                    "compaction": s.git_context_compaction,
                    "seen": s.git_seen,
                }
            )
        if s.last_system_digest:
            lines.append({"type": "system_digest", "sha256": s.last_system_digest})
        if s.expanded_tools:
            lines.append({"type": "expanded_tools", "names": list(s.expanded_tools)})
        lines += [{"type": "omitted_body", **b.model_dump()} for b in s.omitted_bodies]
        redacted = redact_rows(lines, values=secret_values())
        root = Path(s.workspace_root)
        jsonl.write_atomic(self.path_for(root, s.session_id), redacted)
        ensure_state_gitignore(root)

    def load(self, path: Path) -> Session:
        """JSONL からセッションを復元する。"""
        try:
            rows = jsonl.read(path)
        except OSError as e:
            raise SessionError(ui("session.store.cannot_read_session", path=path, error=e)) from e
        return self._from_rows(rows, path)

    def prune(self, root: str | Path, max_sessions: int = DEFAULT_MAX_SESSIONS) -> int:
        """ワークスペースのセッションが上限を超えたら古い順（mtime）に削除する。"""
        if max_sessions <= 0:
            return 0
        sdir = self.path_for(Path(root), "_").parent
        if not sdir.is_dir():
            return 0
        entries = [e for e in os.scandir(sdir) if e.is_file() and e.name.endswith(".jsonl")]
        excess = len(entries) - max_sessions
        if excess <= 0:
            return 0
        entries.sort(key=lambda e: e.stat().st_mtime)
        removed = 0
        for e in entries[:excess]:
            try:
                os.remove(e.path)
                removed += 1
            except OSError:
                pass
        return removed

    def fork(self, s: Session) -> Session:
        """メッセージ履歴を引き継いだ新 session_id のコピーを作る。"""
        ts = now_ms()
        return s.model_copy(
            update={
                "session_id": uuid.uuid4().hex,
                "created_at_ms": ts,
                "updated_at_ms": ts,
                "messages": [m.model_copy(deep=True) for m in s.messages],
                "prompt_history": list(s.prompt_history),
            },
            deep=True,
        )

    @staticmethod
    def _from_rows(rows: list[dict[str, Any]], path: Path) -> Session:
        meta: dict[str, Any] | None = None
        prompts: list[str] = []
        messages: list[ConversationMessage] = []
        compaction: SessionCompaction | None = None
        todos: list[TodoItem] = []
        mode: ModeState | None = None
        legacy_plan: dict[str, Any] | None = None
        plan_steps: list[str] = []
        plan_step_index = 0
        plan_step_estimates: list[int | None] = []
        plan_budget_lines = 0
        git: dict[str, Any] = {}
        system_digest = ""
        expanded_tools: list[str] = []
        omitted_bodies: list[OmittedBody] = []
        try:
            for row in rows:
                kind = row.get("type")
                body = {k: v for k, v in row.items() if k != "type"}
                if kind == "session_meta":
                    meta = body
                elif kind == "prompt_history":
                    prompts.append(str(row.get("text", "")))
                elif kind == "message":
                    messages.append(ConversationMessage.model_validate(body))
                elif kind == "compaction":
                    compaction = SessionCompaction.model_validate(body)
                elif kind == "todos":
                    todos = [TodoItem.model_validate(it) for it in row.get("items", [])]
                elif kind == "mode":
                    mode = ModeState.model_validate(body)
                elif kind == "plan_steps":
                    plan_steps = [str(it) for it in row.get("items", [])]
                    plan_step_index = int(row.get("index", 0))
                    plan_step_estimates = [
                        None if it is None else int(it) for it in row.get("estimates", [])
                    ]
                    plan_budget_lines = int(row.get("budget_lines", 0))
                elif kind == "git_context":
                    git = {
                        "git_context": str(row.get("system", "")),
                        "git_context_compaction": int(row.get("compaction", 0)),
                        "git_seen": str(row.get("seen", "")),
                    }
                elif kind == "system_digest":
                    system_digest = str(row.get("sha256", ""))
                elif kind == "expanded_tools":
                    expanded_tools = [str(it) for it in row.get("names", [])]
                elif kind == "omitted_body":
                    omitted_bodies.append(OmittedBody.model_validate(body))
                elif kind == "plan_mode":
                    legacy_plan = body
            if mode is None and legacy_plan is not None:
                mode = _mode_from_legacy_plan(legacy_plan)
        except Exception as e:
            raise SessionError(ui("session.store.malformed_session", path=path, error=e)) from e
        if meta is None:
            raise SessionError(ui("session.store.missing_session_meta", path=path))
        return Session(
            **meta,
            prompt_history=prompts,
            messages=messages,
            compaction=compaction,
            todos=todos,
            mode=mode if mode is not None else ModeState(),
            plan_steps=plan_steps,
            plan_step_index=plan_step_index,
            plan_step_estimates=plan_step_estimates,
            plan_budget_lines=plan_budget_lines,
            **git,
            last_system_digest=system_digest,
            expanded_tools=expanded_tools,
            omitted_bodies=omitted_bodies,
        )


def _mode_from_legacy_plan(body: dict[str, Any]) -> ModeState:
    """旧 ``plan_mode`` 行を :class:`ModeState` へ変換する（互換シム）。"""
    active = bool(body.get("active", False))
    prev = body.get("previous_mode")
    sub = PermissionMode(prev) if prev else PermissionMode.WORKSPACE_WRITE
    return ModeState(main="plan" if active else "code", sub=sub)


def _meta(s: Session) -> dict[str, Any]:
    """session_meta 行に載せるスカラメタ（メッセージ本体は別行）。"""
    return {
        "version": s.version,
        "session_id": s.session_id,
        "created_at_ms": s.created_at_ms,
        "updated_at_ms": s.updated_at_ms,
        "workspace_root": s.workspace_root,
        "model": s.model,
        "language": s.language,
    }
