"""外部の satsumaestro 子プロセス（worker）をモデルから操作する 7 つのツール。"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

from satsumaestro.security.enforcer import PermissionMode
from satsumaestro.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from satsumaestro.tools.schema import model_tool_schema
from satsumaestro.util.errors import OrchestrationError

if TYPE_CHECKING:
    from satsumaestro.orchestration.runner import Orchestrator
    from satsumaestro.orchestration.worker import TaskReceipt, Worker
    from satsumaestro.orchestration.worker_supervisor import WorkerSupervisor

_UNAVAILABLE = "worker control plane is not available in this context"


def _json(payload: Any) -> ToolResult:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    return ToolResult(output=json.dumps(payload, ensure_ascii=False))


def _brief(worker: Worker) -> dict[str, Any]:
    """worker_get 以外が返す要約（記録の列を除いて短く保つ）。"""
    return {
        "worker_id": worker.worker_id,
        "state": worker.state,
        "cwd": worker.cwd,
        "session_id": worker.session_id,
        "dispatches": worker.dispatches,
        "error": None if worker.error is None else worker.error.model_dump(mode="json"),
    }


class _WorkerTool(Tool):
    """監督を解いて :meth:`apply` に渡す共通の枠。"""

    async def run(self, inp: BaseModel, ctx: ToolContext) -> ToolResult:
        from satsumaestro.orchestration.ids import UnknownIdError
        from satsumaestro.orchestration.runner import Orchestrator
        from satsumaestro.orchestration.worker_supervisor import WorkerSupervisor
        from satsumaestro.tools.builtins.orchestration_tools import known_ids_hint

        orch: Orchestrator | None = (
            ctx.orchestrator if isinstance(ctx.orchestrator, Orchestrator) else None
        )
        sup = orch.workers if orch is not None else None
        if orch is None or not isinstance(sup, WorkerSupervisor):
            return ToolResult(output=_UNAVAILABLE, is_error=True)
        try:
            return await self.apply(inp, sup)
        except OrchestrationError as e:
            text = str(e)
            if isinstance(e, UnknownIdError):
                text = f"{text} {known_ids_hint(e.kind, orch)}"
            return ToolResult(output=text, is_error=True)

    async def apply(self, inp: BaseModel, sup: WorkerSupervisor) -> ToolResult:
        raise NotImplementedError


class CreateWorker(_WorkerTool):
    """worker を登録し、作業ディレクトリの信頼をその場で決める（子はまだ起動しない）。"""

    class Input(BaseModel):
        cwd: str = Field(
            default="",
            description=("Working directory for the worker. Empty uses the current workspace."),
        )
        trusted_roots: list[str] = Field(
            default_factory=list,
            description=(
                "Extra trusted root directories for this call. A worker whose cwd is "
                "inside a trusted root skips the trust gate."
            ),
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="worker_create",
        description=(
            "Register an external CLI worker and check its working directory against the "
            "trusted roots right away. The state is ready (send a prompt with "
            "worker_send_prompt) or awaiting_trust (approve it with worker_resolve_trust). "
            "No process is started yet."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, sup: WorkerSupervisor) -> ToolResult:
        assert isinstance(inp, CreateWorker.Input)
        return _json(_brief(sup.create(inp.cwd, inp.trusted_roots)))


class GetWorker(_WorkerTool):
    """worker のすべて（記録・出力を含む）を返す。"""

    class Input(BaseModel):
        worker_id: str = Field(description="ID returned by worker_create.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="worker_get",
        description="Get the full state of a worker (state, session_id, output, error, log).",
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, sup: WorkerSupervisor) -> ToolResult:
        assert isinstance(inp, GetWorker.Input)
        return _json(sup.get(inp.worker_id))


class CheckWorkerReady(_WorkerTool):
    """いま指示を渡せるか・止まっているか・送り直しが構えてあるかを返す。"""

    class Input(BaseModel):
        worker_id: str = Field(description="ID of the worker to check.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="worker_await_ready",
        description=(
            "Check whether a worker can take a prompt now, is blocked (awaiting trust or "
            "failed), or has a replay armed."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.READ_ONLY,
        parallel_safe=True,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, sup: WorkerSupervisor) -> ToolResult:
        assert isinstance(inp, CheckWorkerReady.Input)
        return _json(sup.readiness(inp.worker_id))


class ApproveWorkerTrust(_WorkerTool):
    """承認待ちの worker の作業ディレクトリを手で承認する。"""

    class Input(BaseModel):
        worker_id: str = Field(description="ID of the worker in awaiting_trust state.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="worker_resolve_trust",
        description=(
            "Approve the working directory of a worker in awaiting_trust state so it becomes ready."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, sup: WorkerSupervisor) -> ToolResult:
        assert isinstance(inp, ApproveWorkerTrust.Input)
        return _json(_brief(sup.approve_trust(inp.worker_id)))


class SendWorkerPrompt(_WorkerTool):
    """worker に指示を渡して子を起動する（空なら構えてある送り直し）。"""

    class Input(BaseModel):
        worker_id: str = Field(description="ID of the worker (its state must be ready).")
        prompt: str = Field(
            default="",
            description=(
                "Task prompt for the worker. Empty resends the armed replay prompt "
                "after a misdelivery."
            ),
        )
        repo: str = Field(default="", description="Task receipt: repository name the task targets.")
        task_kind: str = Field(
            default="", description="Task receipt: kind of task (e.g. docs, bugfix)."
        )
        source_surface: str = Field(
            default="", description="Task receipt: surface that originated the task."
        )
        expected_artifacts: list[str] = Field(
            default_factory=list,
            description="Task receipt: artifacts the task should produce (optional).",
        )
        objective_preview: str = Field(
            default="", description="Task receipt: short preview of the objective."
        )

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="worker_send_prompt",
        description=(
            "Send a prompt to a ready worker and spawn its child process. Optionally "
            "attach a task receipt (repo / task_kind / source_surface / "
            "objective_preview all set together) for misdelivery detection. Poll "
            "worker_get for the result."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, sup: WorkerSupervisor) -> ToolResult:
        assert isinstance(inp, SendWorkerPrompt.Input)
        receipt, problem = _receipt_from(inp)
        if problem is not None:
            return ToolResult(output=problem, is_error=True)
        return _json(_brief(await sup.dispatch(inp.worker_id, inp.prompt or None, receipt)))


class RestartWorker(_WorkerTool):
    """worker を初期化して信頼の判定からやり直す（子が居れば止める）。"""

    class Input(BaseModel):
        worker_id: str = Field(description="ID of the worker to restart.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="worker_restart",
        description=(
            "Restart a worker: kill its child process if any, clear prompt state and "
            "re-evaluate the trust gate."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, sup: WorkerSupervisor) -> ToolResult:
        assert isinstance(inp, RestartWorker.Input)
        return _json(_brief(await sup.restart(inp.worker_id)))


class TerminateWorker(_WorkerTool):
    """worker を止める（SIGTERM → 猶予 → SIGKILL）。"""

    class Input(BaseModel):
        worker_id: str = Field(description="ID of the worker to terminate.")

    spec: ClassVar[ToolSpec] = ToolSpec(
        name="worker_terminate",
        description=(
            "Terminate a worker: stop its child process (SIGTERM, then SIGKILL after a "
            "grace period) and mark it finished."
        ),
        input_schema=model_tool_schema(Input),
        required_mode=PermissionMode.DANGER_FULL_ACCESS,
        parallel_safe=False,
        deferred=True,
    )

    async def apply(self, inp: BaseModel, sup: WorkerSupervisor) -> ToolResult:
        assert isinstance(inp, TerminateWorker.Input)
        return _json(_brief(await sup.terminate(inp.worker_id)))


def _receipt_from(inp: SendWorkerPrompt.Input) -> tuple[TaskReceipt | None, str | None]:
    """入力から受領書を組む（項目は全部そろえるか、どれも渡さない）。"""
    named = {
        "repo": inp.repo,
        "task_kind": inp.task_kind,
        "source_surface": inp.source_surface,
        "objective_preview": inp.objective_preview,
    }
    empty = [name for name, value in named.items() if not value.strip()]
    if len(empty) == len(named) and not inp.expected_artifacts:
        return None, None
    if empty:
        return None, "task receipt fields must be set together; missing: " + ", ".join(empty)
    from satsumaestro.orchestration.worker import TaskReceipt

    return TaskReceipt(**named, expected_artifacts=inp.expected_artifacts), None


WORKER_TOOLS: tuple[type[Tool], ...] = (
    CreateWorker,
    GetWorker,
    CheckWorkerReady,
    ApproveWorkerTrust,
    SendWorkerPrompt,
    RestartWorker,
    TerminateWorker,
)

WORKER_TOOL_NAMES: frozenset[str] = frozenset(t.spec.name for t in WORKER_TOOLS)

WORKER_MUTATING_TOOL_NAMES: frozenset[str] = frozenset(
    t.spec.name for t in WORKER_TOOLS if t.spec.required_mode is PermissionMode.DANGER_FULL_ACCESS
)
