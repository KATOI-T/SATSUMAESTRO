"""分解の静的ゲートと分解モデル 1 往復。"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from pydantic import BaseModel, ValidationError

from satsumaestro.config.models import DecompositionConfig, SamplingConfig
from satsumaestro.engine.tool_call_repair import ToolCallRepairPolicy
from satsumaestro.orchestration.plan import TaskPlan
from satsumaestro.providers.base import ApiRequest, Provider
from satsumaestro.session.models import ConversationMessage, TextBlock, ToolUseBlock

_log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"[。？！?!\n]+")

_PARALLEL_MARKERS: tuple[str, ...] = ("および", "ならびに", "それぞれ", "また、")

_ENUM_LINE = re.compile(r"^\s*(?:[-*・]|[0-9０-９]+[.)．）]|[①-⑳])\s*\S")

_WRITE_VERBS: tuple[str, ...] = (
    "直して",
    "直せ",
    "修正して",
    "実装して",
    "追加して",
    "書いて",
    "作って",
    "作成して",
    "生成して",
    "変更して",
    "書き換えて",
    "リファクタして",
    "リファクタリングして",
    "削除して",
    "消して",
    "リネームして",
    "移動して",
    "適用して",
    "通して",
    "実行して",
    "走らせて",
    "コミットして",
    "マージして",
    "インストールして",
)
_TRAILING_NOISE = re.compile(r"(?:ください|下さい|ね|よ|ほしい|欲しい)+$")


class GateResult(BaseModel):
    single: bool
    reason: str


def static_gate(
    user_text: str,
    cfg: DecompositionConfig,
    *,
    orchestration_enabled: bool = True,
    plan_mode: bool = False,
    breaker_open: bool = False,
) -> GateResult:
    """LLM コスト 0 の一次判定。"""
    if not cfg.enabled or not orchestration_enabled:
        return GateResult(single=True, reason="static_gate:disabled")
    if breaker_open:
        return GateResult(single=True, reason="breaker_open")
    if plan_mode:
        return GateResult(single=True, reason="static_gate:plan_mode")

    text = user_text.strip()
    if len(text) < cfg.min_chars:
        return GateResult(single=True, reason="static_gate:short_prompt")
    if _write_intent(text):
        return GateResult(single=True, reason="static_gate:write_intent")
    if _is_single_sentence(text):
        return GateResult(single=True, reason="static_gate:single_sentence")
    return GateResult(single=False, reason="static_gate:pass")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _is_single_sentence(text: str) -> bool:
    """1 文のみで並列句の目印も無い＝分解しても幅が出ない。"""
    sentences = _sentences(text)
    if len(sentences) != 1:
        return False
    sentence = sentences[0]
    if any(m in sentence for m in _PARALLEL_MARKERS):
        return False
    if sentence.count("、") >= 2:
        return False
    return not _ENUM_LINE.match(sentence)


def _write_intent(text: str) -> bool:
    """書き込み・実行の意図が主（末尾の主節が書き込み動詞で終わる）。"""
    sentences = _sentences(text)
    if not sentences:
        return False
    last = _TRAILING_NOISE.sub("", sentences[-1])
    return any(last.endswith(v) for v in _WRITE_VERBS)


_MAX_INPUT_CHARS = 4_000
_DECOMPOSE_MAX_TOKENS = 1_024

_DECOMPOSE_SYSTEM = (
    "You split a user's request into independent subtasks for parallel execution, "
    "by calling the `decompose_plan` tool exactly once. Subtasks are executed inside "
    "the user's current workspace (a local project checkout) by researchers with "
    "file search/read tools and web access — when the request is about this "
    "repository or codebase, plan local file exploration directly; never plan a "
    "subtask to locate the repository on the web. Rules: prefer read-only "
    "research subtasks — searching/reading files or the web (kind=lookup) or reading "
    "plus reasoning (kind=analysis); keep one topic per subtask — when the request "
    "names several independent questions, emit one subtask per question and never "
    "merge them into a combined subtask; only subtasks with an empty depends_on run "
    "in parallel, so leave depends_on empty unless a subtask truly cannot start "
    "without another's result (a chain of dependent subtasks runs serially and is "
    "usually wrong — prefer a flat list of independent subtasks); mark a subtask "
    "that must edit files as kind=edit "
    "(list target paths in `writes`) and one that must run tests/commands as "
    "kind=verify — those are executed serially by the caller, so keep them few; "
    "write each objective as a "
    "self-contained instruction in the user's language; set single=true with no "
    "subtasks when the request does not split into 2+ independent parts. "
    "Do not answer the request itself. Example arguments:\n"
    '{"single": false, "subtasks": ['
    '{"id": "s1", "objective": "...", "kind": "lookup"}, '
    '{"id": "s2", "objective": "...", "kind": "lookup"}, '
    '{"id": "s3", "objective": "...", "kind": "analysis"}], '
    '"reduce_hint": "..."}'
)

_DECOMPOSE_TOOL: dict[str, object] = {
    "name": "decompose_plan",
    "description": "Report the decomposition of the user's request into subtasks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "single": {
                "type": "boolean",
                "description": "true if the request should not be decomposed",
            },
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "s1, s2, ..."},
                        "objective": {
                            "type": "string",
                            "description": "self-contained instruction for one researcher",
                        },
                        "kind": {
                            "type": "string",
                            "enum": ["lookup", "analysis", "edit", "verify"],
                        },
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "reads": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "paths/URLs the subtask will consult",
                        },
                        "writes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "paths an edit subtask would modify",
                        },
                        "acceptance": {
                            "type": "string",
                            "description": "what finding completes this subtask",
                        },
                        "est_calls": {
                            "type": "integer",
                            "description": "estimated LLM round-trips (1-8)",
                        },
                    },
                    "required": ["id", "objective", "kind"],
                },
            },
            "reduce_hint": {
                "type": "string",
                "description": "how to merge the subtask results into one answer",
            },
        },
        "required": ["single", "subtasks"],
    },
}

_REPAIR_HINT = (
    "The previous `decompose_plan` call had invalid arguments: {error}\n"
    "Call `decompose_plan` again, exactly once, with arguments that match the schema."
)


class Decomposer:
    """役割モデル 1 往復で :class:`TaskPlan` を得る（圧縮の ``_summarize_llm`` と同型）。"""

    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        policy: ToolCallRepairPolicy,
        sampling: SamplingConfig | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._policy = policy
        self._sampling = sampling

    async def decompose(self, user_text: str, *, max_subtasks: int) -> TaskPlan | None:
        """分解計画を返す。"""
        from satsumaestro.engine.events import closing_stream, fold_events

        messages = [
            ConversationMessage(
                role="user",
                blocks=[
                    TextBlock(
                        text=(
                            f"Split into at most {max_subtasks} subtasks:\n\n"
                            f"{user_text[:_MAX_INPUT_CHARS]}"
                        )
                    )
                ],
            )
        ]
        repairs_left = self._policy.max_repairs_per_turn if self._policy.enabled else 0
        while True:
            req = ApiRequest(
                model=self._model,
                system=_DECOMPOSE_SYSTEM,
                messages=messages,
                tools=[dict(_DECOMPOSE_TOOL)],
                max_tokens=_DECOMPOSE_MAX_TOKENS,
                thinking=False,
                sampling=self._sampling,
            )
            try:
                async with closing_stream(self._provider.stream(req)) as events:
                    folded = await fold_events(events)
            except Exception:
                _log.debug("decomposer: provider call failed", exc_info=True)
                return None
            plan, error = _parse_plan(folded.blocks)
            if plan is not None:
                return plan
            if repairs_left <= 0:
                _log.debug("decomposer: giving up after invalid plan: %s", error)
                return None
            repairs_left -= 1
            messages = [
                *messages,
                folded,
                ConversationMessage(
                    role="user",
                    blocks=[TextBlock(text=_REPAIR_HINT.format(error=error))],
                ),
            ]


def _parse_plan(blocks: Sequence[object]) -> tuple[TaskPlan | None, str]:
    """応答ブロックから ``decompose_plan`` の引数を取り出して検証する。"""
    for block in blocks:
        if isinstance(block, ToolUseBlock) and block.name == "decompose_plan":
            try:
                return TaskPlan.model_validate(_normalize_args(block.input)), ""
            except ValidationError as exc:
                return None, str(exc)[:500]
    return None, "no decompose_plan tool call in the response"


_OBJECTIVE_ALIASES: tuple[str, ...] = ("task", "description", "goal", "instruction")


def _normalize_args(raw: object) -> object:
    """検証前の決定的な補正（LLM 再往復より安い正規化。tool_call 修復と同じ思想）。"""
    if not isinstance(raw, dict):
        return raw
    subtasks = raw.get("subtasks")
    if not isinstance(subtasks, list):
        return raw
    used_ids = {s["id"] for s in subtasks if isinstance(s, dict) and isinstance(s.get("id"), str)}
    counter = 1
    normalized: list[object] = []
    for entry in subtasks:
        if isinstance(entry, dict):
            entry = dict(entry)
            if not isinstance(entry.get("objective"), str) or not entry.get("objective"):
                for alias in _OBJECTIVE_ALIASES:
                    value = entry.get(alias)
                    if isinstance(value, str) and value.strip():
                        entry["objective"] = value
                        break
            if not isinstance(entry.get("id"), str) or not entry["id"]:
                while (candidate := f"s{counter}") in used_ids:
                    counter += 1
                entry["id"] = candidate
                used_ids.add(candidate)
        normalized.append(entry)
    return {**raw, "subtasks": normalized}
