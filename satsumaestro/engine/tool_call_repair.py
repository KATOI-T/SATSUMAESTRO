"""tool_call 破損の 1 回リカバリ。"""

from __future__ import annotations

from dataclasses import dataclass

from satsumaestro.config.models import RuntimeConfig
from satsumaestro.providers.base import ApiRequest
from satsumaestro.providers.retry import is_tool_call_parse_error
from satsumaestro.session.models import ConversationMessage, TextBlock

_REPAIR_HINT = (
    "The previous tool call had malformed JSON arguments and was rejected by the server. "
    "Re-issue the tool call with strictly valid JSON. Emit only the arguments you need; "
    "omit optional ones entirely rather than guessing values."
)


@dataclass(frozen=True)
class ToolCallRepairPolicy:
    """リカバリの有効・上限。"""

    enabled: bool = True
    max_repairs_per_turn: int = 1

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> ToolCallRepairPolicy:
        return cls(
            enabled=config.tool_schema.repair_tool_calls,
            max_repairs_per_turn=config.tool_schema.max_repairs_per_turn,
        )


class ToolCallRepairState:
    """1 ターン分のリカバリ回数を持つ可変状態（``run_turn`` ごとに 1 個）。"""

    def __init__(self, policy: ToolCallRepairPolicy | None = None) -> None:
        self.policy = policy if policy is not None else ToolCallRepairPolicy()
        self.repairs = 0
        self.window = 0

    def new_section(self) -> None:
        """区間（段を渡す前・各段）が変わった。"""
        self.window = 0

    def should_repair(self, exc: BaseException) -> bool:
        """この失敗を誘導付きで投げ直すか。"""
        if not self.policy.enabled or self.window >= self.policy.max_repairs_per_turn:
            return False
        return is_tool_call_parse_error(exc)

    def repaired_request(self, req: ApiRequest) -> ApiRequest:
        """誘導を末尾に足した**コピー**を返し、リカバリ回数を 1 消費する。"""
        self.repairs += 1
        self.window += 1
        return req.model_copy(update={"messages": [*req.messages, hint_message()]})


def hint_message() -> ConversationMessage:
    """誘導を user ロールの一時メッセージとして組む（履歴には残さない）。"""
    return ConversationMessage(role="user", blocks=[TextBlock(text=_REPAIR_HINT)])
