"""SATSUMAESTRO 例外階層。"""

from __future__ import annotations


class SatsumaError(Exception):
    """SATSUMAESTRO 由来例外の基底。"""


class ConfigError(SatsumaError):
    """設定の読込・マージ・検証に失敗した。"""


class SessionError(SatsumaError):
    """セッションの読込・保存・整合性に関する失敗。"""


class PathEscapeError(SatsumaError):
    """パスがワークスペース jail の外を指した（`..`/絶対/symlink 突破など）。"""


class StateDirError(PathEscapeError):
    """パスがエージェントの状態ディレクトリ（``.satsuma/``）を指した。"""


class ProviderError(SatsumaError):
    """LLM プロバイダとの通信・応答処理に関する失敗。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = retryable
        self.step: tuple[int, int] | None = None
        self.timeout_recoveries = 0
        self.timeout_s = 0.0


class RetriesExhaustedError(ProviderError):
    """リトライ上限（回数）に達して諦めた。"""

    def __init__(self, attempts: int, elapsed_s: float, *, last_error: str = "") -> None:
        self.attempts = attempts
        self.elapsed_s = elapsed_s
        self.last_error = last_error[:200]
        detail = f": {self.last_error}" if self.last_error else ""
        from satsumaestro.i18n.text import ui

        super().__init__(
            ui(
                "util.errors.gave_up_after_the_retry",
                attempts=attempts,
                elapsed_s=elapsed_s,
                detail=detail,
            ),
            retryable=False,
        )


class ProviderTimeoutError(RetriesExhaustedError):
    """リトライ総経過が ``api_timeout_s`` を超えて打ち切った。"""

    def __init__(self, attempts: int, elapsed_s: float, *, last_error: str = "") -> None:
        super().__init__(attempts, elapsed_s, last_error=last_error)
        detail = f": {self.last_error}" if self.last_error else ""
        from satsumaestro.i18n.text import ui

        self.args = (
            ui(
                "util.errors.ran_out_of_the_api",
                attempts=attempts,
                elapsed_s=elapsed_s,
                detail=detail,
            ),
        )


class ToolError(SatsumaError):
    """ツールの実行・入力検証に関する失敗。"""


class RagError(SatsumaError):
    """RAG の埋め込み・索引アクセスに関する失敗。"""


class EvalAgentError(SatsumaError):
    """評価対象の外部エージェント CLI の起動・実行に失敗した。"""


class MaxIterationsError(SatsumaError):
    """会話ループが ``max_iterations`` に達しても終了しなかった。"""

    def __init__(
        self, limit: int, *, last_text: str = "", step: tuple[int, int] | None = None
    ) -> None:
        self.limit = limit
        self.last_text = last_text[:200]
        self.step = step
        where = f" (in plan step {step[0]} of {step[1]})" if step is not None else ""
        super().__init__(f"reached max_iterations={limit} without end_turn{where}")


class RepetitionLoopError(SatsumaError):
    """応答が同じ内容を繰り返し続け、打ち切っても直らなかった。"""

    def __init__(self, repeats: int, snippet: str, *, recovered_before: bool) -> None:
        self.repeats = repeats
        self.snippet = snippet
        self.recovered_before = recovered_before
        why = (
            "it repeated again after being asked to write the conclusion only once"
            if recovered_before
            else "it repeated and the iteration limit left no room to recover"
        )
        super().__init__(
            f"stopped the turn on repeated content ({why}; the same paragraph {repeats} times: "
            f"{snippet!r})"
        )


class PlanExitLoopError(SatsumaError):
    """plan でないのに exit_plan_mode だけを呼ぶ反復が続いた。"""

    def __init__(self, calls: int, last_text: str = "") -> None:
        self.calls = calls
        self.last_text = last_text[:200]
        super().__init__(
            f"stopped the turn: exit_plan_mode was called {calls} times in a row outside plan mode"
        )


class UnknownToolLoopError(SatsumaError):
    """存在しないツール名を、強めた返答の後も続けて呼んだ。"""

    def __init__(self, name: str, calls: int, last_text: str = "") -> None:
        self.name = name
        self.calls = calls
        self.last_text = last_text[:200]
        super().__init__(
            f"stopped the turn: the unknown tool name {name} was called {calls} times in a row"
        )


class ContextOverflowError(SatsumaError):
    """送る前の見積もりが、モデルの窓に収まらない。"""

    def __init__(self, estimate: int, limit: int, *, dropped_request: bool) -> None:
        self.estimate = estimate
        self.limit = limit
        self.dropped_request = dropped_request
        tail = (
            "the request was not sent and was removed from the conversation"
            if dropped_request
            else "it did not fit even after compaction and trimming the tool results"
        )
        super().__init__(
            f"the request does not fit in the model's context window (estimated ~{estimate:,} "
            f"tokens, limit ~{limit:,} tokens); {tail}"
        )


class ContextTruncatedError(SatsumaError):
    """サーバが要求を黙って切り詰めた（応答の usage が、送った量の下限より大きく少ない）。"""

    def __init__(self, sent: int, read: int) -> None:
        self.sent = sent
        self.read = read
        super().__init__(
            f"the server truncated the request (sent >= ~{sent:,} tokens, the model read "
            f"{read:,}); the reply was discarded because the model did not read the whole request"
        )


class OrchestrationError(SatsumaError):
    """並列オーケストレーション（Task/Team/Cron 登録簿・バックグラウンド実行）の失敗。"""


class ToolCallingUnsupportedError(SatsumaError):
    """モデルがネイティブ tool_calls を使わず本文テキストにツール呼び出しを出力した。"""

    def __init__(self, model: str, tool: str = "") -> None:
        self.model = model
        self.tool = tool
        from satsumaestro.i18n.text import ui

        examples = "llama3.1:8b / llama3.3:70b / mistral-large"
        super().__init__(
            ui(
                "util.errors.tool_calling_unsupported_with_tool",
                model=model,
                tool=tool,
                examples=examples,
            )
            if tool
            else ui("util.errors.tool_calling_unsupported", model=model, examples=examples)
        )
