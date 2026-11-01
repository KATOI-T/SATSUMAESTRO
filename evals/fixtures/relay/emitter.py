"""監査ログの出力。emit("message") の形で 1 行を記録する。"""

_LOG: list[str] = []


def emit(message: str) -> None:
    _LOG.append(message)
