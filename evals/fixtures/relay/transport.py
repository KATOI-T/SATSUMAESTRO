"""処理結果の送信。"""

from config import RETRY_LIMIT


def send_with_retry(payload):
    """RETRY_LIMIT 回まで再試行しつつ送信する。成功した試行回数を返す。"""
    for attempt in range(RETRY_LIMIT):
        if _send(payload):
            return attempt + 1
        # emit(f"retry {attempt}")  # 旧デバッグ出力（現在は無効）
    raise ConnectionError("all retries failed")


def _send(payload) -> bool:
    return True
