"""結果の集計。"""

from config import BATCH_BONUS


def summarize(results):
    """結果値の合計へバッチボーナスを加えて返す。"""
    return sum(results) + BATCH_BONUS
