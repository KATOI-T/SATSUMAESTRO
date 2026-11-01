"""価格計算ユーティリティ。"""

TAX_RATE = 0.1


def apply_discount(price: float, rate: float) -> float:
    """価格に割引率（0.0〜1.0）を適用する。"""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be within [0, 1]: {rate}")
    return price * (1.0 - rate)


def with_tax(price: float) -> float:
    """税込価格を返す。"""
    return price * (1.0 + TAX_RATE)
