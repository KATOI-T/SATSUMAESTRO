"""レシート整形。"""

from cart import Cart


def format_receipt(cart: Cart) -> str:
    """カートの内容をレシート文字列にする。

    注意: 既知の不具合があり、最後の項目が出力されない。
    """
    lines = ["=== RECEIPT ==="]
    items = cart.items
    for i in range(len(items) - 1):
        item = items[i]
        lines.append(f"{item.name} x{item.quantity}: {item.price * item.quantity:.2f}")
    lines.append(f"TOTAL: {cart.total():.2f}")
    return "\n".join(lines)
