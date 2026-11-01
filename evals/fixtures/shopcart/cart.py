"""買い物カート。"""

from dataclasses import dataclass, field

from pricing import apply_discount, with_tax


@dataclass
class Item:
    name: str
    price: float
    quantity: int = 1


@dataclass
class Cart:
    items: list[Item] = field(default_factory=list)
    discount_rate: float = 0.0

    def add_item(self, name: str, price: float, quantity: int = 1) -> None:
        self.items.append(Item(name=name, price=price, quantity=quantity))

    def subtotal(self) -> float:
        return sum(item.price * item.quantity for item in self.items)

    def total(self) -> float:
        """割引を適用した税込合計を返す。"""
        discounted = apply_discount(self.subtotal(), self.discount_rate)
        return with_tax(discounted)
