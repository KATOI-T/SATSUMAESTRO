"""イベントのドメイン型。"""

from dataclasses import dataclass


@dataclass
class Event:
    name: str
    size: int
    route: int
    status: str = "NEW"
