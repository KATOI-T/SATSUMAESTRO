"""タスクのドメイン型。"""

from dataclasses import dataclass


@dataclass
class Task:
    name: str
    priority: int
    tags: list
    deps: tuple


def make_task(name: str, priority: int = 0, tags: list = [], deps: tuple = ()) -> Task:
    """タスクを生成する（tags / deps は省略可能）。"""
    return Task(name=name, priority=priority, tags=tags, deps=deps)
