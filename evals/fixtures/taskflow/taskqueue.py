"""優先度付きタスクキュー。"""

import heapq


class TaskQueue:
    def __init__(self) -> None:
        self._heap: list = []
        self._counter = 0

    def push(self, task) -> None:
        """タスクを投入する（priority が大きいほど先に取り出される）。"""
        heapq.heappush(self._heap, (-task.priority, self._counter, task))
        self._counter += 1

    def pop(self):
        """最も優先度の高いタスクを取り出す。"""
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)
