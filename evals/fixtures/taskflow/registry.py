"""タスクの大域レジストリ。"""

_TASKS: dict = {}


def register(task):
    """タスクを登録する（同名は上書き）。"""
    _TASKS[task.name] = task
    return task


def find_by_tag(tag: str) -> list:
    """タグが一致するタスクを列挙する。"""
    return [t for t in _TASKS.values() if tag in t.tags]


def all_tasks() -> list:
    return list(_TASKS.values())
