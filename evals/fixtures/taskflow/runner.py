"""登録済みタスクの一括実行。"""

from registry import all_tasks, find_by_tag
from scheduler import resolve_order
from taskqueue import TaskQueue


def run_all() -> int:
    """登録済みタスクを実行し、完了数を返す。

    依存関係の充足順を resolve_order で求めてからキューへ投入する。
    """
    tasks = all_tasks()
    by_name = {t.name: t for t in tasks}
    queue = TaskQueue()
    for name in resolve_order(tasks):
        queue.push(by_name[name])
    done = 0
    while len(queue):
        task = queue.pop()
        task.tags.append("done")
        done += 1
    return done


def pending_of(tag: str) -> int:
    """タグ一致のうち未完了のタスク数。"""
    return sum(1 for t in find_by_tag(tag) if "done" not in t.tags)
