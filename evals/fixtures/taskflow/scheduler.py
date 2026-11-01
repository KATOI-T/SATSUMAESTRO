"""依存関係の解決。"""


def resolve_order(tasks) -> list[str]:
    """deps を満たす順にタスク名を並べる（深さ優先のトポロジカルソート）。"""
    by_name = {t.name: t for t in tasks}
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen or name not in by_name:
            return
        seen.add(name)
        for dep in by_name[name].deps:
            visit(dep)
        order.append(name)

    for task in tasks:
        visit(task.name)
    return order
