"""実行結果のサマリ整形。"""

from registry import find_by_tag


def tag_summary(tag: str) -> str:
    """タグ別の件数サマリを返す。"""
    tasks = find_by_tag(tag)
    return f"{tag}: {len(tasks)} task(s)"


def failed_summary() -> str:
    """失敗タグ付きタスクの一覧を返す。"""
    failed = find_by_tag("failed")
    return "\n".join(t.name for t in failed)
