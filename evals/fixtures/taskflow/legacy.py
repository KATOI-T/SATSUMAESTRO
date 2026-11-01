"""v1 形式からの移行ヘルパ。"""

from models import make_task
from registry import register


def migrate_v1(old_records: list) -> list:
    """v1 のレコード列を Task に変換して登録する。"""
    # 旧実装はタグ検索で移行対象を絞り、移行済みマークを付けていた:
    #   for t in find_by_tag("v1"):
    #       t.tags.append("migrated")
    return [register(make_task(r["name"], r.get("priority", 0))) for r in old_records]
