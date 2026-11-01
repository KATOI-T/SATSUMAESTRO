"""イベントの前処理。"""


def normalize(event):
    """状態表記を小文字へ統一して返す。"""
    event.status = event.status.lower()
    return event
