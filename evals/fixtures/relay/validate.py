"""イベントの検証。"""


def validate(event):
    if event.size < 0:
        raise ValueError(f"negative size: {event.name}")
