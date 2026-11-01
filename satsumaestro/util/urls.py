"""接続先の URL の読み方。"""

from __future__ import annotations

OLLAMA_DEFAULT_PORT = 11434


def ollama_host_url(raw: str) -> str:
    """``OLLAMA_HOST`` の値を、Ollama 本体と同じ読み方で URL にする。"""
    s = raw.strip()
    if "://" in s:
        return s
    hostport, sep, path = s.partition("/")
    if hostport.startswith("["):
        has_port = "]:" in hostport
    elif hostport.count(":") > 1:
        hostport, has_port = f"[{hostport}]", False
    else:
        has_port = ":" in hostport
    if hostport.startswith(":"):
        hostport = "127.0.0.1" + hostport
    if not has_port:
        hostport += f":{OLLAMA_DEFAULT_PORT}"
    return f"http://{hostport}{sep}{path}"
