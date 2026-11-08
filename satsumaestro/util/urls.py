"""接続先の URL の読み方。"""

from __future__ import annotations

from urllib.parse import urlparse

OLLAMA_DEFAULT_PORT = 11434


def ollama_host_url(raw: str) -> str:
    """``OLLAMA_HOST`` の値を、Ollama 本体と同じ読み方で URL にする。"""
    return host_url(raw, OLLAMA_DEFAULT_PORT)


def host_url(raw: str, default_port: int) -> str:
    """``OLLAMA_HOST`` と同じ読み方で、ホスト（ポート・パスは任意）を URL にする。"""
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
        hostport += f":{default_port}"
    return f"http://{hostport}{sep}{path}"


def openai_base(url: str) -> str:
    """OpenAI 互換の接続先にする（裸のホストには ``/v1`` を補い、明示のパスはそのまま）。"""
    trimmed = url.strip().rstrip("/")
    if "://" not in trimmed:
        trimmed = f"http://{trimmed}"
    if not urlparse(trimmed).path:
        return trimmed + "/v1"
    return trimmed
