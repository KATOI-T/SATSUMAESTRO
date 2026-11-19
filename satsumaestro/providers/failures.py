"""アカウントの事情で断られた失敗（残高切れ・利用の上限・鍵の拒否）の見分け。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from satsumaestro.providers.retry import causes, describe

FailureKind = Literal["no_credits", "rate_limited", "key_rejected"]

VENDOR_HOSTS = {"api.openai.com": "openai", "api.anthropic.com": "anthropic", "api.x.ai": "xai"}
_NO_CREDIT_CODES = frozenset({"insufficient_quota", "credit_balance_exhausted"})
_NO_CREDIT_PHRASES = ("credit balance is too low", "used all available credits")
_NO_CREDIT_STATUS = frozenset({400, 402, 403, 429})
_KEY_PHRASES = ("incorrect api key",)


@dataclass(frozen=True)
class AccountFailure:
    """断られた理由と、プロバイダの文。"""

    kind: FailureKind
    status: int
    vendor: str
    host: str
    message: str


def account_failure(exc: BaseException) -> AccountFailure | None:
    """SDK の例外の状態と本文から、アカウントの事情による失敗を見分ける（ほかは ``None``）。"""
    for e in causes(exc):
        response = getattr(e, "response", None)
        status = getattr(response, "status_code", None)
        if not isinstance(status, int):
            continue
        codes, message = body_fields(getattr(e, "body", None), getattr(e, "code", None))
        kind = failure_kind(status, codes, message)
        if kind is None:
            return None
        host = _host(response)
        vendor = "anthropic" if type(e).__module__.startswith("anthropic") else ""
        return AccountFailure(
            kind=kind,
            status=status,
            vendor=vendor or VENDOR_HOSTS.get(host, ""),
            host=host,
            message=message.strip() or describe(e),
        )
    return None


def failure_kind(status: int, codes: set[str], message: str) -> FailureKind | None:
    """状態と本文の code・文から、アカウントの事情による失敗の種類（ほかは ``None``）。"""
    text = message.casefold()
    no_credits = codes & _NO_CREDIT_CODES or any(p in text for p in _NO_CREDIT_PHRASES)
    if no_credits and status in _NO_CREDIT_STATUS:
        return "no_credits"
    if status == 429:
        return "rate_limited"
    # xAI は鍵の誤りを 400 で返す
    if status == 401 or (status == 400 and any(p in text for p in _KEY_PHRASES)):
        return "key_rejected"
    return None


def body_fields(body: object, code: object = None) -> tuple[set[str], str]:
    """本文の code と type、プロバイダの文（OpenAI・Anthropic・xAI の形を読む）。"""
    if isinstance(body, str):
        return {str(code or "")}, body
    if not isinstance(body, dict):
        return {str(code or "")}, ""
    inner = body.get("error")
    if isinstance(inner, str):
        return {str(body.get("code") or "")}, inner
    if isinstance(inner, dict):
        body = inner
    codes = {str(body.get(name) or "") for name in ("code", "type")}
    return codes, str(body.get("message") or "")


def _host(response: object) -> str:
    try:
        return str(response.request.url.host)  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        return ""


__all__ = [
    "VENDOR_HOSTS",
    "AccountFailure",
    "FailureKind",
    "account_failure",
    "body_fields",
    "failure_kind",
]
