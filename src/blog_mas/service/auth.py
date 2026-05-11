"""Ch10 §1.2 — API authentication (service-level, defence-in-depth layer).

Ch10 §1.2 states that in enterprise deployment the API is protected and managed
through an API gateway such as AWS API Gateway, Kong, or Istio, which enforces:
  • Authentication and authorization (API keys, OAuth tokens, JWTs)
  • Rate limiting and throttling
  • SSL termination

This module implements the service-level authentication check — a second layer
of defence behind the gateway.  The gateway is the primary enforcer; this
module means the service can still refuse unauthenticated requests even if the
gateway is bypassed (e.g. internal mis-routing, dev environments without a
gateway).

Two schemes are supported:
  1. API key in the 'x-api-key' header (simplest, default for internal services)
  2. Bearer JWT stub (skeleton, not validated cryptographically here — the
     gateway validates and proxies a trusted header in production)

Usage in FastAPI:
  from blog_mas.service.auth import require_api_key
  @app.get("/secure", dependencies=[Depends(require_api_key)])
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from blog_mas.config import get_settings

logger = logging.getLogger(__name__)

# ── Scheme definitions ─────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(
    name="x-api-key",
    auto_error=False,
    description="Static API key for service-to-service auth (enforced at API gateway level).",
)

_bearer_scheme = HTTPBearer(auto_error=False)


# ── API-key validation ─────────────────────────────────────────────────────

async def require_api_key(
    api_key: Annotated[str | None, Security(_api_key_header)],
) -> str:
    """FastAPI dependency that validates the 'x-api-key' header.

    Ch10 §1.2: "Authentication and authorization: Validating API keys, OAuth
    tokens, or JWTs to ensure only authorized clients can access the engine."

    If API_KEY is not configured (local dev), the check is skipped with a
    warning so the service remains usable without credentials.
    """
    settings = get_settings()
    configured_key = settings.security.api_key

    if not configured_key:
        # API_KEY not set → local dev mode, skip check.
        logger.warning(
            "[auth] API_KEY not configured — skipping key validation. "
            "Set API_KEY in .env or K8s Secret for production."
        )
        return "dev-unauthenticated"

    if not api_key or api_key != configured_key:
        logger.warning(
            "[auth] Rejected request: missing or invalid x-api-key header."
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid or missing API key. "
                "Provide a valid key in the 'x-api-key' header."
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


# ── Bearer / JWT skeleton ─────────────────────────────────────────────────

async def require_bearer(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer_scheme)],
) -> dict:
    """FastAPI dependency for Bearer token auth (JWT skeleton).

    In production the API gateway validates the JWT and proxies a trusted
    'x-user-id' header.  This stub demonstrates the pattern and provides a
    place to add library-level validation (e.g. python-jose) if the gateway
    is absent.

    Ch10 §1.2: "Validating … OAuth tokens, or JWTs"
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    # In a real deployment: decode + verify the JWT signature here.
    # e.g. from jose import jwt; payload = jwt.decode(token, PUBLIC_KEY, algorithms=["RS256"])
    # For now, return a stub payload so the pattern is clear.
    logger.debug("[auth] Bearer token received (signature verification stub).")
    return {"sub": "stub-user", "token": token[:8] + "..."}


# ── Rate-limit annotation (gateway concern, documented stub) ──────────────

class RateLimitExceeded(HTTPException):
    """Raised when the service detects traffic above its own threshold.

    The primary rate-limiting is the API gateway's job (Kong/Istio/AWS API GW).
    This exception class exists so application code can model the error shape
    and unit-test the error path without a live gateway.

    Ch10 §1.2: "Rate limiting and throttling: Protecting the engine from
    excessive load or denial-of-service attacks."
    """

    def __init__(self, limit: int, window_s: int = 60) -> None:
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {limit} requests per {window_s}s. "
                "This is enforced at the API gateway; if you see this from the "
                "service directly, the gateway may be mis-configured."
            ),
            headers={"Retry-After": str(window_s)},
        )
