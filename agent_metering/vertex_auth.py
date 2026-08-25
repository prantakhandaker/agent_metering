"""Mint OAuth access tokens from a GCP service-account JSON (Vertex AI)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Union

CredentialsSource = Union[str, Path, dict[str, Any]]

_CLOUD_PLATFORM_SCOPE = ("https://www.googleapis.com/auth/cloud-platform",)

# Cache key → (token, expiry_monotonic)
_TOKEN_CACHE: dict[str, tuple[str, float]] = {}


def _cache_key(source: CredentialsSource) -> str:
    if isinstance(source, dict):
        return f"inline:{source.get('client_email', '')}:{source.get('private_key_id', '')}"
    return f"path:{Path(source).resolve()}"


def _load_info(source: CredentialsSource) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    return json.loads(path.read_text(encoding="utf-8"))


def get_access_token(
    credentials_json: CredentialsSource,
    *,
    force_refresh: bool = False,
) -> str:
    """Return a Bearer access token for Vertex / Google APIs.

    Requires optional dependency ``google-auth``::

        pip install "agent-metering[vertex]"
    """
    key = _cache_key(credentials_json)
    now = time.monotonic()
    if not force_refresh and key in _TOKEN_CACHE:
        token, expiry = _TOKEN_CACHE[key]
        if now < expiry - 60:
            return token

    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "google-auth is required for Vertex AI service-account JSON. "
            'Install with: pip install "agent-metering[vertex]"'
        ) from exc

    info = _load_info(credentials_json)
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=list(_CLOUD_PLATFORM_SCOPE),
    )
    credentials.refresh(Request())
    token = credentials.token
    if not token:
        raise RuntimeError("Failed to obtain Vertex access token from service account")

    # google-auth expiry is datetime; fall back to ~55 minutes
    expiry_mono = now + 3300
    if credentials.expiry is not None:
        import datetime as dt

        remaining = (credentials.expiry - dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining > 0:
            expiry_mono = now + remaining

    _TOKEN_CACHE[key] = (token, expiry_mono)
    return token


def clear_token_cache() -> None:
    _TOKEN_CACHE.clear()
