"""Extract user ids from HTTP requests and LLM call kwargs (never raises)."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

_HEADER_NAMES = ("x-user-id", "x-customer-id")


def _as_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        text = str(value).strip()
        return text or None
    return None


def _from_user_object(user: Any) -> Optional[str]:
    if user is None:
        return None
    if isinstance(user, (str, int)):
        return _as_str(user)

    is_auth = getattr(user, "is_authenticated", None)
    if is_auth is False:
        return None
    if callable(is_auth):
        try:
            if not is_auth():
                return None
        except Exception:
            return None

    for attr in ("id", "pk", "username", "email"):
        try:
            found = _as_str(getattr(user, attr, None))
        except Exception:
            found = None
        if found is not None:
            return found

    try:
        text = str(user).strip()
    except Exception:
        return None
    if text and text not in ("AnonymousUser", "None"):
        return text
    return None


def user_from_object(user: Any) -> Optional[str]:
    """Public alias for extracting an id from a user-like object."""
    try:
        return _from_user_object(user)
    except Exception:
        return None


def _headers_map(request: Any) -> Mapping[str, str]:
    headers = getattr(request, "headers", None)
    if headers is None:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in headers.items()}
    except Exception:
        return {}


def _user_from_jwt_authorization(auth: Optional[str]) -> Optional[str]:
    if not auth or not isinstance(auth, str):
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    segments = token.split(".")
    if len(segments) < 2:
        return None
    payload_b64 = segments[1]
    pad = "=" * (-len(payload_b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_b64 + pad)
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("sub", "user_id", "uid"):
        found = _as_str(payload.get(key))
        if found is not None:
            return found
    return None


def user_from_request(request: Any) -> Optional[str]:
    """Best-effort user id from a web request object."""
    try:
        state = getattr(request, "state", None)
        if state is not None:
            for attr in ("user_id", "user"):
                try:
                    val = getattr(state, attr, None)
                except Exception:
                    val = None
                if attr == "user":
                    found = (
                        _from_user_object(val)
                        if not isinstance(val, (str, int))
                        else _as_str(val)
                    )
                else:
                    found = _as_str(val)
                if found is not None:
                    return found

        # Starlette request.user asserts without AuthenticationMiddleware — catch all.
        user = None
        try:
            user = request.user
        except Exception:
            user = None

        found = _from_user_object(user)
        if found is not None:
            return found

        headers = _headers_map(request)
        for name in _HEADER_NAMES:
            found = _as_str(headers.get(name))
            if found is not None:
                return found

        auth = headers.get("authorization")
        if auth is None:
            meta = getattr(request, "META", None) or {}
            try:
                auth = meta.get("HTTP_AUTHORIZATION")
            except Exception:
                auth = None
        return _user_from_jwt_authorization(auth if isinstance(auth, str) else None)
    except Exception:
        logger.debug("agent_metering: user_from_request failed", exc_info=True)
        return None


def user_from_llm_kwargs(kwargs: Optional[Mapping[str, Any]]) -> Optional[str]:
    """OpenAI ``user=`` or Anthropic ``metadata.user_id`` / ``metadata.user``."""
    if not kwargs:
        return None
    try:
        found = _as_str(kwargs.get("user"))
        if found is not None:
            return found
        metadata = kwargs.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("user_id", "user"):
                found = _as_str(metadata.get(key))
                if found is not None:
                    return found
        return None
    except Exception:
        logger.debug("agent_metering: user_from_llm_kwargs failed", exc_info=True)
        return None
