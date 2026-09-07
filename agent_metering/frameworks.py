"""Auto-patch FastAPI/Starlette, Flask, and Django to set metering user context."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from agent_metering.context import clear_user, reset_user, set_user
from agent_metering.user_detect import user_from_object, user_from_request

logger = logging.getLogger(__name__)

# key -> (owner, attr_name, original)
_ORIG: dict[str, tuple[Any, str, Any]] = {}


def _patch_starlette() -> None:
    key = "starlette.Starlette.__call__"
    if key in _ORIG:
        return
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
    except ImportError:
        return
    except Exception:
        logger.debug("agent_metering: Starlette not patchable", exc_info=True)
        return

    orig = Starlette.__call__

    async def wrapped(self: Any, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            await orig(self, scope, receive, send)
            return
        token = None
        try:
            request = Request(scope, receive=receive)
            uid = user_from_request(request)
            if uid:
                token = set_user(uid)
            await orig(self, scope, receive, send)
        finally:
            if token is not None:
                reset_user(token)
            else:
                clear_user()

    _ORIG[key] = (Starlette, "__call__", orig)
    Starlette.__call__ = wrapped  # type: ignore[method-assign]


def _flask_user(request: Any) -> Optional[str]:
    found = user_from_request(request)
    if found:
        return found
    try:
        from flask import g

        for attr in ("user_id", "user"):
            val = getattr(g, attr, None)
            if val is None:
                continue
            obj = user_from_object(val)
            if obj:
                return obj
    except Exception:
        pass
    try:
        from flask_login import current_user  # type: ignore

        return user_from_object(current_user)
    except Exception:
        return None


def _patch_flask() -> None:
    key = "flask.Flask.full_dispatch_request"
    if key in _ORIG:
        return
    try:
        from flask import Flask, request as flask_request
    except ImportError:
        return
    except Exception:
        logger.debug("agent_metering: Flask not patchable", exc_info=True)
        return

    orig = Flask.full_dispatch_request

    def wrapped(self: Any) -> Any:
        token = None
        try:
            uid = _flask_user(flask_request)
            if uid:
                token = set_user(uid)
            return orig(self)
        finally:
            if token is not None:
                reset_user(token)
            else:
                clear_user()

    _ORIG[key] = (Flask, "full_dispatch_request", orig)
    Flask.full_dispatch_request = wrapped  # type: ignore[method-assign]


def _patch_django() -> None:
    key = "django.BaseHandler.get_response"
    if key in _ORIG:
        return
    try:
        from django.core.handlers.base import BaseHandler
    except ImportError:
        return
    except Exception:
        logger.debug("agent_metering: Django not patchable", exc_info=True)
        return

    orig = BaseHandler.get_response

    def wrapped(self: Any, request: Any) -> Any:
        token = None
        try:
            uid = user_from_request(request)
            if uid:
                token = set_user(uid)
            return orig(self, request)
        finally:
            if token is not None:
                reset_user(token)
            else:
                clear_user()

    _ORIG[key] = (BaseHandler, "get_response", orig)
    BaseHandler.get_response = wrapped  # type: ignore[method-assign]


def enable_frameworks() -> None:
    """Patch supported web frameworks (no-op if not installed)."""
    _patch_starlette()
    _patch_flask()
    _patch_django()


def disable_frameworks() -> None:
    """Restore original framework methods."""
    for key, (owner, attr, orig) in list(_ORIG.items()):
        setattr(owner, attr, orig)
        del _ORIG[key]
