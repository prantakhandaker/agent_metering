"""Request-scoped user / feature attribution via contextvars."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Generator, Optional

_user_id: ContextVar[Optional[str]] = ContextVar("agent_metering_user_id", default=None)
_feature: ContextVar[Optional[str]] = ContextVar("agent_metering_feature", default=None)


def set_user(user_id: Optional[str]) -> Token:
    """Set the current user/customer id for auto-instrumented LLM calls."""
    return _user_id.set(user_id)


def get_user() -> Optional[str]:
    return _user_id.get()


def clear_user() -> None:
    _user_id.set(None)


def set_feature(feature: Optional[str]) -> Token:
    """Set the current feature tag for auto-instrumented LLM calls."""
    return _feature.set(feature)


def get_feature() -> Optional[str]:
    return _feature.get()


def clear_feature() -> None:
    _feature.set(None)


@contextmanager
def user_context(
    user_id: Optional[str] = None,
    feature: Optional[str] = None,
) -> Generator[None, None, None]:
    """Temporarily set user/feature for the current async/task context."""
    user_token: Optional[Token] = None
    feature_token: Optional[Token] = None
    try:
        if user_id is not None:
            user_token = set_user(user_id)
        if feature is not None:
            feature_token = set_feature(feature)
        yield
    finally:
        if feature_token is not None:
            _feature.reset(feature_token)
        if user_token is not None:
            _user_id.reset(user_token)
