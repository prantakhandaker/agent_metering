"""Auto-instrument OpenAI / Anthropic SDKs — meter usage without wrapping call sites."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent_metering.config import default_config_path, get_config, reset_config
from agent_metering.context import get_feature, get_user
from agent_metering.core import Meter

logger = logging.getLogger(__name__)

_ENABLED = False
_METER: Optional[Meter] = None

# Original unbound methods so we can wrap / unwrap idempotently.
_ORIG: dict[str, Any] = {}


def get_meter() -> Meter:
    global _METER
    if _METER is None:
        _METER = Meter()
    return _METER


def set_meter(meter: Meter) -> None:
    """Replace the shared Meter (useful in tests)."""
    global _METER
    _METER = meter


def is_enabled() -> bool:
    return _ENABLED


def _attribution() -> tuple[str, str]:
    cfg = get_config()
    user = get_user() or cfg.customer_id or "unknown"
    feature = get_feature() or cfg.feature or "unknown"
    return user, feature


def _record_openai_response(response: Any) -> None:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        # Responses API style
        if prompt is None:
            prompt = getattr(usage, "input_tokens", 0) or 0
        if completion is None:
            completion = getattr(usage, "output_tokens", 0) or 0
        prompt = int(prompt or 0)
        completion = int(completion or 0)
        if prompt == 0 and completion == 0:
            return
        model = getattr(response, "model", "unknown") or "unknown"
        customer_id, feature = _attribution()
        with get_meter().track(customer_id=customer_id, feature=feature) as t:
            t.record(
                provider="openai",
                model=str(model),
                input_tokens=prompt,
                output_tokens=completion,
            )
    except Exception:
        logger.exception("agent_metering: failed to record OpenAI usage")


def _record_anthropic_response(response: Any) -> None:
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = int(getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "output_tokens", 0) or 0)
        if prompt == 0 and completion == 0:
            return
        model = getattr(response, "model", "unknown") or "unknown"
        customer_id, feature = _attribution()
        with get_meter().track(customer_id=customer_id, feature=feature) as t:
            t.record(
                provider="anthropic",
                model=str(model),
                input_tokens=prompt,
                output_tokens=completion,
            )
    except Exception:
        logger.exception("agent_metering: failed to record Anthropic usage")


def _wrap_sync(orig, recorder):
    def wrapped(self, *args, **kwargs):
        response = orig(self, *args, **kwargs)
        recorder(response)
        return response

    return wrapped


def _wrap_async(orig, recorder):
    async def wrapped(self, *args, **kwargs):
        response = await orig(self, *args, **kwargs)
        recorder(response)
        return response

    return wrapped


def _patch_openai() -> None:
    try:
        import openai
    except ImportError:
        return

    targets: list[tuple[str, Any, Any, bool]] = []
    # (key, owner_class, method_name, is_async)
    try:
        from openai.resources.chat.completions import Completions, AsyncCompletions

        targets.append(("openai.chat.completions.create", Completions, "create", False))
        targets.append(
            ("openai.chat.completions.create_async", AsyncCompletions, "create", True)
        )
    except Exception:
        logger.debug("agent_metering: OpenAI chat.completions not patchable", exc_info=True)

    try:
        from openai.resources.responses import Responses, AsyncResponses

        targets.append(("openai.responses.create", Responses, "create", False))
        targets.append(("openai.responses.create_async", AsyncResponses, "create", True))
    except Exception:
        logger.debug("agent_metering: OpenAI responses not patchable", exc_info=True)

    for key, cls, method_name, is_async in targets:
        if key in _ORIG:
            continue
        orig = getattr(cls, method_name, None)
        if orig is None:
            continue
        _ORIG[key] = (cls, method_name, orig)
        if is_async:
            setattr(cls, method_name, _wrap_async(orig, _record_openai_response))
        else:
            setattr(cls, method_name, _wrap_sync(orig, _record_openai_response))


def _patch_anthropic() -> None:
    try:
        import anthropic  # noqa: F401
        from anthropic.resources.messages import Messages, AsyncMessages
    except ImportError:
        return
    except Exception:
        logger.debug("agent_metering: Anthropic messages not patchable", exc_info=True)
        return

    for key, cls, is_async in (
        ("anthropic.messages.create", Messages, False),
        ("anthropic.messages.create_async", AsyncMessages, True),
    ):
        if key in _ORIG:
            continue
        orig = getattr(cls, "create", None)
        if orig is None:
            continue
        _ORIG[key] = (cls, "create", orig)
        if is_async:
            setattr(cls, "create", _wrap_async(orig, _record_anthropic_response))
        else:
            setattr(cls, "create", _wrap_sync(orig, _record_anthropic_response))


def enable(*, force: bool = False) -> bool:
    """Patch OpenAI/Anthropic SDKs and load config. Idempotent.

    Returns True if instrumentation is active.
    """
    global _ENABLED
    if _ENABLED and not force:
        return True
    if force and _ENABLED:
        disable()
    reset_config()
    get_config(force_reload=True)
    _patch_openai()
    _patch_anthropic()
    _ENABLED = True
    return True


def disable() -> None:
    """Restore original SDK methods (for tests / shutdown)."""
    global _ENABLED
    for key, (cls, method_name, orig) in list(_ORIG.items()):
        setattr(cls, method_name, orig)
        del _ORIG[key]
    _ENABLED = False


def maybe_auto_enable() -> bool:
    """Enable if config file exists or AGENT_METERING_AUTO is truthy."""
    import os

    auto = os.environ.get("AGENT_METERING_AUTO", "").strip().lower()
    if auto in ("0", "false", "no", "off"):
        return False
    if auto in ("1", "true", "yes", "on"):
        return enable()
    path = default_config_path()
    if path.is_file():
        return enable()
    return False
