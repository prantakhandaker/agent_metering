"""Site .pth entry — enable metering without any app import.

Installed as ``agent_metering.pth`` in site-packages so every Python
process in that environment auto-instruments OpenAI / Anthropic SDKs.
Opt out with ``AGENT_METERING_AUTO=0``.
"""

from __future__ import annotations


def _autoload() -> None:
    try:
        from agent_metering.instrument import maybe_auto_enable

        maybe_auto_enable()
    except Exception:
        # Never break host interpreter startup.
        pass


_autoload()
