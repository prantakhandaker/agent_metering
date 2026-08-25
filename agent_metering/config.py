"""Plug-and-play metering config — API keys and/or GCP service-account JSON."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

ENV_CONFIG = "AGENT_METERING_CONFIG"
DEFAULT_CONFIG_NAME = "agent_metering.config.json"

CredentialsJson = Union[str, dict[str, Any]]


@dataclass
class ProviderCredentials:
    """Credentials for one upstream provider."""

    api_key: Optional[str] = None
    project_id: Optional[str] = None
    location: Optional[str] = None
    credentials_json: Optional[CredentialsJson] = None  # path or inline SA JSON


@dataclass
class MeteringConfig:
    customer_id: str = "unknown"
    feature: str = "unknown"
    providers: dict[str, ProviderCredentials] = field(default_factory=dict)
    config_path: Optional[Path] = None


_CONFIG: Optional[MeteringConfig] = None


def default_config_path() -> Path:
    override = os.environ.get(ENV_CONFIG)
    if override:
        return Path(override)
    return Path.cwd() / DEFAULT_CONFIG_NAME


def _parse_provider(raw: Any) -> Optional[ProviderCredentials]:
    if not isinstance(raw, dict):
        return None
    api_key = raw.get("api_key")
    project_id = raw.get("project_id")
    location = raw.get("location")
    credentials_json = raw.get("credentials_json")
    if api_key is not None:
        api_key = str(api_key)
    if project_id is not None:
        project_id = str(project_id)
    if location is not None:
        location = str(location)
    return ProviderCredentials(
        api_key=api_key,
        project_id=project_id,
        location=location,
        credentials_json=credentials_json,
    )


def load_config(path: Optional[Path] = None) -> MeteringConfig:
    """Load metering config from JSON. Missing file → empty defaults."""
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.is_file():
        return MeteringConfig(config_path=config_path)

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return MeteringConfig(config_path=config_path)

    if not isinstance(raw, dict):
        return MeteringConfig(config_path=config_path)

    providers: dict[str, ProviderCredentials] = {}
    for name, cfg in (raw.get("providers") or {}).items():
        parsed = _parse_provider(cfg)
        if parsed is not None:
            providers[str(name)] = parsed

    customer = raw.get("customer_id")
    feature = raw.get("feature")
    return MeteringConfig(
        customer_id=str(customer) if customer else "unknown",
        feature=str(feature) if feature else "unknown",
        providers=providers,
        config_path=config_path,
    )


def get_config(force_reload: bool = False) -> MeteringConfig:
    """Cached config (reload when force_reload or AGENT_METERING_CONFIG changes path)."""
    global _CONFIG
    if _CONFIG is not None and not force_reload:
        return _CONFIG
    _CONFIG = load_config()
    return _CONFIG


def reset_config() -> None:
    global _CONFIG
    _CONFIG = None


def resolve_api_key(provider_name: str, config: Optional[MeteringConfig] = None) -> Optional[str]:
    """API key for provider: env override first, then config file."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure": "AZURE_OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_name = env_map.get(provider_name)
    if env_name:
        from_env = os.environ.get(env_name)
        if from_env:
            return from_env
        # Gemini often uses GOOGLE_API_KEY
        if provider_name == "gemini":
            google_key = os.environ.get("GOOGLE_API_KEY")
            if google_key:
                return google_key

    cfg = config if config is not None else get_config()
    creds = cfg.providers.get(provider_name)
    if creds and creds.api_key:
        return creds.api_key
    return None


def resolve_vertex_settings(
    config: Optional[MeteringConfig] = None,
) -> Optional[ProviderCredentials]:
    """Vertex settings from config, with GOOGLE_APPLICATION_CREDENTIALS fallback."""
    cfg = config if config is not None else get_config()
    creds = cfg.providers.get("vertex")
    if creds is None:
        creds = ProviderCredentials()

    project_id = creds.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
        "GCLOUD_PROJECT"
    )
    location = creds.location or os.environ.get("VERTEX_LOCATION") or os.environ.get(
        "GOOGLE_CLOUD_LOCATION", "us-central1"
    )
    credentials_json: Optional[CredentialsJson] = creds.credentials_json
    if credentials_json is None:
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if gac:
            credentials_json = gac

    if not project_id and not credentials_json:
        return None

    return ProviderCredentials(
        api_key=creds.api_key,
        project_id=project_id,
        location=location or "us-central1",
        credentials_json=credentials_json,
    )


def vertex_openai_compatible_base_url(
    project_id: str,
    location: str,
    proxy_base: Optional[str] = None,
) -> str:
    """Upstream or proxied OpenAI-compatible Vertex base URL (no trailing slash)."""
    path = (
        f"v1/projects/{project_id}/locations/{location}/endpoints/openapi"
    )
    if proxy_base:
        return f"{proxy_base.rstrip('/')}/proxy/vertex/{path}"
    return f"https://{location}-aiplatform.googleapis.com/{path}"
