"""Provider registry for the universal LLM proxy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

StreamMode = Literal["openai_sse", "anthropic_sse", "none"]
UsageExtractor = Literal["openai", "anthropic", "gemini", "generic"]


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    provider_label: str
    auth_headers: list[str] = field(default_factory=lambda: ["Authorization"])
    usage_extractor: UsageExtractor = "openai"
    stream_mode: StreamMode = "openai_sse"
    model_path: Optional[str] = None
    input_tokens_path: Optional[str] = None
    output_tokens_path: Optional[str] = None


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _builtin_providers() -> dict[str, ProviderConfig]:
    azure_base = os.getenv(
        "AZURE_OPENAI_BASE_URL",
        "https://your-resource.openai.azure.com/openai/v1",
    )
    return {
        "openai": ProviderConfig(
            name="openai",
            base_url="https://api.openai.com",
            provider_label="openai",
            auth_headers=["Authorization"],
            usage_extractor="openai",
            stream_mode="openai_sse",
        ),
        "anthropic": ProviderConfig(
            name="anthropic",
            base_url="https://api.anthropic.com",
            provider_label="anthropic",
            auth_headers=["x-api-key", "anthropic-version", "Authorization"],
            usage_extractor="anthropic",
            stream_mode="anthropic_sse",
        ),
        "azure": ProviderConfig(
            name="azure",
            base_url=_normalize_base_url(azure_base),
            provider_label="azure_openai",
            auth_headers=["Authorization", "api-key"],
            usage_extractor="openai",
            stream_mode="openai_sse",
        ),
        "gemini": ProviderConfig(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com",
            provider_label="google",
            auth_headers=["Authorization", "x-goog-api-key"],
            usage_extractor="gemini",
            stream_mode="none",
        ),
        # Vertex AI — base host only; OpenAI-compatible path includes project/location.
        # Auth via OAuth from service-account JSON (see agent_metering.vertex_auth).
        "vertex": ProviderConfig(
            name="vertex",
            base_url=_vertex_host(),
            provider_label="vertex_ai",
            auth_headers=["Authorization"],
            usage_extractor="openai",
            stream_mode="openai_sse",
        ),
    }


def _vertex_host() -> str:
    """Vertex AI API host from config/env location (default us-central1)."""
    location = "us-central1"
    try:
        from agent_metering.config import resolve_vertex_settings

        settings = resolve_vertex_settings()
        if settings and settings.location:
            location = settings.location
    except Exception:
        location = os.getenv("VERTEX_LOCATION") or os.getenv(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        )
    return f"https://{location}-aiplatform.googleapis.com"


def _load_yaml_providers(config_path: Path) -> dict[str, ProviderConfig]:
    if not config_path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    providers: dict[str, ProviderConfig] = {}
    for name, cfg in (raw.get("providers") or {}).items():
        if not isinstance(cfg, dict):
            continue
        providers[name] = ProviderConfig(
            name=name,
            base_url=_normalize_base_url(str(cfg.get("base_url", ""))),
            provider_label=str(cfg.get("provider_label", name)),
            auth_headers=list(cfg.get("auth_headers") or ["Authorization"]),
            usage_extractor="generic",
            stream_mode=cfg.get("stream_mode", "openai_sse"),
            model_path=cfg.get("model_path"),
            input_tokens_path=cfg.get("input_tokens_path"),
            output_tokens_path=cfg.get("output_tokens_path"),
        )
    return providers


_REGISTRY: dict[str, ProviderConfig] | None = None


def get_registry(config_path: Optional[Path] = None) -> dict[str, ProviderConfig]:
    global _REGISTRY
    if _REGISTRY is not None and config_path is None:
        return _REGISTRY

    merged = _builtin_providers()
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "providers.yaml"
    merged.update(_load_yaml_providers(config_path))
    if config_path is None or config_path == Path(__file__).resolve().parent.parent / "providers.yaml":
        _REGISTRY = merged
    return merged


def get_provider(name: str, config_path: Optional[Path] = None) -> ProviderConfig:
    registry = get_registry(config_path)
    if name not in registry:
        raise KeyError(f"Unknown provider: {name}")
    return registry[name]


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None
