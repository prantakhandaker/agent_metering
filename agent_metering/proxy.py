"""Universal LLM proxy — meter usage for OpenAI, Anthropic, Azure, Gemini, and custom APIs."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from agent_metering.core import Meter
from agent_metering.providers.extractors import (
    extract_usage,
    parse_stream_event,
    usage_from_stream_dict,
)
from agent_metering.providers.registry import ProviderConfig, get_provider

logger = logging.getLogger(__name__)

meter = Meter()

# Headers set by the proxy — do not forward to upstream.
METERING_HEADERS = frozenset(
    h.lower()
    for h in (
        "X-Customer-Id",
        "X-Feature",
        "host",
        "content-length",
        "transfer-encoding",
    )
)

# Proxy-process env defaults when request headers are absent (zero-code path).
ENV_CUSTOMER_ID = "AGENT_METERING_CUSTOMER_ID"
ENV_FEATURE = "AGENT_METERING_FEATURE"


def _attribution_defaults() -> tuple[str, str]:
    return (
        os.environ.get(ENV_CUSTOMER_ID, "unknown"),
        os.environ.get(ENV_FEATURE, "unknown"),
    )


def create_app() -> FastAPI:
    app = FastAPI(title="agent_metering proxy", version="0.2.0")

    @app.api_route(
        "/proxy/{provider}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def proxy_provider(provider: str, path: str, request: Request) -> Response:
        try:
            cfg = get_provider(provider)
        except KeyError:
            return Response(
                content=json.dumps({"error": f"Unknown provider: {provider}"}),
                status_code=404,
                media_type="application/json",
            )
        return await _handle_proxy_request(request, cfg, path)

    @app.post("/v1/chat/completions")
    async def legacy_openai_chat(request: Request) -> Response:
        """Backward-compatible alias for OpenAI chat completions."""
        cfg = get_provider("openai")
        return await _handle_proxy_request(request, cfg, "v1/chat/completions")

    return app


def _build_upstream_url(cfg: ProviderConfig, path: str) -> str:
    clean_path = path.lstrip("/")
    return f"{cfg.base_url}/{clean_path}"


def _collect_forward_headers(request: Request, cfg: ProviderConfig) -> dict[str, str]:
    allowed = {h.lower() for h in cfg.auth_headers}
    allowed.update({"content-type", "accept", "accept-encoding", "user-agent"})
    forward: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in METERING_HEADERS:
            continue
        if key.lower() in allowed or key.lower().startswith("x-"):
            if key.lower() not in METERING_HEADERS:
                forward[key] = value
    if "content-type" not in {k.lower() for k in forward}:
        forward["Content-Type"] = request.headers.get("content-type", "application/json")
    return forward


def _request_model(body: bytes) -> str:
    try:
        payload = json.loads(body)
        return str(payload.get("model", "unknown"))
    except json.JSONDecodeError:
        return "unknown"


def _is_streaming(body: bytes) -> bool:
    try:
        return bool(json.loads(body).get("stream"))
    except json.JSONDecodeError:
        return False


def _log_usage(
    *,
    cfg: ProviderConfig,
    customer_id: str,
    feature: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    estimated_output_chars: int = 0,
) -> None:
    if input_tokens == 0 and output_tokens == 0 and estimated_output_chars > 0:
        output_tokens = max(1, estimated_output_chars // 4)
        logger.info(
            "Streaming usage estimated from content length (%d chars); "
            "enable provider stream usage options for exact counts.",
            estimated_output_chars,
        )
    if input_tokens == 0 and output_tokens == 0:
        return
    with meter.track(customer_id=customer_id, feature=feature) as t:
        t.record(
            provider=cfg.provider_label,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


async def _handle_proxy_request(
    request: Request,
    cfg: ProviderConfig,
    path: str,
) -> Response:
    body = await request.body()
    default_customer, default_feature = _attribution_defaults()
    customer_id = request.headers.get("X-Customer-Id") or default_customer
    feature = request.headers.get("X-Feature") or default_feature
    forward_headers = _collect_forward_headers(request, cfg)
    upstream_url = _build_upstream_url(cfg, path)
    request_model = _request_model(body)
    is_stream = _is_streaming(body)

    if is_stream and cfg.stream_mode != "none":
        return await _forward_stream(
            method=request.method,
            upstream_url=upstream_url,
            body=body,
            forward_headers=forward_headers,
            cfg=cfg,
            customer_id=customer_id,
            feature=feature,
            request_model=request_model,
        )

    return await _forward_request(
        method=request.method,
        upstream_url=upstream_url,
        body=body,
        forward_headers=forward_headers,
        cfg=cfg,
        customer_id=customer_id,
        feature=feature,
        request_model=request_model,
    )


async def _forward_request(
    *,
    method: str,
    upstream_url: str,
    body: bytes,
    forward_headers: dict[str, str],
    cfg: ProviderConfig,
    customer_id: str,
    feature: str,
    request_model: str,
) -> Response:
    with meter.track(customer_id=customer_id, feature=feature) as t:
        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream = await client.request(
                method,
                upstream_url,
                content=body,
                headers=forward_headers,
            )

        if upstream.status_code >= 400:
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get(
                    "content-type", "application/json"
                ),
            )

        try:
            data = upstream.json()
            model, prompt, completion = extract_usage(cfg, data, request_model=request_model)
            if prompt or completion:
                t.record(
                    provider=cfg.provider_label,
                    model=model,
                    input_tokens=prompt,
                    output_tokens=completion,
                )
        except (json.JSONDecodeError, ValueError):
            pass

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )


async def _forward_stream(
    *,
    method: str,
    upstream_url: str,
    body: bytes,
    forward_headers: dict[str, str],
    cfg: ProviderConfig,
    customer_id: str,
    feature: str,
    request_model: str,
) -> Response:
    client = httpx.AsyncClient(timeout=120.0)
    req = client.build_request(
        method, upstream_url, content=body, headers=forward_headers
    )
    stream_ctx = client.send(req, stream=True)
    response = await stream_ctx.__aenter__()

    if response.status_code >= 400:
        error_body = await response.aread()
        await stream_ctx.__aexit__(None, None, None)
        await client.aclose()
        return Response(
            content=error_body,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    async def stream_generator() -> AsyncIterator[bytes]:
        usage: Optional[dict[str, Any]] = None
        resolved_model = request_model
        content_chars = 0
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
                text = chunk.decode("utf-8", errors="replace")
                for line in text.splitlines():
                    model, event_usage, chars = parse_stream_event(
                        cfg, line, request_model=request_model
                    )
                    if model:
                        resolved_model = model
                    if event_usage:
                        usage = event_usage
                    content_chars += chars
        finally:
            await stream_ctx.__aexit__(None, None, None)
            await client.aclose()
            prompt, completion = (0, 0)
            if usage:
                prompt, completion = usage_from_stream_dict(usage)
            _log_usage(
                cfg=cfg,
                customer_id=customer_id,
                feature=feature,
                model=resolved_model,
                input_tokens=prompt,
                output_tokens=completion,
                estimated_output_chars=content_chars if usage is None else 0,
            )

    return StreamingResponse(
        stream_generator(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
    )


app = create_app()
