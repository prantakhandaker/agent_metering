"""CLI for zero-code metering: inject provider base URLs and run a child process."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Sequence

from agent_metering.config import (
    ENV_CONFIG,
    get_config,
    reset_config,
    resolve_vertex_settings,
    vertex_openai_compatible_base_url,
)
from agent_metering.proxy import ENV_CUSTOMER_ID, ENV_FEATURE

DEFAULT_PROXY_HOST = "127.0.0.1"
DEFAULT_PROXY_PORT = 8787


def normalize_proxy_url(url: str) -> str:
    return url.rstrip("/")


def build_child_env(
    proxy_url: str,
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Build env for a child app so official SDKs hit the metering proxy."""
    base = normalize_proxy_url(proxy_url)
    env = dict(environ if environ is not None else os.environ)
    env["OPENAI_BASE_URL"] = f"{base}/proxy/openai/v1"
    env["ANTHROPIC_BASE_URL"] = f"{base}/proxy/anthropic"
    # Azure OpenAI Python SDK / OpenAI-compatible clients
    env["AZURE_OPENAI_ENDPOINT"] = f"{base}/proxy/azure"
    env["AZURE_OPENAI_BASE_URL"] = f"{base}/proxy/azure/v1"
    # Common Gemini / Google Generative Language base overrides
    env["GOOGLE_GEMINI_BASE_URL"] = f"{base}/proxy/gemini"
    env["GEMINI_API_BASE"] = f"{base}/proxy/gemini"

    vertex = resolve_vertex_settings()
    if vertex and vertex.project_id and vertex.location:
        env["VERTEX_OPENAI_BASE_URL"] = vertex_openai_compatible_base_url(
            vertex.project_id,
            vertex.location,
            proxy_base=base,
        )
    return env


def proxy_ready(proxy_url: str, timeout: float = 0.5) -> bool:
    base = normalize_proxy_url(proxy_url)
    # FastAPI exposes /docs by default; any HTTP response means the server is up.
    for path in ("/docs", "/openapi.json", "/"):
        try:
            urllib.request.urlopen(f"{base}{path}", timeout=timeout)
            return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
    return False


def wait_for_proxy(proxy_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proxy_ready(proxy_url):
            return
        time.sleep(0.2)
    raise SystemExit(
        f"Timed out waiting for metering proxy at {normalize_proxy_url(proxy_url)}"
    )


def start_proxy_process(
    *,
    host: str,
    port: int,
    customer: Optional[str],
    feature: Optional[str],
    config_path: Optional[str] = None,
    environ: Optional[MutableMapping[str, str]] = None,
) -> subprocess.Popen[bytes]:
    env = dict(environ if environ is not None else os.environ)
    if config_path:
        env[ENV_CONFIG] = str(Path(config_path).resolve())
    if customer:
        env[ENV_CUSTOMER_ID] = customer
    if feature:
        env[ENV_FEATURE] = feature
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "agent_metering.proxy:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(cmd, env=env)


def _strip_leading_separator(command: Sequence[str]) -> list[str]:
    cmd = list(command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    return cmd


def run_command(
    command: Sequence[str],
    *,
    proxy_url: str,
    start_proxy: bool,
    host: str,
    port: int,
    customer: Optional[str],
    feature: Optional[str],
    config_path: Optional[str] = None,
) -> int:
    cmd = _strip_leading_separator(command)
    if not cmd:
        raise SystemExit("Pass a command after options, e.g. -- python my_app.py")

    if config_path:
        os.environ[ENV_CONFIG] = str(Path(config_path).resolve())
        reset_config()
        get_config(force_reload=True)

    proxy_proc: Optional[subprocess.Popen[bytes]] = None
    try:
        if start_proxy:
            cfg = get_config()
            effective_customer = customer or (
                cfg.customer_id if cfg.customer_id != "unknown" else None
            )
            effective_feature = feature or (
                cfg.feature if cfg.feature != "unknown" else None
            )
            if effective_customer is None and effective_feature is None and not config_path:
                print(
                    "Note: starting proxy without --customer/--feature/--config; "
                    f"set {ENV_CUSTOMER_ID}/{ENV_FEATURE} or a config JSON for attribution.",
                    file=sys.stderr,
                )
            proxy_proc = start_proxy_process(
                host=host,
                port=port,
                customer=effective_customer,
                feature=effective_feature,
                config_path=config_path,
            )
            wait_for_proxy(proxy_url)
        elif not proxy_ready(proxy_url):
            raise SystemExit(
                f"No metering proxy at {normalize_proxy_url(proxy_url)}. "
                "Start one with uvicorn, or pass --start-proxy."
            )
        elif customer or feature:
            print(
                "Note: --customer/--feature only apply when using --start-proxy. "
                f"Set {ENV_CUSTOMER_ID}/{ENV_FEATURE} on the existing proxy process.",
                file=sys.stderr,
            )

        child_env = build_child_env(proxy_url)
        if config_path:
            child_env[ENV_CONFIG] = str(Path(config_path).resolve())
        resolved = shutil.which(cmd[0]) or cmd[0]
        full_cmd = [resolved, *cmd[1:]]
        completed = subprocess.run(full_cmd, env=child_env)
        return int(completed.returncode)
    finally:
        if proxy_proc is not None and proxy_proc.poll() is None:
            proxy_proc.terminate()
            try:
                proxy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_proc.kill()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-metering",
        description=(
            "Zero-code LLM metering: point official SDK base-URL env vars at the "
            "proxy and run your existing app unchanged."
        ),
    )
    sub = parser.add_subparsers(dest="command_name", required=True)

    run_p = sub.add_parser(
        "run",
        help="Inject OPENAI_BASE_URL / ANTHROPIC_BASE_URL (etc.) and run a command",
    )
    run_p.add_argument(
        "--proxy-url",
        default=f"http://{DEFAULT_PROXY_HOST}:{DEFAULT_PROXY_PORT}",
        help=f"Metering proxy base URL (default: http://{DEFAULT_PROXY_HOST}:{DEFAULT_PROXY_PORT})",
    )
    run_p.add_argument(
        "--start-proxy",
        action="store_true",
        help="Start a local proxy before running the command",
    )
    run_p.add_argument(
        "--host",
        default=DEFAULT_PROXY_HOST,
        help=f"Host for --start-proxy (default: {DEFAULT_PROXY_HOST})",
    )
    run_p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PROXY_PORT,
        help=f"Port for --start-proxy (default: {DEFAULT_PROXY_PORT})",
    )
    run_p.add_argument(
        "--customer",
        default=None,
        help=f"Default customer id for a proxy started with --start-proxy ({ENV_CUSTOMER_ID})",
    )
    run_p.add_argument(
        "--feature",
        default=None,
        help=f"Default feature for a proxy started with --start-proxy ({ENV_FEATURE})",
    )
    run_p.add_argument(
        "--config",
        default=None,
        dest="config_path",
        help=(
            f"Path to agent_metering.config.json (API keys and/or GCP service-account JSON). "
            f"Sets {ENV_CONFIG} for the proxy process."
        ),
    )
    run_p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run (use -- before it), e.g. -- python my_app.py",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command_name == "run":
        proxy_url = args.proxy_url
        if args.start_proxy:
            proxy_url = f"http://{args.host}:{args.port}"
        return run_command(
            args.command,
            proxy_url=proxy_url,
            start_proxy=args.start_proxy,
            host=args.host,
            port=args.port,
            customer=args.customer,
            feature=args.feature,
            config_path=args.config_path,
        )
    parser.error(f"Unknown command: {args.command_name}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
