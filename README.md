# agent_metering

[![CI](https://github.com/prantakhandaker/agent_metering/actions/workflows/ci.yml/badge.svg)](https://github.com/prantakhandaker/agent_metering/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Import-once **LLM cost metering** for B2B SaaS: put `customer_id` / `feature` in a config JSON, `import agent_metering`, optionally `set_user` per request — existing OpenAI / Anthropic calls are tracked automatically.

**MIT open source** — [CONTRIBUTING](CONTRIBUTING.md) · [SECURITY](SECURITY.md) · [Code of Conduct](CODE_OF_CONDUCT.md)

## Who this is for / not for

**For:** Multi-tenant SaaS teams that need simple **per-customer / per-feature** LLM spend visibility with minimal integration (import + config).

**Not for:** Full tracing/evals (use Langfuse), multi-provider gateways with routing (use LiteLLM / Portkey), or finance-grade invoice reconciliation. This library meters and attributes cost; it does not replace those stacks.

## Why

Flat API rate limits do not protect margin. Agent workloads are open-ended: tool loops, retries, and long contexts can burn thousands of tokens per interaction. A single runaway agent loop can quietly wipe out customer margin before anyone notices.

## Install

```bash
pip install "git+https://github.com/prantakhandaker/agent_metering.git"
# or from a clone:
pip install -e ".[dashboard,dev,example]"
```

When published to PyPI: `pip install agent-metering` (see [CONTRIBUTING.md](CONTRIBUTING.md) for maintainer upload steps).

## Product owner setup (primary)

1. Copy config and set defaults (and optional proxy keys if you use the sidecar later):

```bash
cp examples/agent_metering.config.example.json agent_metering.config.json
```

```json
{
  "customer_id": "acme_corp",
  "feature": "default",
  "providers": {
    "openai": { "api_key": "sk-..." }
  }
}
```

On the **import** path, your app still uses its own `OPENAI_API_KEY` / `api_key=` on the client. Config `customer_id` / `feature` are defaults for attribution. Provider `api_key` entries are used by the **optional proxy**.

2. In your app entrypoint:

```python
import agent_metering  # auto-enables when agent_metering.config.json exists

# Optional: once per HTTP request — NOT around every LLM call
@app.middleware("http")
async def metering_user(request, call_next):
    agent_metering.set_user(getattr(request.state, "user_id", None) or "anonymous")
    agent_metering.set_feature("api")
    return await call_next(request)

# Existing OpenAI() / Anthropic() code unchanged
```

Without middleware, all calls use `customer_id` / `feature` from the config JSON.

```bash
python examples/auto_instrument_example.py
```

3. View spend:

```bash
python -m streamlit run examples/dashboard.py
```

Force on/off: `AGENT_METERING_AUTO=1` or `0`, or `agent_metering.enable()` / `disable()`.  
Do not commit `agent_metering.config.json` or service-account JSON (gitignored).

## Optional: proxy / sidecar (no import)

Use when you cannot import into the app process. Point SDK base-URL env vars at the proxy; keys can live in the config JSON for the proxy to inject.

```bash
python -m agent_metering run --config agent_metering.config.json --start-proxy -- python your_app.py
```

Or:

```bash
export AGENT_METERING_CONFIG=./agent_metering.config.json
uvicorn agent_metering.proxy:app --port 8787
export OPENAI_BASE_URL=http://127.0.0.1:8787/proxy/openai/v1
python your_app.py
```

Sole proxy demo: `examples/proxy_env_only_example.py`.  
Docker: `docker compose -f examples/docker-compose.sidecar.yml up --build`.

| Provider | App `base_url` / env |
|----------|----------------------|
| OpenAI | `.../proxy/openai/v1` · `OPENAI_BASE_URL` |
| Anthropic | `.../proxy/anthropic` · `ANTHROPIC_BASE_URL` |
| Azure | `.../proxy/azure/v1` |
| Gemini | `.../proxy/gemini` |
| Vertex | `.../proxy/vertex/v1/projects/{PROJECT}/locations/{LOCATION}/endpoints/openapi` · `VERTEX_OPENAI_BASE_URL` |

Custom providers: [`agent_metering/providers.yaml`](agent_metering/providers.yaml). Vertex needs `pip install "agent-metering[vertex]"`.

## Project layout

**Primary**

```
agent_metering/
  __init__.py      # import + auto-enable
  instrument.py    # OpenAI / Anthropic patches
  context.py       # set_user / set_feature
  config.py
  core.py
  storage.py
  pricing.py
  alerts.py
examples/
  agent_metering.config.example.json
  auto_instrument_example.py
  dashboard.py
```

**Optional (proxy)**

```
agent_metering/
  proxy.py
  cli.py
  vertex_auth.py
  providers/
examples/
  proxy_env_only_example.py
  docker-compose.sidecar.yml
  Dockerfile.proxy
```

## Tests

```bash
pytest
```

## Roadmap

- Streaming call metering on the import path
- Hosted multi-tenant dashboard
- Postgres storage backend

## License

[MIT](LICENSE)
