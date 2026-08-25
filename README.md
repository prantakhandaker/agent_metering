# agent_metering

Universal **LLM metering proxy** for B2B SaaS: track **cost per customer and per feature** without changing your LLM call sites — point your client at `/proxy/{provider}/...`.

## Why

Flat API rate limits do not protect margin. Agent workloads are open-ended: tool loops, retries, and long contexts can burn thousands of tokens per interaction. A single runaway agent loop can quietly wipe out customer margin before anyone notices. `agent_metering` sits in front of LLM APIs, records every call with customer/feature tags, prices it locally, and surfaces spend in a dashboard.

## Install

### From GitHub

```bash
git clone https://github.com/prantakhandaker/agent_metering.git
cd agent_metering
pip install -e ".[dashboard,dev,example]"
uvicorn agent_metering.proxy:app --port 8787
```

Or without cloning:

```bash
pip install "git+https://github.com/prantakhandaker/agent_metering.git"
uvicorn agent_metering.proxy:app --port 8787
```

### From a local checkout

```bash
pip install -r requirements.txt
# or: pip install -e ".[dashboard,dev,example]"
```

## Zero code change

If your app uses the official OpenAI / Anthropic SDKs **without** a hardcoded `base_url`, you can add metering with **no application source changes**.

1. Run the proxy with attribution defaults (customer / feature for all traffic):

```bash
export AGENT_METERING_CUSTOMER_ID=acme_corp
export AGENT_METERING_FEATURE=support_bot
uvicorn agent_metering.proxy:app --port 8787
```

2. Point SDK base-URL env vars at the proxy (deploy config, shell, or sidecar):

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8787/proxy/openai/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/proxy/anthropic
# then start your existing app unchanged
python my_app.py
```

Or let the CLI inject those env vars and optionally start the proxy:

```bash
python -m agent_metering run --start-proxy --customer acme_corp --feature support_bot -- python my_app.py
```

(`agent-metering` works after `pip install` when console scripts are allowed; on some Windows setups the `.exe` shim is blocked — use `python -m agent_metering` instead.)

Docker Compose sidecar (app image unchanged; only env overrides):

```bash
export OPENAI_API_KEY=sk-...
docker compose -f examples/docker-compose.sidecar.yml up --build
```

Env-only example (client constructed with no `base_url`):

```bash
python -m agent_metering run --start-proxy --customer acme_corp --feature support_bot -- `
  python examples/proxy_env_only_example.py
```

Limitation: apps that **hardcode** `base_url` to the real provider bypass env injection. Request headers `X-Customer-Id` / `X-Feature` still override proxy env defaults when present.

## How customers connect (minimal code)

After the proxy is running, point your LLM client at it — change only `base_url` (and optional attribution headers). No other call-site changes.

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="http://localhost:8787/proxy/openai/v1",
    default_headers={
        "X-Customer-Id": "acme_corp",
        "X-Feature": "support_bot",
    },
)
```

Then view spend:

```bash
python -m streamlit run examples/dashboard.py
```

## Supported providers

| Provider | Customer `base_url` | Upstream |
|----------|---------------------|----------|
| OpenAI | `http://localhost:8787/proxy/openai/v1` | `api.openai.com` |
| Anthropic | `http://localhost:8787/proxy/anthropic` | `api.anthropic.com` |
| Azure OpenAI | `http://localhost:8787/proxy/azure/v1` | `AZURE_OPENAI_BASE_URL` env on proxy |
| Gemini | `http://localhost:8787/proxy/gemini` | Google Generative Language API |
| Custom | `http://localhost:8787/proxy/{name}/...` | Defined in `agent_metering/providers.yaml` |

Legacy alias (OpenAI only): `base_url="http://localhost:8787/v1"` still works.

Zero-code env equivalents (set on the **app** process):

| Provider | Env var |
|----------|---------|
| OpenAI | `OPENAI_BASE_URL` |
| Anthropic | `ANTHROPIC_BASE_URL` |
| Azure | `AZURE_OPENAI_BASE_URL` / `AZURE_OPENAI_ENDPOINT` |
| Gemini | `GOOGLE_GEMINI_BASE_URL` / `GEMINI_API_BASE` |

Proxy attribution defaults (set on the **proxy** process): `AGENT_METERING_CUSTOMER_ID`, `AGENT_METERING_FEATURE`.

### OpenAI example

```bash
python examples/proxy_client_example.py
```

### Anthropic example

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
python examples/proxy_anthropic_example.py
```

### Azure OpenAI example (responses.create)

Set on the **proxy** host:

```bash
export AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1
export AZURE_OPENAI_API_KEY=...
python examples/proxy_azure_example.py
```

### Custom provider via `providers.yaml`

Edit [`agent_metering/providers.yaml`](agent_metering/providers.yaml):

```yaml
providers:
  my_custom_llm:
    base_url: https://llm.mycompany.com
    provider_label: my_custom_llm
    auth_headers: [Authorization]
    model_path: model
    input_tokens_path: usage.prompt_tokens
    output_tokens_path: usage.completion_tokens
    stream_mode: openai_sse
```

Customer connects: `base_url="http://localhost:8787/proxy/my_custom_llm/v1"`.

## Demo (no API key)

```bash
python examples/demo_no_api_key.py
```

## Project layout

```
agent_metering/
  proxy.py
  cli.py
  providers/
    registry.py
    extractors.py
  providers.yaml
  core.py
  storage.py
  pricing.py
  alerts.py
examples/
  proxy_client_example.py
  proxy_env_only_example.py
  proxy_anthropic_example.py
  proxy_azure_example.py
  docker-compose.sidecar.yml
  Dockerfile.proxy
  dashboard.py
  demo_no_api_key.py
tests/
pyproject.toml
requirements.txt
LICENSE
README.md
```

## Tests

```bash
pytest
```

## Roadmap

- Hosted proxy URL (multi-tenant SaaS)
- Postgres storage backend
- Auth / API for hosted dashboard
