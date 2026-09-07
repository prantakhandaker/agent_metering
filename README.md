# agent_metering

[![CI](https://github.com/prantakhandaker/agent_metering/actions/workflows/ci.yml/badge.svg)](https://github.com/prantakhandaker/agent_metering/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Language-agnostic LLM cost metering** for B2B SaaS — point any OpenAI / Anthropic client at the HTTP proxy (Node, Go, Java, PHP, Python, curl, …).

**MIT open source** — [CONTRIBUTING](CONTRIBUTING.md) · [SECURITY](SECURITY.md) · [Code of Conduct](CODE_OF_CONDUCT.md)

## Install

```bash
pip install agent-metering
# optional extras:
# pip install "agent-metering[dashboard]"
# pip install "agent-metering[example]"
```

From GitHub (latest main):

```bash
pip install "git+https://github.com/prantakhandaker/agent_metering.git"
```

## Any language (recommended)

Run the proxy once, then set your SDK **base URL** to it. No SDK install in the app language required.

```bash
pip install agent-metering
python -m uvicorn agent_metering.proxy:app --host 0.0.0.0 --port 8787
```

| Provider | Base URL / env |
|----------|----------------|
| OpenAI | `http://127.0.0.1:8787/proxy/openai/v1` → `OPENAI_BASE_URL` |
| Anthropic | `http://127.0.0.1:8787/proxy/anthropic` → `ANTHROPIC_BASE_URL` |
| Azure | `.../proxy/azure/v1` → `AZURE_OPENAI_BASE_URL` |

Optional per-user / feature headers (stripped before upstream):

- `X-User-Id` (preferred) or `X-Customer-Id`
- `X-Feature`
- Or OpenAI body field `user` / Anthropic `metadata.user_id`

Defaults: env `AGENT_METERING_CUSTOMER_ID` / `AGENT_METERING_FEATURE`, else `default`.

Spend → local SQLite `agent_metering.db`. Dashboard:

```bash
python -m streamlit run examples/dashboard.py   # pip install "agent-metering[dashboard]"
```

### Node

```js
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://127.0.0.1:8787/proxy/openai/v1",
  defaultHeaders: { "X-User-Id": "user_42" },
});

await client.chat.completions.create({
  model: "gpt-4o-mini",
  messages: [{ role: "user", content: "Hi" }],
});
```

Full script: [`examples/proxy_node_example.mjs`](examples/proxy_node_example.mjs).

### curl

```bash
curl http://127.0.0.1:8787/proxy/openai/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user_42" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hi"}]}'
```

### Go

```go
client := openai.NewClient(
  option.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
  option.WithBaseURL("http://127.0.0.1:8787/proxy/openai/v1"),
  option.WithHeader("X-User-Id", "user_42"),
)
```

### Docker sidecar

```bash
docker compose -f examples/docker-compose.sidecar.yml up --build
```

App containers only need `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` pointing at `http://metering-proxy:8787/proxy/...`.

Or wrap a local process:

```bash
python -m agent_metering run --start-proxy -- python your_app.py
```

## Python-only shortcut (optional)

Same venv install auto-patches OpenAI / Anthropic SDKs (no base URL change):

```bash
pip install agent-metering
# run your Python app — no import required
```

Opt out: `AGENT_METERING_AUTO=0`. Demo: `python examples/auto_instrument_example.py`.

Also detects FastAPI/Flask/Django request users and OpenAI `user=` when present.

## Who this is for / not for

**For:** Any stack that can set an LLM HTTP base URL (or env) and needs **per-customer / per-user / per-feature** spend.

**Not for:** Full tracing/evals (Langfuse), or replacing multi-provider gateways you already run (LiteLLM / Portkey) unless you put this proxy in front.

## Why

Flat API rate limits do not protect margin. Agent workloads are open-ended: tool loops and long contexts can burn tokens quietly. Metering per customer/feature surfaces that before margin disappears.

## Project layout

**Primary (any language):** `proxy.py`, `providers/`, `cli.py`  
**Python convenience:** `autoload.py` (`.pth`), `instrument.py`, `user_detect.py`, `frameworks.py`  
**Shared:** `config.py`, `core.py`, `storage.py`, `context.py`

## Tests

```bash
pytest
```

## Releasing

Maintainers publish to PyPI via GitHub Actions Trusted Publishing (no API token in secrets).

1. One-time on [pypi.org](https://pypi.org): **Publishing → Pending publisher**
   - Project: `agent-metering`
   - Owner: `prantakhandaker`
   - Repository: `agent_metering`
   - Workflow: `publish.yml`
2. Bump `version` in `pyproject.toml` to match the release tag.
3. Tag and release:

```bash
# version in pyproject.toml must match the tag
git tag v0.3.0
git push origin v0.3.0
# then GitHub → Releases → Draft release from that tag → Publish
```

Publishing workflow: [`.github/workflows/publish.yml`](.github/workflows/publish.yml).

## License

[MIT](LICENSE)
