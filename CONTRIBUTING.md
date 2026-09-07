# Contributing to agent_metering

Thanks for contributing. This project is an import-once LLM cost metering library (MIT).

## Setup

```bash
git clone https://github.com/prantakhandaker/agent_metering.git
cd agent_metering
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dashboard,dev,example]"
```

## Tests

```bash
pytest
```

CI runs the same command on every push and pull request.

## Pull requests

1. Fork and create a branch from `main`.
2. Keep changes focused (one concern per PR).
3. Add or update tests when behavior changes.
4. Run `pytest` locally before opening the PR.
5. Do not commit secrets (`agent_metering.config.json`, API keys, service-account JSON).

## Code style

- Python 3.10+
- Match existing module layout: import path (`instrument`, `context`) is primary; proxy/CLI is optional
- Prefer small, readable functions; avoid drive-by refactors

## Publishing to PyPI (maintainers)

```bash
pip install build twine
python -m build
twine upload dist/*
```

Requires a PyPI account and Trusted Publishing for the `llm-agent-metering` project (see README Releasing).

## Security

See [SECURITY.md](SECURITY.md) for how to report vulnerabilities.
