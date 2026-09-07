# Security Policy

## Supported versions

Security fixes are applied on the latest `main` branch and released versions of `llm-agent-metering` when practical.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Instead, email the maintainer via the contact listed on the [GitHub profile](https://github.com/prantakhandaker) for the repository owner, or open a **private** security advisory on GitHub if enabled for this repo.

Include:

- Description of the issue
- Steps to reproduce
- Impact (e.g. secret leakage, auth bypass on the optional proxy)
- Any suggested fix

## Secrets

Never commit:

- `agent_metering.config.json` with real keys
- Service-account JSON / cloud credentials
- `.env` files with production secrets

Report accidental secret commits immediately so keys can be rotated.
