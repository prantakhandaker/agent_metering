"""Allow `python -m agent_metering` (avoids blocked console .exe shims on some Windows setups)."""

from agent_metering.cli import main

raise SystemExit(main())
