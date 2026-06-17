# ecliptous AI — Productivity Agent

A lightweight background activity-tracking agent that records timestamped events into an SQLite database and produces session snapshots for analysis and UI consumption.

## Features

- Event logging (active window, input engagement, idle, system signals)
- Minute-granular sessionization and intent attribution
- CLI for starting/stopping the agent and querying UI endpoints
- Lightweight inference hooks and analytics modules

## Quick Start

Prerequisites: Python and pip. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the agent in the foreground (useful for debugging):

```bash
python main.py
```

Manage the agent via the CLI (background/process manager):

```bash
python cli.py agent start
python cli.py agent stop
python cli.py agent status
```

Query UI endpoints (JSON output):

```bash
python cli.py ui now --json
python cli.py ui dump --date YYYY-MM-DD --json
```

## Data & Storage

- Events are stored in an SQLite database at `agent/storage/events.db`.
- The agent appends completed session snapshots to `sessions.json` for downstream reporting.

## Important Files

- `cli.py` — Command-line interface for controlling the agent and UI endpoints
- `main.py` — Foreground agent runner (useful for development)
- `agent/process_manager.py` — Background start/stop/status utilities
- `agent/storage/db.py` — SQLite helpers and event logging
- `agent/session/sessionizer.py` — Canonical session model and `SessionManager`
- `agent/ui/contract.py` — Versioned UI JSON contract consumed by UIs

## Development

- Run tests:

```bash
pytest -q
```

- Useful development commands and notes are in `.github/copilot-instructions.md`.

## Contributing

Contributions welcome. Open issues or pull requests with focused changes. For major changes, please discuss first so we can preserve backwards-compatible CLI semantics (JSON `version` fields used by UI consumers).

## License
Copyright 2026 Arcady Oboukhov All Rights Reserved
---
