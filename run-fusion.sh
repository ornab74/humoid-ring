#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON="${PYTHON:-python3}"; fi
if [[ "${1:-}" == "--tui" ]]; then
  exec "$PYTHON" -m humoid_tui
fi
exec "$PYTHON" main.py
