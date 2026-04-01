#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root as the directory where this script lives.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# If you place this script elsewhere, uncomment and adjust:
# REPO_ROOT="/home/algoritmi/Projects/HyperCEZ"

VENV_PY="$REPO_ROOT/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: Python not found at $VENV_PY"
  exit 1
fi

# Mimic PyCharm content-root behavior.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"

# Pass through all CLI args to run_cl.py
exec "$VENV_PY" "$REPO_ROOT/continual_world/run_cl.py" --tasks CW10 --seed 42 "$@"