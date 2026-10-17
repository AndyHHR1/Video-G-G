#!/usr/bin/env bash
# Arranca el panel con el venv del proyecto. Acepta las mismas opciones que
# `python -m backend` (por ejemplo: scripts/run.sh --host 0.0.0.0 --no-sr).
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
export PYTHONNOUSERSITE=1          # no mezclar paquetes de ~/.local
exec "$VENV/bin/python" -m backend "$@"
