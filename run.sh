#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-3600}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Activate virtualenv if present
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES=1

exec uvicorn main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --log-level "$LOG_LEVEL" \
  --no-access-log # > /dev/null 2>&1 &