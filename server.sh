#!/bin/bash
lsof -ti :3600 | xargs -r kill -9

# Activate virtualenv if present
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES=1
uvicorn main:app \
  --port 3600 \
  --host 0.0.0.0 \
  --workers 1 > /dev/null 2>&1 &
