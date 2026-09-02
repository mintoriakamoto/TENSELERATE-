#!/usr/bin/env bash
# Cooklabs: plan + llama-server on 127.0.0.1:8080 for Hercules provider tenselerate.
# Usage: bash scripts/cooklabs_serve.sh MODEL.gguf [3060|2080ti|1660ti]
set -euo pipefail
MODEL="${1:?model gguf path}"
GPU="${2:-3060}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f scripts/svmi-plan.py ]]; then
  python3 scripts/svmi-plan.py "$MODEL" --gpu "$GPU" || true
fi
BIN="$ROOT/build/bin/llama-server"
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN — cmake --build build --target llama-server first" >&2
  exit 1
fi
exec "$BIN" -m "$MODEL" --host 127.0.0.1 --port 8080
