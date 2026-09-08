#!/usr/bin/env bash
# Save or restore a llama-server slot's KV cache to disk, so the prefilled
# Hercules system prompt (~35K tokens, ~40 s of prefill) survives a server
# restart. Needs the server started with --slot-save-path DIR (the launch's
# --slot-save-path / SLOT_SAVE_PATH); files land in that directory.
#
# Usage: bash scripts/hercules_slots.sh save    [SLOT] [NAME]   # after the first turn has run
#        bash scripts/hercules_slots.sh restore [SLOT] [NAME]   # right after the server is healthy
#        bash scripts/hercules_slots.sh erase   [SLOT]
# Env: PORT (8080) HOST (127.0.0.1). NAME defaults to hercules-system.bin.
set -euo pipefail
ACTION="${1:?save|restore|erase}"
SLOT="${2:-0}"
NAME="${3:-hercules-system.bin}"
URL="http://${HOST:-127.0.0.1}:${PORT:-8080}/slots/${SLOT}?action=${ACTION}"
case "$ACTION" in
  save|restore)
    curl -sS -X POST "$URL" -H "Content-Type: application/json" -d "{\"filename\":\"${NAME}\"}" ;;
  erase)
    curl -sS -X POST "$URL" ;;
  *) echo "unknown action: $ACTION" >&2; exit 2 ;;
esac
echo
