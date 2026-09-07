#!/usr/bin/env bash
# Hercules side server: a small model on the RTX 3060 for Hermes delegation
# children and the compaction summarizer, so that traffic stops occupying
# 170HX slots at the 27B's ~30 ms/token. Same launch builder as the main
# server (tenselerate/backends/llamacpp.py), pinned with CUDA_VISIBLE_DEVICES.
#
# Usage: bash scripts/hercules_side_serve.sh SIDE_MODEL.gguf [extra tenselerate serve flags]
#   e.g. a Qwen3.5-9B Q4_K_M (~5.5 GiB + KV; ~40-55 tok/s single stream on the 3060)
# Env overrides: DEVICE (CUDA index of the 3060, 1) PORT (8081) ALIAS (side) NP (4)
#                CTX (pool tokens, 262144 - the builder requires one full window; a 9B hybrid model keeps ~2 GiB of KV at that size) KV (q8_0) REASONING (low) SAMPLING (greedy)
#                MTP (draft depth; default 1 on an -MTP- GGUF, else 0)
# Then point Hermes at it (HERCULES.md, "Two servers, two cards"):
#   delegation.base_url / auxiliary.compression.base_url: http://127.0.0.1:8081/v1, model: side
set -euo pipefail
MODEL="${1:?side model gguf path}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
args=(serve --backend llamacpp --model "$MODEL" --alias "${ALIAS:-side}"
      --device "${DEVICE:-1}" --port "${PORT:-8081}"
      --slots "${NP:-4}" --ctx-pool "${CTX:-262144}" --kv "${KV:-q8_0}"
      --reasoning "${REASONING:-low}" --sampling "${SAMPLING:-greedy}")
if [[ -n "${MTP:-}" ]]; then args+=(--mtp-draft "$MTP"); fi
exec python3 -m tenselerate "${args[@]}" "${@:2}"
