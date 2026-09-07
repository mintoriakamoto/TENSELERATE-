#!/usr/bin/env bash
# Cooklabs: llama-server on 127.0.0.1:8080 as the Hercules model provider.
#
# Shaped by what was measured on the raven-9950x box (benches/cmp170hx-3060/):
#   * one operator = one sequential main loop + up to 3 parallel subagents
#     (Hermes default delegation.max_concurrent_children) -> 4 slots
#   * --kv-unified: one shared pool, so the main session can hold the full
#     262K window while subagents stay small; aggregate follows live tokens
#   * no MTP: 7-11% acceptance on the DavidAU merge, slower than plain
#   * reasoning_effort=low: fewer thinking tokens beats faster tokens on an
#     agent loop (Qwen3.8 defaults to xhigh from its chat template)
#   * prefill is 855 tok/s, so a cache MISS on a 60K agent context costs ~70 s;
#     prompt-cache hit rate is the latency lever - keep slots sticky and the
#     pool large enough that Hercules' 50% auto-compression fires rarely
#
# Usage: bash scripts/cooklabs_serve.sh MODEL.gguf
# Env overrides: NP (slots, 4) CTX (pool tokens, 524288) KV (q8_0) PORT (8080)
#                REASONING (low) NO_MMVQ (unset; set to 1 once the N=1/2/4
#                GGML_CUDA_NO_MMVQ runs confirm the tensor-core path wins)
set -euo pipefail
MODEL="${1:?model gguf path}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BIN="${LLAMA_SERVER:-$ROOT/build/bin/llama-server}"
if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN - cmake --build build --target llama-server first" >&2
  exit 1
fi
NP="${NP:-4}"; CTX="${CTX:-524288}"; KV="${KV:-q8_0}"; PORT="${PORT:-8080}"
REASONING="${REASONING:-low}"
if [[ -n "${NO_MMVQ:-}" ]]; then export GGML_CUDA_NO_MMVQ="$NO_MMVQ"; fi
exec "$BIN" -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  -ngl 999 --main-gpu 0 -fa on \
  -c "$CTX" -np "$NP" --kv-unified -cb \
  -ctk "$KV" -ctv "$KV" \
  -b 2048 -ub 512 --cache-reuse 256 \
  --chat-template-kwargs "{\"reasoning_effort\":\"$REASONING\"}" \
  "${@:2}"
