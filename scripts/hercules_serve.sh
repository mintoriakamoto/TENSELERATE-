#!/usr/bin/env bash
# Hercules: llama-server on 127.0.0.1:8080 as the Hercules model provider.
#
# Thin wrapper over `tenselerate serve --backend llamacpp`, which builds the
# launch from tenselerate/backends/llamacpp.py - the configuration the box
# measured (benches/cmp170hx-3060/): 4 slots on a unified KV pool, MTP depth 1 opt-in,
# reasoning_effort=low, chunked prefill with cache reuse.
#
# Usage: bash scripts/hercules_serve.sh MODEL.gguf [extra tenselerate serve flags]
# Env overrides: NP (slots, 4) CTX (pool tokens, 524288) KV (q8_0) PORT (8080) ALIAS (tenselerate)
#                REASONING (low) NO_MMVQ (1 = force the tensor-core MMQ path)
#                MTP (draft depth on an -MTP- GGUF; 1 = measured +35%, deeper loses)
set -euo pipefail
MODEL="${1:?model gguf path}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
args=(serve --backend llamacpp --model "$MODEL" --alias "${ALIAS:-tenselerate}"
      --slots "${NP:-4}" --ctx-pool "${CTX:-524288}" --kv "${KV:-q8_0}"
      --port "${PORT:-8080}" --reasoning "${REASONING:-low}")
if [[ "${NO_MMVQ:-}" == "1" ]]; then args+=(--no-mmvq); fi
if [[ -n "${MTP:-}" ]]; then args+=(--mtp-draft "$MTP"); fi   # MTP=1 on an -MTP- GGUF: measured +35%
exec python3 -m tenselerate "${args[@]}" "${@:2}"
