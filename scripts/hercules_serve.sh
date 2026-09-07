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
#                MTP (draft depth; default 1 on an -MTP- GGUF, deeper loses today)
#                MMVQ_MAX (0..8; 1 keeps batch-1 decode on dp4a, routes draft
#                verification to MMQ tensor cores - the fork's GGML_CUDA_MMVQ_MAX)
set -euo pipefail
MODEL="${1:?model gguf path}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
args=(serve --backend llamacpp --model "$MODEL" --alias "${ALIAS:-tenselerate}"
      --slots "${NP:-4}" --ctx-pool "${CTX:-524288}" --kv "${KV:-q8_0}"
      --port "${PORT:-8080}" --reasoning "${REASONING:-low}")
if [[ "${NO_MMVQ:-}" == "1" ]]; then args+=(--no-mmvq); fi
if [[ -n "${MTP:-}" ]]; then args+=(--mtp-draft "$MTP"); fi       # default: 1 on an -MTP- GGUF (+13..38%)
if [[ -n "${MMVQ_MAX:-}" ]]; then args+=(--mmvq-max "$MMVQ_MAX"); fi  # 1 = verification on MMQ, decode on MMVQ
exec python3 -m tenselerate "${args[@]}" "${@:2}"
