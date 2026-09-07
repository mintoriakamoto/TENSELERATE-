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
#                MTP_MODEL (retrained head GGUF from scripts/mtp-head-train.py, served with -md at depth 3)
#                MMVQ_MAX (0..8; 1 keeps batch-1 decode on dp4a, routes draft
#                verification to MMQ tensor cores - the fork's GGML_CUDA_MMVQ_MAX)
#                SAMPLING (greedy, default: server-default temp 0 / repeat-penalty 1.0,
#                the only sampling under which the MTP draft pays - 46.2 vs 29.9 tok/s,
#                but the merge loops in <think>; dry = greedy + DRY loop guard;
#                low = temp 0.3 min-p 0.1; client = leave llama.cpp's defaults)
set -euo pipefail
MODEL="${1:?model gguf path}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
args=(serve --backend llamacpp --model "$MODEL" --alias "${ALIAS:-tenselerate}"
      --slots "${NP:-4}" --ctx-pool "${CTX:-524288}" --kv "${KV:-q8_0}"
      --port "${PORT:-8080}" --reasoning "${REASONING:-low}")
if [[ "${NO_MMVQ:-}" == "1" ]]; then args+=(--no-mmvq); fi
if [[ -n "${MTP:-}" ]]; then args+=(--mtp-draft "$MTP"); fi       # default: 1 on an -MTP- GGUF (+13..38%)
if [[ -n "${MTP_MODEL:-}" ]]; then args+=(--mtp-model "$MTP_MODEL"); fi  # retrained sidecar head (-md), depth 3 default
if [[ -n "${MMVQ_MAX:-}" ]]; then args+=(--mmvq-max "$MMVQ_MAX"); fi  # 1 = verification on MMQ, decode on MMVQ
if [[ -n "${SAMPLING:-}" ]]; then args+=(--sampling "$SAMPLING"); fi   # default greedy (draft acceptance is exact-match)
exec python3 -m tenselerate "${args[@]}" "${@:2}"
