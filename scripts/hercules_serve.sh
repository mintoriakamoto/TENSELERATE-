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
#                NGRAM (n-gram draft depth 1..15, default off; replays a run already
#                seen in this context - free tokens on re-emitted code, a hash lookup
#                on a miss. Tried before the MTP head, so a miss costs nothing)
#                NGRAM_MIN (shortest replay worth drafting; default min(4, NGRAM))
#                MTP_MODEL (retrained head GGUF from scripts/mtp-head-train.py, served with -md at depth 3)
#                CACHE_RAM (host-RAM prompt cache MiB, 16384) SLOT_SIMILARITY (LCP fraction, 0.1)
#                SLOT_SAVE_PATH (dir for /slots save|restore; scripts/hercules_slots.sh)
#                MMVQ_MAX (0..8; 1 keeps batch-1 decode on dp4a, routes draft
#                verification to MMQ tensor cores - the fork's GGML_CUDA_MMVQ_MAX)
#                ATTN_WINDOW (tokens; bounds the attention layers, e.g. 65536 -> NP=9 CTX=589824)
#                ATTN_SINKS (pinned leading positions, default 4; 36000 pins the Hermes system prompt)
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
if [[ -n "${NGRAM:-}" ]]; then args+=(--ngram-draft "$NGRAM"); fi        # n-gram replay drafter, tried before the MTP head
if [[ -n "${NGRAM_MIN:-}" ]]; then args+=(--ngram-min "$NGRAM_MIN"); fi  # default min(4, NGRAM); llama.cpp's own 48 would draft nothing
if [[ -n "${CACHE_RAM:-}" ]]; then args+=(--cache-ram "$CACHE_RAM"); fi          # host-RAM prompt cache MiB (16384)
if [[ -n "${SLOT_SIMILARITY:-}" ]]; then args+=(--slot-similarity "$SLOT_SIMILARITY"); fi
if [[ -n "${SLOT_SAVE_PATH:-}" ]]; then mkdir -p "$SLOT_SAVE_PATH"; args+=(--slot-save-path "$SLOT_SAVE_PATH"); fi  # /slots save|restore
if [[ -n "${MMVQ_MAX:-}" ]]; then args+=(--mmvq-max "$MMVQ_MAX"); fi  # 1 = verification on MMQ, decode on MMVQ
if [[ -n "${SAMPLING:-}" ]]; then args+=(--sampling "$SAMPLING"); fi   # default greedy (draft acceptance is exact-match)
if [[ -n "${ATTN_WINDOW:-}" ]]; then args+=(--attn-window "$ATTN_WINDOW"); fi  # bounded attention window: many slots, unbounded sequences
if [[ -n "${ATTN_SINKS:-}" ]]; then args+=(--attn-sinks "$ATTN_SINKS"); fi      # pinned leading positions (default 4; = system prompt length to pin it)
exec python3 -m tenselerate "${args[@]}" "${@:2}"
