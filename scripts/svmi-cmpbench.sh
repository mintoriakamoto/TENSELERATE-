#!/usr/bin/env bash
# svmi-cmpbench - find where a CMP card stops being throttled.
#
# The CUDA backend has two INT8 matmul paths and the CMP firmware throttle only
# hits one of them:
#
#   batch <= 8   mul_mat_vec_q  (vecdotq.cuh)  -> __dp4a          THROTTLED ~16x
#   batch >  8   MMQ            (mmq.cu)       -> mma.sync s8     tensor-core IMMA
#
# ggml_cuda_should_use_mmq() returns true immediately on turing_mma_available(),
# and the 170HX is CC 8.0, so every matmul wider than MMVQ_MAX_BATCH_SIZE (8) is
# already on the tensor-core path. If the throttle is dp4a-specific, decode
# throughput per sequence stops degrading - or improves - as the number of
# parallel sequences crosses 8, and the answer for the card is concurrency and
# speculative decoding rather than kernel flags.
#
# Decode batching is what matters here, so this drives llama-batched-bench with
# -npl (parallel sequences); llama-bench's -b/-ub only size the prompt batch and
# its tg test is single-sequence, which cannot show this effect.
#
# Usage:
#   scripts/svmi-cmpbench.sh -m model-int8.gguf [-ngl 999] [extra args]
#   NPL="1 2 4 8 16 32" NTG=128 scripts/svmi-cmpbench.sh -m model-int8.gguf
#   BENCH_DP4A=../build-dp4a/bin/llama-batched-bench scripts/svmi-cmpbench.sh -m m.gguf
#
# Build the comparison binary (dp2a emulation) with:
#   cmake -B build-dp4a -DGGML_CUDA=ON -DGGML_CUDA_DISABLE_DP4A=ON && cmake --build build-dp4a -j

set -euo pipefail

BENCH=${LLAMA_BATCHED_BENCH:-./build/bin/llama-batched-bench}
NPL=${NPL:-"1 2 4 8 16 32"}
NPP=${NPP:-512}
NTG=${NTG:-128}
CTX=${CTX:-8192}

if [ $# -eq 0 ]; then
    grep '^#' "$0" | sed -n '2,28p' | sed 's/^# \{0,1\}//'
    exit 1
fi
[ -x "$BENCH" ] || {
    echo "error: $BENCH not found (set LLAMA_BATCHED_BENCH or build llama-batched-bench)" >&2
    exit 1
}

# S_TG t/s (aggregate decode throughput) for one parallel-sequence count.
# batched-bench prints: |PP|TG|B|N_KV|T_PP s|S_PP t/s|T_TG s|S_TG t/s|T s|S t/s|
stg_for() {
    local bin=$1 pl=$2; shift 2
    "$bin" "$@" -c "$CTX" -npp "$NPP" -ntg "$NTG" -npl "$pl" 2>/dev/null |
        awk -F'|' '$0 ~ /^\|[ 0-9]/ {gsub(/ /,"",$9); v=$9} END {print v+0}'
}

sweep() {
    local bin=$1 label=$2; shift 2
    printf '=== %s (%s) ===\n' "$label" "$bin"
    printf '%6s  %14s  %16s  %s\n' npl "S_TG tok/s" "per sequence" "note"
    local base_per="" per tot note
    for pl in $NPL; do
        tot=$(stg_for "$bin" "$pl" "$@")
        per=$(awk -v t="$tot" -v n="$pl" 'BEGIN {printf "%.2f", n ? t/n : 0}')
        [ -z "$base_per" ] && base_per=$per
        note=$(awk -v a="$base_per" -v b="$per" 'BEGIN {
            if (a > 0) printf "%.2fx per-seq vs npl=1", b/a
        }')
        printf '%6s  %14s  %16s  %s\n' "$pl" "$tot" "$per" "$note"
    done
    printf '\n'
}

sweep "$BENCH" "stock build" "$@"

if [ -n "${BENCH_DP4A:-}" ] && [ -x "${BENCH_DP4A}" ]; then
    sweep "$BENCH_DP4A" "GGML_CUDA_DISABLE_DP4A build" "$@"
fi

cat <<'EOF'
reading the table:
  per-sequence throughput falls ~linearly with npl
      -> ordinary bandwidth-bound behaviour, no path change. Aggregate S_TG is
         what improves; compare single-stream against the roofline
         (HBM GB/s / model GB = the batch-1 token ceiling).
  per-sequence throughput holds up, or S_TG jumps, crossing npl=8
      -> confirmed: batch-1 decode sits on the throttled dp4a path while batched
         decode does not. Invest in concurrency (llama-server -np N) and
         speculative decoding (--spec-type draft-dspark), which convert batch-1
         decode into batched verification on the tensor-core path.
  the DISABLE_DP4A build only helps at low npl
      -> expected: the dp2a emulation replaces __dp4a in vecdotq.cuh and
         fattn-common.cuh, both small-batch paths. It does nothing for MMQ.
EOF
