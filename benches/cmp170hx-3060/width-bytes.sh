#!/usr/bin/env bash
# width-bytes - does the decode path read the weights ONCE per step, or once
# per batch column?
#
# At batch 1 the card must read all 15.4 GB of weights to emit one token: that
# is the 10.3 ms / 95 tok/s physics floor in docs/physics.md. At batch N it
# must still read 15.4 GB - once - and multiply it against N activation
# columns. The extra columns are nearly free in bandwidth terms. That is the
# entire reason batching is the throughput lever, and it is why aggregate
# throughput for N concurrent Hercules agents should be close to N x the
# single-stream rate rather than flat.
#
# It is not flat on this box, and the two kernel paths fail differently:
#
#   MMVQ (dp4a)  18.5 + 11.5*(N-1) ms   - linear in N: re-reads weights per column
#   MMQ (tensor) ~55 ms, flat N=2..16   - amortizes columns, but the fixed cost
#                                         is 3.2x the 17.3 ms an at-achieved-
#                                         bandwidth read of 15.4 GB would take
#
# Both numbers come from benches/cmp170hx-3060/README.md. This script measures
# the thing that tells you WHY, and therefore whether a read-once N-column
# kernel is worth writing (docs/kernel-readonce.md):
#
#   Part A (always runs, no profiler): a width sweep with llama-batched-bench.
#     Time per decode step vs N. Slope tells re-read from amortized.
#   Part B (needs ncu and permission to profile): DRAM bytes actually read per
#     decode step at one width, compared against the model's weight bytes.
#     This is the measurement that distinguishes "reads weights 3x" from
#     "reads weights once and is latency-bound".
#
# PREDICTIONS, written before the run (grade them after, per AGENTS.md rule 2):
#   A1. MMVQ time is linear in N with slope ~11.5 ms/column.
#   A2. MMQ time is flat in N from 2 to 16 at ~55 ms.
#   B1. MMVQ at N=8 reads ~8 x weight_bytes (re-read confirmed).
#   B2. MMQ at N=8 reads either ~3 x weight_bytes (excess traffic - a
#       read-once kernel wins ~3x) or ~1 x (traffic is already minimal, the
#       55 ms is latency/occupancy - a read-once kernel wins nothing and the
#       idea is dead).
#   B2 is the one that decides whether to write the kernel. Either answer is
#   worth having; the second one saves days.
#
# Usage:
#   MODEL=/models/Qwen3.8-27B-TURBO-MTP-Q4_K_M.gguf bash benches/cmp170hx-3060/width-bytes.sh
#   WIDTHS=1,2,4,8,16 MODEL=... bash .../width-bytes.sh
#   SKIP_NCU=1 MODEL=... bash .../width-bytes.sh      # Part A only
#   bash benches/cmp170hx-3060/width-bytes.sh --self-test   # no GPU, no model; exit 0
#
# Env: MODEL (required) BATCHED_BENCH (binary; default beside LLAMA_SERVER or
#      build/bin) NCU (ncu binary) WIDTHS (1,2,4,8,16) NCU_WIDTH (8)
#      PP (512) TG (32) NGL (99) RESULTS (width-bytes-<date>.md) SKIP_NCU DRY

set -euo pipefail

BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BENCH_DIR/../.." && pwd)

WIDTHS="${WIDTHS:-1,2,4,8,16}"
NCU_WIDTH="${NCU_WIDTH:-8}"
PP="${PP:-512}"
TG="${TG:-32}"
NGL="${NGL:-99}"
SKIP_NCU="${SKIP_NCU:-}"
DRY="${DRY:-}"
RESULTS="${RESULTS:-$BENCH_DIR/width-bytes-$(date +%F).md}"

# ggml's quantized matmul kernels. MMVQ is mul_mat_vec_q, MMQ is mul_mat_q;
# both names have been stable across the syncs this fork has done.
KERNEL_RE="regex:mul_mat_vec_q|mul_mat_q"

die() { echo "width-bytes: $*" >&2; exit 1; }

find_bin() {
    local name=$1 c
    for c in "${BATCHED_BENCH:-}" \
             "$(dirname "${LLAMA_SERVER:-/nonexistent}")/$name" \
             "$ROOT/build/bin/$name" \
             "$(command -v "$name" 2>/dev/null || true)"; do
        [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}

# Weight bytes actually resident, from the GGUF on disk. The -MTP- file carries
# the draft head too (16.5 GB vs 15.4 GB of base weights), so the ratio below is
# reported against the file size and the 15.4 GB base is noted, not assumed.
model_bytes() {
    local m=$1
    if [ -f "$m" ]; then stat -c %s "$m" 2>/dev/null || stat -f %z "$m"; else echo 0; fi
}

# Parse `ncu --csv` output: sum dram__bytes_read.sum over every captured launch.
# ncu prints one row per kernel per metric, and the value column is
# locale-formatted with thousands separators, so it is stripped before float().
sum_dram_bytes() {
    python3 - "$1" <<'PY'
import csv, sys
total, launches = 0.0, 0
with open(sys.argv[1], newline="") as fh:
    rows = list(csv.reader(fh))
# ncu emits a preamble; the header row is the first one containing "Metric Name"
hdr = next((i for i, r in enumerate(rows) if "Metric Name" in r), None)
if hdr is None:
    print("0 0"); sys.exit(0)
cols = {name: i for i, name in enumerate(rows[hdr])}
mi, vi = cols.get("Metric Name"), cols.get("Metric Value")
for r in rows[hdr + 1:]:
    if mi is None or vi is None or len(r) <= max(mi, vi):
        continue
    if r[mi].strip() == "dram__bytes_read.sum":
        try:
            total += float(r[vi].replace(",", "").replace('"', ""))
        except ValueError:
            continue
        launches += 1
print(f"{total:.0f} {launches}")
PY
}

self_test() {
    local tmp; tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' RETURN

    # a synthetic ncu CSV in the shape ncu actually emits, with a preamble,
    # thousands separators, and an unrelated metric that must be ignored
    cat > "$tmp/ncu.csv" <<'CSV'
==PROF== Connected to process 1234
"ID","Kernel Name","Metric Name","Metric Value"
"0","mul_mat_vec_q","dram__bytes_read.sum","1,000,000"
"0","mul_mat_vec_q","sm__cycles_elapsed.avg","999,999"
"1","mul_mat_q","dram__bytes_read.sum","2,500,000"
CSV
    read -r bytes launches <<<"$(sum_dram_bytes "$tmp/ncu.csv")"
    [ "$bytes" = "3500000" ] || die "self-test: dram sum $bytes != 3500000"
    [ "$launches" = "2" ] || die "self-test: launch count $launches != 2"

    # a CSV with no header must be survivable, not a crash
    echo "garbage" > "$tmp/empty.csv"
    read -r bytes launches <<<"$(sum_dram_bytes "$tmp/empty.csv")"
    [ "$bytes" = "0" ] || die "self-test: headerless sum $bytes != 0"

    # ratio arithmetic: 8 x a 16 GB model at N=8 is a per-column re-read
    local ratio
    ratio=$(python3 -c "print(f'{(8*16.5e9)/16.5e9:.2f}')")
    [ "$ratio" = "8.00" ] || die "self-test: ratio $ratio != 8.00"

    # the width list must parse to distinct positive integers
    python3 - <<PY || die "self-test: WIDTHS parse"
w = [int(x) for x in "$WIDTHS".split(",")]
assert w == sorted(set(w)) and all(x > 0 for x in w), w
PY
    echo "width-bytes: self-test OK (csv parse, ratio arithmetic, width list)"
}

[ "${1:-}" = "--self-test" ] && { self_test; exit 0; }

MODEL="${MODEL:-}"
[ -n "$MODEL" ] || die "MODEL is required (path to the GGUF). --self-test needs neither."
[ -f "$MODEL" ] || die "MODEL not found: $MODEL"

BB=$(find_bin llama-batched-bench) || die "llama-batched-bench not found; set BATCHED_BENCH"
MBYTES=$(model_bytes "$MODEL")

{
    echo "# width-bytes - $(date -u +%FT%TZ)"
    echo
    echo "Model: \`$(basename "$MODEL")\` ($(python3 -c "print(f'{$MBYTES/1e9:.1f}')") GB on disk)"
    echo "Binary: \`$BB\`"
    echo
    echo "Question: does a decode step read the weights once, or once per batch column?"
    echo "Physics: 15.4 GB of base weights at the achieved 890 GB/s is 17.3 ms per step,"
    echo "at any width, if the read is amortized. See docs/physics.md and docs/kernel-readonce.md."
    echo
} | tee "$RESULTS"

run_sweep() {
    local label=$1; shift
    echo "## Part A - width sweep ($label)" | tee -a "$RESULTS"
    echo | tee -a "$RESULTS"
    if [ -n "$DRY" ]; then
        echo "DRY: $* $BB -m $MODEL -npp $PP -ntg $TG -npl $WIDTHS -ngl $NGL" | tee -a "$RESULTS"
        return 0
    fi
    # batched-bench prints its own table; keep it verbatim, it is the evidence
    env "$@" "$BB" -m "$MODEL" -npp "$PP" -ntg "$TG" -npl "$WIDTHS" -ngl "$NGL" \
        2>&1 | tee -a "$RESULTS" || die "batched-bench failed ($label)"
    echo | tee -a "$RESULTS"
}

# Default routing first (whatever the fork's MMVQ_MAX picks), then each path
# forced, so the slope of each is separable rather than a blend of the two.
run_sweep "default routing"
run_sweep "MMVQ forced (dp4a)"   GGML_CUDA_MMVQ_MAX=99
run_sweep "MMQ forced (no mmvq)" GGML_CUDA_NO_MMVQ=1

if [ -n "$SKIP_NCU" ]; then
    echo "Part B skipped (SKIP_NCU set)." | tee -a "$RESULTS"
    exit 0
fi

NCU="${NCU:-$(command -v ncu 2>/dev/null || true)}"
if [ -z "$NCU" ]; then
    {
        echo "## Part B - DRAM bytes per step: SKIPPED"
        echo
        echo "\`ncu\` not found. Part A's slope still separates re-read (linear in N)"
        echo "from amortized (flat), which is most of the answer; Part B is what"
        echo "distinguishes MMQ's 55 ms being excess traffic from it being latency."
        echo "Install the CUDA profiler, or run with SKIP_NCU=1 to silence this."
    } | tee -a "$RESULTS"
    exit 0
fi

echo "## Part B - DRAM bytes per decode step at N=$NCU_WIDTH" | tee -a "$RESULTS"
echo | tee -a "$RESULTS"

ncu_one() {
    local label=$1 csv; shift
    csv=$(mktemp)
    if [ -n "$DRY" ]; then
        echo "DRY: $NCU --csv --metrics dram__bytes_read.sum --kernel-name $KERNEL_RE ..." \
            | tee -a "$RESULTS"
        rm -f "$csv"; return 0
    fi
    # -c bounds the capture: profiling every launch of a 64-layer model would
    # take hours because ncu serialises and replays kernels.
    if ! env "$@" "$NCU" --csv --metrics dram__bytes_read.sum \
            --kernel-name "$KERNEL_RE" --launch-count 400 --target-processes all \
            "$BB" -m "$MODEL" -npp "$PP" -ntg 4 -npl "$NCU_WIDTH" -ngl "$NGL" \
            > "$csv" 2>/dev/null; then
        echo "- $label: ncu failed (profiling may need CAP_SYS_ADMIN or" \
             "\`NVreg_RestrictProfilingToAdminUsers=0\`)" | tee -a "$RESULTS"
        rm -f "$csv"; return 0
    fi
    read -r bytes launches <<<"$(sum_dram_bytes "$csv")"
    rm -f "$csv"
    python3 - "$label" "$bytes" "$launches" "$MBYTES" <<'PY' | tee -a "$RESULTS"
import sys
label, total, launches, model = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
if launches == 0:
    print(f"- {label}: no matmul launches captured (kernel filter missed; check ggml kernel names)")
    raise SystemExit
# 64 layers x ~7 matmuls each is the per-step launch count; the capture is
# bounded, so normalise by however many launches were actually seen rather than
# assuming a whole step was captured.
print(f"- {label}: {total/1e9:.1f} GB over {launches} matmul launches "
      f"({total/launches/1e6:.1f} MB per launch); model file is {model/1e9:.1f} GB")
PY
}

ncu_one "MMVQ forced (dp4a)"   GGML_CUDA_MMVQ_MAX=99
ncu_one "MMQ forced (no mmvq)" GGML_CUDA_NO_MMVQ=1

{
    echo
    echo "### How to read Part B"
    echo
    echo "Sum the per-launch bytes over one full decode step (64 layers) and compare"
    echo "against 15.4 GB of base weights:"
    echo
    echo "- **~1x** - the weight read is already minimal. MMQ's 55 ms is latency or"
    echo "  occupancy, not traffic, and a read-once kernel wins nothing. Close"
    echo "  docs/kernel-readonce.md as refuted and chase occupancy instead."
    echo "- **~3x** - MMQ moves three times the necessary bytes. A read-once"
    echo "  N-column kernel is worth writing; the target is ~17-20 ms at N=8."
    echo "- **~Nx** - the path re-reads the weights per column, as MMVQ's 11.5 ms"
    echo "  slope already implies. Expected for MMVQ; damning if MMQ does it too."
    echo
    echo "Append the verdict to benches/cmp170hx-3060/README.md as a measured row,"
    echo "and grade predictions A1, A2, B1 and B2 from the header of this script."
} | tee -a "$RESULTS"

echo "width-bytes: wrote $RESULTS"
