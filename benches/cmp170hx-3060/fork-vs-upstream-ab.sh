#!/usr/bin/env bash
# Settle "is the fork's GDN/MTP path slower than upstream?" in one command.
#
# Builds (or reuses) an upstream llama.cpp binary and runs the identical
# measurement against it and against the fork's release binary: single-stream
# decode, 8-slot aggregate, and MTP draft acceptance from the server log.
#
# Usage:
#   MODEL=/path/model.gguf bash benches/cmp170hx-3060/fork-vs-upstream-ab.sh
#
# Env:
#   MODEL        (required) the GGUF both sides serve
#   FORK_BIN     fork llama-server      (default: build/bin/llama-server)
#   UP_BIN       upstream llama-server  (default: build it into ../llamacpp-upstream)
#   UP_REF       upstream ref to build  (default: master)
#   NP           slots for the aggregate run (default 8)
#   CTX          pool tokens            (default 262144)
#   KV           cache type             (default q8_0)
#   MTP          draft depth            (default 1)
#   NGEN         tokens per request     (default 128)
set -euo pipefail

MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FORK_BIN="${FORK_BIN:-$ROOT/build/bin/llama-server}"
UP_DIR="${UP_DIR:-$ROOT/../llamacpp-upstream}"
UP_BIN="${UP_BIN:-$UP_DIR/build/bin/llama-server}"
UP_REF="${UP_REF:-master}"
NP="${NP:-8}"; CTX="${CTX:-262144}"; KV="${KV:-q8_0}"; MTP="${MTP:-1}"; NGEN="${NGEN:-128}"
PORT="${PORT:-8099}"
OUT="${OUT:-$ROOT/benches/cmp170hx-3060/ab-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"

if [[ ! -x "$UP_BIN" ]]; then
    echo "== building upstream ($UP_REF) into $UP_DIR"
    [[ -d "$UP_DIR/.git" ]] || git clone -q https://github.com/ggml-org/llama.cpp.git "$UP_DIR"
    git -C "$UP_DIR" fetch -q origin "$UP_REF" && git -C "$UP_DIR" checkout -q FETCH_HEAD
    cmake -S "$UP_DIR" -B "$UP_DIR/build" -DGGML_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES="80;86" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$UP_DIR/build" -j --target llama-server >/dev/null
fi

# one measurement pass against a running server
measure() {  # $1 = tag, $2 = n_parallel
    local tag="$1" np="$2" t0 t1 total=0
    local body='{"model":"m","prompt":"Write a detailed technical explanation of memory bandwidth in GPUs.","n_predict":'"$NGEN"',"temperature":0,"repeat_penalty":1.0,"stream":false}'
    t0=$(date +%s.%N)
    for ((i=0;i<np;i++)); do
        curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
             -d "$body" -o "$OUT/$tag-$i.json" &
    done
    wait
    t1=$(date +%s.%N)
    local wall; wall=$(echo "$t1 - $t0" | bc)
    # agg-report.py prints the aggregate AND refuses to report one that exceeds
    # the sum of per-request rates or that came from requests which never
    # overlapped - the two ways a concurrency number goes wrong.
    python3 "$ROOT/benches/cmp170hx-3060/agg-report.py" --label "$tag np=$np" \
        --wall "$wall" "$OUT/$tag"-*.json 2>&1 | tee -a "$OUT/summary.txt" || true
}

run_side() {  # $1 = tag, $2 = binary, $3 = extra args
    local tag="$1" bin="$2"; shift 2
    echo "== $tag: $bin" | tee -a "$OUT/summary.txt"
    "$bin" -m "$MODEL" --host 127.0.0.1 --port "$PORT" -ngl 999 -fa on \
           -c "$CTX" -np "$NP" --kv-unified -cb -ctk "$KV" -ctv "$KV" "$@" \
           > "$OUT/$tag-server.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
        sleep 1
    done
    measure "$tag-single" 1
    measure "$tag-np$NP"  "$NP"
    # draft acceptance straight from the server log
    grep -o "draft acceptance rate = [0-9.]*" "$OUT/$tag-server.log" | tail -3 | tee -a "$OUT/summary.txt" || true
    kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
    sleep 3
}

: > "$OUT/summary.txt"
SPEC=(); [[ "$MTP" -gt 0 ]] && SPEC=(--spec-type draft-mtp --spec-draft-n-max "$MTP")
run_side upstream "$UP_BIN" "${SPEC[@]}" --temp 0 --repeat-penalty 1.0
run_side fork     "$FORK_BIN" "${SPEC[@]}" --temp 0 --repeat-penalty 1.0

echo; echo "== RESULT ($OUT/summary.txt)"; cat "$OUT/summary.txt"
echo
echo "Read: if fork and upstream agree within noise, the fork's GDN/MTP path is not"
echo "the regression. If upstream is materially faster, capture both server logs and"
echo "open an issue with this summary - that is a real regression to bisect."
