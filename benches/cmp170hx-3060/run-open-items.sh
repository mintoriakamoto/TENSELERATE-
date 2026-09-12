#!/usr/bin/env bash
# run-open-items - run the still-unmeasured configurations from
# benches/cmp170hx-3060/README.md ("Still to measure") unattended, one
# llama-server per configuration, and append rows to results-<date>.md.
#
#   MODEL=/models/Qwen3.8-27B-TURBO-MTP-Q4_K_M.gguf \
#     bash benches/cmp170hx-3060/run-open-items.sh              # everything, in order
#   EXPERIMENTS=loop_guard,deep_kv MODEL=... bash .../run-open-items.sh   # a subset
#   DRY=1 MODEL=/x/Foo-MTP.gguf bash .../run-open-items.sh     # print what would run (uses --dry-run)
#   bash benches/cmp170hx-3060/run-open-items.sh --self-test    # no GPU, no server; exit 0
#
# Experiments (functions exp_<name>, run in this order):
#   loop_guard       --sampling greedy | dry | low, plus low with client temp 0.15 / 0.5:
#                    tok/s, draft acceptance, <think> loop check per shape. Picks a winner.
#   client_override  greedy server, client sends temp 0.7 + repeat_penalty 1.15
#   deep_kv          1 slot x 262144, kv q8_0 vs f16, ~250K-token prefill, decode at depth
#   slot4_width      4 slots x 524288, MTP depth 1, default routing vs --mmvq-max 3 vs --no-mmvq
#   mtp_control      MTP off vs on under the winning sampling
#   spec_depth       where draft width can come from at temp 0: a deeper MTP draft
#                    (predicted to lose) vs the ngram-mod replay drafter at depth
#                    8 and 15, on a rewrite workload and on a code control
#   graph_thrash     are CUDA graphs alive under speculation? ggml keys its graph cache
#                    on a pointer, and a width change resets the warmup to direct execution
#   backend_sampling -bs (sample on the GPU) vs the host copy: a 248,320-float logit
#                    row over PCIe Gen2 x4, twice per verify step at MTP depth 1
#   tree_gate        is the MTP head shallow, or is a linear draft betting on one branch?
#                    Logs conditional per-position acceptance; decides the tree drafter.
#   kv_codec_gate    whether a heavier KV codec is affordable at all: q4_0 vs q8_0
#                    at depth, with the packed vector kernel off and on. Decides
#                    one thing before any codec work starts - see docs/kernel-work.md
#                    item 4. Refutation is pre-registered in the function.
#
# Env: MODEL (required) LLAMA_SERVER (binary; default tenselerate's) PORT (8089)
#      LLAMA_BENCH (llama-bench binary; default next to LLAMA_SERVER or build/bin)
#      LLAMA_BENCH_OLD (an older llama-bench for the regression A/B: take the
#                       oldest release still on the Releases page - do not name a
#                       tag here, the release workflow prunes old ones)
#      MODEL_ALT (a second GGUF of the same model, e.g. Q4_0 or IQ4_XS, for the quant A/B)
#      N (requests per shape, 3) MAX_TOKENS (400) HEALTH_TIMEOUT (600 s)
#      VRAM_FREE_MB (GPU 0 must be below this before a launch, 2000) VRAM_TIMEOUT (180 s)
#      GPU (0) WIN_SAMPLING (mode[:temp], overrides loop_guard's pick for mtp_control)
#      DEEP_PREFILL (250000) RESULTS (results-<date>.md) LOGS (logs/) EXPERIMENTS DRY
#
# Every server launch goes through `python3 -m tenselerate serve --backend llamacpp`
# on 127.0.0.1:PORT (a production server on 8080 is untouched). Server logs are
# kept next to the experiment logs; `grep "draft acceptance" logs/*server*.log`.

set -euo pipefail

BENCH_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$BENCH_DIR/../.." && pwd)
MEASURE="$BENCH_DIR/measure.py"
HOST=127.0.0.1
PORT="${PORT:-8089}"
BASE_URL="http://$HOST:$PORT"
N="${N:-3}"
MAX_TOKENS="${MAX_TOKENS:-400}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
VRAM_FREE_MB="${VRAM_FREE_MB:-2000}"
VRAM_TIMEOUT="${VRAM_TIMEOUT:-180}"
GPU="${GPU:-0}"
DEEP_PREFILL="${DEEP_PREFILL:-250000}"
DRY="${DRY:-}"
LOGS="${LOGS:-$BENCH_DIR/logs}"
RESULTS="${RESULTS:-$BENCH_DIR/results-$(date +%F).md}"
ALL_EXPERIMENTS="bench_ab prefill_ubatch quant_ab loop_guard client_override deep_kv slot4_width mtp_control spec_depth kv_codec_gate tree_gate backend_sampling graph_thrash"
EXPERIMENTS="${EXPERIMENTS:-$ALL_EXPERIMENTS}"
SUMMARY=""          # "name: status" lines, printed at the end
SERVER_PID=""

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
ts() { date +%Y%m%d-%H%M%S; }

# ---------------------------------------------------------------- results

results_header() {
    [ -s "$RESULTS" ] && return 0
    mkdir -p "$(dirname "$RESULTS")"
    printf '# CMP 170HX + RTX 3060 - open items, %s\n\n' "$(date +%F)" > "$RESULTS"
    printf 'Rows appended by `run-open-items.sh`; fold the keepers into README.md.\n\n' >> "$RESULTS"
    printf '| date | quantity | value | how | notes |\n| --- | --- | --- | --- | --- |\n' >> "$RESULTS"
}

# add_row QUANTITY VALUE HOW NOTES   (pipes in the text are escaped)
add_row() {
    results_header
    local q v h n
    q=${1//|/\\|}; v=${2//|/\\|}; h=${3//|/\\|}; n=${4//|/\\|}
    printf '| %s | %s | %s | %s | %s |\n' "$(date +%F)" "$q" "$v" "$h" "$n" >> "$RESULTS"
    log "row: $q -> $v"
}

# ---------------------------------------------------------------- GPU

vram_used_mb() {
    have nvidia-smi || { echo 0; return; }
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null |
        head -n 1 | tr -dc '0-9'
}

# block until GPU 0 is below VRAM_FREE_MB (a stale server OOMed the MTP launch once)
wait_vram_free() {
    [ -n "$DRY" ] && { log "(dry) wait for VRAM on GPU $GPU < ${VRAM_FREE_MB} MiB"; return 0; }
    have nvidia-smi || { log "nvidia-smi not found; skipping the VRAM check"; return 0; }
    local deadline=$((SECONDS + VRAM_TIMEOUT)) used
    while :; do
        used=$(vram_used_mb); used=${used:-0}
        if [ "$used" -lt "$VRAM_FREE_MB" ]; then
            log "GPU $GPU: ${used} MiB used, clear"
            return 0
        fi
        [ "$SECONDS" -lt "$deadline" ] || { log "GPU $GPU still holds ${used} MiB after ${VRAM_TIMEOUT}s"; return 1; }
        sleep 3
    done
}

# ---------------------------------------------------------------- server

# serve_args EXTRA... -> the tenselerate serve argv (no launch)
serve_args() {
    printf '%s\n' python3 -m tenselerate serve --backend llamacpp --model "$MODEL" \
        --host "$HOST" --port "$PORT"
    [ -n "${LLAMA_SERVER:-}" ] && printf '%s\n' --llama-server "$LLAMA_SERVER"
    printf '%s\n' "$@"
}

# start_server SERVER_LOG EXTRA...   ; leaves SERVER_PID set (the process group)
start_server() {
    local slog=$1; shift
    local -a cmd
    mapfile -t cmd < <(serve_args "$@")
    if [ -n "$DRY" ]; then
        log "(dry) would launch: ${cmd[*]}"
        # validate the configuration through the backend's own checks
        (cd "$ROOT" && "${cmd[@]}" --dry-run) >> "$slog" 2>&1 || die "dry-run rejected: ${cmd[*]}"
        return 0
    fi
    wait_vram_free || return 1
    log "launching: ${cmd[*]}"
    log "server log: $slog"
    # setsid: llama-server is a child of the python launcher; killing the group gets both
    (cd "$ROOT" && exec setsid "${cmd[@]}") > "$slog" 2>&1 &
    SERVER_PID=$!
    printf '%s\n' "$SERVER_PID" > "$LOGS/.server.pid"
    wait_health
}

wait_health() {
    local deadline=$((SECONDS + HEALTH_TIMEOUT))
    log "waiting for $BASE_URL/health (up to ${HEALTH_TIMEOUT}s; the load is slow over PCIe Gen2 x4)"
    while :; do
        if python3 "$MEASURE" --base-url "$BASE_URL" --health; then
            log "server healthy after $((SECONDS))s"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "server process exited before becoming healthy; last lines:"
            tail -n 15 "$LOGS/.current-server.log" 2>/dev/null || true
            return 1
        fi
        [ "$SECONDS" -lt "$deadline" ] || { log "health timeout"; return 1; }
        sleep 5
    done
}

stop_server() {
    if [ -n "$DRY" ]; then log "(dry) would stop the server"; return 0; fi
    local pid=${SERVER_PID:-$(cat "$LOGS/.server.pid" 2>/dev/null || true)}
    if [ -n "$pid" ]; then
        log "stopping server group $pid"
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        local i
        for i in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    # anything else still bound to our port (never the production 8080 server)
    pkill -f -- "llama-server .*--port $PORT( |$)" 2>/dev/null || true
    SERVER_PID=""
    rm -f "$LOGS/.server.pid"
    wait_vram_free || log "warning: VRAM did not come back; the next launch may OOM"
}

# last "draft acceptance = X" the server logged (empty when no drafter ran)
grep_acceptance() {
    [ -f "$1" ] || return 0
    sed -n 's/.*draft acceptance = \([0-9.]*\).*/\1/p' "$1" | tail -n 1
}

# ---------------------------------------------------------------- measure

MEASURE_VALUE=""
MEASURE_KV=""
# run_measure LOG ARGS... ; sets MEASURE_VALUE / MEASURE_KV from measure.py's trailer lines
run_measure() {
    local mlog=$1; shift
    MEASURE_VALUE=""; MEASURE_KV=""
    if [ -n "$DRY" ]; then
        log "(dry) would measure: python3 $MEASURE --base-url $BASE_URL $*"
        MEASURE_VALUE="(dry)"; MEASURE_KV="tok_s=0	acceptance=-	loops=0	n=0"
        return 0
    fi
    python3 "$MEASURE" --base-url "$BASE_URL" --n "$N" --max-tokens "$MAX_TOKENS" "$@" 2>&1 |
        tee -a "$mlog"
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
    MEASURE_VALUE=$(sed -n 's/^VALUE\t//p' "$mlog" | tail -n 1)
    MEASURE_KV=$(sed -n 's/^KV\t//p' "$mlog" | tail -n 1)
    [ -n "$MEASURE_VALUE" ]
}

kv_get() { printf '%s\n' "$MEASURE_KV" | tr '\t' '\n' | sed -n "s/^$1=//p" | head -n 1; }

# one server + one or more measurements, rows appended, server stopped in every case
# measure_config NAME QUANTITY HOW NOTES -- SERVE_ARGS... -- MEASURE_ARGS...
measure_config() {
    local name=$1 quantity=$2 how=$3 notes=$4; shift 4
    [ "$1" = "--" ] && shift
    local -a sargs=() margs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do sargs+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    margs=("$@")
    local stamp slog mlog acc
    stamp=$(ts); slog="$LOGS/$name-server-$stamp.log"; mlog="$LOGS/$name-$stamp.log"
    ln -sf "$slog" "$LOGS/.current-server.log"
    if ! start_server "$slog" "${sargs[@]}"; then
        add_row "$quantity" "FAILED to start" "$how" "$notes; see $(basename "$slog")"
        stop_server; return 1
    fi
    local rc=0
    run_measure "$mlog" "${margs[@]}" || rc=1
    acc=$(grep_acceptance "$slog")
    [ -n "$acc" ] && notes="$notes; server log draft acceptance $acc"
    if [ $rc -eq 0 ]; then
        add_row "$quantity" "$MEASURE_VALUE" "$how" "$notes"
    else
        add_row "$quantity" "FAILED" "$how" "$notes; see $(basename "$mlog")"
    fi
    stop_server
    return $rc
}

# ---------------------------------------------------------------- experiments

PROD_ARGS=(--slots 4 --ctx-pool 524288 --kv q8_0 --mtp-draft 1)   # the Hercules launch shape
WINNER_FILE=""

# track the best loop-free sampling for mtp_control: "mode" or "mode:temp"
note_candidate() {
    local label=$1 tok loops
    tok=$(kv_get tok_s); loops=$(kv_get loops)
    printf '%s\t%s\t%s\n' "$label" "${tok:-0}" "${loops:-0}" >> "$WINNER_FILE"
}

pick_winner() {
    [ -s "$WINNER_FILE" ] || return 0
    # loop-free first, then highest tok/s
    awk -F'\t' '$3 == 0' "$WINNER_FILE" | sort -t$'\t' -k2,2gr | head -n 1 | cut -f1
}

exp_loop_guard() {
    WINNER_FILE="$LOGS/loop-guard-candidates.tsv"; : > "$WINNER_FILE"
    local how="llama-server via tenselerate serve, 4x524288 q8_0, MTP depth 1, measure.py --loop-check, N=$N per shape"
    local mode
    for mode in greedy dry low; do
        measure_config "loop_guard-$mode" "loop guard: --sampling $mode (MTP depth 1)" "$how" \
            "server default sampling; client sends none" \
            -- "${PROD_ARGS[@]}" --sampling "$mode" -- --loop-check || continue
        note_candidate "$mode"
    done
    local temp
    for temp in 0.15 0.5; do
        measure_config "loop_guard-low-t$temp" "loop guard: --sampling low, client temp $temp (MTP depth 1)" "$how" \
            "client overrides temperature only; min-p 0.1 and repeat-penalty 1.0 from the server" \
            -- "${PROD_ARGS[@]}" --sampling low -- --loop-check --send-sampling "temp=$temp" || continue
        note_candidate "low:$temp"
    done
    local w
    w=$(pick_winner)
    if [ -n "$w" ]; then
        printf '%s\n' "$w" > "$LOGS/winning-sampling"
        add_row "loop guard winner" "$w" "highest mean tok/s among loop-free candidates" \
            "candidates in logs/loop-guard-candidates.tsv; mtp_control uses it unless WIN_SAMPLING is set"
    else
        add_row "loop guard winner" "none loop-free" "all candidates looped or failed" "mtp_control falls back to dry"
    fi
}

exp_client_override() {
    measure_config "client_override" \
        "client override: server --sampling greedy, request temp 0.7 + repeat_penalty 1.15 (MTP depth 1)" \
        "measure.py --send-sampling temp=0.7,rp=1.15 against the greedy server" \
        "expect ~0.22 acceptance and <34 tok/s if the request wins over the server default" \
        -- "${PROD_ARGS[@]}" --sampling greedy -- --loop-check --send-sampling temp=0.7,rp=1.15
}

exp_deep_kv() {
    local kv
    for kv in q8_0 f16; do
        measure_config "deep_kv-$kv" \
            "262K single stream, kv $kv: decode at ~${DEEP_PREFILL} tokens" \
            "1 slot x 262144, no spec, measure.py --prefill-tokens $DEEP_PREFILL (synthetic numbered paragraphs), N=2" \
            "prediction: f16 ~22 tok/s (MMA_F16 + GQA) vs q8_0 12.4 (vec kernel, 6x redundant KV read)" \
            -- --slots 1 --ctx-pool 262144 --kv "$kv" --mtp-draft 0 --sampling greedy \
            -- --prefill-tokens "$DEEP_PREFILL" --slot-size 262144 --n 2 --max-tokens 200 --loop-check || true
    done
}

exp_slot4_width() {
    local how="4 slots x 524288 q8_0, MTP depth 1, greedy, measure.py --concurrency 4 (distinct 1500-token prefixes)"
    measure_config "slot4_width-default" "4 slots active, MTP depth 1, default MMVQ routing" "$how" \
        "verification width 8 on the dp4a path" \
        -- "${PROD_ARGS[@]}" --sampling greedy -- --concurrency 4 --loop-check || true
    measure_config "slot4_width-mmvq3" "4 slots active, MTP depth 1, --mmvq-max 3" "$how" \
        "GGML_CUDA_MMVQ_MAX=3: width 8 goes to MMQ, single-slot turns stay on dp4a; predicted ~70" \
        -- "${PROD_ARGS[@]}" --sampling greedy --mmvq-max 3 -- --concurrency 4 --loop-check || true
    measure_config "slot4_width-nommvq" "4 slots active, MTP depth 1, --no-mmvq" "$how" \
        "GGML_CUDA_NO_MMVQ=1: every width on MMQ" \
        -- "${PROD_ARGS[@]}" --sampling greedy --no-mmvq -- --concurrency 4 --loop-check || true
}

exp_spec_depth() {
    # Where draft width can come from, at temp 0.
    #
    # The MTP depth sweep already answered one half: the head is shallow, not
    # broken. n-max 1 is +35%, n-max 3 is -15%, n-max 5 is -22%, because
    # position 1 accepts at 0.88 and positions 2+ do not, and a rejected draft
    # column is paid in full. So depth cannot come from the MTP head, and
    # `spec_depth-mtp8` is here as the falsifier for exactly that claim.
    #
    # ngram-mod is the other half. It drafts by replaying a run already present
    # in this context, so it runs no model: a miss is a hash lookup, not a
    # forward pass. It is ordered BEFORE draft-mtp (common_speculative takes the
    # first implementation that returns a draft and never concatenates), so a
    # miss falls straight through to the measured depth-1 path.
    #
    # Two shapes, because the two directions falsify different things:
    #   rewrite - the answer is mostly verbatim replay. This is where a hit
    #             pays, and where the flat MMQ region (55 ms from N=2 to N=16)
    #             turns 9 verify columns into ~9 tokens for one weight read.
    #   code    - the answer never appeared in the prompt, so ngram-mod MUST
    #             miss. This is the control: the fall-through is only free if
    #             this row lands within noise of the baseline.
    #
    # Predictions, written before the run (baseline is 46.2 tok/s greedy MTP-1):
    #   rewrite baseline   ~46      (MTP depth 1, one accepted token per read)
    #   rewrite ngram 8     90-160  (a hit drafts up to 8 free columns; the
    #                                range is wide because per-round hit rate,
    #                                not per-token acceptance, is what decides)
    #   rewrite ngram 15   >= ngram 8, and the gap between them says whether the
    #                      MMQ region is still flat at width 16
    #   code ngram 8       42-48    (a miss must be free). BELOW 42 REFUTES the
    #                      ordering claim: it would mean the ngram attempt costs
    #                      a real pass, not a lookup, and ngram must then be
    #                      turned on per-workload rather than by default
    #   code mtp 8         25-30    (below the 34.4 no-MTP baseline, per -22% at
    #                      n-max 5). ABOVE 46.2 REFUTES the shallow-head reading
    #                      and the whole depth argument is wrong
    local how base
    base="4 slots x 524288 q8_0, greedy (temp 0), measure.py N=$N"
    how="$base, shape rewrite (verbatim replay) and code (control, must miss)"
    for shape in rewrite code; do
        measure_config "spec_depth-$shape-baseline" \
            "$shape: MTP depth 1, no n-gram draft (production today)" "$how" \
            "prediction: ~46 tok/s, the greedy MTP-1 baseline" \
            -- "${PROD_ARGS[@]}" --sampling greedy \
            -- --shapes "$shape" --n "$N" --loop-check || true
        measure_config "spec_depth-$shape-ngram8" \
            "$shape: ngram-mod depth 8 ahead of MTP depth 1" "$how" \
            "prediction: rewrite 90-160, code 42-48; code below 42 refutes the free-miss claim" \
            -- "${PROD_ARGS[@]}" --sampling greedy --ngram-draft 8 \
            -- --shapes "$shape" --n "$N" --loop-check || true
    done
    measure_config "spec_depth-rewrite-ngram15" \
        "rewrite: ngram-mod depth 15 (the edge of the flat MMQ region)" "$how" \
        "prediction: >= depth 8; the gap grades whether MMQ is still flat at width 16" \
        -- "${PROD_ARGS[@]}" --sampling greedy --ngram-draft 15 \
        -- --shapes rewrite --n "$N" --loop-check || true
    measure_config "spec_depth-code-mtp8" \
        "code: MTP draft depth 8, the falsifier for the shallow-head reading" "$how" \
        "prediction: 25-30, below the 34.4 no-MTP baseline; above 46.2 refutes the depth argument" \
        -- --slots 4 --ctx-pool 524288 --kv q8_0 --mtp-draft 8 --sampling greedy \
        -- --shapes code --n "$N" --loop-check || true
}

exp_mtp_control() {
    local w mode temp
    w=${WIN_SAMPLING:-$(cat "$LOGS/winning-sampling" 2>/dev/null || true)}
    w=${w:-dry}
    mode=${w%%:*}; temp=""
    [ "$mode" != "$w" ] && temp=${w#*:}
    local -a margs=(--loop-check)
    [ -n "$temp" ] && margs+=(--send-sampling "temp=$temp")
    local how="4x524288 q8_0, --sampling $mode${temp:+ + client temp $temp}, measure.py N=$N per shape"
    measure_config "mtp_control-off" "control: MTP off, sampling $w" "$how" "--mtp-draft 0" \
        -- --slots 4 --ctx-pool 524288 --kv q8_0 --mtp-draft 0 --sampling "$mode" -- "${margs[@]}" || true
    measure_config "mtp_control-on" "control: MTP depth 1, sampling $w" "$how" "--mtp-draft 1" \
        -- --slots 4 --ctx-pool 524288 --kv q8_0 --mtp-draft 1 --sampling "$mode" -- "${margs[@]}" || true
}

# ---------------------------------------------------------------- llama-bench experiments

bench_bin() {
    if [ -n "${LLAMA_BENCH:-}" ]; then printf '%s' "$LLAMA_BENCH"; return; fi
    if [ -n "${LLAMA_SERVER:-}" ] && [ -x "$(dirname "$LLAMA_SERVER")/llama-bench" ]; then
        printf '%s' "$(dirname "$LLAMA_SERVER")/llama-bench"; return
    fi
    printf '%s' "$ROOT/build/bin/llama-bench"
}

# run_bench NAME LOG ENVSPEC MODEL ARGS... ; prints "pp=<t/s> tg=<t/s>" from llama-bench -o jsonl
# ENVSPEC is "-" or VAR=value[,VAR=value]
run_bench() {
    local name=$1 blog=$2 envspec=$3 model=$4; shift 4
    local bin; bin=$(bench_bin)
    local -a envs=()
    [ "$envspec" != "-" ] && IFS=',' read -r -a envs <<< "$envspec"
    if [ -n "$DRY" ]; then
        log "(dry) $name: env ${envs[*]:-none} $bin -m $model -ngl 999 -fa on -o jsonl $*"
        printf 'pp=0 tg=0'
        return 0
    fi
    [ -x "$bin" ] || { log "no llama-bench at $bin (set LLAMA_BENCH)"; return 1; }
    env "${envs[@]}" "$bin" -m "$model" -ngl 999 -fa on -o jsonl "$@" 2>>"$blog" | tee -a "$blog" |
        python3 -c '
import json, sys
pp = tg = None
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
        tg = r.get("avg_ts")
    elif r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
        pp = r.get("avg_ts")
print("pp=%s tg=%s" % ("-" if pp is None else "%.1f" % pp, "-" if tg is None else "%.1f" % tg))'
}

# the regression bisect: new binary with the packed attention off / on, and an older binary if given
# Are CUDA graphs alive during speculation, or is every step re-launching ~1000 kernels?
#
# ggml keys its CUDA graph cache on `cgraph->nodes[0]` - a POINTER, not a shape
# (ggml_cuda_graph_get_key). Every batch width therefore lands in the same cache slot.
# ggml_cuda_graph_update_required then memcmp's every node's ne/nb and source pointers
# against the previous step, and on any difference ggml_backend_cuda_graph_compute does
# this:
#
#     if (properties_changed) { graph->warmup_complete = false; }   // execute directly
#
# so a shape change does not merely update the graph - it drops to direct execution AND
# resets the warmup, which needs two consecutive stable calls to re-enter. A workload whose
# batch width alternates never gets two in a row and runs permanently un-graphed.
#
# Decode submits draft+1 tokens. MTP depth 1 is always 2, constant. `ngram-mod` with
# n-min 4 / n-max 8 emits 4..8 tokens on a hit and nothing on a miss, so the width walks
# 1, 5, 6, 7, 8, 9 - a different shape most steps. The n-gram configuration recommended
# earlier in this repo may therefore be paying for its free tokens in launch overhead, and
# nobody has looked.
#
# The "CUDA graphs reused = 247, not disabled" row in README.md was taken WITHOUT
# speculation, where the width is constant at 1. It says nothing about this.
#
# ~1000 kernels per token (64 layers, 48 of them GDN) at a few microseconds of dispatch
# each is the right order of magnitude for the ~11.4 ms of the 29.9 ms step that the
# two-regime decomposition has never accounted for.
#
# Pre-registered, counting log lines per generated token at -v:
#   CONFIRMS  "CUDA graph warmup reset" appears at a rate near one per step under
#             ngram+MTP and not under MTP alone. Graph thrash is real; the fix is a
#             constant-width draft (pad to a fixed column count - free, since MMQ is flat
#             from N=2 to N=16) rather than a wider one.
#   REFUTES   resets are rare in every arm. Graphs are alive, the residual is elsewhere,
#             and kernel-work.md item 3's nsys profile is the next step instead.
exp_graph_thrash() {
    local how="production args at -v; count 'CUDA graph warmup reset' and 'CUDA Graph id' lines per generated token in the server log"
    measure_config "graph_thrash-nospec" "no speculation: constant width 1" "$how" \
        "baseline: the 247-reused row was taken here" \
        -- "${PROD_ARGS[@]}" --mtp-draft 0 --sampling greedy --extra=-v -- --loop-check || true
    measure_config "graph_thrash-mtp1" "MTP depth 1: constant width 2" "$how" \
        "prediction: as stable as no-spec, because draft+1 does not vary" \
        -- "${PROD_ARGS[@]}" --mtp-draft 1 --sampling greedy --extra=-v -- --loop-check || true
    measure_config "graph_thrash-ngram" "ngram-mod 8 ahead of MTP 1: width walks 1,5..9" "$how" \
        "prediction: warmup resets near one per step; if so the n-gram win is partly cancelled by launch overhead" \
        -- "${PROD_ARGS[@]}" --mtp-draft 1 --ngram-draft 8 --ngram-min 4 --sampling greedy --extra=-v -- --loop-check || true
    log "graph_thrash: grep -c 'warmup reset' and 'Graph id' in the server logs and divide by tokens generated. One reset per step means graphs are off."
}

# Does sampling on the GPU pay on a 2 GB/s link with a 248,320-token vocabulary?
#
# llama.cpp samples on the host by default: the logit row is copied device-to-host every
# sampled position. That row is vocab_size floats - this model's vocabulary is 248,320, so
# ~0.99 MB - and this box's link is PCIe Gen2 x4 at ~2 GB/s. That is ~0.6 ms per sampled
# position, and under MTP depth 1 there are two positions per verify step. An x16 Gen4
# machine moves the same row in ~30 us and nobody has ever had a reason to care, which is
# why `-bs` is still marked experimental and off by default upstream. The product of an
# unusually large vocabulary and an unusually narrow link is what makes it worth a run here.
#
# It is also a blocking copy, so the cost is not only bytes: the sync each step can break
# CUDA-graph replay batching. That part cannot be predicted from the file, only measured.
#
# Pre-registered. Greedy MTP depth 1 is 46.2 tok/s.
#   CONFIRMS  -bs is faster. Size the win against ~0.6 ms/position: a gain far larger than
#             ~1.2 ms per step means the sync, not the bytes, was the cost - which is a
#             more interesting result and worth a profile.
#   REFUTES   -bs is equal or slower. The copy was already overlapped, and this closes.
# Read the server log before trusting either number: llama.cpp silently disables backend
# sampling for a grammar or a reasoning budget ("backend sampling is not compatible"), and
# Hermes tool calls use grammars. A run whose log carries that line measured nothing.
exp_backend_sampling() {
    measure_config "backend_sampling-off" \
        "greedy MTP depth 1, host sampling (default)" \
        "production args, --sampling greedy" \
        "baseline: 46.2 tok/s measured" \
        -- "${PROD_ARGS[@]}" --sampling greedy -- --loop-check || true
    measure_config "backend_sampling-on" \
        "greedy MTP depth 1, GPU sampling (-bs)" \
        "same, --backend-sampling" \
        "prediction: +0.6 ms/position of PCIe saved, ~1.2 ms/step under depth 1; check the log for the grammar/reasoning-budget disable line" \
        -- "${PROD_ARGS[@]}" --sampling greedy --backend-sampling -- --loop-check || true
    log "backend_sampling: grep the server log for 'backend sampling is not compatible' - if it is there, the ON run silently used the host path and measured nothing."
}

# Is the MTP head shallow, or is a linear draft just betting on one branch?
#
# The depth sweep (n-max 1 +35%, 3 -15%, 5 -22%) was read as "the head is shallow:
# position 1 accepts reliably, positions 2+ do not". That reading is confounded. A linear
# draft asks the head to predict position 2 given ITS OWN position-1 guess, and that guess
# is wrong ~12% of the time at 0.881 acceptance. When it is wrong, position 2 cannot be
# accepted however good the head is. The sweep fused two questions and blamed the first:
#   (1) can the head predict position 2 at all?
#   (2) did position 1 happen to be right?
#
# The server now logs both, so this run separates them. "acc per pos" is unconditional -
# position i counted only when everything before it was accepted. "acc given prev" is
# n_accepted_per_pos[i] / n_accepted_per_pos[i-1], which is P(i accepted | i-1 accepted):
# the head's real position-2 skill with the branch-luck divided out.
#
# Why it matters beyond the reading: MMQ is flat at ~55 ms from N=2 to N=16, so one weight
# read serves sixteen columns at no extra cost, and depth-1 MTP uses two of them. If the
# head can predict position 2 when position 1 is right, a tree draft (top-k at position 1,
# expanded and verified in one batch behind a tree mask) turns the other fourteen free
# columns into accepted tokens. If it cannot, no tree helps and the existing reading stands.
#
# Pre-registered. Mean accepted length today is 1.88 at 46.2 tok/s.
#   CONFIRMS  acc given prev at position 2 is >= 0.7. The head is not the problem; the
#             linear draft is. Build the tree drafter (docs/tree-speculation.md).
#   REFUTES   acc given prev at position 2 is <= 0.3. The head really is shallow, a tree
#             covers branches that were never going to be accepted anyway, and this line
#             closes. Between 0.3 and 0.7 is a real answer too - it sets how wide the
#             position-1 fan has to be before the tree pays, so record it rather than
#             rerunning until it lands somewhere convenient.
# Depth 2 is the cheapest shape that produces the number; depth 3 shows whether the
# conditional rate holds up or decays, which is what sets usable tree depth.
exp_tree_gate() {
    local d
    for d in 2 3; do
        measure_config "tree_gate-d$d" \
            "MTP draft depth $d, greedy: conditional per-position acceptance" \
            "production args, --mtp-draft $d --sampling greedy; read 'acc given prev' from the server log" \
            "prediction: acc given prev at position 2 >= 0.7 if the depth failure is branch luck, <= 0.3 if the head is shallow" \
            -- "${PROD_ARGS[@]}" --mtp-draft "$d" --sampling greedy -- --loop-check || true
    done
    log "tree_gate: the number that decides this is 'acc given prev', NOT 'acc per pos'. The first position of 'acc given prev' is position 2 given position 1."
}

# Is a heavier KV codec affordable at all? One question, four cells, answered before
# any codec kernel is written.
#
# The claim under test: on the vec path the KV dequant is paid once per QUERY head,
# not once per KV head. This model has gqa_ratio 6 (n_head 24 / n_head_kv 4), so the
# same bytes are decoded six times per token - the 51 ms KV term at 262K against ~11 ms
# of actual bytes. If that is right, it explains why q4_0 halves the KV bytes and still
# measures -8%/token at depth: the saving is counted once and the dequant six times.
#
# The packed vector kernel (GGML_CUDA_FATTN_VEC_GQA) dequantizes each K/V tile once per
# block and reuses it across the packed query heads. That divides the dequant term while
# leaving the byte term whole - so it should move q4_0 relative to q8_0, not just move
# both. A codec that is Nx more expensive to decode than q8_0 is affordable only if that
# division is real, which is the whole reason to run this before building one.
#
# Pre-registered. q4_0 is 18 KiB/token against q8_0's 34, and today measures -8% at depth.
#   CONFIRMS  the q4_0-vs-q8_0 gap with GQA=1 is better than the gap with GQA=0, i.e. the
#             dequant term shrank. Heavier codecs get cheaper the more they are amortized,
#             and kernel-work.md item 4 is worth building.
#   REFUTES   the gap is unchanged or worse with GQA=1. The dequant was never the binding
#             term, every byte saved is paid back in something else, and no codec - 4-bit,
#             2-bit or codebook - will help. Close item 4 and do not write the kernel.
# Note the refutation does not depend on GQA=1 being faster in absolute terms: it is the
# GAP BETWEEN THE TWO KV TYPES that carries the claim. Record all four numbers.
exp_kv_codec_gate() {
    local blog="$LOGS/kv_codec_gate-$(ts).log" depth="${KV_GATE_DEPTH:-131072}" kv gqa r
    for kv in q8_0 q4_0; do
        for gqa in 0 1; do
            r=$(run_bench "kvgate-$kv-gqa$gqa" "$blog" "GGML_CUDA_FATTN_VEC_GQA=$gqa" "$MODEL" \
                    -p "$depth" -n 32 -r 1 -ctk "$kv" -ctv "$kv") || return 1
            add_row "llama-bench tg32 at ${depth} depth, $kv KV, packed attention $([ "$gqa" = 1 ] && echo ON || echo OFF)" \
                "$r" "$(bench_bin) -p $depth -n 32 -r 1 -ctk $kv -ctv $kv, GGML_CUDA_FATTN_VEC_GQA=$gqa" \
                "cell $kv/GQA=$gqa. What decides item 4 is (q4_0 - q8_0) at GQA=1 versus the same gap at GQA=0, not any single cell"
        done
    done
    log "kv_codec_gate: compare the two GAPS, not the four cells. Gap improves -> build the codec; gap flat or worse -> close docs/kernel-work.md item 4."
}

exp_bench_ab() {
    local blog="$LOGS/bench_ab-$(ts).log" r
    r=$(run_bench "new-gqa0" "$blog" "GGML_CUDA_FATTN_VEC_GQA=0" "$MODEL" -p 512 -n 64 -r 3) || return 1
    add_row "llama-bench pp512/tg64, this binary, packed attention OFF" "$r" \
        "$(bench_bin) -p 512 -n 64 -r 3, GGML_CUDA_FATTN_VEC_GQA=0" "baseline for the synced tree; pre-sync tg64 was 33.5"
    r=$(run_bench "new-gqa1" "$blog" "GGML_CUDA_FATTN_VEC_GQA=1" "$MODEL" -p 512 -n 64 -r 3) || return 1
    add_row "llama-bench pp512/tg64, this binary, packed attention FORCED at 512 tokens" "$r" \
        "same, GGML_CUDA_FATTN_VEC_GQA=1" "if this is below the OFF row the packed kernel loses at shallow depth (auto gate is 32K)"
    r=$(run_bench "new-depth-gqa0" "$blog" "GGML_CUDA_FATTN_VEC_GQA=0" "$MODEL" -p 65536 -n 32 -r 2 -ctk q8_0 -ctv q8_0) || return 1
    add_row "llama-bench tg32 at 64K depth, q8_0 KV, packed attention OFF" "$r" "-p 65536 -n 32 -ctk q8_0 -ctv q8_0" "pre-sync 65K was 23.5"
    r=$(run_bench "new-depth-gqa1" "$blog" "GGML_CUDA_FATTN_VEC_GQA=1" "$MODEL" -p 65536 -n 32 -r 2 -ctk q8_0 -ctv q8_0) || return 1
    add_row "llama-bench tg32 at 64K depth, q8_0 KV, packed attention ON" "$r" "same, GGML_CUDA_FATTN_VEC_GQA=1" "the kernel's own A/B; prediction: higher than OFF"
    if [ -n "${LLAMA_BENCH_OLD:-}" ]; then
        r=$(LLAMA_BENCH="$LLAMA_BENCH_OLD" run_bench "old" "$blog" "-" "$MODEL" -p 512 -n 64 -r 3) || return 1
        add_row "llama-bench pp512/tg64, OLD binary ($LLAMA_BENCH_OLD)" "$r" "same flags" \
            "if OLD > new-OFF the regression is in upstream's kernels: nsys both and diff the top kernels"
    else
        log "LLAMA_BENCH_OLD not set: skipping the old-binary row (point it at the llama-bench from the oldest release still on the Releases page)"
    fi
}

# prefill batch: the 35K system prompt costs ~40 s per cache miss at 855 tok/s
exp_prefill_ubatch() {
    local blog="$LOGS/prefill_ubatch-$(ts).log" ub r
    for ub in 512 1024 2048; do
        r=$(run_bench "ub$ub" "$blog" "-" "$MODEL" -p 4096 -n 0 -r 2 -b 2048 -ub "$ub") || return 1
        add_row "llama-bench pp4096, -b 2048 -ub $ub" "$r" "$(bench_bin) -p 4096 -r 2" \
            "prefill vs micro-batch; the launch uses 512. Pick the fastest that fits the compute buffer"
    done
}

# weight bytes: decode is bytes-bound, so a smaller or cheaper-to-dequant quant is a direct speedup
exp_quant_ab() {
    [ -n "${MODEL_ALT:-}" ] || { log "MODEL_ALT not set (a Q4_0 or IQ4_XS GGUF of the same model): skipping"; return 0; }
    local blog="$LOGS/quant_ab-$(ts).log" r
    r=$(run_bench "quant-main" "$blog" "-" "$MODEL" -p 512 -n 64 -r 3) || return 1
    add_row "llama-bench tg64, $(basename "$MODEL")" "$r" "-p 512 -n 64 -r 3" "the served quant"
    r=$(run_bench "quant-alt" "$blog" "-" "$MODEL_ALT" -p 512 -n 64 -r 3) || return 1
    add_row "llama-bench tg64, $(basename "$MODEL_ALT")" "$r" "same flags" \
        "bytes per weight is the lever: Q4_0 ~+10-15% (cheapest dequant), IQ4_XS ~+12% (14 GB). Quality needs a tokens-to-answer A/B"
}

# ---------------------------------------------------------------- driver

run_experiment() {
    local name=$1 elog rc=0
    case " $EXPERIMENTS " in
        *" $name "*) ;;
        *) SUMMARY+="$name: skipped"$'\n'; return 0 ;;
    esac
    elog="$LOGS/$name-$(ts).log"
    log "=== experiment $name (log $elog)"
    local fn="exp_$name"
    [ "${FAIL_EXPERIMENT:-}" = "$name" ] && fn=false    # self-test fault injection
    set +e
    { "$fn"; } 2>&1 | tee -a "$elog"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 0 ]; then SUMMARY+="$name: ok"$'\n'; else SUMMARY+="$name: FAILED (rc $rc, see $elog)"$'\n'; fi
    return 0
}

cleanup() {
    [ -n "$DRY" ] && return 0
    local pid
    pid=$(cat "$LOGS/.server.pid" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill -TERM -- "-$pid" 2>/dev/null || true
        sleep 2
        kill -KILL -- "-$pid" 2>/dev/null || true
        rm -f "$LOGS/.server.pid"
    fi
    pkill -f -- "llama-server .*--port $PORT( |$)" 2>/dev/null || true
}

main() {
    [ -n "${MODEL:-}" ] || die "MODEL=/path/to/model.gguf is required (or --self-test)"
    [ -n "$DRY" ] || [ -f "$MODEL" ] || die "MODEL not found: $MODEL"
    [ -f "$MEASURE" ] || die "missing $MEASURE"
    EXPERIMENTS=${EXPERIMENTS//,/ }
    local e
    for e in $EXPERIMENTS; do
        case " $ALL_EXPERIMENTS " in *" $e "*) ;; *) die "unknown experiment '$e'; have: $ALL_EXPERIMENTS" ;; esac
    done
    mkdir -p "$LOGS"
    results_header
    trap cleanup EXIT INT TERM
    if [ -z "$DRY" ]; then
        python3 "$MEASURE" --base-url "$BASE_URL" --health && \
            die "something already answers on $BASE_URL; pick another PORT"
    fi
    log "model: $MODEL"; log "results: $RESULTS"; log "logs: $LOGS"; log "experiments: $EXPERIMENTS"
    for e in $ALL_EXPERIMENTS; do
        run_experiment "$e"
    done
    printf '\n==> summary\n%s' "$SUMMARY"
    printf '==> rows in %s:\n' "$RESULTS"
    grep -c '^| 20' "$RESULTS" || true
    printf 'draft acceptance per server: grep "draft acceptance" %s/*server*.log\n' "$LOGS"
}

self_test() {
    bash -n "$0" || die "bash -n failed"
    python3 "$MEASURE" --self-test || die "measure.py self-test failed"

    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    # unit checks of the pure helpers
    printf 'x\nslot 0: draft acceptance = 0.88100 (  335 accepted /   380 generated), mean len =  1.88\n' > "$tmp/s.log"
    [ "$(grep_acceptance "$tmp/s.log")" = "0.88100" ] || die "grep_acceptance failed"
    [ -z "$(grep_acceptance "$tmp/none.log")" ] || die "grep_acceptance on a missing log"
    MEASURE_KV=$'tok_s=44.40\tacceptance=0.85\tloops=1\tn=9'
    [ "$(kv_get tok_s)" = "44.40" ] && [ "$(kv_get loops)" = "1" ] || die "kv_get failed"
    WINNER_FILE="$tmp/c.tsv"
    printf 'greedy\t46.2\t2\ndry\t45.1\t0\nlow\t41.0\t0\nlow:0.15\t44.0\t0\n' > "$WINNER_FILE"
    [ "$(pick_winner)" = "dry" ] || die "pick_winner failed: $(pick_winner)"
    RESULTS="$tmp/r.md"; add_row "q|x" "v" "h" "n" >/dev/null
    grep -q '^| .* | q\\|x | v | h | n |$' "$RESULTS" || die "add_row failed: $(cat "$RESULTS")"
    [ "$(grep -c '^| --- ' "$RESULTS")" = "1" ] || die "results header"

    # the full driver in DRY mode: every launch is validated with tenselerate --dry-run
    local out
    out=$(DRY=1 MODEL="$tmp/Fake-27B-TURBO-MTP-Q4_K_M.gguf" LOGS="$tmp/logs" RESULTS="$tmp/dry.md" \
          bash "$0") || die "DRY driver failed:
$out"
    printf '%s\n' "$out" | grep -q 'loop_guard: ok' || die "dry summary missing loop_guard"
    printf '%s\n' "$out" | grep -q 'mtp_control: ok' || die "dry summary missing mtp_control"
    local rows; rows=$(grep -c '^| 20' "$tmp/dry.md")
    [ "$rows" -ge 13 ] || die "expected >= 13 dry rows, got $rows"
    grep -q -- '--spec-draft-n-max 1' "$tmp"/logs/slot4_width-mmvq3-server-*.log || die "dry-run argv not logged"
    grep -q 'GGML_CUDA_MMVQ_MAX=3' "$tmp"/logs/slot4_width-mmvq3-server-*.log || die "mmvq-max not in env prefix"
    grep -q -- '-ctk f16' "$tmp"/logs/deep_kv-f16-server-*.log || die "f16 kv not in argv"
    grep -q -- '--port 8089' "$tmp"/logs/loop_guard-greedy-server-*.log || die "port not 8089"
    grep -q -- '--host 127.0.0.1' "$tmp"/logs/loop_guard-greedy-server-*.log || die "not loopback"
    out=$(DRY=1 MODEL=x EXPERIMENTS=deep_kv LOGS="$tmp/logs2" RESULTS="$tmp/sub.md" bash "$0")
    printf '%s\n' "$out" | grep -q 'loop_guard: skipped' || die "EXPERIMENTS subset not honoured"
    [ "$(grep -c '^| 20' "$tmp/sub.md")" = "2" ] || die "subset row count"
    ! DRY=1 MODEL=x EXPERIMENTS=bogus LOGS="$tmp/logs3" RESULTS="$tmp/b.md" bash "$0" >/dev/null 2>&1 || die "unknown experiment accepted"
    # one failing experiment is reported and does not stop the rest
    out=$(DRY=1 MODEL=x EXPERIMENTS=client_override,mtp_control FAIL_EXPERIMENT=client_override \
          LOGS="$tmp/logs4" RESULTS="$tmp/f.md" bash "$0")
    printf '%s\n' "$out" | grep -q 'client_override: FAILED' || die "failed experiment not reported:
$out"
    printf '%s\n' "$out" | grep -q 'mtp_control: ok' || die "a failure stopped the following experiment"

    printf 'self-test OK: bash -n, measure.py stub run, acceptance grep, kv parse, winner pick,\n'
    printf '              row escaping, DRY driver (all launches validated by tenselerate --dry-run),\n'
    printf '              EXPERIMENTS subset, unknown experiment rejected, failure isolation\n'
}

case "${1:-}" in
    --self-test) self_test ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//' ;;
    "") main ;;
    *) die "unknown option: $1 (try --help)" ;;
esac
