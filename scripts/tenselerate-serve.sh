#!/usr/bin/env bash
# tenselerate-serve.sh — serve the RavenX Chaos Agent (Qwen3.8-27B, Q4_K_M) on
# 127.0.0.1:8080 with an OpenAI-compatible /v1 endpoint, so Hermes can point at
#     model.base_url  http://127.0.0.1:8080/v1
#
# One script, both machines. It reads the GPU from nvidia-smi and picks the
# right placement without you having to know the hardware. Run it on either box
# after building llama-server for that box's preset.
#
#   ./scripts/tenselerate-serve.sh            # sane defaults (8K ctx), download+serve
#   ./scripts/tenselerate-serve.sh --max      # push this machine to its ceiling
#   MODEL=/path/model.gguf ./scripts/...       # use a local GGUF instead of -hf
#   CTX=131072 ./scripts/...                    # override context explicitly
#   PORT=9000 ./scripts/...                     # different port
#
# Why --max works at all: this model is the qwen3_5 HYBRID. Of its 64 layers only
# 16 are full-attention; the other 48 are GDN linear-attention that keep a small
# FIXED state instead of a growing KV cache. So KV accrues on 16 layers, not 64 —
# 34 KiB/token at q8_0, 18 KiB at q4_0 — and the model's native 262,144-token
# (256K) context fits far more easily than a dense 27B would. Past 256K needs
# RoPE scaling (YaRN); --max stops at native, no rope.
set -euo pipefail

MODE="serve"
for a in "$@"; do
    case "$a" in
        --max)  MODE="max" ;;
        -h|--help)
            sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $a (try --max or --help)" >&2; exit 2 ;;
    esac
done

HF_REPO="deadbydawn101/RavenXAiLabs-Chaos-Agent-Qwen3.8-27B-Frontier-Intelligence-Injected-OBLITERATED-GGUF"
ALIAS="${HF_REPO}:Q4_K_M"          # exactly what `hermes config set model.default` expects
HOST="${HOST:-127.0.0.1}"          # loopback only — never expose this port
PORT="${PORT:-8080}"
NATIVE_CTX=262144                  # config.json max_position_embeddings; past this needs YaRN

# llama-server: prefer one already on PATH, else this repo's build dirs.
find_server() {
    if command -v llama-server >/dev/null 2>&1; then command -v llama-server; return; fi
    for d in build-rtx-5070+3060 build-rtx-blackwell build-rtx-ampere \
             build-cmp170hx-int8 build-cmp90hx-int8 build-cmp100-210-int8 build; do
        [ -x "$d/bin/llama-server" ] && { echo "$PWD/$d/bin/llama-server"; return; }
    done
    echo "ERROR: llama-server not found. Build it for this machine first:" >&2
    echo "  CMP 170HX : cmake --preset cmp170hx-int8 && cmake --build build-cmp170hx-int8 -j" >&2
    echo "  5070+3060 : cmake --preset rtx-5070+3060 && cmake --build build-rtx-5070+3060 -j" >&2
    exit 1
}
SERVER="$(find_server)"

# Model source: a local MODEL= path, or download+cache from HF by repo id.
if [ -n "${MODEL:-}" ]; then
    SRC=(-m "$MODEL")
else
    SRC=(-hf "${HF_REPO}:Q4_K_M")   # 15.7 GB, cached under ~/.cache/llama.cpp
fi

# ---- machine detection -----------------------------------------------------
GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
case "$GPUS" in
    *"5070"*3060*|*3060*5070*)  MACHINE="fallen" ;;
    *"CMP 170HX"*|*"CMP170HX"*) MACHINE="cmp170hx" ;;
    "")                         MACHINE="cpu" ;;
    *)                          MACHINE="other" ;;
esac

# ---- placement + limits, per machine ---------------------------------------
# Defaults (safe): 8K context, single slot, q8_0 KV. --max overrides per machine
# from the numbers in docs/svmi.md ("Pushing both machines to the limit"), which
# are computed from the exact hybrid geometry (16/64 full-attn, head_dim 256,
# 4 KV heads). KV type is chosen so the target context actually fits.
CTX="${CTX:-8192}"
PARALLEL=1
CTK="q8_0"; CTV="q8_0"
PLACE=(-ngl 99)

case "$MACHINE" in
    fallen)
        # shellcheck disable=SC2054  # 0.68,0.32 is one llama.cpp arg, not two elements
        PLACE=(-ngl 99 --tensor-split 0.68,0.32)
        echo "detected: RTX 5070 + RTX 3060 — resident split, 5070 first" >&2
        if [ "$MODE" = max ]; then
            # ~5.6 GiB free for KV after weights+state. q8_0 caps ~174K; q4_0
            # reaches the full native 256K with room to spare. So --max = 256K
            # on q4_0 KV, single stream.
            CTX="$NATIVE_CTX"; CTK="q4_0"; CTV="q4_0"
            echo "  --max: full 256K native context, single stream, q4_0 KV" >&2
            echo "  q4_0 K loses a little fidelity; recover most of it with a one-time bias:" >&2
            echo "    tools/kv-mean-center — see docs/kv-mean-center.md" >&2
        fi
        ;;
    cmp170hx)
        echo "detected: CMP 170HX — model resident on one card" >&2
        echo "  reminder: the HBM2e unlock is volatile. If nvidia-smi shows 8/10 GiB," >&2
        echo "  the model will NOT fit — re-run cmpunlocker before serving." >&2
        if [ "$MODE" = max ]; then
            # ~46 GiB free for KV. One 256K q8_0 sequence is 8.5 GiB, so the card
            # holds several at once. --max = full 256K AND 5 parallel slots, which
            # is the throughput ceiling (5 agents x 256K, all resident).
            CTX="$NATIVE_CTX"; PARALLEL=5; CTK="q8_0"; CTV="q8_0"
            echo "  --max: 5 parallel slots, each up to 256K, q8_0 KV (~42 GiB of KV)" >&2
            echo "  (one client wanting a single 256K stream still gets full speed;" >&2
            echo "   the extra slots serve concurrent agents at no cost to a lone request.)" >&2
        fi
        ;;
    cpu)
        echo "warning: no NVIDIA GPU detected — CPU only, slow. --max not advised." >&2
        [ "$MODE" = max ] && { CTX="$NATIVE_CTX"; CTK="q4_0"; CTV="q4_0"; }
        ;;
    other)
        echo "detected GPU(s): $GPUS — offloading all layers (-ngl 99)" >&2
        [ "$MODE" = max ] && { CTX="$NATIVE_CTX"; echo "  --max: 256K context (adjust CTK/CTV if it will not fit)" >&2; }
        ;;
esac

# ---- launch ----------------------------------------------------------------
# Thinking OFF (the model card is emphatic: Qwen 3.8 otherwise burns the whole
# token budget on reasoning loops), and the card's sampling: temp 0, rp 1.15.
# --parallel N gives N independent slots; total context is split across them, so
# --max on the CMP sizes -c as N x 256K.
CTX_TOTAL=$(( CTX * PARALLEL ))
echo "serving on http://${HOST}:${PORT}/v1  (Hermes base_url)" >&2
echo "  context ${CTX} tok x ${PARALLEL} slot(s)  |  KV ${CTK}/${CTV}" >&2
exec "$SERVER" "${SRC[@]}" \
    --alias "$ALIAS" \
    --host "$HOST" --port "$PORT" \
    -c "$CTX_TOTAL" --parallel "$PARALLEL" "${PLACE[@]}" \
    --jinja --reasoning-format none \
    --temp 0 --repeat-penalty 1.15 \
    -fa on -ctk "$CTK" -ctv "$CTV"
