#!/usr/bin/env bash
# tenselerate-serve.sh — serve the RavenX Chaos Agent (Qwen3.8-27B, Q4_K_M) on
# 127.0.0.1:8080 with an OpenAI-compatible /v1 endpoint, so Hermes can point at
#     model.base_url  http://127.0.0.1:8080/v1
#
# One script, both machines. It reads the GPU from nvidia-smi and picks the
# right -ngl / --tensor-split without you having to know the hardware. Run it
# on either box after building llama-server for that box's preset.
#
#   ./scripts/tenselerate-serve.sh                 # download the model, serve
#   MODEL=/path/to/model.gguf ./scripts/...        # use a local GGUF instead
#   PORT=9000 ./scripts/tenselerate-serve.sh       # different port
#
# The model name Hermes sends (the long HF id with :Q4_K_M) is set as --alias so
# the server answers to exactly that id; llama-server serves whatever is loaded
# regardless, but matching it keeps strict clients happy.
set -euo pipefail

HF_REPO="deadbydawn101/RavenXAiLabs-Chaos-Agent-Qwen3.8-27B-Frontier-Intelligence-Injected-OBLITERATED-GGUF"
ALIAS="${HF_REPO}:Q4_K_M"          # exactly what `hermes config set model.default` expects
HOST="${HOST:-127.0.0.1}"          # loopback only — never expose this port
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"                 # matches the model card's server example

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

# Per-GPU placement. -ngl 99 offloads every layer; the model is ~15.7 GiB, so a
# single 24-40-64 GiB card holds it outright, and a 12+12 pair needs a split
# that fills the faster card first (a layer split buys capacity, not speed —
# see scripts/svmi-gpucheck.py).
GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
PLACE=(-ngl 99)
case "$GPUS" in
    *"5070"*3060*|*3060*5070*)
        # RTX 5070 (12 GiB, faster) + RTX 3060 (12 GiB). Fill the 5070 first.
        # --tensor-split is indexed by visible-device order; if device 0 is not
        # the 5070, reorder with CUDA_VISIBLE_DEVICES.
        # shellcheck disable=SC2054  # 0.68,0.32 is one llama.cpp arg, not two elements
        PLACE=(-ngl 99 --tensor-split 0.68,0.32)
        echo "detected: RTX 5070 + RTX 3060 — resident split, 5070 first" >&2
        ;;
    *"CMP 170HX"*|*"CMP170HX"*)
        echo "detected: CMP 170HX — model resident on one card" >&2
        echo "  reminder: the HBM2e unlock is volatile. If nvidia-smi shows 8/10 GiB," >&2
        echo "  the model will NOT fit — re-run cmpunlocker before serving." >&2
        ;;
    "") echo "warning: no NVIDIA GPU detected — will run on CPU (slow)." >&2 ;;
    *)  echo "detected GPU(s): $GPUS — offloading all layers (-ngl 99)" >&2 ;;
esac

# Thinking OFF (the model card is emphatic: Qwen 3.8 otherwise burns the whole
# token budget on reasoning loops), and the card's sampling: temp 0, rp 1.15.
echo "serving ${ALIAS%%:*}... on http://${HOST}:${PORT}/v1  (Hermes base_url)" >&2
exec "$SERVER" "${SRC[@]}" \
    --alias "$ALIAS" \
    --host "$HOST" --port "$PORT" \
    -c "$CTX" "${PLACE[@]}" \
    --jinja --reasoning-format none \
    --temp 0 --repeat-penalty 1.15 \
    -fa on -ctk q8_0 -ctv q8_0
