#!/bin/bash
# Production boot: CMP 170HX / Ampere sm_80. Port 8083.
# Default GGUF: Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf
# CUDA toolkit: 12.8 measured; 13.3 and 14.4 accepted. 12.4 rejected.
# Runtime: GGML_CUDA_MMVQ_MAX=32  -c 262144 -np 8 -kvu  MTP n-max 4
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${TENSELERATE_BIN:-$ROOT/build-deploy-cmp170hx/bin/llama-server}"
# Exact GGUF this fork is measured and served with (DavidAU TURBO + MTP, Q4_K_M).
GGUF_NAME="Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"
GGUF_HF_REPO="DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
_find_model() {
  local c
  for c in \
    "${MODEL:-}" \
    "${TENSELERATE_MODEL:-}" \
    "$ROOT/models/$GGUF_NAME" \
    "$HOME/models/$GGUF_NAME" \
    "/home/ai/models/$GGUF_NAME"
  do
    [ -n "$c" ] && [ -f "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}
MODEL="$(_find_model || true)"
PORT="${PORT:-8083}"
ALIAS="${ALIAS:-hermes38-tenselerate}"
LOGDIR="${TENSELERATE_LOGDIR:-$HOME/.hermes/logs}"
SLOTDIR="${TENSELERATE_SLOTDIR:-$HOME/.hermes/slotcache}"
# shellcheck source=cuda-root.sh
source "$(cd "$(dirname "$0")" && pwd)/cuda-root.sh"
CUDA_ROOT="${TENSELERATE_CUDA_ROOT:-}"

mkdir -p "$LOGDIR" "$SLOTDIR"
LOG="$LOGDIR/boot-tenselerate-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== TENSELERATE boot $(date) ==="

if [ -z "$CUDA_ROOT" ]; then
  echo "FATAL: need CUDA toolkit 12.8 (measured), 13.3, or 14.4. 12.4 is rejected."
  echo "  export CUDAToolkit_ROOT=/usr/local/cuda-12.8"
  exit 1
fi
export PATH="$CUDA_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDAToolkit_ROOT="$CUDA_ROOT"
export GGML_CUDA_MMVQ_MAX="${GGML_CUDA_MMVQ_MAX:-32}"

if [ ! -x "$BIN" ]; then
  echo "FATAL: $BIN missing."
  echo "  cmake --preset deploy-cmp170hx && cmake --build $ROOT/build-deploy-cmp170hx -j\$(nproc) --target llama-server"
  exit 1
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "FATAL: missing $GGUF_NAME"
  echo "  huggingface-cli download $GGUF_HF_REPO $GGUF_NAME --local-dir $ROOT/models"
  echo "  or: bash scripts/fetch-model.sh"
  exit 1
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
pkill -9 -f "$BIN" 2>/dev/null || true
sleep 1

if command -v nvidia-smi >/dev/null 2>&1; then
  sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 || true
  sudo -n nvidia-smi -pl "${GPU_POWER_LIMIT:-250}" >/dev/null 2>&1 || true
fi

echo "CUDA 12.8  MMVQ_MAX=$GGML_CUDA_MMVQ_MAX  -c 262144 -np 8 -kvu  MTP n-max 4  :$PORT"
echo "model $MODEL"

NUMA=( )
if command -v numactl >/dev/null 2>&1; then
  NUMA=(numactl --membind=0)
fi

"${NUMA[@]}" "$BIN" \
  -m "$MODEL" \
  -ngl 999 --main-gpu 0 -kvo \
  -c 262144 -np 8 -cb -kvu \
  -ctk q8_0 -ctv q8_0 -fa on \
  -b 8192 -ub 2048 \
  --cache-idle-slots --slot-prompt-similarity 0.1 --cache-ram "${CACHE_RAM:-16384}" \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 \
  --slot-save-path "$SLOTDIR" \
  --slots --metrics --jinja --no-reasoning-preserve \
  --op-offload --stream-decode \
  --poll 0 --numa isolate \
  -t 16 -tb 8 \
  --temp 0 --repeat-penalty 1.0 \
  --host 127.0.0.1 --port "$PORT" --alias "$ALIAS" \
  > "$LOGDIR/llama-server-tenselerate.log" 2>&1 &

ok=0
for i in $(seq 1 180); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q ok; then
    ok=1
    break
  fi
  if ! kill -0 $! 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ "$ok" != 1 ]; then
  echo "Status: ✗"
  tail -40 "$LOGDIR/llama-server-tenselerate.log"
  exit 1
fi
echo "Status: ✓  http://127.0.0.1:${PORT}/v1  alias $ALIAS"
