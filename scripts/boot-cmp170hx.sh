#!/bin/bash
# Production boot: CMP 170HX, TENSELERATE :8083. Do not start llama-upstream (:8082).
LOG="$HOME/.hermes/logs/boot-tenselerate-$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$HOME/.hermes/logs" "$HOME/.hermes/slotcache"
exec > >(tee -a "$LOG") 2>&1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build-deploy-cmp170hx/bin/llama-server"
MODEL="/home/ai/models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"
PORT=8083
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_CUDA_MMVQ_MAX=3

echo "=== TENSELERATE boot $(date) ==="
if [ ! -x "$BIN" ]; then
  echo "FATAL: $BIN missing. cmake --preset deploy-cmp170hx && cmake --build ${ROOT}/build-deploy-cmp170hx -j\$(nproc) --target llama-server"
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "FATAL: $MODEL missing"
  exit 1
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
pkill -9 -f '/TENSELERATE/build-deploy-cmp170hx/bin/llama-server' 2>/dev/null || true
sleep 1

sudo -n nvidia-smi -pm 1 >/dev/null 2>&1 || true
sudo -n nvidia-smi -pl 250 2>&1 || echo "WARNING: could not set 250W"

echo "Build: CUDA 12.8 FORCE_MMQ=ON CUBLAS=OFF DISABLE_DP4A=ON sm_80-real"
echo "Run: MMVQ_MAX=3  -c 262144 -np 8 -kvu  MTP n-max 4  q8_0  -b 8192 -ub 2048  :${PORT}"

numactl --membind=0 "$BIN" \
  -m "$MODEL" \
  -ngl 999 --main-gpu 0 -kvo \
  -c 262144 -np 8 -cb -kvu \
  -ctk q8_0 -ctv q8_0 -fa on \
  -b 8192 -ub 2048 \
  --cache-idle-slots --slot-prompt-similarity 0.1 --cache-ram 16384 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 \
  --slot-save-path /home/ai/.hermes/slotcache \
  --slots --metrics --jinja --no-reasoning-preserve \
  --op-offload --stream-decode \
  --poll 0 --numa isolate \
  -t 16 -tb 8 \
  --temp 0 --repeat-penalty 1.0 \
  --host 127.0.0.1 --port "$PORT" --alias hermes38-tenselerate \
  > "$HOME/.hermes/logs/llama-server-tenselerate.log" 2>&1 &

ok=0
for i in $(seq 1 120); do
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
  tail -40 "$HOME/.hermes/logs/llama-server-tenselerate.log"
  exit 1
fi
echo "Status: ✓  256K unified / 8 slots / n-max 4 / MMVQ_MAX=3  :${PORT}"
