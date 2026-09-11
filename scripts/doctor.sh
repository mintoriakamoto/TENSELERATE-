#!/bin/bash
# Clone-health check. Measured toolkit is 12.8; 13.3 and 14.4 are accepted. 12.4 is not.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF_NAME="Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"
fail=0
say() { printf '  %-8s %s\n' "$1" "$2"; }

# shellcheck source=cuda-root.sh
source "$(cd "$(dirname "$0")" && pwd)/cuda-root.sh"

echo "TENSELERATE doctor ($ROOT)"

if [ -n "${TENSELERATE_CUDA_ROOT:-}" ]; then
  _ver=$("$TENSELERATE_CUDA_ROOT/bin/nvcc" --version | awk '/release/{print $NF}')
  say OK "CUDA toolkit $_ver at $TENSELERATE_CUDA_ROOT (measured path is 12.8; 13.3/14.4 accepted)"
else
  say FAIL "need CUDA 12.8, 13.3, or 14.4 nvcc (12.4 rejected). Driver 13.3/14.4 is fine."
  fail=1
fi

if command -v cmake >/dev/null; then
  say OK "cmake $(cmake --version | head -1)"
else
  say FAIL "cmake not on PATH"
  fail=1
fi

if command -v ninja >/dev/null; then
  say OK "ninja $(ninja --version)"
else
  say FAIL "ninja not on PATH (preset generator is Ninja)"
  fail=1
fi

if command -v nvidia-smi >/dev/null; then
  say OK "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  say WARN "nvidia-smi not found (build still works; serve needs a driver)"
fi

BIN="$ROOT/build-deploy-cmp170hx/bin/llama-server"
if [ -x "$BIN" ]; then
  say OK "binary $BIN"
else
  say WARN "llama-server not built yet — cmake --preset deploy-cmp170hx && cmake --build build-deploy-cmp170hx -j\$(nproc) --target llama-server"
fi

found=""
for c in "${MODEL:-}" "${TENSELERATE_MODEL:-}" "$ROOT/models/$GGUF_NAME" "$HOME/models/$GGUF_NAME" "/home/ai/models/$GGUF_NAME"; do
  if [ -n "$c" ] && [ -f "$c" ]; then found="$c"; break; fi
done
if [ -n "$found" ]; then
  say OK "model $found ($(du -h "$found" | awk '{print $1}'))"
else
  say WARN "model $GGUF_NAME not on disk — bash scripts/fetch-model.sh"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "doctor: FAIL (need CUDA 12.8/13.3/14.4 + cmake + ninja; flags: MMQ ON, MMVQ_MAX=3)"
  exit 1
fi
echo "doctor: OK"
exit 0
