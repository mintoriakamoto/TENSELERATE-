#!/bin/bash
# Clone-health check. CUDA 12.8 is required. Exit 1 if this tree cannot build/serve.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF_NAME="Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"
CUDA128="${CUDAToolkit_ROOT:-/usr/local/cuda-12.8}"
fail=0
say() { printf '  %-8s %s\n' "$1" "$2"; }

echo "TENSELERATE doctor ($ROOT)"

_nvcc=""
if [ -x "$CUDA128/bin/nvcc" ]; then
  _nvcc="$CUDA128/bin/nvcc"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
  _nvcc=/usr/local/cuda/bin/nvcc
fi
if [ -n "$_nvcc" ] && "$_nvcc" --version 2>/dev/null | grep -q 'release 12.8'; then
  say OK "CUDA 12.8 nvcc: $("$_nvcc" --version | awk '/release/{print $NF}') ($_nvcc)"
else
  say FAIL "CUDA 12.8 toolkit required (got: ${CUDA128} / PATH nvcc). 12.4 and 13.x toolkits are rejected."
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
  echo "doctor: FAIL (CUDA 12.8 + cmake + ninja required)"
  exit 1
fi
echo "doctor: OK"
exit 0
