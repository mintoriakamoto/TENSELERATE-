#!/usr/bin/env bash
# One-shot: deps → CUDA 12.8 (or 13.3/14.4) → cmake --preset deploy-cmp170hx →
# llama-server → exact DavidAU GGUF. Does NOT install Hermes.
#
#   bash install.sh
#   bash install.sh --skip-model
#   bash install.sh --serve
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"
SKIP_MODEL=0
SERVE=0
for a in "$@"; do
  case "$a" in
    --skip-model) SKIP_MODEL=1 ;;
    --serve) SERVE=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

say() { printf '==> %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

say "1/5 host packages (cmake ninja python3)"
need=()
have cmake || need+=(cmake)
have ninja || need+=(ninja-build)
have python3 || need+=(python3)
have pip3 || have pip || need+=(python3-pip)
have curl || need+=(curl)
have git || need+=(git)
if [ "${#need[@]}" -gt 0 ]; then
  if have sudo; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${need[@]}" build-essential pkg-config
  else
    echo "install: ${need[*]}  (no sudo)" >&2
    exit 1
  fi
fi

say "2/5 CUDA toolkit (12.8 measured; 13.3/14.4 ok; 12.4 no)"
# shellcheck source=scripts/cuda-root.sh
source "$ROOT/scripts/cuda-root.sh"
if [ -z "${TENSELERATE_CUDA_ROOT:-}" ]; then
  echo "FATAL: install CUDA 12.8 (preferred), 13.3, or 14.4 under /usr/local/cuda-*" >&2
  echo "  https://developer.nvidia.com/cuda-12-8-0-download-archive" >&2
  exit 1
fi
export PATH="$TENSELERATE_CUDA_ROOT/bin:$PATH"
export CUDAToolkit_ROOT="$TENSELERATE_CUDA_ROOT"
export LD_LIBRARY_PATH="$TENSELERATE_CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "    using $TENSELERATE_CUDA_ROOT"

say "3/5 pip -r requirements-install.txt"
PIP=pip3
have pip3 || PIP=pip
"$PIP" install -q -r "$ROOT/requirements-install.txt"

say "4/5 cmake --preset deploy-cmp170hx (FORCE_MMQ ON, CUBLAS OFF, DISABLE_DP4A ON, sm_80, MMVQ_MAX=3 at runtime)"
cmake --preset deploy-cmp170hx
cmake --build "$ROOT/build-deploy-cmp170hx" -j"$JOBS" --target llama-server

if [ "$SKIP_MODEL" -eq 0 ]; then
  say "5/5 fetch DavidAU TurboFCFusion Q4_K_M MTP GGUF"
  bash "$ROOT/scripts/fetch-model.sh"
else
  say "5/5 skip model"
fi

bash "$ROOT/scripts/doctor.sh" || true

echo
echo "build done. server is not Hermes — boot with:"
echo "  bash scripts/boot-cmp170hx.sh"
echo "  # MMVQ_MAX=3  -np 8 -c 262144 -kvu  MTP n-max 4  :8083"

if [ "$SERVE" -eq 1 ]; then
  exec bash "$ROOT/scripts/boot-cmp170hx.sh"
fi
