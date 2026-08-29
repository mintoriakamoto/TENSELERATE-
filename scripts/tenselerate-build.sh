#!/usr/bin/env bash
# tenselerate-build - compile the engine end to end.
#
# This is a llama.cpp fork, so a full build is two halves:
#   1. the llama.cpp targets (llama-server, llama-cli) via the top-level CMake,
#      with CUDA auto-detected from nvcc;
#   2. the TENSELERATE int8 kernels + the ctypes bridge shared lib
#      (tenselerate/csrc), which the Python engine dlopens.
#
#   scripts/tenselerate-build.sh              # build everything, auto-detect CUDA
#   scripts/tenselerate-build.sh --cpu        # force a CPU-only build
#   scripts/tenselerate-build.sh --cuda       # require CUDA (fail if no nvcc)
#   scripts/tenselerate-build.sh --kernels    # only the TENSELERATE csrc kernels
#
# Honours BUILD_DIR (default ./build) and JOBS (default: nproc).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
KERNEL_DIR="${KERNEL_DIR:-$ROOT/build-kernels}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

MODE="auto"          # auto | cpu | cuda | kernels
for a in "$@"; do
    case "$a" in
        --cpu) MODE="cpu" ;;
        --cuda) MODE="cuda" ;;
        --kernels) MODE="kernels" ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) printf 'error: unknown option %s\n' "$a" >&2; exit 2 ;;
    esac
done

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
say() { printf '==> %s\n' "$*"; }

have cmake || die "cmake is required"

# -- decide CUDA -----------------------------------------------------------
CUDA="OFF"
case "$MODE" in
    cuda)
        have nvcc || die "--cuda given but nvcc not found"
        CUDA="ON" ;;
    cpu|kernels) CUDA="OFF" ;;
    auto)
        if have nvcc; then CUDA="ON"; say "nvcc found - CUDA build";
        else say "no nvcc - CPU-only build"; fi ;;
esac

# -- 1. the TENSELERATE kernels (always; fast, and the bridge needs the .so) --
say "building TENSELERATE int8 kernels ($KERNEL_DIR)"
cmake -S "$ROOT/tenselerate/csrc" -B "$KERNEL_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$KERNEL_DIR" -j "$JOBS"
if [ -x "$KERNEL_DIR/tenselerate_gemm_test" ]; then
    "$KERNEL_DIR/tenselerate_gemm_test" >/dev/null && say "kernel reference test: passed"
fi

if [ "$MODE" = "kernels" ]; then
    say "kernels-only build done: $KERNEL_DIR"
    exit 0
fi

# -- 2. the llama.cpp targets (the server + cli) ---------------------------
say "configuring llama.cpp build ($BUILD_DIR, CUDA=$CUDA)"
cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA="$CUDA" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF
say "building llama-server + llama-cli (-j $JOBS)"
cmake --build "$BUILD_DIR" -j "$JOBS" --target llama-server llama-cli

say "build complete"
for b in llama-server llama-cli; do
    p=$(find "$BUILD_DIR/bin" "$BUILD_DIR" -name "$b" -type f 2>/dev/null | head -1 || true)
    [ -n "$p" ] && printf '    %s -> %s\n' "$b" "$p"
done
printf '    kernels  -> %s\n' "$KERNEL_DIR/libtenselerate_int8_gemm_c.so"
