# shellcheck shell=bash
# Pick a CUDA toolkit. Measured/default is 12.8. 13.3 and 14.4 are accepted.
# 12.4 is not. Driver UMD 13.3/14.4 is never a reject (nvidia-smi, not nvcc).
# Usage: source this file; it sets TENSELERATE_CUDA_ROOT.

_tenselerate_nvcc_ok() {
  local nvcc="$1/bin/nvcc"
  [ -x "$nvcc" ] || return 1
  "$nvcc" --version 2>/dev/null | grep -Eq 'release (12\.8|13\.3|14\.4)'
}

_tenselerate_pick_cuda() {
  local c
  for c in \
    "${CUDAToolkit_ROOT:-}" \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-13.3 \
    /usr/local/cuda-14.4 \
    /usr/local/cuda
  do
    [ -n "$c" ] || continue
    if _tenselerate_nvcc_ok "$c"; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

TENSELERATE_CUDA_ROOT="$(_tenselerate_pick_cuda || true)"
