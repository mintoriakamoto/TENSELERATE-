// TEST FIXTURE - not part of the engine's real build products.
//
// Implements the exact same C ABI as the CUDA shared library's
// `tenselerate_int8_gemm` (see int8_gemm.h), but computes it on the CPU via
// int8_gemm_ref. tests/tenselerate/test_backend_bridge.py compiles this into a
// shared library with a plain host g++ (no CUDA needed) and points
// tenselerate/backend/int8_gemm.py at it via TENSELERATE_INT8_GEMM_LIB.
//
// The point: it validates the ENTIRE Python<->native marshaling path - pointer
// types, array contiguity, the calling convention, error-code propagation -
// using the identical mechanism that talks to the real .cu kernel, with only
// the computation swapped for the CPU reference. Nothing about the ctypes
// bridge is CUDA-specific, so this is a legitimate end-to-end test of it.
#include "int8_gemm.h"

extern "C" int tenselerate_int8_gemm(
    const int8_t* A, const float* a_scale,
    const int8_t* B, const float* b_scale,
    float* C, int M, int N, int K) {
    if (K % 4 != 0) return -1;   // mirror the real wrapper's contract exactly
    tenselerate::int8_gemm_ref(A, a_scale, B, b_scale, C, M, N, K);
    return 0;
}
