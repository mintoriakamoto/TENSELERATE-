// int8 symmetric-quantized GEMM: the matmul path the TENSELERATE engine leans on.
//
// C[m,n] = sum_k A[m,k] * B[n,k]  with A,B int8, accumulated in int32, then
// rescaled by per-row float scales:  out[m,n] = acc * a_scale[m] * b_scale[n].
//
// This mirrors an IMMA tensor-core GEMM. On the CMP 170HX the int8 tensor-core
// (IMMA) path is NOT firmware-throttled, unlike dp4a dispatch and FP16 GEMM, so
// this is the fast lane on that hardware; on healthy consumer cards it is also
// competitive for decode. The header carries a portable C++ reference so the
// semantics can be tested with no GPU (tenselerate/csrc/gemm_test.cpp); the CUDA
// kernel in int8_gemm.cu must match it bit-for-bit on the integer accumulator.
#pragma once
#include <cstdint>
#include <cstddef>

namespace tenselerate {

// Row-major: A is [M,K], B is [N,K] (weight, row = output channel), C is [M,N].
// a_scale is [M], b_scale is [N]. Integer accumulation is exact.
inline void int8_gemm_ref(
    const int8_t* A, const float* a_scale,
    const int8_t* B, const float* b_scale,
    float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            const int8_t* a = A + (size_t)m * K;
            const int8_t* b = B + (size_t)n * K;
            for (int k = 0; k < K; ++k) {
                acc += (int32_t)a[k] * (int32_t)b[k];
            }
            C[(size_t)m * N + n] = (float)acc * a_scale[m] * b_scale[n];
        }
    }
}

// CUDA entry point (defined in int8_gemm.cu). Pointers are device pointers.
// Returns 0 on success. A no-op stub is provided when built without CUDA so the
// symbol always links.
int int8_gemm_cuda(
    const int8_t* dA, const float* d_a_scale,
    const int8_t* dB, const float* d_b_scale,
    float* dC, int M, int N, int K, void* stream);

}  // namespace tenselerate
