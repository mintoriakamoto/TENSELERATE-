// CUDA int8 GEMM for the TENSELERATE engine.
//
// C[m,n] = (sum_k A[m,k]*B[n,k]) * a_scale[m] * b_scale[n],  A[M,K], B[N,K] int8.
//
// This first kernel is a tiled dp4a implementation: correct and portable across
// every CC >= 6.1 device, validated against int8_gemm_ref. It is the CORRECTNESS
// baseline. The hardware win on the CMP 170HX comes from the mma.sync s8 IMMA
// path (CC >= 7.5), which is not firmware-throttled the way dp4a dispatch is;
// that specialized kernel is the next step (see docs/tenselerate-engine.md
// roadmap) and will be validated against this same reference before it replaces
// this one on that hardware.
#include "int8_gemm.h"

#include <cuda_runtime.h>

namespace tenselerate {

namespace {

constexpr int TILE = 16;   // TILE x TILE output tile, K stepped in blocks of 4

// One thread computes one C[m,n]. K is walked 4 lanes at a time through __dp4a
// (int8x4 dot + int32 accumulate), the same primitive an IMMA GEMM issues in
// bulk. Shared-memory staging keeps global reads coalesced.
__global__ void int8_gemm_kernel(
    const int8_t* __restrict__ A, const float* __restrict__ a_scale,
    const int8_t* __restrict__ B, const float* __restrict__ b_scale,
    float* __restrict__ C, int M, int N, int K) {
    const int row = blockIdx.y * TILE + threadIdx.y;   // m
    const int col = blockIdx.x * TILE + threadIdx.x;   // n

    __shared__ int8_t As[TILE][TILE * 4];
    __shared__ int8_t Bs[TILE][TILE * 4];

    int32_t acc = 0;
    const int kstep = TILE * 4;
    for (int k0 = 0; k0 < K; k0 += kstep) {
        // stage a [TILE, kstep] slab of A and of B into shared memory
        for (int j = threadIdx.x; j < kstep; j += TILE) {
            const int kk = k0 + j;
            As[threadIdx.y][j] =
                (row < M && kk < K) ? A[(size_t)row * K + kk] : (int8_t)0;
        }
        for (int j = threadIdx.y; j < kstep; j += TILE) {
            const int kk = k0 + j;
            Bs[threadIdx.x][j] =
                (col < N && kk < K) ? B[(size_t)col * K + kk] : (int8_t)0;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kstep; kk += 4) {
            // pack 4 int8 into one int32 lane for __dp4a
            int a4, b4;
            const int8_t* ap = &As[threadIdx.y][kk];
            const int8_t* bp = &Bs[threadIdx.x][kk];
            a4 = (ap[0] & 0xff) | ((ap[1] & 0xff) << 8) |
                 ((ap[2] & 0xff) << 16) | ((ap[3] & 0xff) << 24);
            b4 = (bp[0] & 0xff) | ((bp[1] & 0xff) << 8) |
                 ((bp[2] & 0xff) << 16) | ((bp[3] & 0xff) << 24);
            acc = __dp4a(a4, b4, acc);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[(size_t)row * N + col] =
            (float)acc * a_scale[row] * b_scale[col];
    }
}

}  // namespace

int int8_gemm_cuda(
    const int8_t* dA, const float* d_a_scale,
    const int8_t* dB, const float* d_b_scale,
    float* dC, int M, int N, int K, void* stream) {
    if (K % 4 != 0) return -1;   // K padded to a multiple of 4 by the caller
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    int8_gemm_kernel<<<grid, block, 0, s>>>(
        dA, d_a_scale, dB, d_b_scale, dC, M, N, K);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}

}  // namespace tenselerate

// C ABI bridge (see int8_gemm.h): owns the device buffers around
// int8_gemm_cuda so a ctypes caller only ever touches host memory. Errors from
// any CUDA call short-circuit and clean up what was already allocated.
extern "C" int tenselerate_int8_gemm(
    const int8_t* A, const float* a_scale,
    const int8_t* B, const float* b_scale,
    float* C, int M, int N, int K) {
    if (K % 4 != 0) return -1;

    int8_t *dA = nullptr, *dB = nullptr;
    float *d_as = nullptr, *d_bs = nullptr, *dC = nullptr;
    cudaError_t err = cudaSuccess;

    auto fail = [&]() {
        cudaFree(dA); cudaFree(dB); cudaFree(d_as); cudaFree(d_bs); cudaFree(dC);
        return -2;
    };

    err = cudaMalloc(&dA, (size_t)M * K * sizeof(int8_t));
    if (err != cudaSuccess) return fail();
    err = cudaMalloc(&dB, (size_t)N * K * sizeof(int8_t));
    if (err != cudaSuccess) return fail();
    err = cudaMalloc(&d_as, (size_t)M * sizeof(float));
    if (err != cudaSuccess) return fail();
    err = cudaMalloc(&d_bs, (size_t)N * sizeof(float));
    if (err != cudaSuccess) return fail();
    err = cudaMalloc(&dC, (size_t)M * N * sizeof(float));
    if (err != cudaSuccess) return fail();

    err = cudaMemcpy(dA, A, (size_t)M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail();
    err = cudaMemcpy(dB, B, (size_t)N * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail();
    err = cudaMemcpy(d_as, a_scale, (size_t)M * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail();
    err = cudaMemcpy(d_bs, b_scale, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return fail();

    int rc = tenselerate::int8_gemm_cuda(dA, d_as, dB, d_bs, dC, M, N, K, nullptr);
    if (rc != 0) { fail(); return rc; }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return fail();

    err = cudaMemcpy(C, dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return fail();

    cudaFree(dA); cudaFree(dB); cudaFree(d_as); cudaFree(d_bs); cudaFree(dC);
    return 0;
}
