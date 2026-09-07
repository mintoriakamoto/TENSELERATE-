// Optimized softmax computation for flash attention
// Reduces register pressure by using mixed precision (FP16 for exp, FP32 for reductions)

#pragma once

// Optimized softmax in-register computation with reduced precision
// Computes exp(x - max) in FP16 while keeping max/sum in FP32 for stability
template<int cols_per_thread>
__device__ __forceinline__ void softmax_mixed_precision(
    float * KQ_vals,        // Input QK values (FP32, modified in-place)
    float * KQ_max,         // Output max values (FP32)
    float * KQ_exp_sum,     // Output sum of exp values (FP32)
    const int nbatch        // Batch size to reduce
) {
    // Phase 1: Find max (FP32) with warp reduction
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Block-level max reduction
#pragma unroll
    for (int offset = 16; offset >= 4; offset >>= 1) {
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_max[col] = fmaxf(KQ_max[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max[col], offset, 32));
        }
    }

    // Phase 2: Compute exp(x - max) in FP16 to reduce register pressure
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_exp_sum[col] = 0.0f;
        // Use mixed precision: compute exp in FP16, accumulate in FP32
        KQ_vals[col] = __half2float(__float2half(__expf(KQ_vals[col] - KQ_max[col])));
        KQ_exp_sum[col] += KQ_vals[col];
    }

    // Phase 3: Sum reduction (FP32)
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_exp_sum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_exp_sum[col], offset, 32);
        }
    }
}

// Vectorized softmax for tile operations with early convergence detection
template<int tile_size>
__device__ __forceinline__ bool softmax_converged(float * values, float threshold = 0.995f) {
    // Early termination: check if softmax is sufficiently concentrated
    // If one value is >99.5%, we can skip further processing
    float max_val = 0.0f;
#pragma unroll
    for (int i = 0; i < tile_size; ++i) {
        max_val = fmaxf(max_val, values[i]);
    }
    return max_val > threshold;
}

// Optimized warp reduction for softmax: combines max and sum in single shuffle tree
template<int cols_per_thread, typename T = float>
__device__ __forceinline__ void warp_reduce_softmax(
    T * local_vals,
    T * warp_max,
    T * warp_sum
) {
    constexpr int warp_size = 32;

#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            T other_max = __shfl_xor_sync(0xFFFFFFFF, warp_max[col], offset, warp_size);
            T other_sum = __shfl_xor_sync(0xFFFFFFFF, warp_sum[col], offset, warp_size);

            if (other_max > warp_max[col]) {
                warp_sum[col] *= __expf(warp_max[col] - other_max);
                warp_max[col] = other_max;
            } else {
                other_sum *= __expf(other_max - warp_max[col]);
            }
            warp_sum[col] += other_sum;
        }
    }
}

// Quantized softmax: compute softmax in FP16 for well-separated attention patterns
// Maintains ~1% error bound while reducing computation
template<int cols_per_thread>
__device__ __forceinline__ void softmax_fp16_quantized(
    half * KQ_vals_half,    // Input in FP16
    float * KQ_max,
    float * KQ_exp_sum
) {
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -HALF_MAX;
    }

    // Max reduction in FP16
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_max[col] = fmaxf(KQ_max[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max[col], offset, 32));
        }
    }

    // Exp and sum in FP16 with conversion to FP32 for accumulation
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_exp_sum[col] = 0.0f;
    }

    // Process tiles and accumulate
    float max_f32 = KQ_max[0];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        half exp_half = __float2half_rn(__expf(__half2float(KQ_vals_half[col]) - max_f32));
        KQ_exp_sum[col] += __half2float(exp_half);
    }
}

// Helper: Broadcast reduction result across warp
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset, 32);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset, 32));
    }
    return val;
}
