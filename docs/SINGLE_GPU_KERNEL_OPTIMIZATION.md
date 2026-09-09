# Single-GPU Kernel Optimization Strategy (CMP170HX A100)

## Executive Summary

With 1 GPU unlocked via cmpunlocker2, optimize for **latency-critical prefill** (264K context) and **throughput-critical decode**. Single GPU eliminates all-reduce overhead, but memory pressure is acute.

**Target improvements:**
- Prefill: 51ms → 38ms GQA (25% latency reduction)
- Decode: Sustained 1200+ tok/s with K=4 MTP (speculative)
- Memory efficiency: 75GB usable → aggressive quantization + paging

**Kernel fusion strategy:** Combine GQA + rotary + output projection into single fused kernel to eliminate intermediate tensor writes to HBM.

---

## 1. Reverse-Engineered GQA Fusion Pattern

### Current Pipeline (51ms measured @ 262K context)
```
1. Embed input              (batch=1, seq_len=262144)
2. RoPE (rotary embedding)  (parallel to embed, 2ms overhead)
3. QKV projection           (3 separate matmuls: Q, K, V)
4. Attention compute        (GQA: reduce K/V heads)
5. Attention output         (single head reduce)
6. Output projection        (MLP/dense)
Total: 51ms
```

### Physics Floor
```
Model: Qwen3.6-27B-AWQ (quantized, ~13.6GB resident)
Compute: 55.3 TFLOPS FP32 (A100 SM80)

Bottleneck analysis:
- HBM2e bandwidth: 1935 GB/s (vs 13.6GB model = ~200 FLOPS:byte ratio)
- Attention is compute-bound on Q@K^T (FLOPs >> memory)
- KV cache access is memory-bound (GQA helps: fewer K/V heads)

Expected latency floor:
  - Compute: 27B model × sparse_attention_flops / 55.3e12 ≈ 13.4ms
  - Memory: KV load (8 heads × 262K × 16 bytes × 27 layers) / 1935 GB/s ≈ 4.1ms
  Total floor: ~17.7ms (achievable with perfect fusion, zero overhead)

Measured 51ms vs floor 17.7ms = 2.88× headroom
Goal: 38ms = 2.15× headroom (realistic with fusion)
```

### Fusion Strategy: Single Fused Kernel

Combine QKV→Attention→Output into **one kernel launch** to eliminate HBM round-trips:

```cuda
// Pseudo-code: fused_gqa_attention_kernel
__global__ void fused_gqa_attention(
    float *embed,      // [batch, seq, hidden]
    float *q_proj, *k_proj, *v_proj, *out_proj,  // Weights
    float *rope_cos, *rope_sin,  // Rotary embeddings (precomputed)
    float *kv_cache,   // GQA cache: K/V reduced heads
    float *output,     // Final output
    int seq_len, int hidden, int num_heads, int kv_heads)
{
    // Each thread block: one attention head (or partial attention)
    // Shared memory: load K/V for this chunk from HBM ONCE
    // Compute: Q @ K^T, softmax, @ V, all in shared mem
    // Write output once

    __shared__ float shared_k[CHUNK_K][HIDDEN];  // K for this chunk
    __shared__ float shared_v[CHUNK_K][HIDDEN];  // V for this chunk
    __shared__ float shared_attn[CHUNK_Q][CHUNK_K];  // Attention scores

    // 1. Load K/V from cache (or compute on-the-fly if training)
    // 2. Load Q for this chunk
    // 3. Apply RoPE to Q and cached K (in registers)
    // 4. Compute Q @ K^T → shared_attn
    // 5. Softmax(shared_attn)
    // 6. Multiply by V → output
    // 7. Write output to HBM

    // Key: All intermediate tensors stay in shared mem or registers
    // No HBM writes except final output
}
```

**Expected latency: 38ms** (vs 51ms current)
- Kernel launch overhead: 0.5ms (vs 11ms with separate kernels)
- Memory stalls: only on final write
- Compute utilization: >85%

### Implementation Checklist
- [ ] Extract RoPE computation from separate kernel → fused
- [ ] Merge QKV projections into single GEMM with split output
- [ ] Implement tiled attention (Triton or CUTLASS template)
- [ ] Test with nsys profiling (target: 38ms ±0.5ms)
- [ ] Benchmark token/sec on decode (should see 10-15% throughput gain)

---

## 2. MMQ (Small-Batch Quantization) Optimization

### Current Bottleneck (55ms for batch=1)
```
MMQ kernel: matrix multiply with quantized weight matrices
Measured: 55ms @ seq_len=1, batch=1 (single decode token)
Physics floor: 38ms (with tile optimization)
Overhead: 11ms (GDN block launch + synchronization)
```

### Block-Level GDN Hybrid Strategy

Tensor: shape [batch=1, seq=1, hidden=11008]
Kernel split:
1. **Tile A** (fine-grained): 8×8 → kernel A0, A1, ...
2. **Tile B** (block-level): 1024×1024 → kernel B0, B1, ...
3. **GDN gate**: Route to appropriate kernel based on input distribution

```cuda
// GDN gate: measure variance to decide kernel path
float compute_gdn_gate(float *tile_a, int tile_size) {
    // Compute variance of A (quantization-aware)
    float mean = 0, var = 0;
    #pragma unroll
    for (int i = 0; i < tile_size; i++) {
        float x = tile_a[i];
        mean += x / tile_size;
    }
    #pragma unroll
    for (int i = 0; i < tile_size; i++) {
        float x = tile_a[i];
        var += (x - mean) * (x - mean) / tile_size;
    }

    // Route based on variance
    if (var < LOW_VAR_THRESHOLD) {
        return kernel_a;  // Fine-grained (better for low-variance)
    } else {
        return kernel_b;  // Block-level (better for high-variance)
    }
}
```

**Expected result: 8ms per tile** (vs 11ms current GDN overhead)
- Eliminates 3ms of unnecessary synchronization
- Better cache locality within tile

### Implementation Checklist
- [ ] Implement variance-based GDN gate in MMQ kernel
- [ ] Benchmark tile kernel A vs B on representative AWQ matrices
- [ ] Profile nsys: target <8ms per tile
- [ ] Integrate into decode loop

---

## 3. Speculative Decoding (MTP) Optimization for Single GPU

### Challenge
- K=4 MTP is aggressive (66% acceptance)
- Wasted compute on rejected predictions
- Memory bandwidth for parallel batch of K drafts

### Single-GPU Strategy: Serial Accept-Reject with Pipelined Prefill

```
Iteration 1:
  Prefill: "Hello world how are" → hidden state H_4
  Draft:   Run decoder 4× in parallel on GPU (K=4 heads)
           Compute logits: L_0, L_1, L_2, L_3 (one per token)
  Verify:  Compare L_0..L_3 vs target model logits
           Accept/reject: L_i ≈ L_target → accept, move to next token

Iteration 2:
  Prefill: Compute H_5 from H_4 (single token, small)
  Draft:   4× parallel decoder
  Verify:  Compare + accept/reject
```

**Key insight:** Single GPU doesn't parallelize K drafts across different GPUs (no all-reduce), so we run them serially with full compute utilization.

### Memory Optimization for MTP
- Keep decoder cache paged (only hot K/V heads in HBM)
- Swap cold heads to system RAM (PCIe 16GB/s)
- Accept/reject decision < 1ms → minimal PCIe stall

```cuda
// Pseudo-code: paged kv cache swap
struct PagedKVCache {
    float *hot_kv;       // Current head K/V on HBM
    void *cold_kv_file;  // mmap to system RAM (cold heads)
    
    void prefetch_next_head(int head_id) {
        // Async copy: cold_kv_file → HBM for next head
        // Overlaps with current head decode
    }
};
```

**Expected throughput improvement:**
- Base: 1024 tok/s (K=1, single draft)
- MTP K=4: 2560 tok/s (66% × 4 + baseline)
- With paging: sustain 2500+ tok/s (no swap stalls)

### Implementation Checklist
- [ ] Implement paged KV cache (mmap + async prefetch)
- [ ] Add accept/reject logic with 66% target acceptance
- [ ] Benchmark: measure accept rate + throughput
- [ ] Profile memory bandwidth (target: <20% PCIe usage)

---

## 4. Memory Pooling & Fragmentation Reduction

### Single GPU Pain Point
- KV cache grows to 75GB usable (80GB - OS/system overhead)
- Multiple models/quantizations in parallel → fragmentation
- Malloc/free overhead on every context switch

### Strategy: Pre-Allocated Ring Buffer

```cuda
struct MemoryPool {
    float *ring_buffer[RING_SIZE];  // Pre-allocated, fixed chunks
    int head = 0, tail = 0;         // Circular pointers
    
    float* allocate(size_t size) {
        // O(1): return next chunk in ring
        float* ptr = ring_buffer[head];
        head = (head + 1) % RING_SIZE;
        return ptr;
    }
    
    void deallocate(float* ptr) {
        // O(1): rotate tail forward
        tail = (tail + 1) % RING_SIZE;
    }
};
```

**Benefits:**
- Zero fragmentation (fixed chunk size)
- No malloc/free overhead
- Predictable memory layout (better prefetching)
- ~5-8% latency reduction on context switches

### Pre-Allocation Sizing
```
For Qwen3.6-27B + 262K context:
- KV cache per layer: 256 × 262144 × 4 bytes × 27 layers ≈ 72GB
- Overhead (~4%): 3GB
- Pre-allocate 32 chunks of 2.4GB each → covers 4 parallel contexts
- Total: ~77GB (fits in unlocked 80GB A100)
```

### Implementation Checklist
- [ ] Implement ring buffer memory pool
- [ ] Pre-allocate on startup (measure time)
- [ ] Benchmark context switch latency (target: <1ms)
- [ ] Test fragmentation on 48-hour continuous inference

---

## 5. Attention Windowing (SWA) for 262K Context

### Current Implementation
- `tenselerate-attn-window.cpp` already implements sliding window attention
- Reduces KV cache size from 75GB → ~15GB (for 4K window)
- Trade-off: context awareness limited to recent tokens

### Hybrid Window Strategy: Sink Tokens + Recent Context

```
Layout for 262K context:
[Sink tokens (128)] + [Recent context (4096)] = 4224 total attention

Tokens 0-127:     Sink (high importance, attend all)
Tokens 128-258K:  Evicted
Tokens 258K-262K: Recent (keep full attention)

Attention pattern:
- Query @ Sink: full attention (128 × 4224 sparse)
- Query @ Recent: full attention (4096 × 4096 dense)
- Total KV: only 4224 heads tracked
```

**Memory savings:**
- Full 262K: 75GB
- Windowed 4K: 4GB
- Hybrid sink+recent: 6GB (good trade-off between context and memory)

### ROPEFreq Tuning for Long Context

Rotary embeddings need adjustment for frequencies when using windowing:

```python
# From llama-hparams.cpp
rope_freq_base_train_swa = rope_freq_base_train  # Assume same as full attention
# Better: scale based on effective context window
rope_freq_base_swa = rope_freq_base_train * sqrt(effective_window / full_context)
```

### Implementation Checklist
- [ ] Profile current SWA: measure attention matrix size
- [ ] Tune sink token count (currently 4, try 64-256)
- [ ] Benchmark: accuracy loss on long-context tasks
- [ ] Compare: windowed vs full attention on perplexity

---

## 6. Profiling & Validation Plan

### Tools
1. **nsys** (NVIDIA profiler)
   ```bash
   nsys profile --stats=true --output=profile_%h_%t ./tenselerate
   ```
   Measure: kernel runtime, memory bandwidth, launch latency

2. **NCU** (NVIDIA Compute Profiler)
   ```bash
   ncu --set default --kernel_name "regex:gqa|mmq|attn" ./tenselerate
   ```
   Measure: SM throughput, L2 cache hit rate, memory BW utilization

3. **Custom benchmarks**
   ```bash
   # Measure end-to-end prefill latency
   time ./tenselerate --model qwen-3.6-awq --tokens 262144 --profile
   
   # Measure decode throughput
   ./tenselerate --model qwen-3.6-awq --decode-only --count 100
   ```

### Acceptance Criteria

| Optimization | Before | Target | Measurement |
|---|---|---|---|
| GQA + fusion | 51ms | 38ms | nsys kernel time |
| MMQ tile | 55ms | 8ms/tile | nsys kernel time |
| MTP K=4 | 1024 tok/s | 2560 tok/s | tokens/sec (decode) |
| Memory pool | N/A | <1ms | context switch latency |
| SWA hybrid | 75GB | 6GB | peak memory usage |

---

## 7. Deployment Checklist for Single GPU

- [ ] Compile with `-DCMAKE_CUDA_ARCHITECTURES=80 -DENABLE_TENSOR_CORE_INTRINSICS=ON`
- [ ] Unlock GPU memory with cmpunlocker2 (verify 80GB available)
- [ ] Verify CUDA 12.2+, cuDNN 8.9.6+ installed
- [ ] Run all micro-benchmarks (GQA, MMQ, MTP, memory pool)
- [ ] End-to-end prefill benchmark: 262K tokens < 40ms
- [ ] Sustained decode: >2400 tok/s for 1 hour continuous
- [ ] Memory stability: no OOM on 75GB workload
- [ ] Optional: Enable SVMI memory management (handles overflow gracefully)

---

## References

- NVIDIA A100 Tensor Core Performance: https://www.nvidia.com/en-us/data-center/a100/
- Flash Attention v2: https://arxiv.org/abs/2307.08691
- GQA (Group Query Attention): https://arxiv.org/abs/2305.13245
- Speculative Decoding: https://arxiv.org/abs/2211.17192
- Paged Attention (vLLM): https://arxiv.org/abs/2309.06180
