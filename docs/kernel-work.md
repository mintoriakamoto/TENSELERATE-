# CUDA kernel work that is still open - what, where, how to prove it

Three changes the measurements point at. None is written; each needs the card
in the loop, so this is the spec a session with `nvcc` and the 170HX works
from. Measured baselines: `benches/cmp170hx-3060/README.md`.

## 1. GQA-aware vector attention for quantized KV (the 51 ms KV term at 262K)

**Status: written 2026-09-07, compiled by CI for sm_80, NOT yet run on the card.**
`fattn-vec.cuh` gained a `gqa_pack` kernel variant: the block's columns are the
Q heads sharing one K/V head at a single token, so K/V are read and
dequantized once per block; `launch_fattn<D, 1, ncols2>` supplies the
(sequence, K/V head, gqa tile) grid the MMA kernel already uses. The host gate
`ggml_cuda_fattn_vec_gqa_cols` (fattn-common.cuh) turns it on for quantized
K/V, one query token, a mask, no sinks, no ALiBi, D >= 128, GQA ratio even,
and KV length >= 4096 (`GGML_CUDA_FATTN_VEC_GQA=1` forces it at any depth, `=0`
disables). Before trusting a number:

```
GGML_CUDA_FATTN_VEC_GQA=1 build/bin/test-backend-ops -o FLASH_ATTN_EXT -b CUDA0   # correctness vs CPU
GGML_CUDA_FATTN_VEC_GQA=0 llama-bench -m $GGUF -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 -p 262144 -n 32
GGML_CUDA_FATTN_VEC_GQA=1 llama-bench -m $GGUF -ngl 999 -fa 1 -ctk q8_0 -ctv q8_0 -p 262144 -n 32
```

**Symptom.** Single-stream decode at 262K depth is 12.4 tok/s with q8_0 KV; the
KV read costs 51 ms where the bytes say ~11. q4_0 is 8% slower still.

**Where.** `ggml/src/ggml-cuda/fattn.cu`, `ggml_cuda_get_best_fattn_kernel`.
On sm_80 (`turing_mma_available`, not Ada) with quantized K/V and one query
column the dispatch returns `BEST_FATTN_KERNEL_VEC`. The vec kernels
(`fattn-vec*.cuh`) read the KV head once per *query head*; with GQA ratio 6
the same K/V bytes are read six times. `gqa_opt_applies` is computed but the
vec branch for quantized KV does not use it. The MMA path
(`fattn-mma-f16.cuh`) batches the ratio into `ncols2` but takes only f16 K/V.

**Change.** Either (a) a vec kernel variant that maps `gqa_ratio` query heads
onto one block, dequantizing each K/V tile once into registers/shared memory
and dotting all `gqa_ratio` Q vectors against it (the softmax state is per
query head; the shared part is the K/V load and dequant), dispatched when
`ggml_is_quantized(K->type) && gqa_opt_applies && Q->ne[1] == 1`; or (b)
dequantize K/V tiles to f16 on the fly for the MMA kernel (upstream has
discussed this; check `fattn-mma-f16.cuh` upstream before writing it).

**Prove.** `llama-bench -p 262144 -n 32` q8_0 vs f16 KV before and after. The
target is the f16 number (~22 tok/s predicted) at q8_0's memory, and no change
at depth 0 (33.5).

## 2. MMQ at small width (the ~55 ms floor)

**Symptom.** With `GGML_CUDA_NO_MMVQ=1` a decode step costs ~55 ms whether the
batch is 2 or 16 columns; dp4a costs 18.5 + 11.5 per extra column. Crossover
at width 3-4. MTP depth 1 (width 2) on MMQ is 30.6 tok/s against 46.6 on dp4a.

**Where.** `ggml/src/ggml-cuda/mmq.cuh`. `launch_mul_mat_q` picks `mmq_x`
from 8 upward in steps of 8 to minimise tile count (`mmq_x_best`), so a
width-2 batch already runs the 8-column tile. The floor is therefore not
padding to 128; it is the per-tile pipeline: every block loads and converts a
full weight tile (`load_tiles_*`, MMQ_ITER_K = 256) into shared memory for
`MMQ_NWARPS` = 8 warps, then issues MMAs whose N dimension is mostly empty.
At width 2 the kernel is dequant/shared-memory bound, the same work MMVQ does
in registers with no round trip.

**Change.** A small-N specialisation: for `ne11 <= 8`, a tile config with
fewer warps per block (2-4), no shared-memory staging of the activations, and
the weight tile streamed straight from global memory through the dequant into
the MMA fragments (the A operand), so the block's cost tracks bytes read, not
tile size. Keep the existing path for `ne11 > 8`. The dispatch point is
already there: `ggml_cuda_mmvq_max_batch()` (this fork, `mmvq.cuh`) decides
which widths reach MMQ.

**Prove.** The NO_MMVQ MTP sweep (n-max 1..5) should come out monotone with
width and beat dp4a from width 2: target ~20-25 ms at width 2 (from 55), which
puts a retrained depth-3 head near 85-95 tok/s instead of ~70
(`docs/mtp-realign-davidau.md`).

## 3. The GDN block at batch 1 - what the arithmetic says before anyone writes a kernel

**Claim to test (from a box session):** a specialised n_tokens = 1 GDN kernel
with the state in shared memory and `mma.sync` for S^T k would take single
stream from 33 to 45-55 tok/s.

**What the current kernel does** (`ggml/src/ggml-cuda/gated_delta_net.cu`):
one warp per state column, the 128x128 per-head state held in registers
(4 floats per lane), two warp reductions per token (S^T k, then S^T q after the
rank-1 update). Grid = heads x sequences x S_v/4 blocks of 128 threads. At
n_tokens = 1 that is already the minimal form of the recurrence: a rank-1
update and two matvecs per head.

**The bytes and flops:** state per layer = H x 128 x 128 x 4 B = 2-4 MB
(H = 32-64); read once, write once, 48 GDN layers -> ~0.2-0.4 GB per token,
~0.3-0.5 ms at the measured bandwidth. Flops are ~3 x 128 x 128 x H per layer,
microseconds. A matvec has no use for `mma.sync`: tensor cores need an N
dimension, and at one token there is none. The recurrence math cannot be
where the ~11 ms residual (29.9 ms step - 18.5 ms weight read) goes.

**Where it plausibly goes:** kernel count. A GDN block in this model is a
conv1d, several norms, SiLU/gating elementwise ops, the q/k/v/z/b/a
projections, the recurrence, an output norm and gate - roughly 15-25 launches
per layer, ~1000 per token across 48 layers, plus the attention layers. Even
under CUDA graphs (reused = 247, so graphs are on) each small kernel costs a
few microseconds of latency and tail; 1000 x 5 us = 5 ms is the right order
of magnitude, and it does not shrink with a faster recurrence. This is also
consistent with the width sweep: the residual amortises across sequences
because those same launches then carry N tokens each.

**So the fix is fusion, not tensor cores:** fold the conv + SiLU + gating +
norm chain of the GDN block into one or two kernels per layer (and the
attention-layer elementwise chain likewise), cutting launches per token by
several hundred. Expected: a few ms off the 29.9 ms step -> ~38-42 tok/s
single stream before speculation, and the same saving under MTP. 45-55 from
the recurrence alone is not supported by the byte/flop count.

**Prove before writing:** `nsys profile --stats=true llama-bench -n 16`
(or `-b 1` in llama-cli) and read two numbers: kernels per token and the
summed time of kernels under 20 us. If the sum of the small kernels is not
several ms, this whole section is wrong and the residual is somewhere else
(the attention path, the MTP verify columns, or a host-side gap).

## Order

The profile in 3 first: it is one command and it decides whether the
single-stream residual is launches (fuse), or something else. Then 2: it
changes what the retrained head is worth and has a measured harness (the
NO_MMVQ sweep). 1 is written; its A/B is on the open-items list.
