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
K/V, one query token, a mask, no sinks, no ALiBi, D >= 128, GQA ratio >= 2,
and KV length >= 32768 (`GGML_CUDA_FATTN_VEC_GQA=1` forces it at any depth, `=0`
disables; the auto threshold was raised from 4096 until the A/B is measured).

The width the gate returns is the **smallest instantiated `ncols2` that holds the
whole ratio** (2, 4, 6 or 8), not the largest power-of-two that divides it. The
divisor rule sent this model's ratio of 6 to width 2 - three passes over K/V,
a third of the amortization the whole item is for - even though the kernel tiles
as `ceil(gqa_ratio / ncols2)` and masks leftover columns, so a non-dividing width
was always correct. `ncols2 = 6` is instantiated for the exact fit. Before trusting a number:

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

## 4. A heavier KV codec, paid for by the packing in 1 (more resident context, same VRAM)

**Status: gated. `EXPERIMENTS=kv_codec_gate` decides whether to build it; no kernel yet
and none should be written until that run reports.**

**The inversion.** `q4_0` KV is 18 KiB/token against `q8_0`'s 34, and still measures
**-8%/token at depth**. The bench README records that as "the dequant costs more than the
bytes save", which is true but incomplete: on the vec path the dequant is paid **once per
query head** and the bytes are saved **once**. With `gqa_ratio` 6 the saving is counted
once against a cost counted six times. That is the same accounting behind the 51 ms KV
term in 1 - it is not a fact about `q4_0`, it is a fact about the kernel.

Item 1 divides the dequant term and leaves the byte term whole. So the packing is not only
a bandwidth fix; it is a **budget**. A codec that is several times more expensive to decode
than `q8_0` is unaffordable today and roughly free once each K/V tile is decoded once per
block instead of once per query head. The heavier the codec, the more the amortization is
worth - the opposite of the usual direction.

**What it buys, in this box's units.** 16 of 64 layers carry KV
(`full_attention_interval` 4). At `q8_0` that is 9.1 GB per 262K slot, so roughly three
deep slots fit the free VRAM. At an effective ~2.5 bits it is about 3 GB per slot, so
roughly nine. Same card, same weights: **the eight-slot configuration at a true 256K each
instead of a shared pool split eight ways.** That is the whole point - not tokens per
second, tokens *resident*.

**Why it is not free elsewhere.** On silicon whose MMA path already batches GQA, nobody
pays the per-query-head dequant, so nobody has this budget to spend and the trade looks
unattractive. This card is on the vec path *because* the KV is quantized - quantizing KV
selects the kernel that punishes quantized KV. Item 1 breaks that loop, and what is left
over is the thing to spend.

**Composes with what is already here.** `--kv-mean-center` subtracts a per-(head,channel)
bias before `Q4_0` K quantization and is softmax-invariant - it exists to buy back exactly
the fidelity an aggressive K codec gives up. It has no measured row either; it should be
graded in the same sitting.

**Change (only if the gate confirms).** A codebook/grouped codec for K and V decoded once
per block into shared memory inside the packed vec kernel from 1, dispatched on the same
predicate (`ggml_is_quantized(K->type) && Q->ne[1] == 1 && gqa_opt_applies`). Correctness
against CPU via `test-backend-ops -o FLASH_ATTN_EXT` before any timing, as in 1.

**Prove before writing:** `EXPERIMENTS=kv_codec_gate MODEL=... bash
benches/cmp170hx-3060/run-open-items.sh`. Four cells: `q8_0`/`q4_0` x
`GGML_CUDA_FATTN_VEC_GQA` 0/1. The claim lives in the **gap between the two KV types**,
not in any single cell - if the gap does not improve when the packing is on, the dequant
was never the binding term, no codec will help, and this item closes.

**Note on 1's packing factor (fixed).** The width ladder used to return the largest
power-of-two that *divided* the ratio, so this model's 6 took the `% 2` rung: 2 of 6 heads
per block, every K/V byte read three times, a third of the amortization this item depends
on. That was a selection bug, not a kernel limit - the kernel tiles the ratio as
`ceil(gqa_ratio / ncols)` and masks the leftover columns, so a width that does not divide
the ratio was always correct. The ladder now returns the smallest instantiated width that
*holds* the ratio, and `ncols2 = 6` is instantiated, so 6 packs in one block with no idle
lanes. Grade the codec against that, not against the old behaviour.

## Order

The profile in 3 first: it is one command and it decides whether the
single-stream residual is launches (fuse), or something else. Then 2: it
changes what the retrained head is worth and has a measured harness (the
NO_MMVQ sweep). 1 is written; its A/B is on the open-items list. 4 is gated
behind 1's A/B and behind its own `kv_codec_gate` run - it is the largest
prize here and the one most likely to be closed by a single measurement, so
it is cheap to settle and expensive to assume.
