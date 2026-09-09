# The missing middle: a read-once, N-column decode kernel

**Status: a proposal with a pre-registered falsification test.** Nothing here is
measured on the box yet. `benches/cmp170hx-3060/width-bytes.sh` is the ten-minute
experiment that decides whether to build it, and one of its two outcomes closes
this document rather than opening it.

## The observation

At batch 1 the card must read all 15.4 GB of weights to emit one token. That is
the physics floor: 15.4 GB / 1493 GB/s = 10.3 ms = 95 tok/s, and 17.3 ms at the
890 GB/s the weight read actually achieves (`docs/physics.md`).

At batch N it must **still read 15.4 GB - once** - and multiply it against N
activation columns. The extra columns are nearly free in bandwidth terms. This
is the whole reason batching is the throughput lever, and why aggregate
throughput for N concurrent Hercules agents ought to approach N x the
single-stream rate rather than flattening.

It does not, and the two paths fail in different ways
(`benches/cmp170hx-3060/README.md`):

| path | width 1 | per extra column | width 8 | shape |
|---|---|---|---|---|
| MMVQ (dp4a) | 18.5 ms | +11.5 ms | ~99 ms | linear in N |
| MMQ (tensor core) | - | ~5.6 ms | **~55 ms, flat N=2..16** | flat in N |
| physics, achieved BW | 17.3 ms | ~0 | **~17.3 ms** | flat in N |

Two separate diagnoses fall out of the shapes:

- **MMVQ is linear in N with a slope two-thirds of its intercept.** A kernel
  that held the weights and looped over columns would be flat. A slope that
  size says it re-streams most of the weights for every column. It is a vector
  kernel doing a vector kernel's job N times.
- **MMQ is flat in N, so it does amortize the columns** - and still sits at
  3.2x the achieved-bandwidth weight read. Flatness rules out per-column
  re-reads; it does not distinguish "moves 3x the bytes" from "moves the right
  bytes but is latency- or occupancy-bound".

That second ambiguity is the only thing standing between here and a decision,
and it is one `ncu` metric wide.

## Why this is the number that matters

Hercules runs many agents, not one. The measured aggregate is 134 tok/s at
N=32. If a decode step at width 8 cost what the weight read costs - ~17.3 ms -
eight tokens per step is ~460 tok/s. The gap is roughly **3x on aggregate
throughput**, which is larger than every other open item on this box combined:

| lever | best case | status |
|---|---|---|
| read-once N-column kernel | ~3x aggregate | this document, undecided |
| CUDA-graph residual (#53) | up to 11.4 ms/token at batch 1 | 30-second A/B, unrun |
| trellis quant (EXL3-style) | ~27% fewer weight bytes | changes numerics |
| MTP depth/sampling | 33.5 -> 46.2 tok/s single stream | measured, shipped |

It is also the only one that scales with the number of agents rather than
helping a single stream.

## The kernel

A Q4_K decode GEMM specialised for **N = 2..16**, sitting between upstream's two
existing modes. Upstream has a vector kernel (N=1, re-streams for more) and a
tiled kernel (efficient at large N). The middle - many small concurrent
sequences on one card - is empty, and it is exactly where a multi-agent server
lives.

Shape, per threadblock:

1. Quantize the N activation columns to `q8_1` once per step, as MMVQ already
   does, into shared memory. At N=16 and K_tile=256 that is 256 x 16 x 1 byte
   plus scales - a few KB. It never leaves shared memory again.
2. Stream the Q4_K super-blocks for a tile of M output rows from global memory,
   coalesced, **exactly once**. Dequantize in registers.
3. For each weight block, run N `dp4a` chains against the N columns already
   resident in shared memory. Accumulate into N per-row registers.
4. Never pad N. The tile is over M and K; N is a register-resident dimension,
   so width 3 does three columns of work, not sixty-four.

`dp4a` rather than FP16 tensor cores because `docs/rig-cmp170hx-3060.md:24`
records INT8/dp4a as uncrippled on this card - there is no throttled-integer
story here to route around.

### Does the arithmetic close?

At N=8 the kernel does 27B parameters x 8 columns x 2 flops = ~432 GFLOP per
step. At the card's INT8 rate that is on the order of 1-2 ms, against a 17.3 ms
weight read. **Still firmly memory-bound at N=8**, which is the answer you want:
it means the design target is the weight read itself, and the compute the extra
columns add disappears underneath it. The arithmetic intensity only becomes
interesting somewhere past N=32, which is beyond this kernel's range.

Predicted: **~18-22 ms at width 8**, against ~55 ms measured for MMQ.

## What would falsify this

Run `benches/cmp170hx-3060/width-bytes.sh`. It carries four pre-registered
predictions; B2 is the decisive one:

- **MMQ at N=8 reads ~1x the weight bytes** - the traffic is already minimal.
  The 55 ms is latency or occupancy, this kernel wins nothing, and this document
  should be closed as refuted. Chase occupancy instead.
- **MMQ at N=8 reads ~3x the weight bytes** - the excess is real and a
  read-once kernel is worth the days it costs.

Either result is worth having. The second one is a 3x; the first one saves the
week that would have been spent discovering it the expensive way.

## Order of work

1. `width-bytes.sh` on the box. Ten minutes. Decides everything below.
2. Only if B2 comes back high: the kernel, in a fork-owned file
   (`ggml/src/ggml-cuda/tenselerate-mmq-narrow.cu`), env-gated and off by
   default, graded against MMQ at widths 2, 4, 8, 16 with bit-exactness checked
   against the reference oracle first.
3. Re-run the N=32 aggregate and grade it against the ~460 tok/s ceiling.

Step 1 costs ten minutes and step 2 costs days, which is the correct ratio for a
proposal whose central claim is still unmeasured.
