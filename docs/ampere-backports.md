# What sm_90 and sm_120 have that sm_80 can also have

The CMP 170HX is a GA100 at sm_80. Hopper (sm_90) and Blackwell (sm_100/120)
carry a lot that this card physically cannot run, and a few things it can. This
page separates the two, because "backport the new architecture's work" is only
a real plan for the second group.

## Cannot be backported: instructions the silicon does not have

| feature | arch | why not |
| --- | --- | --- |
| `wgmma` warpgroup MMA | sm_90 | new instruction class; sm_80 has `mma.sync m16n8k16` and nothing wider |
| TMA (`cp.async.bulk`, tensor maps) | sm_90 | sm_80's ancestor is `cp.async`, already used in `fattn-mma-f16.cuh` and `cp-async.cuh` |
| thread-block clusters, distributed shared memory | sm_90 | no cluster hardware on sm_80 |
| `setmaxnreg` (dynamic register reallocation) | sm_90 | no |
| `tcgen05`, FP4/FP6 datatypes | sm_100/120 | no |

The fork's own `mmq-hopper-q1.cu` is in this group: it is gated to Hopper and
never executes on this box.

## Can be backported: scheduling gated on compute capability

These are *decisions*, not instructions. The code compiles and runs on sm_80;
only a heuristic tuned on other cards excludes it. The 170HX has an unusual
balance (FP16 tensor cores restored at 162-170 TFLOPS, dp4a throttled, 1493
GB/s HBM2e, PCIe Gen2 x4), so inheriting another card's heuristic is a guess.
Each is opt-in, off by default, and exists to be measured.

### 1. The Ada+ flash-attention kernel choice (`GGML_CUDA_FATTN_ADA_GATE=1`)

`fattn.cu` picks between the vector kernel and the MMA kernel partly by arch:

```
if (cc >= GGML_CUDA_CC_ADA_LOVELACE) { if (Q->ne[1] <= 2) return VEC; }
else                                 { if (Q->ne[1] == 1) return VEC; }
```

With quantized KV, Ada and newer keep batch **width 2** on the vector kernel;
Ampere drops to the MMA kernel at width 2. Width 2 is exactly an **MTP depth-1
verify pass** - the hot path of every speculative step we measure. The vector
kernel has a `ncols == 2` instance (`fattn-vec.cuh`) with no arch-specific
instructions in it, and this fork's GQA-packed path lives there too. Whether
it wins at width 2 on this card is a measurement, not a deduction.

`GGML_CUDA_FATTN_ADA_GATE=1` makes Ampere take the Ada+ branch.

### 2. Stream-k work decomposition (`GGML_CUDA_FATTN_STREAM_K=1|0`)

`fattn-common.cuh`:

```
use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || amd_wmma_available(cc)
               || tiles_efficiency_percent < 75;
```

Stream-k splits the KV loop across SMs and fixes up the partial results. It is
scheduling, and it runs on sm_80. At batch 1 the efficiency term already turns
it on (one output tile against ~108 SMs). The interesting region is **8-32
slots**, where the tile count rises, efficiency crosses 75%, and Ampere loses
stream-k while Ada and newer keep it. That is the width where our aggregate
goes flat.

`GGML_CUDA_FATTN_STREAM_K=1` forces it on, `=0` off, overriding the heuristic.

### 3. Not gated by arch, just unused: L2 persistence (an sm_80 feature)

`cudaAccessPolicyWindow` / `cudaStreamSetAttribute` let a kernel pin a byte
range in L2. Ampere introduced it; **nothing in `ggml-cuda` uses it**. GA100
has 40 MB of L2. The GDN recurrent state is ~150 MiB per sequence in total but
only ~3.2 MiB per layer, and every layer reads and writes it every single token.
Pinning the live state window while the weight stream flows past it is the kind
of thing L2 persistence exists for. Untried; issue #71.

### 4. The MMVQ per-arch table inherits Turing's row for Ampere

`get_mmvq_mmid_max_batch` returns `MMVQ_MAX_BATCH_SIZE` unconditionally for Ada
and newer, uses a Turing table for Turing through Ampere, and a Pascal table
below. So Ampere's MUL_MAT_ID thresholds are Turing's, chosen on a card whose
dp4a is not throttled. This fork already overrides the plain-matmul side with
`GGML_CUDA_MMVQ_MAX`; the MUL_MAT_ID side is untuned and only matters for MoE,
which this model is not.

## How to measure (release binary, on the box)

```bash
# 1. the Ada gate, at the MTP verify width. Run with MTP depth 1 so Q->ne[1]==2.
llama-bench -m MODEL.gguf -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -r 3
GGML_CUDA_FATTN_ADA_GATE=1 llama-bench -m MODEL.gguf -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -r 3
#    then the same pair through the server with --spec-type draft-mtp --spec-draft-n-max 1

# 2. stream-k at the widths where the heuristic drops it
for n in 8 16 32; do
  GGML_CUDA_FATTN_STREAM_K=0 ...   # forced off
  GGML_CUDA_FATTN_STREAM_K=1 ...   # forced on
done
#    use benches/cmp170hx-3060/fork-vs-upstream-ab.sh with NP=$n; it prints the
#    concurrency factor next to the aggregate

# 3. token identity for anything that wins
test-backend-ops -o FLASH_ATTN_EXT       # with and without the env var set
```

Both flags change *which correct kernel runs*, not the math, so a win is free.
Record the rows in `benches/cmp170hx-3060/README.md` next to these predictions:
the Ada gate is the one with a named mechanism behind it (the verify pass is
width 2 and currently leaves the vector kernel), so it is the one to run first.
