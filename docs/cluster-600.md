# 600 tok/s on 10x CMP 170HX: what the engine already does, and what the numbers allow

This maps the TENSELERATE optimization spec onto this engine and gives the measured-
arithmetic answer to the target. Two things up front, because they change the plan:

1. **The spec is written for a PyTorch engine** (`tenselerate/engine/generation.py`,
   `torch.utils.cpp_extension`, `.item()`/`torch.cuda.synchronize()`, NVTX ranges).
   This repo is llama.cpp: C++/CUDA, no PyTorch in the inference path. Most spec
   items are therefore not "implement X" but "X already exists, here is its name".
2. **600 tok/s aggregate and 256K context per sequence are individually reachable
   and mutually exclusive at the same time** on this hardware. The arithmetic is
   below, and `scripts/svmi-cluster.py` reproduces it for any configuration.

## Spec -> engine status

| Spec item | Status in this engine |
| --- | --- |
| 1. Async execution, no blocking in the decode loop | **Exists.** The C++ loop has no Python-side sync points; CUDA graphs are on by default (`GGML_CUDA_GRAPHS`) |
| 1. Pinned ring buffers for transfers | **Exists.** SVMI pinned weight store + staging ring: `GGML_CUDA_REGISTER_HOST=1`, `--stream-weights N` |
| 1. Double-buffered overlap of transfer and compute | **Exists.** Upload queues, one per DMA copy engine: `GGML_SCHED_STREAM_QUEUES`, `GGML_SCHED_STREAM_PREFETCH` |
| 2. Continuous batching, 8-32 streams | **Exists.** `llama-server -np N -cb` (continuous batching is the default) |
| 2. PagedAttention KV manager | **Partial.** Slot-based unified KV cache with defragmentation, not page-table indirection. Fragmentation is handled; arbitrary physical paging is not |
| 2. Non-blocking request injection mid-generation | **Exists.** That is what continuous batching does in `llama-server` |
| 3. MTP head decoding | **Exists.** `--spec-type draft-mtp` reads the Multi-Token-Prediction heads out of the main model - including the Qwen3.5/3.6 hybrid path (`LLM_GRAPH_TYPE_DECODER_MTP`) |
| 3. Speculative 2-3 draft tokens, parallel verify | **Exists.** Same flag; `--spec-draft-n-max` sets the draft length |
| 4. Q8_0 KV cache | **Exists.** `-fa on -ctk q8_0 -ctv q8_0` (quantized V requires flash-attention) |
| 4. FP8 KV cache | **Missing.** No FP8 KV type in ggml. `q8_0` is the same footprint |
| 4. Chunked prefill | **Exists.** `-b` / `-ub` size the logical and physical prefill batches |
| 4. RoPE/YaRN to 1M | **Exists.** `--rope-scaling yarn --rope-freq-base 1000000 --yarn-orig-ctx <train-ctx>` |
| 5. Ampere tensor-core MMA instead of dp4a | **Exists, and it is the default above batch 8.** MMQ uses `mma.sync` s8 IMMA on CC >= 7.5. Batch <= 8 used the dp4a vector path until `GGML_CUDA_NO_MMVQ=1` (added for these cards) |
| 5. FP16 logit preservation | **Different by design.** llama.cpp keeps logits in F32, which is strictly more precise than FP16 |
| 6. Pipeline parallel over RPC | **Exists.** `ggml-rpc-server` + `--rpc host:port,...` layer-splits across nodes; sizing via `scripts/svmi-net.py` |
| 6. OpenAI `/v1/chat/completions` | **Exists.** `llama-server` implements it, streaming included |

Genuinely missing and worth building, in order: FP8 KV (only if it beats `q8_0`
in a measured A/B - same size, so the case is accuracy, not capacity), and true
page-table KV if slot defragmentation proves insufficient under 32-way load.

## The arithmetic

27B at the mixed `INT8` quant is 17.6 GiB, so the model fits one 40 GiB card with
22 GiB to spare. What does *not* fit is deep context: at 64 layers / 8 KV heads /
head_dim 128, `q8_0` KV costs **136 KiB per token**, so one 256K sequence needs
**34 GiB** - more than the free VRAM after weights.

```sh
python3 scripts/svmi-cluster.py --profile 27b --nodes 10 --gpu cmp170hx-40 \
    --ctx 262144 --slots 16 --target 600
```

Decode is bandwidth-bound, and weights are read once per step regardless of how
many sequences share it, so per-node throughput is
`B * HBM_BW / (weights + B * KV_per_seq)`. That gives the envelope:

| Context | Slots that fit | Per node | Aggregate (10 nodes) | 600 target |
| --- | --- | --- | --- | --- |
| 8K | 19 | 475 tok/s | **4748** | met, 8x over |
| 32K | 4 | 109 tok/s | **1092** | met |
| 64K | 2 | 55 tok/s | 546 | just under |
| 128K | 1 | 27 tok/s | 273 | missed |
| 256K (`q8_0` KV) | 0 - does not fit one sequence | - | - | missed |
| 256K (`q4_0` KV) | 1 | 27 tok/s | 265 | missed |

The crossover is ~64K. Above it, KV per sequence dominates the bandwidth budget
and no amount of batching helps because the slots do not fit in the first place.

### Hybrid models change these numbers entirely

Qwen3.5/3.6 (and Qwen3-Next) are **hybrid SSM + attention**: `src/models/qwen35.cpp`
sets `is_recr_impl[i] = (i + 1) % full_attn_interval != 0`, so only 1 layer in
`full_attention_interval` keeps a growing KV cache - the rest carry a fixed-size
recurrent state that does not scale with context. At the usual interval of 4 that
is 16 of 64 layers, so KV per token is **4x smaller** than a dense model of the
same shape, and the whole deep-context picture moves:

| Context | KV type | KV/seq | Slots on one 40 GiB card | tok/s |
| --- | --- | --- | --- | --- |
| 250K | `q8_0` | 8.5 GiB | 2 | 96 |
| 250K | `q4_0` | 4.5 GiB | 4 | 186 |
| 500K | `q4_0` | 9.0 GiB | 2 | 93 |
| 1M | `q4_0` | 18.0 GiB | 1 | 46 |

Two consequences. **250K fits at `q8_0`**, so the `q4_0` K-fidelity problem and its
mean-centering calibration are avoidable at that depth - take the quality for free.
And **1M context fits one card**, which the dense arithmetic said needed a cluster.

Pass the GGUF and the planner reads `full_attention_interval` from the file; with
`--profile` give it `--full-attn-interval N` or it assumes a dense model.

### Deep context on a dense model: 250K and 500K

At these depths the card choice decides the answer, and it is the **8 GiB variant
unlocked to 64 GiB** you want - not the 10 GiB variant unlocked to 40 GiB. A 40 GiB
node has 20 GiB free after weights, which holds exactly one 250K sequence at `q4_0`
and cannot hold a 500K one at all. A 64 GiB node has 44 GiB free: two 250K
sequences, or one 500K.

MTP speculation is the only lever left up here, because it adds accepted tokens per
pass without adding a byte of KV. Figures below assume 1.75 accepted tokens/pass
(`--spec-type draft-mtp`), `q4_0` KV, 10 nodes:

| Context | Card | Slots/node | Aggregate | vs 600 |
| --- | --- | --- | --- | --- |
| 250K | 40 GiB | 1 | 464 tok/s | 77% - needs ~13 nodes |
| 250K | **64 GiB** | 2 | **589 tok/s** | **98% - 10 nodes is the right count** |
| 500K | 64 GiB | 1 | 295 tok/s | 49% - needs ~21 nodes |
| 500K | 40 GiB | pipeline, 2 nodes/replica | 44 tok/s | do not do this |

So: 250K at target throughput is a 10-node deployment on 64 GiB cards. 500K at
target is a ~20-node deployment. 500K on 40 GiB cards forces a pipeline split that
costs an order of magnitude - the KV no longer fits, and splitting it does not
reduce the bytes each token must read, it only spreads them across a sequential
chain of nodes.

Host-RAM KV offload (`-nkvo`) is **not** a way out at these depths on this
hardware: every token reads the whole KV, so 18-36 GiB per token over a 1-2 GB/s
PCIe link is seconds per token, not milliseconds. On these cards KV has to live in
VRAM.

## The topology this implies

**Do not pipeline-split the model across nodes for throughput.** The weights fit
one card; splitting them adds a network hop per token per boundary and buys
nothing. Ten independent replicas behind a load balancer is the correct shape,
and it is what reaches 600+.

Pipelining *is* justified for one thing: holding a context that exceeds one card.
That is a KV problem, not a weights problem, and there is a cheaper answer first -
these nodes have 128 GiB of host RAM each, so `-nkvo` puts the KV there.

Run **two pools** and route by request length. The throughput pool has a real
knob: context traded against slots. 8K/19 slots clears the target ~8x over,
32K/4 slots clears it ~1.8x with four times the usable context. Pick by what the
requests actually need - both meet 600.

```sh
# throughput pool (7 nodes) - max aggregate: 8K ctx, 19 slots, ~475 tok/s per node
GGML_CUDA_NO_MMVQ=1 llama-server -m model-int8.gguf -ngl 999 \
    -c 8192 -np 19 -cb -fa on -ctk q8_0 -ctv q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    --host 0.0.0.0 --port 8080
# balanced variant: -c 32768 -np 4   (~109 tok/s per node, 1092 aggregate)

# deep-context pool (3 nodes) - 256K per sequence, q4_0 KV to make it fit
GGML_CUDA_NO_MMVQ=1 llama-server -m model-int8.gguf -ngl 999 \
    -c 262144 -np 1 -cb -fa on -ctk q4_0 -ctv q4_0 \
    --kv-mean-center kbias.gguf \
    --rope-scaling yarn --rope-freq-base 1000000 --yarn-orig-ctx 262144
```

`q4_0` K rows lose fidelity, which this fork has a specific fix for: a one-time
per-channel mean-shift file (`llama-kv-mean-center`, see the KV section of
[svmi.md](svmi.md)) that is attention-invariant, so it costs nothing at run time.
Beyond 256K, add `-nkvo` to page KV into the 128 GiB of host RAM per node.

MTP speculation applies to both pools and is the one lever that raises
*single-stream* speed rather than aggregate: it turns batch-1 decode into batched
verification, which on a CMP card also moves the work onto the un-throttled
tensor-core path.

## Honesty notes

- The 27B shapes in `MODEL_PROFILES` are approximate. Pass the real GGUF to
  `svmi-cluster.py` and it reads hparams from the file instead; the KV-per-token
  figure moves proportionally to `n_layer x n_head_kv x head_dim`, and every row
  of the table with it.
- `BW_EFFICIENCY` is 0.65 of theoretical HBM bandwidth. That is a planning
  assumption, not a measurement on this card.
- Nothing here has been run on CMP silicon. `scripts/svmi-cmpbench.sh` measures
  the one assumption everything else rests on - whether small-batch decode is
  really on the throttled path - and `svmi-bench.sh` covers the streaming side.
