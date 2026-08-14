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

## The topology this implies

**Do not pipeline-split the model across nodes for throughput.** The weights fit
one card; splitting them adds a network hop per token per boundary and buys
nothing. Ten independent replicas behind a load balancer is the correct shape,
and it is what reaches 600+.

Pipelining *is* justified for one thing: holding a context that exceeds one card.
That is a KV problem, not a weights problem, and there is a cheaper answer first -
these nodes have 128 GiB of host RAM each, so `-nkvo` puts the KV there.

Run **two pools** and route by request length:

```sh
# throughput pool (7 nodes) - short context, many slots, one replica per node
GGML_CUDA_NO_MMVQ=1 llama-server -m model-int8.gguf -ngl 999 \
    -c 32768 -np 4 -cb -fa on -ctk q8_0 -ctv q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 3 \
    --host 0.0.0.0 --port 8080

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
