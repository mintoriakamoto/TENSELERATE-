# TENSELERATE — one engine from all of them

The goal in one line: **our own engine to run models**, not a fork of any one of
them, that takes the strongest idea from each source and fuses them so the whole
is faster and pushes further than any single engine on the hardware we run.

This is the map of what we take, from where, and why — and what is uniquely ours.
It is a design charter, not a status report; the honest per-phase status lives in
[tenselerate-engine.md](tenselerate-engine.md).

## What each source does best, and what we take

### llama.cpp — portability and the format
- **GGUF** as the on-disk format. We read it with our **own** loader
  (`tenselerate/gguf/`), so nothing at runtime depends on llama.cpp — only the
  file format is shared. A model made for llama.cpp just runs.
- **k-quant weight formats** (Q4_K, Q6_K, Q8_0) and the **MMQ integer-kernel**
  lesson: on quantized decode, integer tensor-core math beats dequant-to-FP16,
  and on the CMP that is the *only* fast path. We keep the formats, we write our
  own kernels.
- **Hybrid GDN support** (its `qwen35` path proves the model is runnable at all;
  most engines could not load it a year ago).

### vLLM — throughput and memory
- **PagedAttention**: KV in fixed non-contiguous blocks, so many sequences pack a
  GPU without fragmentation. This is the KV manager's backbone.
- **Continuous batching**: sequences join and leave the running batch every step
  instead of waiting for a whole batch to finish — the single biggest throughput
  win for a busy server.
- **The hybrid KV manager**: size the full-attention block pool and the fixed
  linear-state pool so they share physical memory cleanly. Exactly what our
  16-full / 48-linear split needs.

### ExLlamaV3 — quality per bit and single-user speed
- **High-quality low-bit quantization** (the EXL3 idea: better error per bit than
  round-to-nearest k-quants). This is where the biggest context/quality headroom
  on the consumer box comes from.
- **Consumer-GPU kernel scheduling**: the fastest single-user decode on RTX-class
  cards, which is your friend's box exactly.

### FreeToken — the async-copy idea, carefully
- **Batched asynchronous host↔device copy** to overlap weight/KV movement with
  compute. We take the pattern and *avoid the bug we already found in it* (a
  synchronous degradation when the batched copy path falls back), because we own
  the code and validate it against a reference.

## What is uniquely ours

None of the four target the hardware or the exact workload we run. This is the
part no fork gives us:

- **SVMI streaming** — keep the model resident where it fits, stream the overflow
  from host RAM, so a model larger than VRAM still runs instead of failing. Built
  for the CMP's narrow PCIe and the consumer box's 24 GiB ceiling.
- **The CMP int8 lane** — an IMMA `mma.sync` s8 GEMM that rides the tensor-core
  path the CMP 170HX firmware leaves *un*-throttled, while FP16 and dp4a dispatch
  are crippled. Every other engine assumes FP16 works.
- **Two-machine, no-link hosting** — replicas per site, never a shard across a
  WAN (`scripts/svmi-net.py`), because a layer split pays a round trip per token.
- **The reference-oracle method** — every kernel has a CPU reference and a test
  before it is written, so correctness is a bar, not a hope. This is why the
  engine can grow fast without regressing.

## How they fuse into one pipeline

```
  GGUF (ours)  ─►  weights + hybrid config  ─►  scheduler (vLLM-style continuous batch)
                                                   │
                            ┌──────────────────────┴───────────────────────┐
                            ▼                                               ▼
              full-attention layers (16)                       linear GDN layers (48)
              PagedAttention KV blocks                          fixed per-seq state pool
              flash-style causal kernel                         chunked-scan kernel
                            │                                               │
                            └──────────────► int8 GEMM (IMMA on CMP, ─◄─────┘
                                              FP16/consumer tuned elsewhere)
                                                   │
                    ExLlama-grade low-bit weights ─┤  MTP self-speculation (built-in draft)
                    SVMI streaming for overflow  ──┘  NVTX-instrumented throughout
                                                   ▼
                                       OpenAI /v1  (ours, loopback)
```

The dispatch rule that ties it together: **pick the backend per tensor by the
hardware in front of it.** IMMA int8 on the CMP, FP16/consumer kernels on the
GeForce, the same reference semantics behind both — so one engine, one model
file, one API, runs at each machine's ceiling without the user choosing a build.

## The order we build it

Foundations first, each against a reference that already exists: GGUF loader
(done) → real tokenizer → kernel bridge → GDN + attention + IMMA kernels → paged
hybrid KV → continuous batching → MTP → SVMI. See the phase table in
[tenselerate-engine.md](tenselerate-engine.md).
