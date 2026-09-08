# The vLLM backend — the Ampere production path

TENSELERATE's identity is the layer *above* the runtime: the single-model lock,
the product targets (a 1M context floor, the 32K–262K no-RoPE window, a 400
tok/s speed target; none of these is a measured number on the current box),
and the `plan` advisor. That layer is backend-agnostic. There are
two backends:

| backend | what it is | where it runs |
| --- | --- | --- |
| `reference` | the from-scratch NumPy/CUDA engine — the correctness oracle every kernel is checked against | any box, no GPU needed |
| **`vllm`** | drives an upstream vLLM OpenAI server as the compute runtime | the **Ampere** box (CMP 170HX + RTX 3060 12 GiB) |

## Why vLLM, and why now

The target box changed from the dual RTX 2080 Ti (**Turing, sm_75**) to
**CMP 170HX (GA100, sm_80) + RTX 3060 12 GiB (GA106, sm_86)** — both **Ampere**.
That matters: vLLM runs the `qwen3_5` Gated-DeltaNet hybrid natively, with real
Flash-Linear-Attention Triton kernels, FlashAttention-2, and int4/Marlin — all
of which require Ampere+. On Turing those paths fall back or don't build; on
Ampere they are first-class. So on this box vLLM is the production runtime and
the reference engine stays the oracle.

## What the box does — window LOCKED at max recall

The window is **locked at 262,140 (256K)** — the deepest no-RoPE verbatim recall
— and never narrows for speed. So there is no window table to pick from: the box
runs that one window and takes whatever throughput the resulting KV allows.
`tenselerate plan --machine cmp170hx+3060 --kv-bits 4 --spec mtp` at the 1M floor:

| window (locked) | max batch | aggregate | vs 400 target |
| --- | --- | --- | --- |
| **262,140** | 7 | **~301 tok/s** | under, by design |

The box is **below the 400 tok/s target on purpose** — quality is pinned at
maximum, and speed takes what recall leaves. 400 is a target, not a hard gate.
1M+ context still holds (the GDN state carries it). (Roofline at 65% bandwidth
efficiency, pooled 52 GiB — the CMP's unlocked 40 GiB plus the RTX 3060's 12 GiB
— over a PP=2 pipeline; not a measurement.)

## Running it

```
tenselerate serve --backend vllm            # the recommended config, out of the box
tenselerate serve --backend vllm --dry-run  # just print the vllm command
```

**The defaults are the max-recall config:** the locked 256K window, `--kv-bits 4`
(fp8 KV), `--spec mtp` (lossless speculative), `--gpu-memory-utilization 0.92`,
`--max-num-seqs 16` — ~301 tok/s at 256K of verbatim recall. The only lossless
levers are q4 KV, MTP, and (with a head) `--spec eagle3 --eagle-model <head>`;
none reach 400 at this window, because the window does not narrow. Context stays
1M+ regardless.

To trade depth for raw throughput, a narrower window gives far more concurrency
(32K -> ~2,476 tok/s aggregate); to trade throughput for the deepest no-RoPE
recall, 262K -> ~167 tok/s. Context stays 1M+ in every case (the GDN state).

The launcher (`tenselerate/backends/vllm.py`) builds the `vllm serve` argv and
enforces the engine's floors **before vLLM starts** — a sub-floor context or an
over-ceiling window is refused here, so the backend swap changes the runtime,
never the contract.

### How the floors map to vLLM flags — honestly

| engine concept | vLLM flag | note |
| --- | --- | --- |
| context floor | `--max-model-len ≥ 1,000,000` | vLLM's Qwen3-Next carries long range in the GDN state, as the reference does |
| no-RoPE window | *(none)* | intrinsic to the model config; vLLM manages the hybrid KV itself. We still validate the window against the quality floor/ceiling |
| two heterogeneous GPUs, no NVLink | `--pipeline-parallel-size 2` | **pipeline**, never tensor-parallel — TP wants ~equal GPUs on a fast link, which this box is not |
| `--kv-bits 4` | `--kv-cache-dtype fp8` | vLLM has **no int4 KV**; fp8 is its footprint lever. 8→`auto`. The KIVI int4 KV in the research roadmap is not a vLLM feature today |
| `--spec mtp` | `--speculative-config {qwen3_next_mtp}` | vLLM's built-in Qwen3-Next MTP speculative decode (default, lossless) |
| `--spec eagle3 --eagle-model <head>` | `--speculative-config {eagle3, model}` | trained EAGLE-3 draft head - higher acceptance than MTP, still lossless; needs a head (none public for this model yet) |

The one honesty flag: **the CMP 170HX ships with 8 GiB**, which cannot hold the
15.4 GiB Q4_K_M weights — the 40 GiB figure assumes the memory unlock is applied
(see `docs/svmi`). Without it, the box does not serve 27B at all.
