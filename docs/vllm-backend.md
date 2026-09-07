# The vLLM backend — the Ampere production path

TENSELERATE's identity is the layer *above* the runtime: the single-model lock,
the product floors (1M context, the 32K–262K no-RoPE window, the 400 tok/s
speed floor), and the `plan` advisor. That layer is backend-agnostic. There are
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

## What the box can do

The Ampere box **meets all three floors at once** — the first supported box that
does. `tenselerate plan --machine cmp170hx+3060` at the 1M context floor:

| window | max batch | aggregate | vs 400 floor |
| --- | --- | --- | --- |
| 131,072 | 8 | ~182 tok/s | under |
| 65,536 | 16 | ~363 tok/s | under |
| **49,152** | **22** | **~489 tok/s** | **meets** |
| **32,768** | **33** | **~733 tok/s** | **meets** |

So 1M context + the 32K quality floor + the 400 tok/s standard are all
satisfiable here, at a 49K window or narrower. (Roofline at 65% bandwidth
efficiency, pooled 52 GiB — the CMP's VRAM at its unlocked 40 GiB figure plus
the RTX 3060's 12 GiB — over a PP=2 pipeline; not a measurement.)

## Running it

```
tenselerate serve --backend vllm            # the recommended config, out of the box
tenselerate serve --backend vllm --dry-run  # just print the vllm command
```

**The defaults are the deep-and-fast sweet spot for the Ampere box:** `--kv-bits 4`
(fp8 KV), `--spec mtp` (lossless speculative), `--gpu-memory-utilization 0.92`,
`--max-num-seqs 16`. At the default 131K window that plans to **~616 tok/s** at
15 streams with 131K of verbatim recall - past the 400 floor with depth to
spare. Override any of them (`--kv-bits 8`, `--spec none`, `--spec eagle3
--eagle-model <head>`).

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
