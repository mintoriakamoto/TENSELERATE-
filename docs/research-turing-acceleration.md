# Turing acceleration research — what actually maps to the dual 2080 Ti

This is the grounded roadmap for making TENSELERATE faster on its one supported
box: **dual RTX 2080 Ti** (Turing, `sm_75`, 22 GiB pooled, ~1232 GB/s), serving
Qwen3.8-27B (`qwen3_5`) — a hybrid of **48 Gated-DeltaNet (GDN) linear layers +
16 full-attention layers**. Everything below is filtered to that hardware and
that architecture; general LLM-inference tricks that do not run on `sm_75` are
called out as dead ends so we do not spend effort on them.

Nothing here is "100%" in the absolute sense — a roofline is not silicon — but
items are ranked by how directly primary sources tie them to *our* GPU and
*our* head dimensions. None of it involves RoPE scaling or YaRN; the no-RoPE
rule stands.

Baseline: the box plans to ~152 tok/s at the 32K window, below the 400 tok/s
standard. The levers below are how it closes that gap.

---

## Tier 1 — Turing-native, high confidence

### 1. FlashQLA-SM70/75 — a GDN kernel validated on the 2080 Ti
The GDN layers are **48 of 64 (75%)** of the model and carry the whole long
range; their chunked prefill is a large share of the 1M-context ingest cost.

- QwenLM's official [`FlashQLA`](https://github.com/QwenLM/FlashQLA) is **SM90+
  only** — it does *not* build for Turing.
- The community fork
  [`weicj/FlashQLA-SM70-SM75`](https://github.com/weicj/FlashQLA-SM70-SM75)
  backports it, and its **runtime validation target is the RTX 2080 Ti / SM75**.
  It is benchmarked at **B=1, T=512, Hq=16, Hv=32, D=128** — and our config is
  `linear_num_key_heads=16`, `linear_num_value_heads=48`,
  `linear_key_head_dim = linear_value_head_dim = 128`. **D=128 is our exact head
  dim.** Reported **~2.1× kernel / ~2.08× GDN-stage** speedup vs the recurrent
  path.

**Applies to:** GDN chunked-prefill forward. **Caveat:** forward/prefill only
(no decode kernel, no backward), needs CUDA 12.8+ / PyTorch 2.8+. The per-token
decode state update is cheap and stays on the recurrent path; the win is 1M
ingest latency.

### 2. FlashAttention-2 on Turing (fp16) — for the 16 windowed layers
Our full-attention layers attend a bounded 32K–262K window = exactly FA-2's
case.

- FA-2 supports compute capability 7.5 (2080 Ti / T4), **fp16 only** (Turing
  has no BF16), ~1.2–2.4× forward. Turing forks:
  [farnghwai/flash-attention-2080ti](https://github.com/farnghwai/flash-attention-2080ti),
  and llama.cpp's own fp16 FA smem-swizzle work
  ([PR #25635](https://github.com/ggml-org/llama.cpp/pull/25635)).
- **Catch:** mainline llama.cpp / Ollama do *not* ship Turing FA kernels — they
  silently fall back to standard attention. We compile our own `sm_75` SASS, so
  we can build the Turing FA path in rather than inherit the fallback.

**Applies to:** the windowed attention layers — lower latency and fewer
bytes/token read.

### 3. 4-bit KV is the *validated* quality floor — 2-bit is not
This de-risks the existing `--kv-bits 4` lever and points at a free quality
upgrade.

- The KV-quant literature converges on: **4-bit KV preserves accuracy; 2-bit
  degrades it, especially on long-context reasoning.** Sources:
  [KIVI](https://arxiv.org/abs/2402.02750), KVQuant,
  [UltraQuant](https://arxiv.org/html/2606.20474),
  [Kitty](https://arxiv.org/abs/2511.18643).
- **KIVI asymmetric layout** (per-channel keys, per-token values) beats uniform
  q4_0 at the same footprint — a concrete quality bump for the same bytes.

**Applies to:** the resident windowed KV. So `q4_0` is the right precision floor
(never 2-bit), and KIVI's layout is the target quant scheme. A model-specific
A/B still measures the exact delta on RavenX before it is trusted, but 4-bit
itself is no longer a gamble.

---

## Tier 2 — speculative decode: EAGLE-3 ≥ plain MTP

- [EAGLE-3](https://arxiv.org/abs/2503.01840) reports **0.80–0.88 acceptance**
  (above Eagle-2 / plain MTP), zero quality loss (the verify guarantees output
  identical to plain decode), and **~1.37× on Qwen3-Coder-Next** specifically,
  higher on code/math/structured output.
- The model's **built-in MTP head is the zero-training baseline** (what `--spec
  mtp` models today). **EAGLE-3 is the stronger roadmap draft head** — it needs
  a trained head, so it is a build cost, not free. Both multiply decode and
  stack on Tier 1.

---

## Dead ends — do NOT spend effort here

Half of "what will help" is knowing what cannot, so we do not chase it on
`sm_75`:

- **SageAttention (int8 / fp4 attention): unusable on Turing.** It *dropped*
  `sm_75` support; its gains are Ampere+/Blackwell FP4
  ([thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)). The fp16
  FA-2 path (Tier 1 #2) is the correct Turing route for attention — not
  int8-quantized attention.
- **Official `QwenLM/FlashQLA`: SM90+ only.** Only the community `sm_75` fork
  (Tier 1 #1) touches our hardware.

---

## How the levers compose

| lever | what it speeds | quality cost | status |
| --- | --- | --- | --- |
| FlashQLA-SM75 | GDN chunked prefill (1M ingest) | none | roadmap kernel (fork exists) |
| fp16 FA-2 (Turing) | the 16 windowed attention layers | none | roadmap kernel (forks exist) |
| q4_0 → KIVI KV | ~2× concurrency, less KV traffic | 4-bit: negligible (validated) | q4_0 shipped as a `plan` dial |
| MTP self-spec | decode throughput (~1.8×) | none (verify-guaranteed) | modeled by `--spec mtp` |
| EAGLE-3 | decode throughput (~2×+) | none (verify-guaranteed) | roadmap, needs trained head |

FlashQLA + FA-2 are latency/prefill and memory-traffic wins (they also help the
roofline by cutting bytes/token on their layers); q4_0/KIVI buys concurrency;
MTP / EAGLE-3 multiply decode. The `plan` acceleration path already models
`--kv-bits 4 --spec mtp` reaching ~590 tok/s at the 32K window; this document is
the evidence base behind those dials and the next kernels to build.
