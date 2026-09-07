# Rig field guide: CMP 170HX (40 GB) + RTX 3060 on a 9950X

A serving guide for the `raven-9950x` machine (see `scripts/svmi-auto.py` MACHINES):
Ryzen 9 9950X / B650E / 32 GiB DDR5, an NVIDIA CMP 170HX unlocked to 40 GiB, and an
RTX 3060 12 GiB - 52 GiB VRAM total. Target model: the RavenX Chaos Agent
(Qwen3.8-27B, arch `qwen3_5`), served at the 1M context floor.

Everything below is a starting config backed by the sources cited, not a benchmark of
this exact box. Measure the three open items in the last section before trusting numbers.

## The two cards have opposite strengths

This box runs the **full compute unlock** (see next section), so the 170HX behaves close
to an A100, not the throttled mining card the arXiv:2505.03782 case study measured.

| | CMP 170HX (40 GiB, unlocked) | RTX 3060 (12 GiB) |
| --- | --- | --- |
| Silicon | GA100 (A100), sm_80 | GA106, sm_86 |
| VRAM bandwidth | ~1493 GB/s (verify; unlock reports 0.73-1.4 TB/s) | ~360 GB/s |
| PCIe link | 1.1 x4 native; software patch -> Gen2 (~2 GB/s); cap mod -> Gen2 x16 (~8 GB/s) | 4.0 x16 - healthy |
| Tensor cores | restored, but 256-cycle MMA throttle (ILP cannot hide it) | yes (FP16/BF16) |
| FP32 | restored by the unlock | full |
| FP16 / BF16 | restored | full |
| INT8 / dp4a | uncrippled | full |

## Unlock state: full compute unlock (fuse-map reset)

This card was unlocked with the d3dx9/cmpunlocker tool (GSP exploit found 2026-07-16,
released 2026-07-19), which resets the firmware logical fuse map via the Falcon BootROM
`.fwsignature_ga100` load bug. That re-enables ALL factory-disabled compute - FP32,
FP16, BF16, and Tensor Cores - and the HBM2e geometry (10 GiB -> 40 GiB). It is NOT the
older capacity-only unlock, so the throttled-compute workarounds (`--fmad=false`,
forced MMQ, dp4a-disable) do NOT apply here.

Caveats that survive the unlock:
- **Tensor Cores keep a 256 fixed-cycle MMA throttle**: single-instruction latency is
  pinned at 256 cycles and cannot be hidden by ILP/pipeline overlap. So prefill /
  compute-bound GEMM underperforms a real A100; decode (bandwidth-bound) is unaffected.
- **PCIe stays Gen2** (Gen3/4 blocked by OTP fuse `FUSE_PCIE_GEN23_DIS`): ~2 GB/s on
  the software patch, ~8 GB/s with the 12-capacitor x16 hardware mod. Still keep the
  model resident; do not stream weights.
- **The unlock is volatile**: a systemd daemon reapplies it after every reboot / driver
  reload (nvidia-open 580 / 610.43.0x). Re-check `nvidia-smi` after any driver change.

If a given card is only capacity-unlocked (FP32 still ~1/32, no tensor cores), use the
throttled-card path instead: `--fmad=false` + `-DCMAKE_CUDA_ARCHITECTURES=80`, INT8/MMQ,
`GGML_CUDA_NO_MMVQ=1` (see scripts/svmi-gpucheck.py). Tell the two apart with a quick FP16
GEMM / `gpu-burn -tc` bench: ~10-50 TFLOPS FP16 => full unlock; ~6 TFLOPS => capacity-only.

## Card roles (do NOT split the model across both)

- **170HX = resident decode engine.** Holds the whole 27B resident and runs the
  token loop. Decode is memory-bandwidth bound and this card has ~4x the 3060's
  bandwidth. KV cache lives here too. Load once over the slow link, keep resident,
  stream nothing.
- **3060 = draft + prefill assist + I/O.** Runs the MTP draft head (speculative decode)
  and hosts embeddings / any small side model on the card with a live host link. The
  170HX can now prefill too (tensor cores restored), but the 256-cycle MMA throttle
  handicaps its GEMM, so offloading latency-sensitive prefill to the 3060 still helps.
- **9950X + 32 GiB = control plane, not a weight tier.** Tokenization, sampling,
  scheduler, grammar-constrained decode. 32 GiB is too tight to be a streaming/KV
  offload tier - and you do not need one, since the model fits on the 170HX.

A `--tensor-split` layer split runs every token at the slower card's pace for its
share, so adding the 360 GB/s 3060 to a model that already fits on the 1493 GB/s
170HX makes decode slower, not faster (`svmi-gpucheck.plan_split` proves this). Keep
the main model 100% on the 170HX; give the 3060 a separate role.

## Build (full compute unlock)

With FP32/FP16/BF16/tensor cores restored, build a NORMAL sm_80 llama.cpp - no
`--fmad=false`, no forced MMQ, no dp4a-disable, cuBLAS left ON:

```
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80-real;86-real"      # 80 = 170HX/GA100, 86 = 3060
# do NOT set --fmad=false / FORCE_MMQ / GGML_CUDA_NO_MMVQ here - those are the
# capacity-only-unlock workarounds and cost throughput on a compute-unlocked card.
```

`-fa on` (flash attention) now works and is the default - the tensor cores are enabled.
The only compute caveat is the 256-cycle MMA throttle, which shows up in prefill/GEMM,
not in bandwidth-bound decode.

## Run (1M context)

```
CUDA_VISIBLE_DEVICES=<170hx>,<3060> \               # 170HX must be device 0
llama-server -m qwen3.8-27b-UD-Q4_K_M.gguf \
  -ngl 999 --main-gpu 0 \                           # whole model resident on the 170HX
  -c 1000000 \                                      # the 1M floor; GDN carries range past the window
  -ctk q8_0 -ctv q8_0 -fa on \                      # ~8.5 GiB KV at the 262K window
  -b 2048 -ub 512 \                                 # chunked prefill (memory-safe on 32 GiB / 12 GiB)
  -md qwen3.8-27b-mtp-draft.gguf --spec-draft-n-max 5 \  # MTP self-spec: biggest decode lever
  -t 16
# window stays <= 262,140 (engine-enforced). Do NOT pass --swa-full (removes the window).
```

Weights choice: UD-Q4_K_M (~15.4 GiB) gives the fastest decode (fewer bytes/token on a
bandwidth-bound card) and the most KV headroom. Mixed-INT8 (~17.6 GiB, q8_0 attention
over a Q4_K_M body) trades ~2 GiB of bandwidth for higher attention fidelity - A/B it.

## Context and KV at 1M

Context length and KV size are decoupled. Only 16 of the 64 layers are full-attention
and cache KV; the other 48 are Gated-DeltaNet layers whose fixed recurrent state carries
long range with no positional encoding and no KV growth. The full-attention window is
capped at **262,140 tokens** (the trained rotary range minus 4 attention sinks); past
that a position would extrapolate, which needs RoPE scaling - which the engine refuses.

So KV is sized by the window, not the sequence. Per-token KV (16 caching layers, 4 KV
heads, 256 head-dim): ~34 KiB at q8_0, ~18 KiB at q4_0, ~64 KiB at f16.

| Window | q4_0 | q8_0 | f16 |
| --- | --- | --- | --- |
| 128K (default) | ~2.25 GiB | ~4.25 GiB | ~8.0 GiB |
| 262K (max, no-RoPE) | ~4.5 GiB | ~8.5 GiB | ~16 GiB |

This is **constant at 1M** - decode speed at 1M equals decode speed at 128K, because the
resident KV is identical. A literal 1M full-attention KV would be ~32 GiB (q8_0) / ~61 GiB
(f16) AND would require RoPE scaling, so it is never materialized.

170HX VRAM budget at 1M: 15.4 (Q4_K_M) + 8.5 (KV @262K q8_0) + ~3.5 (GDN state + overhead)
= ~27.4 / 40 GiB, ~12 GiB free. Even f16 KV at the max window fits.

Recall tradeoff, stated honestly: verbatim recall inside the 262K window; compressive /
associative recall for the older tokens via the GDN state. No windowed config gives
verbatim recall across the full 1M without RoPE scaling.

The only real cost of long context is the one-time prefill of the prompt (compute-bound
- the 170HX's weak axis). Prefill on the 3060, chunk it (`-b`/`-ub`), and reuse the
cache across turns (`--cache-reuse`) so you pay it once.

## Engine choice: llama.cpp, not vLLM

With compute restored, the 170HX is now a viable vLLM target on its OWN (40 GiB,
A100-class, FP16/tensor cores work) - single-card vLLM for the 27B is plausible if you
want its continuous-batching throughput for many concurrent requests. But TWO-card
tensor parallelism is still gated by the PCIe link: Gen2 x4 (~2 GB/s), or ~8 GB/s with
the capacitor mod - far below what per-layer all-reduce wants, and Gen3/4 are OTP-fused
off. So for the two-card box, llama.cpp still wins: it keeps the model resident on the
170HX, gives the 3060 an asymmetric draft/prefill role, and implements the GDN +
windowing + MTP that make 1M context fit. Use vLLM only for single-170HX high-concurrency
serving, or if you add a matched card on a real x16 link.

## Speculation (MTP)

The Qwen3.8-27B GGUF ships an MTP draft head; `--spec-draft-n-max 5` (10 for structured
output like JSON/HTML/XML) is the single biggest decode lever for this model. It costs
~12-15% prefill throughput, which is the right trade here since the 170HX is a strong
decoder and weak prefiller. Reported at +33-39% up to 3.5x on this model
(Ferrox Labs Field Manual No.12; sudoingX/qwen38-mtp). Probe the real acceptance rate
with `scripts/svmi-bitspec.py`.

## Measure before trusting

1. Confirm the full unlock held (FP16 GEMM / `gpu-burn -tc` ~10-50 TFLOPS, not ~6),
   especially after any driver reload - the unlock is volatile.
2. MTP acceptance rate -> the real speedup and the best `--spec-draft-n-max`.
   Use `scripts/svmi-bitspec.py`.
3. UD-Q4_K_M vs mixed-INT8 for your workload's quality/speed balance.

Then confirm token-identity with `scripts/svmi-verify.sh` before trusting a long run.

## Sources

- d3dx9/cmpunlocker - GA100 fuse-map reset via the Falcon BootROM .fwsignature_ga100 bug (2026-07)
- Tom's Hardware - software mod unlocks 64 GB on the CMP 170HX
- DevQuasar - "the almost A100": CMP 170HX unlocked
- 170th-Street benchmarks - the 256-cycle tensor-core MMA throttle
- arXiv:2505.03782 - Exploration of Cryptocurrency Mining-Specific GPUs in AI: CMP 170HX (throttled/capacity-only state)
- niconiconi - CMP 170HX review / performance lockdown workaround
- Ferrox Labs Field Manual No.12 - Maximising Qwen3.8 on a 5090 Laptop
- sudoingX/qwen38-mtp - one-flag MTP decode speedup for Qwen3.8-27B
- Prima.cpp (arXiv:2504.08791) - heterogeneous home-cluster inference
- llama.cpp #24616 - dp4a emulation via dp2a
- Quesma - Qwen3.8-27B quantizations benchmarked
