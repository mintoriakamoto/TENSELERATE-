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
| PCIe link | Gen2 x4 (~2 GB/s) measured; cap mod -> Gen2 x16 (~8 GB/s) | Gen4 x4 (~8 GB/s) measured - fine for draft/IO, poor for streaming |
| Tensor cores | restored — **measured 162-170 TFLOPS FP16** (dense A100 class; see below) | yes (FP16/BF16) |
| FP32 | restored by the unlock | full |
| FP16 / BF16 | restored | full |
| INT8 / dp4a | uncrippled | full |

## Measured state (this box) and the power-limit lever

Live readout (2026-09-07): 170HX at Gen2 x4, 40 GiB, **250 W** (the earlier 100 W cap
has been lifted), **162-170 TFLOPS FP16 measured** - the 256-cycle MMA gate reported for
other units does not reproduce on this one; 3060 at Gen4 x4, 12 GiB. Persistence ON both.

An earlier readout had the card power limited to 100 W of its 250 W default, and that
was the biggest throughput limiter at the time. Decode is bandwidth-bound and a
GA100 held at 40% of its power budget down-clocks cores and memory, so it will NOT reach
the ~1493 GB/s / A100-class decode the unlock makes possible. Thermal is not the reason -
33 C leaves ~50 C of headroom - so if the PSU can supply it, raise the limit and
re-measure:

```
sudo nvidia-smi -i <170hx> -pl 200      # step 150 -> 200 -> 250 as the PSU allows
# bench decode at 100 W vs the raised cap (scripts/svmi-bitspec.py) to see the cost.
```

If the box is genuinely PSU-limited to 100 W, keep the cap but scale throughput
expectations down - the case-study and "best config" numbers assume full power.

## Unlock state: full compute unlock (fuse-map reset)

This card was unlocked with the d3dx9/cmpunlocker tool (GSP exploit found 2026-07-16,
released 2026-07-19), which resets the firmware logical fuse map via the Falcon BootROM
`.fwsignature_ga100` load bug. That re-enables ALL factory-disabled compute - FP32,
FP16, BF16, and Tensor Cores - and the HBM2e geometry (10 GiB -> 40 GiB). It is NOT the
older capacity-only unlock, so the throttled-compute workarounds (`--fmad=false`,
forced MMQ, dp4a-disable) do NOT apply here.

Caveats that NO unlock removes (dispatch-level gating and OTP fuses, not firmware):
- **The 256-cycle MMA gate does NOT apply to this unit.** 170th-Street reports a
  4-warp/SM issue limit with 256-cycle MMA latency (~1/32 of peak) on the cards they
  measured; this card measures 162-170 TFLOPS FP16 (2026-09-07), i.e. dense A100
  class. Prefill and wide speculative verification run at full speed on the 170HX.
  Re-check with `gpu-burn -tc` after any driver reload - if it ever reads ~5-10
  TFLOPS, the gate is back and prefill should move to the 3060.
- **PCIe stays Gen2** (Gen3/4 blocked by OTP fuse `FUSE_PCIE_GEN23_DIS`): ~2 GB/s on
  the software patch, ~8 GB/s with the 12-capacitor x16 hardware mod. Keep the model
  resident; do not stream weights.

Install (amoghmunikote/cmpunlocker) for this 10 GiB card: `sudo ./install.sh --profile=10gb`
-> 40 GiB (the `--profile=8gb` target is 64 GiB; an experimental `80-new` branch chases
80 GiB, unconfirmed - skip it). The 170th-Street repo/GitBook is the author's reference.

Unlock method (persistence tradeoff): this box runs the d3dx9 daemon-based unlock,
which reapplies after every reboot / driver reload (nvidia-open 580 / 610.43.0x) -
re-check `nvidia-smi` after any driver change. Alternatives with the same result:
abobasixseven/unlock-cmp-170hx (in-driver kernel patches for 610.43.03 - cleaner, no
BootROM exploit or daemon race), or the thaurock vBIOS flash (persistent, no daemon,
but carries brick risk). 40 GiB is the confirmed-stable target for the 10 GiB card;
an 80 GiB target is claimed but unconfirmed - not worth the risk, the 27B fits in 40.

If a given card is only capacity-unlocked (FP32 still ~1/32, no tensor cores), use the
throttled-card path instead: `--fmad=false` + `-DCMAKE_CUDA_ARCHITECTURES=80`, INT8/MMQ,
`GGML_CUDA_NO_MMVQ=1` (see scripts/svmi-gpucheck.py). Tell the two apart with a quick FP16
GEMM / `gpu-burn -tc` bench: ~160+ TFLOPS FP16 => full unlock (this box); ~10-50 => unlocked
but MMA-gated; ~6 TFLOPS => capacity-only.

## Card roles (do NOT split the model across both)

- **170HX = resident decode engine.** Holds the whole 27B resident and runs the
  token loop. Decode is memory-bandwidth bound and this card has ~4x the 3060's
  bandwidth. KV cache lives here too. Load once over the slow link, keep resident,
  stream nothing.
- **3060 = side roles only.** With the 170HX measured at A100-class compute, prefill
  belongs on the 170HX too. The 3060's remaining value is its own 360 GB/s (a
  bandwidth-proportional ~80/20 layer split, to be measured) and hosting a side
  drafter / embeddings on the card with a live host link.
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

`-fa on` (flash attention) now works and is the default - the tensor cores are enabled
and measured at full rate on this unit, so there is no compute caveat left; decode is
purely bandwidth-bound.

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

## Serving Hercules: slots, not batches

Hercules is a fork of Nous Hermes Agent, so its load shape is Hermes': one
sequential main loop per session, plus subagents that run **in parallel** (default
`delegation.max_concurrent_children: 3`, no hard ceiling), plus one session per
gateway chat if the messaging gateway is on. Each of those is an ordinary
streaming `/v1/chat/completions` request - Hermes never sends `n > 1`, so there is
no batch flag on the client side. Concurrency is entirely `llama-server`'s job:

- `-np N` is the number of requests in flight; continuous batching (`-cb`) is
  already the default and merges them into one decode step.
- Without `--kv-unified`, `-c` is split evenly: each slot gets `c / N`. With
  `--kv-unified`, `-c` is one shared pool and any slot may grow to the full
  window while the others stay small - which is exactly a main session plus
  short-lived subagents. Use it.
- Per step the card reads weights + the KV that is actually live, so aggregate
  throughput follows live tokens, not `N x window`.

Hercules auto-compresses at 50% of the model's advertised window
(`compression.threshold: 0.50`), so a session advertised at 262K lives at
<= ~131K in practice.

```
llama-server -m qwen3.8-27b-UD-Q4_K_M.gguf -ngl 999 --main-gpu 0 \
  -c 1048576 -np 8 --kv-unified -cb -fa on \    # 8 slots sharing a 1M-token q4_0 pool (18 GiB)
  -ctk q4_0 -ctv q4_0 \                          # q8_0 halves the pool to 512K; A/B the quality
  --chat-template-kwargs '{"reasoning_effort":"low"}'   # biggest end-to-end lever on an agent loop
```

Sizing from two measured points (single stream 30 ms/step, two streams 41.5
ms/step): each step is ~18.5 ms of weight read (~890 GB/s, 60% of nominal)
plus **~11.5 ms per live sequence** of Gated-DeltaNet recurrence work that does
not batch today. Short prompts; add ~5.4 ms per slot holding a full 256K window
at q4_0 (10.8 ms at q8_0).

| slots | step | per stream | aggregate |
| --- | --- | --- | --- |
| 1 | 30 ms | 33 tok/s | 33 |
| 2 | 41.5 ms | 24 | 48 (measured 48.2) |
| 4 | 64.5 ms | 15.5 | 62 |
| 8 | 110 ms | 9 | 72 |
| 16 | 203 ms | 5 | 79 |

So aggregate saturates near **1 / 11.5 ms = ~87 tok/s** however many slots are
added, and every added slot slows the operator's main loop. For one operator:
`-np 4` (main loop + default 3 subagents) is the ceiling worth paying for;
`-np 2` if the main loop's speed matters more than subagent parallelism. Do not
enable MTP on the DavidAU merge (7-11% acceptance; measured 14.9 tok/s, slower
than plain). The lever that changes this table is a GDN kernel that processes
all live sequences in one launch - if t_seq fell to ~2 ms, 8 slots would give
~230 tok/s aggregate. Until then, `reasoning_effort` (fewer tokens) is worth
more than any slot count.

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

The only real cost of long context is the one-time prefill of the prompt. It runs at
full tensor-core rate on the 170HX; chunk it (`-b`/`-ub`) and reuse the cache across
turns (`--cache-reuse`) so you pay it once.

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
~12-15% prefill throughput, cheap against a bandwidth-bound decode loop. With compute
measured at A100 class, verification depth is limited by draft acceptance, not by the
card, so deeper drafts and tree drafts are worth measuring. Reported at +33-39% up to 3.5x on this model
(Ferrox Labs Field Manual No.12; sudoingX/qwen38-mtp). Probe the real acceptance rate
with `scripts/svmi-bitspec.py`.

## Measure before trusting

1. Confirm the full unlock held (FP16 GEMM / `gpu-burn -tc` ~160+ TFLOPS as measured
   here, not ~6), especially after any driver reload - the unlock is volatile.
2. MTP acceptance rate -> the real speedup and the best `--spec-draft-n-max`.
   Use `scripts/svmi-bitspec.py`.
3. UD-Q4_K_M vs mixed-INT8 for your workload's quality/speed balance.

Then confirm token-identity with `scripts/svmi-verify.sh` before trusting a long run.

## Sources

- Consensus-Protocol/cmp170hx - the authoritative technical wiki (silicon, firmware, unlock, procedures, open problems)
- d3dx9/cmpunlocker - GA100 fuse-map reset via the Falcon BootROM .fwsignature_ga100 bug (2026-07)
- amoghmunikote/cmpunlocker - first public unlock tool; --profile=10gb -> 40 GiB (8gb -> 64, 80-new experimental)
- amoghmunikote/170th-Street (+ 170th-street.gitbook.io/hx) - the author's comprehensive resource + FP16 benchmarks
- abobasixseven/unlock-cmp-170hx - in-driver kernel patches for 610.43.03 (no BootROM exploit)
- thaurock-x/CMP-170HX-64GB-Unlocked-VBIOS - standalone 64 GB vBIOS flash (persistent)
- Tom's Hardware - software mod unlocks 64 GB on the CMP 170HX
- DevQuasar - "the almost A100": CMP 170HX unlocked
- 170th-Street benchmarks - FP16 tensor-core ~1/32 peak (256-cycle MMA + 4-warp gate); NOT reproduced on this unit (162-170 TFLOPS measured, see benches/cmp170hx-3060/)
- arXiv:2505.03782 - Exploration of Cryptocurrency Mining-Specific GPUs in AI: CMP 170HX (throttled/capacity-only state)
- niconiconi - CMP 170HX review / performance lockdown workaround
- Ferrox Labs Field Manual No.12 - Maximising Qwen3.8 on a 5090 Laptop
- sudoingX/qwen38-mtp - one-flag MTP decode speedup for Qwen3.8-27B
- Prima.cpp (arXiv:2504.08791) - heterogeneous home-cluster inference
- llama.cpp #24616 - dp4a emulation via dp2a
- Quesma - Qwen3.8-27B quantizations benchmarked
