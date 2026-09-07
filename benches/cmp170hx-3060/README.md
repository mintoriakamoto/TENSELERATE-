# CMP 170HX (40 GiB, full unlock) + RTX 3060 — measured results

Numbers measured on the `raven-9950x` box. Every figure here supersedes the
modeled constants in `tenselerate/cli.py` and the estimates in
`docs/rig-cmp170hx-3060.md`; if they disagree, this file wins.

| date | quantity | value | how | notes |
| --- | --- | --- | --- | --- |
| 2026-09-07 | 170HX FP16 tensor throughput | 162-170 TFLOPS | FP16 GEMM bench on the box | dense A100 class; the 256-cycle MMA gate does not reproduce on this unit |
| 2026-09-07 | 170HX power limit | 250 W | `nvidia-smi` | earlier 100 W cap lifted |
| 2026-09-07 | 170HX PCIe | Gen2 x4 | `nvidia-smi` | OTP-fused; ~2 GB/s |
| 2026-09-07 | decode, wrong build (capacity-only throttle workarounds) | 22.5 tok/s | llama-server, single stream | card at 74 W / 83% util |
| 2026-09-07 | decode, normal sm_80 build, no spec | **33.3 tok/s** | llama-server, single stream, Q4_K_M, 2x256K slots, q4_0 KV | +48%; card at 218 W / 98% util; prompt 127 -> 158 tok/s on a short prompt |
| 2026-09-07 | MTP launch, first attempt | OOM | `cudaMalloc failed` allocating the MTP context | VRAM not freed from the previous server; clean relaunch succeeded at 29.6 GiB resident |
| 2026-09-07 | decode, normal build + MTP (`--spec-type draft-mtp --spec-draft-n-max 5`) | 33.3 tok/s | llama-server, single stream, q8_0 KV | **no gain**; MTP acceptance measured at **7-11%** on the DavidAU merge |
| 2026-09-07 | decode, 2x256K slots, normal build, no spec | **48.2 tok/s aggregate, 26.5 per stream** | llama-server `-np 2`, q4_0 KV | wrong build gave 16 / 8.8; model predicted 42-54 / 21-27 |
| 2026-09-07 | `llama-bench -ngl 999 -fa 1 -p 4096 -n 64 -r 3` | **pp4096 855.6 tok/s, tg64 33.5 tok/s** | llama-bench | prefill IS A100 class (>800 predicted); tg = pure weight read = 553 GB/s = 37% of nominal |
| 2026-09-07 | clocks under decode | HBM 1215/1215 MHz, SM 1410/1410 MHz, 207-223 W | `nvidia-smi -l 1` | both at max; nothing throttled |
| 2026-09-07 | CUDA graphs | reused = 247, not disabled | server log | launch overhead is not the gap |
| 2026-09-07 | decode, MTP n-max 5, prose | **14.9 tok/s** | llama-server | MTP is SLOWER than plain (33.5): 7-11% acceptance means the verify pass is pure cost. Drop it. |
| 2026-09-07 | width sweep, short prompts, N=1,2,4,8,16 slots x 16K ctx | **141.5 tok/s aggregate at N=16** | `-np N`, q4_0 KV | step time DROPS from N=8 to N=16; the earlier 87 tok/s ceiling below is disproven |
| 2026-09-07 | 4 x 256K, q4_0 KV | 59 tok/s aggregate | llama-server `-np 4` | what fits at the full window today (MMVQ regime) |
| 2026-09-07 | **4 x 256K, q4_0 KV, `GGML_CUDA_NO_MMVQ=1`** | **70.5 tok/s aggregate** | llama-server `-np 4` | **+19% over 59** at the config that fits; model predicted ~63 |
| 2026-09-07 | 16 x 16K, `GGML_CUDA_NO_MMVQ=1` | ~130 tok/s (reproducible) | `-np 16` | baseline 100-141 was noisy; same MMQ path either way, so no change expected and none seen |

## First reading of 33.3 tok/s (superseded)

30 ms per token against 16.5 GB of weights read as "at most 37-48% of nominal
bandwidth, kernel-side time". The diagnostics below then showed clocks maxed
and CUDA graphs on, and the width sweep replaced this with the two-regime
decomposition further down: ~18.5 ms of weight read (~890 GB/s, 60%) plus a
per-sequence cost set by ggml's matmul dispatch.

## Diagnostics to run next (each answers one question)

1. `llama-bench -m <gguf> -ngl 999 -fa 1 -p 4096 -n 64` — `tg` at a short
   context is the pure weight-read rate (bytes / time = achieved GB/s);
   `pp` at 4096 tells whether prefill is really A100 class (expect >800 tok/s)
   or whether the s8 IMMA path is gated in practice despite the FP16 gpu-burn
   number. A 127-token prompt cannot answer that.
2. `nvidia-smi --query-gpu=clocks.mem,clocks.max.mem,clocks.sm --format=csv -l 1`
   during decode — is HBM at its max clock under load?
3. Server log with `--verbose`, grep `graph` — if CUDA graphs are being
   disabled for the hybrid graph, 64 layers of eager launches is a measurable
   slice of the 30 ms.
4. `svmi-bwprofile.py -m <gguf> --gpu cmp170hx-40` — persists the achieved
   GB/s so `plan`/`info` can stop using the 0.65 guess.

## MTP does not work on this model

The DavidAU TURBO merge changed the trunk; the shipped MTP head was trained
against the base Qwen3.8 trunk and now agrees with it only 7-11% of the time.
Speculation throughput is `accepted / pass_time`; at ~1.1 accepted per pass
there is nothing to amortize, and the doc's "up to 3.5x" (base model, aligned
head) does not transfer. Every speculation lever in the plan - chain MTP, tree
MTP, GDN-state forking - is gated on a drafter that agrees with THIS trunk.
Options, cheapest first: an n-gram / prompt-lookup drafter (model-agnostic,
lossless, only helps on copied spans); re-aligning the MTP head by distilling it
against the merged trunk (small training job, head only); a base-model A/B on
tokens-to-answer, since the merge was chosen for fewer thinking tokens and that
claim is as unmeasured as the MTP one was.

## The decode step: two regimes, not one cost

First reading (now superseded): from single stream 30.0 ms and two streams
41.5 ms, `step(N) = 18.5 + 11.5 N` ms, i.e. ~890 GB/s of weight read (60% of
nominal) plus 11.5 ms per live sequence, attributed to the GDN recurrence and
giving an aggregate ceiling of ~87 tok/s. **N=16 measured 141.5 tok/s (113
ms/step), so the linear model is wrong past N=8.** The weight-read figure
survives; the per-sequence attribution does not.

What the other session identified, and the numbers fit: ggml's CUDA backend
dispatches int8 matmuls by batch width - `ne11 <= 8` goes to `mul_mat_vec_q`
(the dp4a vector path, MMVQ), wider goes to MMQ (`mma.sync` tensor-core GEMM).
The "11.5 ms per sequence" at N <= 8 is mostly MMVQ's per-column cost on this
card; above 8 the tensor-core GEMM amortizes columns and the per-sequence cost
falls to ~5.6 ms (113 = 18.5 + ~5 ms of 16K-window KV + 16 x 5.6). That
residual is the real per-sequence work (GDN recurrence + attention + MMQ
column cost); the GDN alone is smaller than first claimed.

```
MMVQ regime (N <= 8):  step = 18.5 + 11.5 N  (+5.4 per slot holding a full 256K q4_0 window)
MMQ  regime (N >  8):  step = 18.5 +  5.6 N  (+5.4 per full slot)
```

The fork already ships the knob that moves the boundary: `GGML_CUDA_NO_MMVQ=1`
forces MMQ at every width. The rig doc told this card NOT to set it (it was a
capacity-only-unlock workaround); on the measurements it is the opposite -
the tensor cores are the fast path here at every N. Predictions if the
hypothesis holds (to be checked against the N=9 / N=12 knee and the
NO_MMVQ runs now in flight):

| config | MMVQ (today) | NO_MMVQ=1 (predicted) |
| --- | --- | --- |
| N=1, short ctx | 33 tok/s (30 ms) | ~41 (24 ms) - only if MMQ at M=1 is not slower; measure |
| N=2, short ctx | 48 | ~67 |
| N=4, short ctx | 62 | ~97 |
| N=4 x full 256K q4_0 | 59 (measured) | ~63 - the KV read (4 x 5.4 ms) is now half the step |
| N=8, short ctx | 72 | ~125 |
| N=9 (first MMQ width), short | - | ~130: a visible knee vs N=8 |
| N=16 x 16K | 141.5 (measured) | same path, same number |

Two things follow. At the full 256K window with several slots, the KV read is
the other half of the step, so after NO_MMVQ the next lever for deep contexts
is bytes (provable page skipping in the reference, q4_0 K fidelity A/B), not
compute. And the aggregate ceiling in the MMQ regime is ~1/5.6 ms = ~180 tok/s
from per-sequence work, before any GDN batching.

**Confirmed at the config that matters (4 x 256K): 59 -> 70.5 tok/s with
`GGML_CUDA_NO_MMVQ=1`.** Step time 56.7 ms against a predicted 62.5 - the KV
read at depth is a little cheaper than the 5.4 ms/slot assumed, which the
depth sweep now running will pin. At N=16 the flag changes nothing (both paths
already MMQ), as expected. Still open: **N=1 with NO_MMVQ** - a single active
slot is the common state of an agent loop between subagent bursts, and MMQ at
M=1 has to be no slower than MMVQ before the flag becomes the default.

Depth-sweep predictions (single stream, llama-bench decode after a deep
prefill; the weight read is ~18.5 ms so the KV term is what moves):
q4_0 KV 34->18 KiB/token: 131K adds ~2.7 ms -> ~30 tok/s; 262K adds ~5.4 ms
-> ~28 tok/s. q8_0: 262K adds ~10.8 ms -> ~24 tok/s. If measured decode at
262K lands near 28-30 the bytes model holds and page-skip / q4_0 K are worth
exactly the fraction of that KV term they remove.

## Still to measure

Leads from this week's scan (drafters that run on upstream llama-server, the
fork being six weeks behind, MTP acceptance disputed by the model author's own
numbers): `docs/research-week-2026-09-07.md`, test plan at the end.


- N=9 and N=12 (locates the MMVQ->MMQ knee; running)
- `GGML_CUDA_NO_MMVQ=1` at **N=1** (decides whether the flag is the default; N=4 x 256K done: 70.5)
- depth sweep: single-stream decode at 131K and 262K filled KV (running; predictions above)
- per-op profile of one MMQ-regime step (`nsys profile`, llama-bench `-n 16`) to
  split the remaining ~5.6 ms/sequence between GDN, attention and GEMM
- tokens-to-answer, DavidAU merge vs base Qwen3.8 (decides the model)
- UD-Q4_K_M vs mixed-INT8; q8_0 vs q4_0 KV at the 262K window
