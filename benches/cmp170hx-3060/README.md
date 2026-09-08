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
| 2026-09-07 | depth sweep, single stream, q8_0 KV, batch 1 | 0: 33.5 / 16K: 30.3 / 65K: 23.5 / 131K: 18.2 / **262K: 12.4 tok/s** (29.9 -> 80.9 ms) | llama-bench, prefill to depth then time decode | prefill 856 -> 327 tok/s over the same range; KV term at 262K is **51 ms**, predicted 10.8 |
| 2026-09-07 | n-gram speculation, single stream | baseline 34.4 / `ngram-cache` **16.9** / `ngram-mod` 34.4 tok/s | llama-server | halved or flat: near-zero acceptance on prose and the verify batch is paid in full (see docs/mtp-realign-davidau.md) |
| 2026-09-07 | **MTP draft-depth sweep, code decode, single stream** | baseline 34.4 / **n-max 1: 46.6 (+35%)** / n-max 2: 39.7 (+15%) / n-max 3: 29.2 (-15%) / n-max 5: 26.8 (-22%) | llama-server `--spec-type draft-mtp` on the -MTP- GGUF | **the head is shallow, not broken**: position 1 accepts reliably, positions 2+ do not. Every earlier "MTP loses" number was n-max 5. Prose/JSON verification running |
| 2026-09-07 | **MTP n-max 1 across output shapes, single stream** | JSON/tool calls **47.5 (+38%)**, code 46.6 (+35%), prose 39.0 (+13%) vs 34.4 | llama-server `--spec-draft-n-max 1` on the -MTP- GGUF | holds on every shape Hermes emits; 8-slot + delegation test running |
| 2026-09-07 | **MTP depth sweep under `GGML_CUDA_NO_MMVQ=1`**, code, single stream | n-max 1: **30.6** / 2: 40.0 / 3: 32.7 / 4: 31.6 / **5: 46.1** (vs MMVQ 46.6 / 39.7 / 29.2 / 26.3 / 26.8) | llama-server | MMQ loses at width 2, wins from width ~4; both paths peak at ~46.5 |
| 2026-09-07 | MTP sweep on a build with MMVQ max batch = 2 | peak 46.6 at n-max 5 | rebuilt llama.cpp | same ceiling by a different route |
| 2026-09-07 | q4_0 vs q8_0 KV at depth, single stream | **21.5 vs 23.5 tok/s (-8%)** | llama-bench | the dequant costs more than the bytes save on the vector attention path; q4_0 only helps fit |
| 2026-09-07 | `--kv-unified`, single stream | no change | llama-server | a batching detail; matters only with several slots |
| 2026-09-07 | `reasoning_effort low` | 0% on tok/s; large on time-to-answer | llama-server | fewer tokens, same speed per token |
| 2026-09-07 | **MTP depth 1 on the real Hermes server** (chat template + reasoning, production sampling and slots) | **33.8 vs 34.4 tok/s - no gain** | llama-server as Hercules uses it | resolved below: the sampling params were fighting the draft |
| 2026-09-07 | **production server, MTP depth 1, greedy** (`--temp 0`, no repeat penalty) | **46.2 tok/s, draft acceptance 0.881** | llama-server, server log `draft acceptance` | the microbench gain is real on the production server |
| 2026-09-07 | production server, MTP depth 1, model-card sampling (temp 0.7, repeat-penalty 1.15) | **29.9 tok/s, draft acceptance 0.218** | same server, same head, same flag | **13% below the no-MTP baseline**: a rejected draft is a paid verify column. Suspect 1 confirmed; `-np 1` not needed |
| 2026-09-07 | single-GPU width test, N=16 streams, `-np 24 -c 393216`, q8_0 KV | **122.4 tok/s aggregate** | llama-server, 170HX only | consistent with the 130-141 MMQ-regime ceiling at 16K ctx; the extra KV depth costs the rest |
| 2026-09-07 | dual-GPU layer split of the 27B, 170HX + 3060 | OOM on first try; conservative retry (split 78/22, `-np 16 -c 262144`) running | llama-server `CUDA_VISIBLE_DEVICES=0,1` | **prediction, written before the result:** slower per token than the 170HX alone - see "Splitting the 27B across both cards" |
| 2026-09-07 | greedy MTP depth 1, no reasoning-budget injection, JSON task | valid JSON at ~46 tok/s; **the merge still loops inside `<think>` under pure greedy** | production server | the loop is the merge under greedy, not the budget cap. Temp 0.15 / 0.3 / 0.5 sweep running for the escape point that keeps acceptance |
| 2026-09-08 | **N=32 streams** (`launch_np32.sh`) | **134 tok/s aggregate** | llama-server, 170HX only | grades the two predictions on record: the box session's 130-180 holds, the 170-220 above was too high. Aggregate is flat from N=16 (122) to N=32 (134): the per-sequence cost (GDN block + KV) stopped shrinking at N=16, as the width sweep said. Tensor-core utilisation is not the lever; bytes are |
| 2026-09-08 | prefill through the server, before / after `-b`/`-ub` raised | 40 -> **316 tok/s** | llama-server, Hermes prompt | the 40 was a misconfiguration (small micro-batch, re-prefilling the 35K prompt on every request). 316 is still 2.7x below llama-bench pp4096 = 856 on the same card: the rest is `-ub 2048 -b 4096` and the prompt cache actually hitting (`--cache-ram`, LCP slot selection); see HERCULES.md |
| 2026-09-08 | Hermes production: draft depth 8 -> 4, KV q8_0 -> q4_0 | server-reported draft acceptance 27-42% -> 46-65%, first-token latency 1800 -> 500-800 ms, prompt-cache thrash (3.8 GB re-prefill per request) gone | llama-server + Hermes, "9.0/10" self-scores | the acceptance figure is the server's accepted/drafted ratio, not the per-position 0.88 (greedy) / 0.22 (temp 0.7) measured here; depth 4 -> more accepted per drafted, not more tokens per second - the tok/s was not reported and must be. q4_0 KV halves KV but measured -8%/token at depth vs q8_0 |
| 2026-09-08 | Hermes delegation: 6 parallel sub-agents against the local server | all 6 dispatched in 170 ms, ran in parallel, returned | Hermes `delegate_task` x6, provider :8082 | the delegation path works end to end. This is the workload the bounded window (PR #62) is for: 6-16 slots, each with the 35K prompt pinned |
| 2026-09-08 | box GDN change (unconditional `ggml_repeat` head broadcast) | no single-stream gain | llama-server | as predicted: the fused kernel already broadcasts in-kernel; the guarded repeat on main is the correct code |

## First reading of 33.3 tok/s (superseded)

30 ms per token against 16.5 GB of weights read as "at most 37-48% of nominal
bandwidth, kernel-side time". The diagnostics below then showed clocks maxed
and CUDA graphs on, and the width sweep replaced this with the two-regime
decomposition further down: ~18.5 ms of weight read (~890 GB/s, 60%; this is the *intercept of the width sweep*, not a direct measurement - end to end, a 29.9 ms batch-1 step over 16.5 GB is 553 GB/s = 37% of nominal, and the ~11 ms difference is the GDN recurrence + attention + launch time that width amortizes) plus a
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

## MTP works at depth 1 - and the verify-cost model predicted the whole curve

The merge broke the head's positions 2+ but left position 1 intact. With
`c ~ 11.5 ms` per verified column on the dp4a path and ~1.9 tokens accepted
per pass at n-max 1 (position 1 plus the bonus token):

| n-max | pass = 18.5 + (n+1) x 11.5 | accepted (est.) | predicted | measured |
| --- | --- | --- | --- | --- |
| 1 | 41.5 ms | ~1.9 | 46 tok/s | **46.6** |
| 2 | 53 ms | ~2.1 | 40 | **39.7** |
| 5 | 87.5 ms | ~2.3 | 26 | **26.8** |

## MTP depth 1 nets zero on the real server - resolved: sampling

**Resolved 2026-09-07.** Suspect 1 below was it. The same production server,
same head, same `--spec-draft-n-max 1`:

| sampling | decode | draft acceptance |
| --- | --- | --- |
| greedy (temp 0, no repeat penalty) | **46.2 tok/s** | **0.881** |
| model card (temp 0.7, repeat-penalty 1.15) | 29.9 tok/s | 0.218 |

The mechanism is in `common/sampling.cpp` (`common_sampler_sample_and_accept_n`):
a draft token is accepted only if the token the *sampler* produces at that
position equals it. The head predicts the argmax; at temperature 0.7 the
sampler leaves the argmax often enough to cut acceptance to ~0.2, and
repeat-penalty 1.15 rewrites the argmax on every token that appeared recently
(code and JSON are nothing but recent tokens). Each rejected draft is a
verify column paid in full, so sampled MTP lands *below* the 34.4 no-MTP
baseline. Greedy is now the server default in the launch
(`--temp 0 --repeat-penalty 1.0`, `--sampling client` to opt out). These are
request defaults: a client that sends `temperature` or `repeat_penalty` wins,
so Hermes must not send them (HERCULES.md).

**The loop showed up.** Pure greedy loops inside `<think>` on this merge
(confirmed with the reasoning-budget injection removed, so it is the merge,
not the cap). Two guards that keep most of the argmax, both in the launch:

- `--sampling dry`: greedy plus DRY (`--dry-multiplier 0.8`, allowed length 2,
  scan window 2048). DRY only touches tokens that would *extend* a repeated
  n-gram, so on non-looping text it is exactly greedy and acceptance should
  stay ~0.88. The scan window is capped at 2048: the default (-1) scans the
  whole 262K context on the CPU for every token.
- `--sampling low`: temp 0.3, min-p 0.1, no repeat penalty. Where the
  distribution is peaked (most code/JSON tokens) the sampled token is still
  the argmax; where it is flat it can leave the loop. Acceptance will sit
  between 0.88 and 0.22; the box's 0.15/0.3/0.5 sweep finds the knee.

Repeat penalty stays at 1.0 in every mode: it rewrites the argmax on every
recently seen token, which is what took acceptance to 0.22.

The original list, kept for the record:

1. **Sampling.** llama.cpp accepts a draft token only if the target's *sampled*
   token matches it. The microbench sampled greedily; Hermes sends temperature
   ~0.7-1.0 with top-p/top-k, so position-1 acceptance falls from ~0.9 toward
   the sampler's own agreement rate, and at ~1.3 accepted per pass the 41.5 ms
   pass is a wash. Run: the production server with `--temp 0` (or Hermes
   `model.temperature: 0.2`) - if the gain returns, sampling is the cause, and
   low temperature is anyway right for tool-calling turns.
2. **Width.** With several slots active, each step verifies 2 columns per
   slot: 4 slots -> width 8 on the dp4a path, ~99 ms per step. MTP on a
   multi-slot dp4a server is the worst quadrant of the cost model above; it
   pays only on MMQ (width >= 4 -> `MMVQ_MAX=3`) or with one slot. Run: `-np 1`
   with the Hermes prompt.
3. **Thinking text.** Reasoning tokens are prose-shaped (+13% in the
   microbench, not +38%), and at `low` they are still a large share of the
   output. Nothing to run; it bounds the upside even when 1 and 2 are fixed.

Run 1 landed and returned the whole gain; run 2 (`-np 1`) is moot for the
single-stream number. Under 4 active slots MTP still verifies 2 columns per
slot on dp4a (width 8, ~99 ms per step) - that is the remaining case for
`MMVQ_MAX=3`, below.

**The ~64 tok/s prediction for NO_MMVQ + depth 1 was wrong: measured 30.6.**
The per-column cost model held for the dp4a path and failed for MMQ. Fitting
all the single-sequence points (accepted per pass ~1.9 at n-max 1, ~2.6 at
n-max 5):

```
MMVQ (dp4a):  pass(m) ~ 18.5 + 11.5 (m-1) ms   linear in the verified width m
MMQ (tensor): pass(m) ~ 55 ms, roughly flat from m = 2 to m ~ 16
```

MMQ is not "5.6 ms per column"; it has a ~55 ms floor at small width on this
card - the int8 MMA tiles are sized for wide batches and the kernel is
dequant/tile-bound, not bandwidth-bound, when only a few columns are live.
The width sweep's "5.6 ms per sequence" was that floor amortized over 16
sequences. Consequences:

- Crossover at m ~ 3-4: the dp4a path wins for widths 1-3, MMQ from ~4 up.
  That is exactly why `NO_MMVQ=1` gave +19% at 4 x 256K (width 4) and -34%
  at n-max 1 (width 2), and why a build with MMVQ capped at width 2 peaks at
  n-max 5 (width 6, on MMQ) at the same 46.6.
- **The right value for the fork's new `GGML_CUDA_MMVQ_MAX` is ~3**, not 1:
  single-slot decode and depth-1 verification stay on dp4a, four-slot steps
  and any wider verification go to MMQ. To be measured; the two extremes are.
- Single-stream ceiling with this head: max over paths of accepted / pass =
  1.9 / 41.5 ms (dp4a, n-max 1) = 46.6, or 2.6 / 56 ms (MMQ, n-max 5) = 46.1.
  Same number by two routes, as the other session found. Nothing in routing
  moves it further; only more accepted tokens per pass do.
- Retrained head (healthy curve 0.86/0.77/0.67 by position): ~3.1 accepted
  at n-max 3 -> 3.1 / 53 ms (dp4a) = ~58 tok/s, or ~3.9 at n-max 5 -> 3.9 /
  56 ms (MMQ) = **~70 tok/s**. That replaces the ~90 written earlier.
- Past that, the kernel lever is MMQ's small-width floor: a tile shape (or
  stream-K split) efficient at 2-8 columns would put verification at
  ~18.5 + small, and the same head would give ~85-95.

Calibration for the retrain (KGP Talkie, base Qwen3.8 UD-Q4_K_XL on a 5090,
45 configs): acceptance by draft position 0.86 / 0.77 / 0.67 / 0.59 / 0.52
for n-max 1..5, throughput peaking at n-max 3 (133.6 tok/s vs 73.6 plain,
1.8x). A healthy head keeps ~0.77 at position 2 and ~0.67 at 3; ours holds
position 1 and loses the rest. That curve is the target for
docs/mtp-realign-davidau.md, and the reason the optimum there is n-max 3-4. Retraining the head
(docs/mtp-realign-davidau.md) then restores positions 2+ and moves the optimum
to n-max 3-4 at ~3 accepted: ~90 tok/s on the MMQ path.

## Speculation is not dead here; the dp4a verify pass is

Every speculative method measured so far lost or broke even: MTP (7-11%
acceptance) and n-gram (`ngram-cache` halved throughput). The other session
read that as structural - "compute-bound, speculation pays the GDN
recurrence". The width sweep gives the actual cost: a verify pass of m
tokens is `18.5 + m * c` ms with c ~ 11.5 ms on the batch<=8 dp4a path and
~5.6 ms on the MMQ path. A 4-draft pass therefore costs 76 ms today
(break-even 2.5 accepted tokens) and 46.5 ms under `GGML_CUDA_NO_MMVQ=1`
(break-even 1.55). With an unmatched head or prose n-grams, ~1 token is
accepted per pass: a 76 ms pass for one token is exactly the halving seen.
A head matched to the merge (3+ accepted, the DimInfer/RadixArk numbers) on
the MMQ path is ~2x. Recipe and the direct verify-cost measurement:
`docs/mtp-realign-davidau.md`.

## Cross-check against a laptop RTX 5090 (Ferrox Field Manual No.12)

Their 5090 laptop: 896 GB/s, 40.8 tok/s single stream on the same model class
(24.5 ms/token). Our 170HX: 1.37-1.49 TB/s nominal, 33.5 tok/s (29.9 ms).
1.5x the bandwidth, slower decode - so the 170HX's step is not the weight
read. Put the two through the same decomposition:

```
5090:  24.5 ms = ~23 ms weight read (16.5 GB at ~80%) + ~1.5 ms per-sequence
170HX: 29.9 ms = ~18.5 ms weight read (60%)          + ~11.5 ms per-sequence
```

Same llama.cpp kernels, 8x the per-sequence cost. The width sweep already
located that cost: the batch<=8 dp4a GEMV path (MMVQ) costs ~11.5 ms per
sequence here and the tensor-core MMQ path ~5.6 ms, so the dp4a path is slow
on this GA100 in a way it is not on a 5090 - whatever the unlock wiki says
about dp4a being "uncrippled". The clean test is `GGML_CUDA_NO_MMVQ=1` at N=1:
the model predicts ~24 ms -> ~41 tok/s, i.e. **parity with the 5090 per
token**. Until that number exists, "compute-bottlenecked by the MMA gate" is
the wrong story - the gate is measured absent (162-170 TFLOPS); the slow path
is the vector GEMV.

The manual's other levers, mapped: an intact MTP head (+137%, +250% on JSON
at n-max 10) - ours measured broken on the merge in this fork, disputed by
the model author's own 60%, decided by the upstream-build / stock-Unsloth
test now running; `reasoning_effort low` (4.2x time-to-answer) - already in
the launch; `GGML_CUDA_F16=ON` at build time (+250% claimed) - a cheap
rebuild, but that flag changes the f16 intermediates of the cuBLAS prompt
path, so expect it on prefill, not decode; n-gram speculation for structured
output - being A/B'd, lossless and model-agnostic, only helps on copied spans.

## Depth: the KV read costs 5x what bytes say, and the code says why

At 262K with q8_0 KV a single stream decodes at 12.4 tok/s: 80.9 ms per token,
of which ~51 ms is the KV read. 8.5 GiB of q8_0 KV at ~1 TB/s should be ~9 ms.
The effective rate is ~178 GB/s - the other session's "latency-bound, not
bandwidth-bound" reading of it. The dispatch in `ggml/src/ggml-cuda/fattn.cu`
(`ggml_cuda_get_best_fattn_kernel`) gives the mechanism on sm_80:

- **quantized K or V + one query token -> `BEST_FATTN_KERNEL_VEC`**, the
  vector kernel, which has no GQA batching: each of the Q heads streams its own
  copy of its KV head. With 6 query heads per KV head that is 6 x 9.1 GB
  ~ 55 GB per token, and 55 GB at ~1.07 TB/s is 51 ms. It is bandwidth-bound
  after all - on six-fold redundant reads.
- **f16 K and V -> `BEST_FATTN_KERNEL_MMA_F16` with the GQA optimization**
  (`gqa_opt_applies`), which reads each KV entry once for all six query heads
  and splits the KV length across blocks (stream-K).

Prediction, single stream at 262K with `-ctk f16 -ctv f16`: KV read ~16.5 GB
once, ~16 ms -> ~46 ms/token -> **~22 tok/s vs 12.4**, i.e. ~1.8x at depth
from a KV-type flag, at the cost of 16 GiB of KV per 262K slot (one deep slot
fits, not four). q8_0 K with f16 V does not help: any quantized tensor selects
the vec path. This is the highest-value single test now on the box; if it
holds, the Hercules choice becomes "one deep f16 slot" vs "four q8_0 slots",
and the real fix is a GQA-aware vec kernel for quantized KV (or upstream having
added one since the fork's July base - check before writing it).

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

## How to run the open items

`run-open-items.sh` works the list below unattended: one `llama-server` per
configuration on `127.0.0.1:8089` (a production server on 8080 is left alone),
launched through `python3 -m tenselerate serve --backend llamacpp`, health-polled
for up to 10 min (the load is slow over PCIe Gen2 x4), measured, killed, and
GPU 0's VRAM watched until it is back under 2 GiB before the next launch (a
stale server OOMed the MTP launch once). Rows land in `results-<date>.md` with
this table's columns; fold the keepers into the table above by hand.

```
MODEL=/models/Qwen3.8-27B-TURBO-MTP-Q4_K_M.gguf bash benches/cmp170hx-3060/run-open-items.sh
EXPERIMENTS=loop_guard,deep_kv MODEL=... bash benches/cmp170hx-3060/run-open-items.sh   # a subset
DRY=1 MODEL=... bash benches/cmp170hx-3060/run-open-items.sh                            # print the launches
bash benches/cmp170hx-3060/run-open-items.sh --self-test                                # no GPU needed
```

Experiments, in order: `loop_guard` (`--sampling greedy` / `dry` / `low`, and
`low` with client temp 0.15 / 0.5 - tok/s, draft acceptance and a `<think>`
loop check per shape; the loop-free mode with the highest tok/s is written to
`logs/winning-sampling`), `client_override` (greedy server, request carries
temp 0.7 + repeat_penalty 1.15), `deep_kv` (1 slot x 262144, q8_0 vs f16,
~250K-token synthetic prefill, decode at depth), `slot4_width` (4 slots, MTP
depth 1, default routing vs `--mmvq-max 3` vs `--no-mmvq`, 4 concurrent
streams), `mtp_control` (MTP off vs on under the winner). One failing
experiment does not stop the rest; a summary prints at the end.

`measure.py` is the stdlib client behind it: `stream: false` chat completions on
three fixed prompt shapes (JSON tool calls, code, prose), reading llama-server's
`timings` object - `predicted_per_second`, `predicted_n`, `prompt_per_second`,
and `draft_n` / `draft_n_accepted` when a drafter ran. It sends no
`temperature` / `top_p` / `repeat_penalty` unless `--send-sampling temp=..,rp=..`
is given, so what it measures is the server's request defaults - the
same contract Hermes has to honour. `--concurrency N` occupies N slots behind
distinct long prefixes and reports aggregate and per-stream tok/s;
`--loop-check` flags a 12-gram repeated 4+ times. Logs: `logs/<experiment>-<ts>.log`
plus the server's own log (`grep "draft acceptance" logs/*server*.log`).

## The N=32 "300-400 tok/s" plan - predictions written before the run

A proposal from another session: `-np 32 -c 1048576` with q4_0 KV plus
`-bs`, `--stream-weights 2`, `--repack`, projecting 300-400 tok/s aggregate
and "99% TFLOPS utilization" at N=64. The measured cost model says otherwise,
and the flags are not what their names suggest:

| flag | what it is | expected effect here |
| --- | --- | --- |
| `-bs` / `--backend-sampling` | samples on the GPU instead of the CPU | the only one that can help: at N=32 the CPU sampler is a few ms per step; worth measuring, maybe +5% |
| `--stream-weights 2` | SVMI host-pinned weight streaming over PCIe for models that do not fit VRAM | the model fits; streaming through Gen2 x4 (~2 GB/s) can only be slower. Expect zero or negative |
| `--repack` | CPU-backend weight repacking (extra buffer types) | no CPU layers with `-ngl 999`; no effect |
| q4_0 KV | fits more, reads slower | measured -8% per token at depth vs q8_0; helps only if q8_0 does not fit 32 x 32K |

The aggregate ceiling is bytes, not TFLOPS. A step at N=32 is the ~55 ms MMQ
floor plus ~5.6 ms per sequence beyond 16, plus the KV read for 32 live
contexts; at 32 x 32K that is on the order of 145-190 ms per step, so
**~170-220 tok/s aggregate**, not 300-400. N=64 adds another ~180 ms of
per-sequence cost and doubles the KV read; the step lengthens faster than the
width grows, so aggregate flattens around 250 at best. Tensor-core
utilization at these widths is 5-10% and stays there; the weight read is
already at ~60% of nominal bandwidth by the width-sweep intercept (37% end to end at batch 1), which is where a Q4_K_M dequant lands.

None of this is Hercules' workload. One operator is 1-4 slots; the number
that matters there is per-turn latency (46 tok/s greedy MTP), and every
extra slot on the same card slows it. Measure N=32 if the goal is a
multi-tenant server; do not tune the Hercules launch by it.

On "the card goes compute-bound at N=16": the sweep shows the *step* still
getting cheaper per sequence from N=8 to N=16 (the MMQ regime) and the
aggregate flattening because the per-sequence cost (~5.6 ms, the GDN block's
per-token work plus KV) stops shrinking. Bytes, not flops: tensor-core
utilisation at N=16 is under 10%. Two predictions for N=32 are on record,
~130-180 (a box session) and ~170-220 (above); they overlap at 170-180 and
the run decides. q4_0 KV cannot be the relief either way: it measured 8%
slower per token than q8_0 at depth.

`--logit-bias TOKEN-INF` for the `<think>` loop bans one token id; a
repetition loop is a phrase, not a token, so it moves the loop rather than
ending it. `--sampling dry` / `--sampling low` are the two guards under test.

## Splitting the 27B across both cards - why it cannot be faster

A layer split runs the two cards *in sequence* for every token: the 170HX
does its layers, ships one hidden state (10 KB) over PCIe Gen2 x4 (~50 us,
negligible), then the 3060 does its layers. Per-token time is the sum, and
the 3060 reads its share at 360 GB/s against the 170HX's ~890 GB/s measured:

```
170HX alone      : 15.5 GB / 890 GB/s              ~ 17.4 ms
78/22 split      : 0.78 x 17.4 + 0.22 x 15.5/360   ~ 13.6 + 9.5 = 23.1 ms   (-25% per token)
any split x      : 17.4 (1 - x) + 43 x             - grows with every layer moved
```

The 3060 has a quarter of the bandwidth, so every layer moved onto it costs
2.5x what it saved on the 170HX. The split buys VRAM (more slots, more KV),
never speed, and on a 40 GiB card that holds 4 x 256K already the VRAM is
not the constraint. The row above states this before the measurement lands;
if the split measures faster, the sequential-execution model is wrong and
this section gets rewritten.

What the 3060 is for: a second, small model (scripts/hercules_side_serve.sh)
running *in parallel* with the main loop, taking Hermes' delegation children
and compaction off the 27B. Parallel, not sequential, is the only way the
second card adds throughput.

### llama-bench experiments in the runner (no server)

`EXPERIMENTS=bench_ab,prefill_ubatch,quant_ab` runs directly through
llama-bench: the regression bisect (this binary with the packed attention off
and forced on, at 512 and at 64K depth, plus `LLAMA_BENCH_OLD=` an older
binary), the prefill micro-batch sweep (`-ub 512/1024/2048` at pp4096), and
the quant A/B (`MODEL_ALT=` a Q4_0 or IQ4_XS GGUF). Rows land in the same
results file.

## The "TENSELERATE's GDN is corrupted" diagnosis - checked, stale

A parallel session concluded that MTP acceptance was 3x lower here than on
upstream because of four code differences, and recommended debugging those
rather than the GGUF. Checked against `main` at 6ac432de, all four are gone or
were misread. PR #52 took the recurrent/GDN/drafter machinery back to upstream
wholesale, which is exactly the surface the diagnosis names.

| claimed difference | state on main |
| --- | --- |
| `d2t` draft-vocab trim | every file touching `d2t` (`llama-model.h`, `dflash.cpp`, `eagle3.cpp`, `llama-arch.cpp`) is identical to upstream |
| multi-layer hidden-state tap | removed in #52 with the capture API; zero references remain |
| `build_recurrent_attn` extra `state_rows` parameter | removed in #52 with the rows-mode GDN op; the signature is upstream's |
| `S_v = v->ne[0]` vs upstream `S_v = s->ne[0]` | a misreading: **both** forms are in upstream's own `delta-net-base.cpp`. The chunking / autoregressive / fused builders take the value tensor (`v->ne[0]`, lines 29/302/386); `build_recurrent_attn` takes the state (`s->ne[0]`, line 541). The file is **byte-identical** to upstream |

`git diff upstream/master -- src/models/delta-net-base.cpp src/llama-graph.* ggml/src/ggml-cuda/gated_delta_net.cu common/speculative.cpp` is empty. There is no fork-side GDN or drafter code left to corrupt: the whole fork footprint in upstream-owned directories is the CUDA perf work, kv-mean-center, the mmap host-pin, `Q4_K_M_INT8`, the SVMI flags, the bounded-window hook and the server guards.

The *numbers* in that report are still worth settling, because they do not match
this ledger: 23.5 tok/s single stream is our **wrong-build** row (card at 74 W),
and 110 tok/s aggregate at np=8 would beat our measured 70.5 at 4x256K. Both
sides of that comparison need to run here, same model, same flags:

```bash
MODEL=/path/model.gguf bash benches/cmp170hx-3060/fork-vs-upstream-ab.sh
```

It builds upstream `master`, serves the same GGUF from each binary in turn, and
prints single-stream tok/s, np=8 aggregate and the server's own draft-acceptance
line for both. If the two agree within noise the fork's GDN path is exonerated
by measurement, not by argument. If upstream is materially faster, that summary
is the bisect ticket.

## Still to measure

Leads from this week's scan (drafters that run on upstream llama-server, the
fork being six weeks behind, MTP acceptance disputed by the model author's own
numbers): `docs/research-week-2026-09-07.md`, test plan at the end.


- N=9 and N=12 (locates the MMVQ->MMQ knee; running). N=32 landed at 134 (row above); the knee is the only open width
- **server prefill vs llama-bench prefill**: 316 vs 856 tok/s on the same card. `-ub 2048 -b 4096` and a prompt-cache hit rate from the server log; the gap is configuration until proven otherwise
- **Hermes production tok/s** at draft depth 4 / q4_0 KV: the box reported acceptance and latency but not tokens per second; the `timings` object of one long completion decides whether depth 4 beats depth 1 greedy (46.2)
- `GGML_CUDA_MMVQ_MAX=3` (the fork's threshold): 4 slots x 256K with MTP depth 1 - predicted to match NO_MMVQ's 70.5 on the step while keeping single-slot turns on the dp4a path
- production server, greedy MTP depth 1 with **Hermes as the client** (not curl): confirms Hermes sends no `temperature`/`repeat_penalty`; the server log's `draft acceptance` line should stay ~0.88
- loop guard: `--sampling dry` vs `--sampling low` at temp 0.15 / 0.3 / 0.5 - tok/s, `draft acceptance`, and whether `<think>` still loops; the winner becomes the default
- 4 slots active, greedy MTP depth 1, dp4a vs `MMVQ_MAX=3`: whether verification width 8 pays on either path
- 8 slots + MTP depth 1 aggregate with `MMVQ_MAX=3`
- stock unsloth/Qwen3.8-27B-UD-Q4_K_M + its MTP head on the same build (acceptance; running) and the same file on an upstream build
- `GGML_CUDA_F16=ON` rebuild: pp4096 and tg64 side by side with the current build
- **262K single stream with `-ctk f16 -ctv f16`** (prediction ~22 tok/s vs 12.4; decides the KV type for deep slots)
- **GQA-packed vector attention** (`docs/kernel-work.md` item 1, in the tree, unrun): `test-backend-ops -o FLASH_ATTN_EXT` with `GGML_CUDA_FATTN_VEC_GQA=1` first, then the 262K q8_0 tg number with the variable at 0 and 1 - prediction: the 51 ms KV term falls toward ~15 ms, 12.4 -> ~20 tok/s at q8_0 without the f16 memory cost
- per-op profile of one MMQ-regime step (`nsys profile`, llama-bench `-n 16`) to
  split the remaining ~5.6 ms/sequence between GDN, attention and GEMM
- tokens-to-answer, DavidAU merge vs base Qwen3.8 (decides the model)
- UD-Q4_K_M vs mixed-INT8; q8_0 vs q4_0 KV at the 262K window
