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

## What 33.3 tok/s says

30 ms per token. The weights alone are 16.5 GB, so the loop is achieving at most
**~550 GB/s on a short prompt (~37% of the 1493 GB/s nominal), ~710 GB/s if the
full 262K q4_0 window was being read (~48%)**. The planners assume 65%
(`BW_EFFICIENCY` in `tenselerate/cli.py`); the box does not deliver that yet.
The card reads 98% util while moving under half its bandwidth, which points at
kernel-side time (launch gaps, the GDN recurrence kernels, graph disablement)
rather than the HBM being the wall. Model prediction was ~45 tok/s; the gap is
the number to chase before any speculation lever.

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

| 2026-09-07 | decode, normal build + MTP (`--spec-type draft-mtp --spec-draft-n-max 5`) | 33.3 tok/s | llama-server, single stream, q8_0 KV | **no gain**; MTP acceptance measured at **7-11%** on the DavidAU merge |

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

| 2026-09-07 | decode, 2x256K slots, normal build, no spec | **48.2 tok/s aggregate, 26.5 per stream** | llama-server `-np 2`, q4_0 KV | wrong build gave 16 / 8.8; model predicted 42-54 / 21-27 |

| 2026-09-07 | `llama-bench -ngl 999 -fa 1 -p 4096 -n 64 -r 3` | **pp4096 855.6 tok/s, tg64 33.5 tok/s** | llama-bench | prefill IS A100 class (>800 predicted); tg = pure weight read = 553 GB/s = 37% of nominal |
| 2026-09-07 | clocks under decode | HBM 1215/1215 MHz, SM 1410/1410 MHz, 207-223 W | `nvidia-smi -l 1` | both at max; nothing throttled |
| 2026-09-07 | CUDA graphs | reused = 247, not disabled | server log | launch overhead is not the gap |
| 2026-09-07 | decode, MTP n-max 5, prose | **14.9 tok/s** | llama-server | MTP is SLOWER than plain (33.5): 7-11% acceptance means the verify pass is pure cost. Drop it. |

## The decode step, decomposed from two points

Single stream 30.0 ms/step; two streams 41.5 ms/step, both with short prompts
(so almost no KV bytes in either). Weights are read once per step, so:

```
step(N) = T_weights + N * t_seq
30.0 = T_w + t_seq        41.5 = T_w + 2 t_seq
=> t_seq = 11.5 ms per sequence     T_w = 18.5 ms  -> 16.5 GB / 18.5 ms = ~890 GB/s (60% of nominal)
```

The implied weight-read efficiency (60%) is right where the planners' 65%
guess sits, so the decomposition is physically credible: **the HBM is fine. The
missing 40% of the token is ~11.5 ms of per-sequence work that is not bytes** -
with clocks maxed and CUDA graphs on, that is the 48 Gated-DeltaNet layers'
recurrence path (state update + conv + gates + norms per layer, ~240 us per
layer per token). At that size it is neither bandwidth nor FLOPs; it is
many small kernels each running far below the card's occupancy. This is
exactly what the other session concluded qualitatively ("genuine per-token
compute, the lever is width"); the two points put a number on it.

Consequences:

- Aggregate throughput saturates at `1 / t_seq` = **~87 tok/s** no matter how
  many slots are added, as long as t_seq does not shrink with batching. The
  1 -> 2 slot step did NOT amortize it (the second sequence cost the full
  11.5 ms), which says the GDN path is serialized per sequence today.
- Add ~5.4 ms per slot that holds a full 256K window at q4_0 (10.8 at q8_0)
  for the KV read; that is the only part of a slot's cost that is bytes.
- The lever is a GDN kernel that processes all live sequences in one launch
  (width). If t_seq fell to ~2 ms, 8 slots would step in ~35 ms -> ~230 tok/s
  aggregate. Single-stream kernel work cannot recover the 11.5 ms; only
  batching can hide it, and only after the kernel batches.
- Prefill needs nothing: 855 tok/s at 4096 means a 32K prompt is ~40 s and a
  full 262K prompt ~5 min, once, then `--cache-reuse` keeps it.

Predicted step times from the model, short prompts (add KV per full slot):

| slots | step | per stream | aggregate |
| --- | --- | --- | --- |
| 1 | 30.0 ms | 33 | 33 |
| 2 | 41.5 ms | 24 | 48 (measured 48.2) |
| 4 | 64.5 ms | 15.5 | 62 |
| 8 | 110.5 ms | 9 | 72 |
| 16 | 202.5 ms | 5 | 79 |

## Still to measure

- npl curve with short prompts: `NPP=512 NTG=128 NPL="1 2 4 8" scripts/svmi-cmpbench.sh -m <gguf>`.
  This is the direct test of the table above: if step time climbs ~11.5 ms per
  slot the GDN path is serialized and the kernel is the next job; if it
  flattens past 2, batching already amortizes it and more slots pay.
- per-op profile of one decode step (`nsys profile` on llama-bench `-n 16`) to
  confirm the 11.5 ms sits in the recurrent layers and see how many kernels it is
- tokens-to-answer, DavidAU merge vs base Qwen3.8 (decides the model)
- UD-Q4_K_M vs mixed-INT8; q8_0 vs q4_0 KV at the 262K window
