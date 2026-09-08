# Speed of light on the 170HX: what physics allows, what we measure, where the gap is

Every number that matters for decode on this card is a byte count divided by a
bandwidth, or a flop count divided by a rate. This page writes those floors
down next to the measured step times so the order of work is set by the size
of the gap, not by what sounds interesting.

Card: CMP 170HX, GA100 die. HBM2e **1493 GB/s** nominal, **~890 GB/s**
achieved by the weight-read kernels (width-sweep intercept). Tensor cores
**162-170 TFLOPS FP16 measured** (INT8 dense is nominally 2x that). Power cap
250 W, held at 207-223 W under decode: not throttling. PCIe Gen2 x4, ~2 GB/s:
irrelevant for decode once the model is resident.

Model: Qwen3.8-27B Q4_K_M, **15.4 GB** of weights (16.5 GB for the -MTP- GGUF).
48 GDN layers with a 150 MiB f32 state per sequence; 16 attention layers with
34 KiB of KV per token at q8_0.

## Single stream, zero depth

| term | physics | at achieved 890 GB/s | measured |
| --- | --- | --- | --- |
| weight read, 15.4 GB | 10.3 ms | 17.3 ms | 18.5 ms (intercept) |
| GDN state, 48 layers, 150 MiB read + write | 0.2 ms | 0.34 ms | part of the residual |
| attention KV at depth 0 | ~0 | ~0 | ~0 |
| launches: ~64 layers x ~15 kernels, ~8-10 us each if not graphed | 0 (graphed) | 0 (graphed) | **8-10 ms if graphs are off** |
| **step** | **10.5 ms = 95 tok/s** | **17.7 ms = 56 tok/s** | **29.9 ms = 33.5 tok/s** |

The measured step is 2.8x the speed of light and 1.7x the achieved-bandwidth
floor. The 11.4 ms between 18.5 and 29.9 is not math: the GDN recurrence for
one token is a third of a millisecond by bytes. It is the shape of launches,
which is exactly what CUDA graphs remove. Whether graphs are actually active
on this model's graph is the first thing to check (below).

## Sixteen streams

| term | physics (1493 GB/s, INT8 330 TOPS) | at achieved 890 GB/s | measured |
| --- | --- | --- | --- |
| weight read (shared by all 16) | 10.3 ms | 17.3 ms | ~55 ms MMQ floor |
| flops, 2 x 27e9 x 16 | 2.6 ms (hidden under bytes) | 2.6 ms | in the floor |
| per sequence: state 0.34 ms + 16K-window KV 0.6 ms | 16 x 0.9 = 15 ms | 15 ms | 16 x 5.6 = 90 ms |
| **step** | **~26 ms = 620 tok/s** | **~33 ms = 490 tok/s** | **~130 ms = 122 tok/s** |

Two gaps, both kernel, neither physics: the MMQ path costs 3x the weight
read at widths 2-16 (dequant to int8 tiles and tile shapes chosen for large N
on a card that only has the integer path), and each live sequence costs 5.6 ms
where bytes say 0.9 (the GDN block's per-sequence launches and the attention
kernel's per-head re-reads). Aggregate is flat from N=16 to N=32 (122 -> 134)
because the per-sequence term, not the weight read, is what the step is made
of. Fix the per-sequence term and the MMQ floor and the same card serves
400-500 tok/s aggregate at 16 streams with no new hardware.

## Depth

| term | physics | at 890 GB/s | measured |
| --- | --- | --- | --- |
| KV read at 262K, q8_0, 8.5 GB | 5.7 ms | 9.5 ms | **51 ms** |

5.4x: the vector flash-attention kernel reads each K/V byte once per Q head
(6 Q heads per KV head). Two fixes exist, one merged: the GQA-packed path
reads once per KV head (`GGML_CUDA_FATTN_VEC_GQA`, ungraded), and the bounded
window (`--attn-window`, merged) removes the depth term entirely by keeping
the read at the window size.

## Prefill

856 tok/s at pp4096 is 2 x 27e9 x 856 = **46 TFLOPS** effective against 165
FP16 or ~330 INT8 available: a 3.6-7x gap. Prefill is paid once per prompt and
mostly hidden by the prompt cache, but the 35K system prompt is 40 s today
and could be 6-11 s. Lower priority than decode; the same MMQ tile work
touches it.

## What cannot be beaten, and the one way around it

At batch 1 the card must read the weights once per emitted token: 15.4 GB at
1493 GB/s is 95 tok/s and nothing in software moves that number except
reading fewer bytes. Two ways to read fewer bytes per emitted token:

- **Smaller weights.** IQ4_XS is 4.25 bpw (-6%); Q3 is -25% at a quality cost
  that needs the same A/B as everything else. Small.
- **More tokens per read: speculation.** The MTP head accepts 0.88 at depth 1
  under greedy (46.2 tok/s measured, 1.38x). A retrained head at depth 3 with
  0.7 acceptance is ~2.4 tokens per weight read: single-stream physics moves
  from 95 to ~230 tok/s, and the practical number from ~56 to ~130 once the
  launch residual is gone. This is the only lossless lever above the
  bandwidth line.

Batching is the same lever for aggregate: one weight read serves N tokens.
Its limit is the per-sequence term above, not bandwidth.

## Order of work, by tok/s per day

1. **Are CUDA graphs on?** 30 seconds on the box: `llama-bench -n 64` with
   and without `GGML_CUDA_DISABLE_GRAPHS=1`. Identical numbers mean graphs
   are already off for this model's graph (a node type the capture rejects,
   or the per-token recurrent-state inputs forcing a re-capture every step),
   and 8-10 ms per token is on the table: 33.5 -> ~45 tok/s single stream.
   Different numbers mean graphs work and the residual is elsewhere; `nsys`
   (issue #53) says where.
2. **Per-sequence 5.6 ms -> ~1 ms** and the **MMQ floor 55 -> ~25 ms**
   (issue #54): the aggregate lever, 122 -> 300-500 at 16 streams. Profile
   first; the GDN block's launch count per sequence is the prime suspect.
3. **MTP head that survives sampling** (issue #56): multiplies whatever 1 and
   2 leave, ~1.5-2x single stream.
4. **Grade the window and the packed attention** (issues #63, #57): the
   depth term, already coded.
5. **Fork, don't fetch** (issue #67): agent start time, not tok/s.

Everything above is measured against the release binary and recorded in
`benches/cmp170hx-3060/README.md` next to the physics number it is chasing.
