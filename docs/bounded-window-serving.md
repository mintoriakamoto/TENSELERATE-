# Bounded-window serving: many slots, unbounded sequences

The goal of this repository is one card serving as many agents as possible,
each with as much context as possible. This page is the method that changes
the shape of that problem on the 27B, the arithmetic behind it, the code that
implements it, and the measurements that decide whether it ships as default.

## The wall today

llama.cpp caches K and V for every token a full-attention layer has seen.
On Qwen3.8-27B that is 16 layers x 4 KV heads x 256 dims x (K+V):

| KV type | per token | 32K | 64K | 128K | 256K |
| --- | --- | --- | --- | --- | --- |
| q8_0 | 34 KiB | 1.06 GiB | 2.1 GiB | 4.25 GiB | 8.5 GiB |
| q4_0 | 18 KiB | 0.56 GiB | 1.1 GiB | 2.25 GiB | 4.5 GiB |

With 15.4 GiB of weights and ~2.5 GiB of compute buffers and runtime, the
40 GiB 170HX has ~22 GiB for slots. At 256K per slot that is **two** slots at
q8_0, four at q4_0, and the server caps every sequence at the 262,144-token
training context. Decode at 262K depth measured **12.4 tok/s** against 33.5
at zero depth: the attention layers read the whole KV every token.

## What the model already is

48 of the 64 layers are Gated-DeltaNet: a fixed-size recurrent state per
sequence (~150 MiB for the 27B: 48 layers x 128 x 6144 f32 plus the conv
window), no positional encoding, no growth with sequence length. Only 16
layers are attention. The long range already lives in a state; the KV cache
only exists to give those 16 layers verbatim recall.

The reference engine in `tenselerate/` was designed around exactly this
(`config.py`: a bounded window for the attention layers, 4 attention sinks,
`kv_bytes_per_token` counts 16 layers). llama.cpp did not have it for this
architecture. Now the fork does.

## The method

Bound the 16 attention layers to a sliding window of W tokens plus S pinned
leading positions (attention sinks), and let the 48 GDN layers carry
everything older. Three consequences:

1. **KV per slot is O(W), not O(sequence).** Slots = 22 GiB / (KV(W) + 0.3
   GiB state and rollback copies).
2. **Sequences are unbounded.** Relative positions inside the window never
   exceed W, so RoPE never extrapolates however long the sequence gets. The
   server's training-context cap is lifted when the window is on.
3. **Decode cost is flat in depth.** The attention layers read W tokens of
   KV per step at 1M depth exactly as at W.

Slots that fit on the 170HX (22 GiB budget, 0.3 GiB per slot for state):

| window | q8_0 slots | q4_0 slots |
| --- | --- | --- |
| 32K | 16 | 25 |
| 64K | 9 | 15 |
| 128K | 4 | 8 |

The pinned sink region is the second half of the idea. Hermes ships ~35K
tokens of system prompt (tools, skills, memory). `LLAMA_ATTN_SINKS=36000`
pins those positions in attention forever, so every slot always attends to
its instructions plus the most recent W tokens, and the GDN state carries the
conversation in between. KV per slot is then (S + W) x 34 KiB: with S=36K,
W=32K that is 2.3 GiB at q8_0, **9 slots**, each with the full tool set
attendable at any depth.

## Predicted speed

From the measured step model (weights 18.5 ms, MMQ floor ~55 ms at widths
2-16, ~5.6 ms per sequence for the GDN block, and the depth sweep 29.9 ->
33 -> 42.5 -> 55 -> 80.9 ms at 0 / 16K / 65K / 131K / 262K):

| config | step | per stream | aggregate |
| --- | --- | --- | --- |
| today: 1 slot at 262K depth | 80.9 ms | 12.4 | 12.4 |
| 1 slot, W=32K, any depth | ~35 ms | ~28 | 28 |
| 4 slots, W=64K, any depth | ~55 + 4 x 15 = 115 ms | ~8.7 | ~35 |
| 9 slots, S=36K + W=32K | ~55 + 9 x 15 = 190 ms | ~4.7 | ~47 |
| 16 slots, W=32K | ~55 + 16 x 11 = 230 ms | ~4.3 | ~70 |

Read: the window does not raise the aggregate ceiling (that is bytes on the
weight read and the MMQ floor, see `docs/kernel-work.md`); it converts the
depth penalty into slots. Sixteen agents at unbounded depth run at the same
aggregate as today's four capped at 262K, and single-stream decode at 1M
depth is 2.3x faster than today's at 262K. These are predictions; the rows
are graded on the box below.

## What it costs

Verbatim recall exists only inside the window plus sinks. Anything older is
recalled through the GDN state: associative, not verbatim. The model was
trained with full attention to 262K, so this is a change to what the model
sees at inference and it must be measured, not assumed:

- Needle-in-a-haystack at depths 64K, 128K, 256K with W=32K and W=64K
  against the unbounded model. The window is a win only where recall holds.
- Sinks pinned by absolute position: at depths past 262K the relative
  distance from the current token to the sink region exceeds the trained
  RoPE range for the 64 rotated dims. StreamingLLM re-anchors sink positions;
  this implementation does not yet. Measure before relying on S at >262K.
- Agent workloads recall recent tool output and the instructions, which is
  exactly what sinks + window keep; long-document verbatim quotation is what
  they lose.

## Implementation

Off unless `LLAMA_ATTN_WINDOW` is set, so the default binary is unchanged.
The logic lives in a fork-owned file; the upstream sources carry one-line
hooks marked `TENSELERATE` (see `docs/upstream-sync.md`, "how to keep
conflicts small"; `scripts/fork-hunks.sh` lists them).

- `src/tenselerate-attn-window.cpp` (fork-owned): reads `LLAMA_ATTN_WINDOW`
  and `LLAMA_ATTN_SINKS`, sets `swa_type = STANDARD`, `n_swa`, marks every
  non-recurrent layer SWA, copies the rope base to the SWA fields, sets the
  sink count. `src/models/qwen35.cpp` calls it once from `load_arch_hparams`.
  `create_memory` then picks `llama_memory_hybrid_iswa`, whose SWA cache is
  `n_swa x n_seq_max + n_ubatch` cells.
- `src/models/qwen35.cpp`: the graph builds the hybrid-iswa input when
  `swa_type` is set and `build_layer_attn` dispatches on the input type
  (two small hunks; no template, so upstream edits to the file merge).
- `src/llama-hparams.h`: `is_masked_swa` never masks positions below
  `n_swa_sink`, which also keeps those cells from being evicted.
- `tools/server/server-context.cpp`: the per-slot context is not capped at
  `n_ctx_train` when the window is on.
- `tenselerate serve --attn-window N --attn-sinks S`, env `ATTN_WINDOW` /
  `ATTN_SINKS` in `scripts/hercules_serve.sh`. The pool floor becomes
  slots x window (the pool is sequence capacity; only the window is KV).
- `tests/test-attn-window.cpp` (ctest, main label): with W >= prompt the
  logits are bit-identical to the unbounded model; with W=32 a sequence
  3x the training context decodes with finite logits.

## Fork, don't fetch: zero-transport agents

Every existing way to give a new agent its context moves bytes across the
170HX's fused PCIe Gen2 x4 link at ~2 GB/s: re-prefill (40 s for the 35K
prompt), the RAM prompt cache (1.2 GB, ~0.6 s), an NVMe restore (same 0.6 s,
the drive is not the bottleneck). The hybrid makes a different move possible:
**fork the sequence on the card and move nothing.**

A sequence on this model is two things: KV cells for the window (shared by
reference between sequences in a unified cache: `llama_memory_seq_cp` adds a
sequence bit to the cells, no copy) and one 150 MiB GDN state cell (shared by
reference too, copied on the first write: the recurrent memory's `src`
copy-on-write). llama-server already forks this way for `n > 1` completions
(`copy_state_to`). Exposed across requests it becomes:

1. **Live fork for delegation.** Hermes delegates a subtask; the child is a
   fork of the parent's sequence at its current position. Cost: cell metadata
   plus a 150 MiB in-VRAM copy on the child's first token, ~0.2 ms at HBM
   speed. The child starts with the parent's whole context, including the
   associative memory of everything past the window, and pays only for its
   own new tokens. Today the same child costs 0.6 s over PCIe or 40 s of
   prefill, and inherits nothing but the system prompt.
2. **Prefix checkpoint.** The 35K system prompt is processed once into a
   template sequence: its window cells plus its GDN state at the prefix
   boundary, held in one reserved recurrent cell. Every new conversation
   forks from the template by reference. VRAM for the shared prefix is paid
   once instead of per slot (1.2 GB, not 9 x 1.2 GB), and the RAM prompt
   cache is never consulted for the prefix. With the sink region pinned to
   the prefix, the shared cells are also never evicted from any fork.

Per-slot VRAM then becomes window KV plus one state: at a 32K window and
q8_0, ~1.25 GiB, so **16 slots** each carrying the full pinned system prompt
on the 40 GiB card, against 9 with per-slot prefix copies.

What is genuinely new here is the combination on this hardware: a bounded
window with a pinned prefix, forked by reference, on a card whose link makes
every fetch expensive. The parts have cousins (vLLM's prefix caching shares
blocks by hash; SGLang caches recurrent states at prefix boundaries for
hybrid models); llama.cpp's server has neither across requests. The work is
in `tools/server`: a `fork_from` field or `/slots/<id>?action=fork&target=j`
endpoint, a template slot for the prefix checkpoint, and one more recurrent
cell (`recurrent_rs_size = n_seq_max + 1`). Issue #67.

Two honest limits. A live fork inherits the parent's GDN state at the fork
point, so a child can only fork from a slot that is at the position it wants;
forking "from 40K tokens ago" needs a state checkpoint there, which is what
the prefix checkpoint provides for the one boundary that matters. And a fork
shares the parent's window cells by reference, so the parent's later
eviction of those cells is per-sequence (a cell frees only when its last
sequence bit drops), which is how the KV cache already accounts them.

## How to run it

```bash
# 9 slots, system prompt pinned, 32K sliding window, q8_0 KV
ATTN_WINDOW=32768 ATTN_SINKS=36000 NP=9 CTX=$((1048576*9)) \
  bash scripts/hercules_serve.sh MODEL.gguf

# 16 slots, plain 32K window (sinks default 4)
ATTN_WINDOW=32768 NP=16 CTX=$((1048576*16)) bash scripts/hercules_serve.sh MODEL.gguf
```

`CTX` is the sequence capacity in cells and costs metadata only; the KV is
the window. Expect `llama_kv_cache_iswa: creating SWA KV cache, size = ...`
in the log and `n_ctx_slot` above 262144.

## Measurements to record

In `benches/cmp170hx-3060/README.md`, next to the predictions above:

1. Single stream, W=32K, decode at depths 0 / 262K / 1M: step time and
   tok/s. Prediction: flat at ~35 ms.
2. 9 slots S=36K W=32K and 16 slots W=32K: aggregate and per-stream at
   depth 100K each. Predictions: ~47 and ~70 aggregate.
3. Needle recall at 64K / 128K / 256K, W=32K and W=64K, vs unbounded.
4. VRAM at rest and under load for each config, against the slot table.
