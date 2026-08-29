# Whole-system memory tiering — max live context, not max VRAM

A design note, thinking from the engine's own first principles rather than
copying a transformer server's playbook.

## The reframe

The usual "long context is VRAM-bound" problem does **not** apply to a single
sequence here, and that changes the whole target:

- The 16 full-attention layers are **windowed**, so their KV is a constant
  ~1 GiB (32K window), never growing with context.
- The 48 Gated-DeltaNet layers carry the real long range in a **fixed
  recurrent state** — position-free, ~75 MB at bf16 (`gdn_state_bytes()`),
  and it does not depend on context length *or* the window at all.

So one 1M-token (or 10M-token) session already fits in ~1.1 GiB and needs no
more VRAM to go longer. Single-sequence context is maxed by construction. The
scarce resource is **how many live long-context sessions you can hold at
once** — and *that* is what the whole system's memory can buy.

## The insight that makes tiering clean here

In a normal transformer, KV grows without bound, so CPU-offload/swap is a
messy heuristic over a moving target. Here, a session's *entire* resident
footprint is a **known constant**:

```
parked_footprint = window_KV(≈1.06 GiB @32K) + GDN_state(≈0.075 GiB bf16)
                 ≈ 1.14 GiB / session        # parked_footprint_bytes()
```

Both terms are bounded and fixed-size, so a second tier holds a
**deterministic** number of sessions — the same memory-first admission the
scheduler already enforces for VRAM (`Scheduler.can_admit`), extended to the
whole box:

| tier | holds | capacity on the deploy box |
| --- | --- | --- |
| **VRAM** (80/64/40 GB CMP) | the *active* working set | ~20–60 sessions hot |
| **host RAM** (X10SRL-F, 2699, up to 512 GB) | *parked* sessions | ~112 / 225 / 450 parked @128/256/512 GB |
| disk/NVMe (optional L3) | cold sessions | effectively unbounded |

A 256 GB host parks ~225 live 1M-token conversations and pages the handful
that are actively decoding into VRAM. That is the "use my whole system"
answer: VRAM sets *concurrency*, host RAM sets *live-session count*.

## Why the paging is cheap — and novel to this architecture

The unit that moves between tiers is `(window KV blocks + GDN state)`, and
both facts about it are gifts of the hybrid:

1. **Bounded size.** Nothing you page grows over the session's life, so a
   parked slot is allocated once and reused — no fragmentation, no
   recompaction, deterministic PCIe transfer time.
2. **Predictable cadence.** A session pages in only when it becomes the active
   decoder (a new turn arrives) and pages out when it yields. Multi-turn chat
   is *mostly idle per session*, so the hot set is tiny and the PCIe traffic
   is per-turn, not per-token. Gen2 x16 (the unlock's link) moves ~1.14 GiB in
   well under a second — invisible next to think-time between turns.
3. **The GDN state is the long memory, and it is what's worth keeping.** Unlike
   evicted KV (gone forever under the sliding window), the recurrent state is
   the session's entire accumulated context in 75 MB. Parking *that* to RAM is
   how a session survives eviction from VRAM without losing a single token of
   its history.

There is even an L3 trick the linearity permits: a fully cold session can be
stored as just its **GDN state + the last window of tokens** (~1.14 GiB), and
if you are willing to trade compute for storage, the window KV can be
*recomputed* from those tokens on wake instead of stored — dropping the cold
footprint toward the 75 MB state alone. Storage-for-compute, enabled because
GDN is a replayable linear recurrence.

## What stays banned

This is a memory-hierarchy trick, not a positional one: every attended
position is still inside the trained rotary range, the window still ≥ the 32K
quality floor, and there is still no RoPE scaling. Tiering changes *where* a
session's bounded state lives, never *what the model attends to*.

## Status and the build order

Foundation landed now (CPU-testable, no kernel):

- `ModelConfig.gdn_state_bytes()` — the previously-missing state size (the
  scheduler defaulted it to 0).
- `ModelConfig.parked_footprint_bytes()` — the constant that a host-RAM tier
  budgets against.

Roadmap, in order:

1. **Two-tier pool** — a host-RAM `ParkedPool` mirroring `KVBlockPool`'s
   budget math, with the scheduler admitting into VRAM and evicting to RAM by
   an LRU-of-idle-sessions policy. Pure bookkeeping; testable on CPU.
2. **Pinned-host transfer path** — page a session's blocks + state across PCIe
   on wake/yield, overlapped with decode (the SVMI pinned ring already exists
   for weights; reuse it).
3. **NVMe L3 + state-only cold storage** — the recompute-on-wake option.

None of this needs the 80 GB unlock to hold: it is precisely what lets 40 GB
of *stable* VRAM serve a large live-session population by leaning on host RAM.
