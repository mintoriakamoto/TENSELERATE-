# Read-state: the model remembers having read it

**Status: specified, gated. Nothing is built. `EXPERIMENTS=read_state_gate` decides whether
the retention is real enough to build on, and it reuses the needle test that issue #63 has
been asking for since the bounded window landed.**

## The observation

`LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (`include/llama.h:921`) serialises *"partial states,
such as SWA KV cache or recurrent cache"* - on this model, the GDN recurrent state without
the attention KV. The server already depends on it: `server-context.cpp` saves and restores
a partial state around **every MTP verify block** (`spec_ckpt`, ten call sites), thousands
of times a minute, bit-exactly. If it did not round-trip, speculation would produce wrong
tokens.

So the mechanism is proven in production on this exact model. What has never been done is
give it **a filename and a lifetime**. Every use today is in-memory and dies with the slot.

## The asymmetry that makes it worth doing

| what | size after reading N tokens |
| --- | --- |
| attention KV, q8_0, 16 of 64 layers | 34 KiB x N - **9.1 GB at 262K** |
| GDN recurrent state | **~150 MiB, flat** (`docs/ampere-backports.md:69`) |

The recurrent state does not grow. It is the same 150 MiB whether the sequence read a
thousand tokens or two million. And it holds what the sequence read *after those tokens
left the attention window* - the `fork_from` work states this directly: a forked child
*"inherits the parent's whole context including the GDN memory past the attention window"*.

That is the capability nobody has built a product around:

> **A fixed-size artifact that carries the model's comprehension of more text than its
> context window can hold.**

The window here is locked at 262,144 and never narrows (`config.py`). Read-state does not
widen it. It carries what the recurrent half already integrated, at constant cost.

## What it is not

- **Not RAG.** No retrieval, no chunking, no re-reading at query time, and no loss of
  cross-chunk structure. The model does not look things up; it has already read them.
- **Not the prompt cache.** That stores KV and grows with tokens - `HERCULES.md` puts the
  35K Hermes prefix at 2.3 GiB per cached copy. Read-state is 150 MiB for any prefix, and
  it survives a restart rather than living in a RAM budget.
- **Not fine-tuning.** Weights are untouched, it takes one reading pass rather than a
  training run, and it is per-corpus and disposable rather than global and permanent.
- **Not `fork_from`.** That is the same idea with no persistence: free, but only from a
  *live* slot, and gone when the server stops.

## What it would be used for here

Hermes delegates to six or more sub-agents. Today each one either re-prefills the ~35K
system prompt (~40 s) or loads a prompt cache (~0.6 s over this box's Gen2 x4 link).
Read-state changes the unit: a named state that absorbed an entire repository is loaded
once per sub-agent at ~150 MiB - **0.075 s over the same link, or near zero with
`LLAMA_STATE_SEQ_FLAGS_ON_DEVICE`** (`include/llama.h:925`), which keeps the tensors in
VRAM and is exactly what a box with 25.7 GiB free and a crippled host link wants.

At 150 MiB each, that free VRAM holds well over a hundred prepared states. One per
repository, per document set, per long-running investigation - switched between in
milliseconds.

## The honest limits

1. **It is lossy, by construction.** The attention KV is not in the snapshot. Verbatim
   recall of an arbitrary span is gone; what remains is whatever the recurrent state
   encoded. That may be gist and long-range association, or it may be very little.
2. **The retention is unmeasured.** Nobody has established what a GDN state retains after
   a million tokens, or whether it saturates. This is the whole risk, and it is an
   empirical question with a cheap answer.
3. **Composition is stacking, not merging.** Read A, snapshot, continue into B and you get
   A-then-B. Averaging or adding two independent states is not sound and is not proposed.
4. **Order matters.** A recurrent state is path-dependent. Two corpora read in different
   orders give different states, and nothing here pretends otherwise.

## The gate

The needle test issue #63 has been asking for is exactly the right instrument, and it has
never run. Prefill a synthetic corpus with facts planted at known depths, snapshot the
partial state, restart, load the state into a fresh sequence, and ask for the facts with
**no corpus in context**.

Pre-registered, measuring recall against depth:

| result | reading | action |
| --- | --- | --- |
| recall holds well past the window | the state carries real long-range content | build the library |
| recall only within the window's worth of recent tokens | the state is a short-horizon summary | useful for warm starts, not for corpus memory - build the narrow version |
| recall at chance | the state carries nothing retrievable on its own | close this document |

The middle outcome is the most likely and is still worth having: a 150 MiB warm start that
skips a 40 s prefill is a real feature even if it remembers nothing beyond the window.

## Sketch, only if the gate confirms

- `tenselerate read <corpus> --into NAME` - stream a corpus through one sequence, snapshot
  the partial state, write `NAME.state` plus a manifest (model hash, token count, order).
- `tenselerate states` / `rm` - the library.
- Server: `POST /slots/<id>?action=restore_state&name=NAME`, alongside the existing slot
  save/restore, with `ON_DEVICE` when the state is already resident.
- A **model-hash guard** in the manifest, refusing to load a state written by different
  weights. A recurrent state is meaningless against another model and the failure would be
  silent - this is the one piece that must exist before anything else ships.

## Order

The gate first. It needs no new code beyond the needle harness #63 already calls for, and
it can close the idea in one run. Nothing else in this document is worth writing until the
retention number exists.
