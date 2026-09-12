# Tree speculation: spending the free columns

**Status: specified, gated. `EXPERIMENTS=tree_gate` decides whether to build it. No code
yet, and none should be written until that run reports.**

## The waste

Two measured facts from `benches/cmp170hx-3060/README.md` that have not been put together:

- **MMQ is flat at ~55 ms from N=2 to N=16.** One weight read serves sixteen columns at no
  extra wall-clock. The weight read is the entire tax; columns inside that region are free.
- **MTP depth 1 uses two of them** - draft plus verify - at 46.2 tok/s, acceptance 0.881,
  mean accepted length 1.88.

Fourteen free columns are discarded every decode step.

## Why the depth sweep is not a reason to stop at two

The sweep (n-max 1 +35%, 3 -15%, 5 -22%) was read as *"the head is shallow: position 1
accepts reliably, positions 2+ do not."* That does not follow from that experiment.

A **linear** draft asks the head to predict position 2 given **its own** position-1 guess.
At 0.881 acceptance that guess is wrong about one step in eight, and when it is wrong,
position 2 cannot be accepted however good the head is. The sweep measured two things at
once - whether the head can predict position 2, and whether position 1 happened to be
right - and charged the whole loss to the first.

The two have different fixes. A shallow head needs retraining. A brittle branch needs a
draft that does not bet everything on one token.

## The change

Draft a **tree** rather than a chain. Take top-k from the MTP head at position 1, expand
each branch, and verify every branch in one batch behind a tree-structured attention mask
so each node attends only to its own ancestors. Position 2 is then conditioned on a *set*
containing the true token far more often than 0.881.

Shape it to the measured flat region - **4 x 2 x 2 = 16 columns**, one weight read, same
~55 ms. The published designs (SpecInfer, Medusa, Sequoia, EAGLE-2) all trade tree width
against latency because on ordinary GPUs extra columns cost time. On this card they do
not, up to 16. That trade simply is not present here, which is why the shape can be picked
from the acceptance numbers instead of from a latency budget.

## What it is worth

Mean accepted length is 1.88 today. With conditional acceptance around 0.85 and a 4-wide
fan at position 1, a 3-deep tree projects to roughly 2.7-3.0 accepted per step against the
same single weight read:

    46.2 tok/s x (2.9 / 1.88) ~ 71 tok/s single stream

That is past the ~67 tok/s ceiling of a depth-1 draft with *perfect* acceptance, because
the tree goes deeper rather than only being righter. It is a projection from two measured
numbers and one unmeasured one, not a result.

## The gate

`EXPERIMENTS=tree_gate MODEL=... bash benches/cmp170hx-3060/run-open-items.sh`

The server logs two per-position lines. `acc per pos` is unconditional - position i is
counted only when everything before it was accepted, so it cannot separate the two
questions above. `acc given prev` is `n_accepted_per_pos[i] / n_accepted_per_pos[i-1]`,
which is P(i accepted | i-1 accepted): the head's real skill at position i with the branch
luck divided out. Both are `SLT_INF`, so they appear at default verbosity.

Pre-registered on the position-2 value of `acc given prev`:

| result | reading | action |
| --- | --- | --- |
| **>= 0.7** | the head predicts position 2 fine when position 1 is right; the depth failure was branch misprediction | build the tree |
| **0.3 - 0.7** | partly both; the value sets how wide the position-1 fan must be before the tree pays | design from the number, do not rerun for a friendlier one |
| **<= 0.3** | the head really is shallow; a tree would cover branches that were never going to be accepted | close this document, the existing reading stands |

Depth 3 in the same run shows whether the conditional rate holds or decays, which is what
sets usable tree depth.

## Where it plugs in

`common/speculative.h` returns `llama_tokens * result` - a flat list. There is no tree
anywhere in the fork, so this is additive rather than a retrofit:

1. **Draft shape.** A new `common_speculative` implementation emitting nodes plus parent
   indices rather than a sequence. The existing implementations are unaffected and the
   ordering rule in `common_speculative` (first non-empty draft wins, drafts are never
   concatenated) still applies.
2. **Mask.** Verification needs each node to attend to its ancestors only. This is the
   part that touches the model path, which is where an unmeasured change has already cost
   a revert - it needs `test-backend-ops` coverage and a CPU-build unit test before any
   timing run.
3. **Accept.** Walk the deepest fully-accepted path, commit it, roll back the rest. The
   GDN recurrent state rollback is the hazard: PR #52 exists because of *"the partial
   `seq_rm` after an MTP verify block on the GDN model"*, and a tree makes that path wider,
   not narrower. Pin it with a test before trusting a tok/s number.

## Order

The gate first - it is one restart and it can close the whole item. Then the mask with its
tests. Then the shape, tuned from the measured conditional rates rather than from the 4x2x2
guess above.
