# Hierarchical agent swarm — controlled emergence on one locked model

The DICE thesis's payload: many agents, forming teams under local rules, with
useful global behaviour emerging. The novelty here is not "run agents" — it is
that two properties of *this* engine make a local swarm cheap in ways a normal
stack cannot match. Reference model: `tenselerate/engine/swarm.py`, CPU-tested
in `tests/tenselerate/test_swarm.py`. It builds on `engine.mesh` (placement)
and `engine.scheduler` (batched decode).

## An agent is a session, not a model

Every agent runs the one locked Qwen3.8-27B, so the weights load **once per
node** and each agent is just a session — its own GDN state and windowed KV —
sharing that copy. The continuous-batching scheduler decodes the whole swarm in
**one weight-read per step**, so N agents cost barely more than one.

The economics, measured on the 80 GB CMP at the 32K window (58 hot slots):

| | value |
| --- | --- |
| swarm size | 58 agents, all one model |
| resident (shared weights) | **81 GiB** |
| naive (a model per agent) | 960 GiB — infeasible |
| saved by weight-sharing | **878 GiB** |
| per-agent decode | ~11 tok/s each (~638 aggregate, one weight-read) |

The single-model lock, which looked like a restriction, is exactly what makes a
58-agent local swarm fit at all. The marginal cost of an agent is one session
footprint (~1.14 GiB), not a 15 GiB model.

## Agent fork — the novel primitive

The GDN state is a fixed-size, position-free ~75 MB summary of an agent's whole
context. So a child agent can be **forked** by cloning the parent's state: it
starts with the parent's entire accumulated context, and skips prefilling it.

| spawn mode | pays | inherits |
| --- | --- | --- |
| **fork** | a ~75 MB state copy | the parent's full context, warm, no prefill |
| cold | a full prefill pass | nothing |

At a 1M-token shared context, fork trades a sub-second 75 MB copy for a
1,000,000-token prefill it never runs. This is `fork()` for agents, and it is
cheap **only** because the state is bounded and self-contained in this hybrid —
a KV-cache transformer has no such portable, fixed-size mind to clone. Whether a
forked state produces a *good* divergent agent is an open empirical question
(same model, so states share a space, but forking mid-context is unproven for
GDN); it is gated on eval, like the KV-precision A/B, before it is trusted for
quality-critical roles.

## Hierarchy and emergence

The swarm is a tree: a `coordinator` root, `planner`/`worker` children, under
two local rules — a parent may spawn a child while the swarm is under its slot
`budget`, and `prune()` reclaims a whole subtree. Global behaviour (team shape,
depth, division of labour) emerges from those local rules; the budget is the
one hard cap, the same memory-first refusal as everywhere else — the swarm
never overcommits a node's session slots.

Placement is the mesh's job: a swarm `fits_on(mesh)` when the mesh can hold one
live session per agent, so a large swarm spreads subtrees across nodes and a
node loss re-places what survivors can hold (bounded footprints again). Forked
children keep their coordinator's context when they migrate, because the mind
they carry is the portable state.

## What stays true

Every agent is the one locked model, at a window ≥ the 32K quality floor, no
RoPE scaling. A swarm changes *how many bounded sessions run and how they are
organised*, never *what the model attends to*. It needs no 80 GB unlock: a
stable 40/64 GB node still runs a substantial swarm from its hot slots, and
host RAM parks the idle members.

## Build order

Landed (CPU-tested): the tree, the budget rule, and the fork/shared-weights
economics above.

Next: (1) real GDN-state clone in the scheduler (copy-on-write on spawn);
(2) an inter-agent message channel (state hand-off or text, whichever a role
needs); (3) an emergence policy — the local spawn/prune rule a coordinator runs
to grow and shrink its team against a live load.
