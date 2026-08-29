# DICE serving mesh — a decentralized, node-loss-resilient collective

Framing borrowed from DARPA **DICE** (Decentralized AI through Controlled
Emergence): robust global behaviour from simple local rules, a collective that
stays on-mission and resilient to the loss of any one agent. Applied to this
engine's already-heterogeneous box (CMP 170HX + 2080 Ti pair + 3060), it turns
a single-scheduler server into a mesh of peers.

Reference model: `tenselerate/engine/mesh.py` (`NodeSpec`, `MeshNode`, `Mesh`),
CPU-tested in `tests/tenselerate/test_mesh.py`. It is capacity/placement
bookkeeping; the `--rpc` layer carries the actual traffic.

## The three local rules

1. **Local admission.** Each node admits from its own queue against its own
   VRAM budget (its `Scheduler`). No single global scheduler exists to lose.
2. **Movable sessions.** A session's footprint is the constant
   `window-KV + GDN-state` (~1.14 GiB at the 32K window,
   `parked_footprint_bytes`), so it migrates between nodes as a fixed-size
   blob — sub-second over the Gen2 link (`MeshNode.migration_seconds`). This is
   the memory-tiering result reused across nodes instead of across tiers.
3. **Resilience as a capacity inequality.** The mesh is **N-1 resilient** at a
   load if the survivors still hold every live session after the single largest
   serving node drops: `load <= resilient_capacity()`.

## Capacity on the deployment box (256 GB host RAM on the CMP)

| window | node | serves | hot | parked |
| --- | --- | --- | --- | --- |
| 32K | cmp170hx | yes | 58 | 225 |
| 32K | 2x2080ti | yes | 4 | 0 |
| 32K | 3060 | no | 0 | 0 |
| | **mesh** | | **62 hot** | **287 live** |

A node holds the weights or it does not serve — the 3060 (12 GiB < 15.41 GiB)
is a smoke-test card, not a mesh member. Host RAM turns the CMP into the bulk
holder: 287 live 32K-window sessions across the mesh, 62 of them hot at once.

## The honest resilience finding

**This box is capacity-lopsided, so it is not meaningfully N-1 resilient.** The
CMP is ~98% of live capacity, so `resilient_capacity()` is only ~4 sessions at
the 32K window: losing the CMP drops almost everything, and the 2080 Ti node
can cover only a handful. DICE-style resilience assumes agents of comparable
weight; one dominant node breaks that.

What actually buys resilience, in order:

1. **A second serving node of comparable size** — a second CMP. Two ~equal
   nodes make the mesh genuinely N-1 (either can fail and the other holds the
   safe load). This is the real recommendation if uptime matters.
2. **Failover, not load-sharing, from the 2080 Ti node.** As-is, treat it as a
   warm spare that keeps a few priority sessions alive through a CMP blip while
   the CMP restarts — the migration is bounded and fast, so a small set moves
   in well under a second.
3. **Graceful degradation as doctrine.** On node loss the mesh re-places what
   it can onto survivors (bounded footprints make this exact) and sheds the
   rest by admission refusal rather than OOM — the single-node memory-first
   rule, meshed. `Mesh.place()` refuses to overcommit for exactly this reason.

## What stays true

The mesh changes *where a session lives*, never *what the model attends to*:
every node runs the one locked model, at a window ≥ the 32K quality floor, with
no RoPE scaling. Migration moves bounded state; it never extends context by
extrapolation. And it needs no 80 GB unlock — it is how a set of *stable* nodes
(40/64 GB CMP + the 2080 Ti node) serve a large, resilient live-session
population together.

## Build order

Landed (CPU-tested): the capacity/placement/resilience model above.

Next: (1) a router that runs `place()` and issues migrations over `--rpc`;
(2) a heartbeat so a node-loss event triggers `capacity_without()` re-placement;
(3) an admission gate per node that refuses at `resilient_capacity()` when
run in resilient mode.
