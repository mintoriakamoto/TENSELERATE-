"""
Decentralized serving mesh - the DICE framing (DARPA "Decentralized AI through
Controlled Emergence") made concrete for this engine.

The deployment is already a heterogeneous collective: the CMP 170HX node, the
2080 Ti pipeline node, the 3060. DICE's principle is that robust global
behaviour emerges from simple local rules, and the collective stays on-mission
and resilient to the loss of any one agent. Here that means:

  * **local admission.** Each node admits from its own queue against its own
    VRAM budget (its `Scheduler`). There is no single global scheduler to lose.
  * **movable sessions.** Thanks to the hybrid, a session's footprint is a
    known constant - a bounded window of KV plus a fixed GDN state
    (`ModelConfig.parked_footprint_bytes`) - so a session migrates between
    nodes as a fixed-size blob, not a growing cache. Placement and re-placement
    are exact bookkeeping.
  * **resilience to node loss.** The mesh is N-1 resilient at a load if the
    survivors can still hold every live session after the single largest
    serving node drops. That is the DICE "resilient to agent loss" property,
    made a capacity inequality this module checks.

This is the reference model: pure capacity/placement bookkeeping, CPU-testable,
no networking. The RPC transport (`scripts/svmi-net.py`, the `--rpc` layer)
carries the actual migrations; this decides who holds what and whether the
mesh survives a loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tenselerate.config import ModelConfig
from tenselerate.engine.scheduler import Scheduler

GiB = 1024 ** 3


@dataclass(frozen=True)
class NodeSpec:
    """One machine in the mesh."""
    name: str
    vram_gib: float
    bw_gbs: float
    host_ram_gib: float = 0.0        # host RAM available to park cold sessions
    weights_gib: float = 15.41
    overhead_gib: float = 1.5


class MeshNode:
    """
    A node's serving capacity, derived - never guessed - from its VRAM and
    host RAM against the model's constant per-session footprint.
    """

    def __init__(self, spec: NodeSpec, cfg: ModelConfig, kv_bpe: float = 1.0625):
        self.spec = spec
        self.cfg = cfg
        # hot slots: sessions actively resident in VRAM, from the real scheduler
        kv_budget = spec.vram_gib - spec.weights_gib - spec.overhead_gib
        self.hot = 0
        if kv_budget > 0:
            try:
                self.hot = Scheduler(cfg, kv_budget_gib=kv_budget).max_concurrent
            except ValueError:
                self.hot = 0
        # parked slots: sessions held in this node's host RAM (only meaningful
        # on a node that can also decode them, i.e. one that holds the weights)
        foot_gib = cfg.parked_footprint_bytes(kv_bpe) / GiB
        self.parked = (int(spec.host_ram_gib / foot_gib)
                       if self.serves and foot_gib > 0 else 0)

    @property
    def serves(self) -> bool:
        """A node serves only if it can hold the weights and one hot slot."""
        return self.hot >= 1

    @property
    def capacity(self) -> int:
        """Live sessions this node can hold at once (hot + parked)."""
        return self.hot + self.parked

    def migration_seconds(self, link_gbs: float) -> float:
        """Wall time to move one session's footprint across a `link_gbs` link."""
        return self.cfg.parked_footprint_bytes() / (link_gbs * 1e9)


class Mesh:
    """A collective of nodes serving the one model under local rules."""

    def __init__(self, cfg: ModelConfig, specs: list[NodeSpec]):
        self.cfg = cfg
        self.nodes = [MeshNode(s, cfg) for s in specs]

    @property
    def serving_nodes(self) -> list[MeshNode]:
        return [n for n in self.nodes if n.serves]

    @property
    def hot_capacity(self) -> int:
        """Sessions the mesh can actively decode at once."""
        return sum(n.hot for n in self.serving_nodes)

    @property
    def live_capacity(self) -> int:
        """Live sessions the mesh can hold at once (hot + parked, all nodes)."""
        return sum(n.capacity for n in self.serving_nodes)

    # -- resilience: the DICE "resilient to agent loss" property ------------
    def capacity_without(self, name: str) -> int:
        """Live capacity if the named node drops."""
        return sum(n.capacity for n in self.serving_nodes if n.spec.name != name)

    def resilient_capacity(self) -> int:
        """
        Live sessions the mesh can guarantee THROUGH the loss of its single
        largest serving node - the N-1 safe load. Placing at or below this
        means no session is dropped when any one node fails.
        """
        serving = self.serving_nodes
        if not serving:
            return 0
        largest = max(n.capacity for n in serving)
        return self.live_capacity - largest

    def is_resilient_at(self, load: int) -> bool:
        """True if `load` live sessions survive the loss of any one node."""
        return load <= self.resilient_capacity()

    # -- placement: hot-first, spill to parked, local to each node ----------
    def place(self, load: int) -> dict[str, tuple[int, int]]:
        """
        Distribute `load` live sessions across the mesh, filling fast hot slots
        before parked ones. Returns {node: (hot_used, parked_used)}. Raises if
        the load exceeds live capacity - the mesh refuses rather than overcommit,
        the same memory-first rule the single-node scheduler follows.
        """
        if load > self.live_capacity:
            raise ValueError(
                f"load {load} exceeds mesh live capacity {self.live_capacity}")
        plan: dict[str, tuple[int, int]] = {}
        remaining = load
        # fill all hot slots first (lowest latency), largest-hot node first
        for n in sorted(self.serving_nodes, key=lambda x: -x.hot):
            h = min(n.hot, remaining)
            plan[n.spec.name] = (h, 0)
            remaining -= h
        # then spill into parked slots
        for n in sorted(self.serving_nodes, key=lambda x: -x.parked):
            if remaining <= 0:
                break
            h, _ = plan[n.spec.name]
            p = min(n.parked, remaining)
            plan[n.spec.name] = (h, p)
            remaining -= p
        return plan
