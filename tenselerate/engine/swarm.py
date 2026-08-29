"""
Hierarchical agent swarm - the DICE "controlled emergence" layer, built on the
serving mesh and on two facts that are gifts of this specific engine.

An "agent" here is NOT a separate model. It is one session on the one locked
model: its own Gated-DeltaNet state and windowed KV, sharing the node's single
resident copy of the weights. Two consequences drive every design choice:

  1. **Weights load once per node.** All agents on a node are batched by the
     existing continuous-batching scheduler into one weight-read per step, so a
     swarm of N agents decodes at nearly the cost of one. A local swarm is
     feasible ONLY because the single-model lock makes every agent the same
     model - the marginal cost of an agent is one session footprint
     (`parked_footprint_bytes`), not a 15 GiB model load.

  2. **The GDN state is a forkable mind.** It is a fixed-size, position-free
     summary of an agent's whole context (~75 MB, `gdn_state_bytes`). A child
     agent can be FORKED by copying the parent's state - it starts with the
     parent's entire accumulated context for the price of a memory copy, with
     ZERO re-prefill compute. That is `fork()` for agents, cheap only because
     the state is bounded and self-contained in this hybrid.

The swarm is a tree of agents under local rules (a parent spawns children
within a shared slot budget); global behaviour emerges from those rules. This
is the reference model - tree bookkeeping and the fork/shared-weights economics,
CPU-tested. Placement across nodes is delegated to `engine.mesh`; batched
decode is the existing `engine.scheduler`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tenselerate.config import ModelConfig
from tenselerate.engine.mesh import Mesh

GiB = 1024 ** 3


class SwarmBudgetError(RuntimeError):
    """Raised when a spawn would exceed the swarm's session-slot budget."""


@dataclass
class Agent:
    agent_id: int
    role: str
    parent_id: int | None
    depth: int
    # True if this agent was forked from its parent's GDN state (warm start,
    # no re-prefill) rather than cold-started and prefilled from scratch.
    forked: bool = False
    children: list[int] = field(default_factory=list)


class Swarm:
    """
    A hierarchical collective of agents, all running the one locked model.

    cfg    : model geometry (supplies the per-agent footprint and state size)
    budget : max concurrent agents = session slots the swarm may hold at once
    """

    def __init__(self, cfg: ModelConfig, budget: int, root_role: str = "coordinator"):
        if budget < 1:
            raise ValueError("budget must allow at least the root agent")
        self.cfg = cfg
        self.budget = budget
        self.agents: dict[int, Agent] = {}
        self._next_id = 0
        self.root_id = self._add(root_role, parent_id=None, depth=0, forked=False)

    def _add(self, role: str, parent_id: int | None, depth: int,
             forked: bool) -> int:
        aid = self._next_id
        self._next_id += 1
        self.agents[aid] = Agent(aid, role, parent_id, depth, forked)
        if parent_id is not None:
            self.agents[parent_id].children.append(aid)
        return aid

    # -- growth under local rules -----------------------------------------
    def can_spawn(self) -> bool:
        return self.size < self.budget

    def spawn(self, parent_id: int, role: str, fork: bool = True) -> Agent:
        """
        A parent spawns a child. `fork=True` clones the parent's GDN state so
        the child inherits its full context with no re-prefill (the cheap,
        novel path); `fork=False` cold-starts a fresh agent that must prefill
        its own context. Refuses past the budget - the swarm never overcommits
        the node's session slots, the same memory-first rule everywhere else.
        """
        if parent_id not in self.agents:
            raise KeyError(f"no agent {parent_id}")
        if not self.can_spawn():
            raise SwarmBudgetError(
                f"swarm at budget {self.budget}; prune before spawning")
        depth = self.agents[parent_id].depth + 1
        aid = self._add(role, parent_id, depth, forked=fork)
        return self.agents[aid]

    def prune(self, agent_id: int) -> int:
        """Remove an agent and its whole subtree; returns the count removed."""
        if agent_id == self.root_id:
            raise ValueError("cannot prune the root")
        removed = 0
        for child in list(self.agents[agent_id].children):
            removed += self.prune(child)
        parent = self.agents[agent_id].parent_id
        if parent is not None:
            self.agents[parent].children.remove(agent_id)
        del self.agents[agent_id]
        return removed + 1

    # -- shape -------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.agents)

    @property
    def depth(self) -> int:
        return max(a.depth for a in self.agents.values())

    # -- the economics that make a local swarm feasible -------------------
    def marginal_agent_bytes(self) -> float:
        """Cost of one more agent: a session footprint, NOT a model."""
        return self.cfg.parked_footprint_bytes()

    def resident_bytes(self, weights_gib: float = 15.41) -> float:
        """
        What the swarm actually costs on a node: the weights ONCE plus one
        footprint per agent - because every agent is the same locked model.
        """
        return weights_gib * GiB + self.size * self.marginal_agent_bytes()

    def naive_bytes(self, weights_gib: float = 15.41) -> float:
        """What it would cost if each agent were its own model load (it is not)."""
        return self.size * (weights_gib * GiB + self.marginal_agent_bytes())

    def weight_sharing_saving(self, weights_gib: float = 15.41) -> float:
        """Bytes saved by sharing one weight copy across the whole swarm."""
        return self.naive_bytes(weights_gib) - self.resident_bytes(weights_gib)

    def fork_clone_bytes(self) -> float:
        """A fork copies the parent's mind (the GDN state), nothing more."""
        return self.cfg.gdn_state_bytes()

    def prefill_tokens_saved_by_fork(self, shared_context_len: int) -> int:
        """
        A forked child inherits `shared_context_len` tokens of context via the
        state clone, so it skips prefilling them - the dominant cost at long
        context. (A cold-started child would save nothing and pay the full
        prefill.) This is the compute win; the memory footprint is the same
        either way, so fork trades a ~75 MB copy for a whole prefill pass.
        """
        return max(0, shared_context_len)

    # -- placement (delegated to the mesh) --------------------------------
    def fits_on(self, mesh: Mesh) -> bool:
        """A swarm fits if the mesh can hold one live session per agent."""
        return self.size <= mesh.live_capacity
