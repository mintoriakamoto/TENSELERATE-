"""
Hierarchical agent swarm on the one locked model: agents are sessions sharing
one weight copy, children fork the parent's GDN state (warm start, no
re-prefill), and the whole tree is placed on the mesh. Reference bookkeeping,
CPU-tested.
"""
from __future__ import annotations

import dataclasses

import pytest

from tenselerate.config import RAVENX_27B
from tenselerate.engine.mesh import Mesh, NodeSpec
from tenselerate.engine.swarm import Swarm, SwarmBudgetError

GiB = 1024 ** 3
MB = 1024 ** 2


def _swarm(budget=64):
    cfg = dataclasses.replace(RAVENX_27B, attention_window=32_768)
    return Swarm(cfg, budget=budget)


def test_root_exists_and_is_the_coordinator():
    s = _swarm()
    assert s.size == 1
    assert s.agents[s.root_id].role == "coordinator"
    assert s.agents[s.root_id].parent_id is None


def test_spawn_builds_a_hierarchy():
    s = _swarm()
    a = s.spawn(s.root_id, "planner")
    b = s.spawn(a.agent_id, "worker")
    assert s.size == 3
    assert s.depth == 2
    assert b.parent_id == a.agent_id
    assert a.agent_id in s.agents[s.root_id].children


def test_budget_is_a_hard_slot_cap():
    s = _swarm(budget=3)
    s.spawn(s.root_id, "w")
    s.spawn(s.root_id, "w")
    assert not s.can_spawn()
    with pytest.raises(SwarmBudgetError):
        s.spawn(s.root_id, "w")


def test_prune_removes_the_whole_subtree():
    s = _swarm()
    a = s.spawn(s.root_id, "planner")
    s.spawn(a.agent_id, "worker")
    s.spawn(a.agent_id, "worker")
    assert s.size == 4
    removed = s.prune(a.agent_id)
    assert removed == 3               # planner + its 2 workers
    assert s.size == 1
    assert a.agent_id not in s.agents[s.root_id].children


def test_cannot_prune_the_root():
    s = _swarm()
    with pytest.raises(ValueError):
        s.prune(s.root_id)


# ---- the economics that make a local swarm feasible ---------------------
def test_marginal_agent_is_a_session_not_a_model():
    s = _swarm()
    # one more agent costs a session footprint (~1.14 GiB), not 15 GiB of weights
    assert s.marginal_agent_bytes() == RAVENX_27B.parked_footprint_bytes() \
        or s.marginal_agent_bytes() > 1.0 * GiB
    assert s.marginal_agent_bytes() < 2.0 * GiB


def test_weight_sharing_saves_the_bulk_of_a_big_swarm():
    s = _swarm(budget=64)
    for _ in range(31):
        s.spawn(s.root_id, "worker")
    assert s.size == 32
    # sharing one weight copy across 32 agents saves ~31x the weights
    saving = s.weight_sharing_saving()
    assert saving > 30 * 15.0 * GiB
    # naive (per-agent model) would be far larger than the real resident cost
    assert s.naive_bytes() > 3 * s.resident_bytes()


def test_fork_trades_a_small_copy_for_a_whole_prefill():
    s = _swarm()
    child = s.spawn(s.root_id, "worker", fork=True)
    assert child.forked is True
    # the clone copies only the ~75 MB mind
    assert s.fork_clone_bytes() / MB < 100
    # and skips prefilling the entire shared context (1M tokens here)
    assert s.prefill_tokens_saved_by_fork(1_000_000) == 1_000_000


def test_cold_spawn_is_marked_not_forked():
    s = _swarm()
    child = s.spawn(s.root_id, "worker", fork=False)
    assert child.forked is False


def test_swarm_fits_on_the_mesh_by_session_slots():
    s = _swarm(budget=64)
    for _ in range(40):
        s.spawn(s.root_id, "worker")
    mesh = Mesh(s.cfg, [
        NodeSpec("cmp170hx", 80.0, 1490.0, host_ram_gib=256.0),
        NodeSpec("2x2080ti", 22.0, 1232.0),
    ])
    # 41 agents fit easily in the mesh's live capacity (hundreds at 32K)
    assert s.fits_on(mesh)
    assert s.size <= mesh.live_capacity
