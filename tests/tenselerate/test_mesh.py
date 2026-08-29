"""
The decentralized serving mesh (DICE framing): local admission, movable
sessions, and resilience to the loss of any one node - all derived from the
model's constant per-session footprint, never guessed.
"""
from __future__ import annotations

import dataclasses

import pytest

from tenselerate.config import RAVENX_27B
from tenselerate.engine.mesh import Mesh, NodeSpec

# the deployment collective: 80 GB CMP with 256 GB host RAM, the 2080 Ti
# pipeline node, and the 3060 (cannot hold the weights)
DEPLOY = [
    NodeSpec("cmp170hx", vram_gib=80.0, bw_gbs=1490.0, host_ram_gib=256.0),
    NodeSpec("2x2080ti", vram_gib=22.0, bw_gbs=1232.0),
    NodeSpec("3060", vram_gib=12.0, bw_gbs=360.0),
]


def _mesh(window=131_072):
    cfg = dataclasses.replace(RAVENX_27B, attention_window=window)
    return Mesh(cfg, DEPLOY)


def test_only_weight_holding_nodes_serve():
    m = _mesh()
    names = {n.spec.name for n in m.serving_nodes}
    assert "cmp170hx" in names and "2x2080ti" in names
    assert "3060" not in names            # 12 GiB < 15.41 GiB weights


def test_hot_capacity_is_the_sum_of_serving_nodes():
    m = _mesh()
    assert m.hot_capacity == sum(n.hot for n in m.serving_nodes)
    assert m.hot_capacity >= 14           # CMP alone gives 14 at 128K


def test_host_ram_adds_parked_capacity_only_on_serving_nodes():
    m = _mesh(window=32_768)              # smaller footprint, more parked
    cmp = next(n for n in m.nodes if n.spec.name == "cmp170hx")
    tiny = next(n for n in m.nodes if n.spec.name == "3060")
    assert cmp.parked > 100               # 256 GiB / ~1.14 GiB per session
    assert tiny.parked == 0               # cannot serve, so cannot park either


def test_live_capacity_exceeds_hot_capacity_thanks_to_host_ram():
    m = _mesh(window=32_768)
    assert m.live_capacity > m.hot_capacity


def test_mesh_is_n_minus_1_resilient_below_resilient_capacity():
    m = _mesh(window=32_768)
    safe = m.resilient_capacity()
    assert safe == m.live_capacity - max(n.capacity for n in m.serving_nodes)
    assert m.is_resilient_at(safe)
    assert not m.is_resilient_at(safe + 1)


def test_losing_a_node_is_survivable_capacity_wise():
    m = _mesh(window=32_768)
    # dropping the 2080 Ti node leaves the CMP holding the whole load
    without = m.capacity_without("2x2080ti")
    assert without == m.capacity_without("2x2080ti")
    assert without >= m.resilient_capacity()


def test_place_fills_hot_before_parked():
    m = _mesh(window=32_768)
    hot = m.hot_capacity
    plan = m.place(hot)                    # exactly the hot slots
    assert sum(p for _, p in plan.values()) == 0     # nothing parked yet
    assert sum(h for h, _ in plan.values()) == hot


def test_place_spills_into_parked_when_hot_is_full():
    m = _mesh(window=32_768)
    load = m.hot_capacity + 5
    plan = m.place(load)
    assert sum(p for _, p in plan.values()) == 5


def test_place_refuses_to_overcommit():
    m = _mesh(window=32_768)
    with pytest.raises(ValueError, match="exceeds mesh live capacity"):
        m.place(m.live_capacity + 1)


def test_migration_is_a_bounded_sub_second_transfer():
    m = _mesh(window=32_768)
    cmp = next(n for n in m.nodes if n.spec.name == "cmp170hx")
    # ~1.14 GiB over a Gen2 x16 link (~8 GB/s effective) is well under a second
    assert cmp.migration_seconds(link_gbs=8.0) < 0.25
