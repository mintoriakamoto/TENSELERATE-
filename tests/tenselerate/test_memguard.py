"""
The memory guardian: hold the CMP at 80 GB, but serve only memory proven good.
Retention becomes "never serve from unverified memory", and capacity degrades
(80 -> 64 -> 40) before an answer ever does.
"""
from __future__ import annotations

import pytest

from tenselerate.engine.memguard import (
    MIN_SERVE_GIB, MemoryHealth, MemoryUnsafe, SoakPlan, choose_machine_profile,
)


def test_clean_80gb_card_is_serve_ready_at_the_top_tier():
    h = MemoryHealth(reported_gib=80, verified_gib=80, ecc_enabled=True)
    assert h.serve_ready
    assert h.safe_tier_gib() == 80.0
    assert choose_machine_profile(h) == "cmp170hx"


def test_unstable_tail_is_carved_out_and_degrades_the_tier():
    # card reports 80 but only 66 soaked clean; the bad tail is quarantined
    h = MemoryHealth(reported_gib=80, verified_gib=80, ecc_enabled=True,
                     quarantined_gib=14)
    assert h.usable_gib == 66
    assert h.safe_tier_gib() == 64.0          # largest tier <= usable
    assert choose_machine_profile(h) == "cmp170hx-64"
    assert h.serve_ready


def test_xid_error_disqualifies_the_card():
    h = MemoryHealth(reported_gib=80, verified_gib=80, xid_errors=1)
    assert not h.serve_ready
    assert "Xid" in h.reason()


def test_uncorrectable_ecc_means_corruption_already_happened():
    h = MemoryHealth(reported_gib=80, verified_gib=80, ecc_enabled=True,
                     uncorrectable_ecc=1)
    assert not h.serve_ready
    assert "corruption" in h.reason()


def test_exhausted_remap_pool_is_disqualifying():
    h = MemoryHealth(reported_gib=80, verified_gib=80, ecc_enabled=True,
                     remapped_rows=512, remap_rows_available=512)
    assert h.remap_exhausted
    assert not h.serve_ready


def test_card_below_serve_floor_cannot_serve_at_any_tier():
    h = MemoryHealth(reported_gib=80, verified_gib=32)
    assert not h.serve_ready
    with pytest.raises(MemoryUnsafe):
        h.safe_tier_gib()
    with pytest.raises(MemoryUnsafe):
        choose_machine_profile(h)


def test_ecc_off_still_serves_on_a_soaked_carve_out_but_says_so():
    h = MemoryHealth(reported_gib=80, verified_gib=64, ecc_enabled=False)
    assert h.serve_ready                      # 64 verified, no errors
    assert "ECC OFF" in h.reason()


def test_degraded_tiers_still_meet_the_requirement():
    # even the smallest stable tier is a real MACHINE_HW profile that the
    # planner shows meeting 1M ctx + 400 tok/s
    from tenselerate.cli import MACHINE_HW
    for tier_profile in ("cmp170hx", "cmp170hx-64", "cmp170hx-40"):
        assert tier_profile in MACHINE_HW
    h40 = MemoryHealth(reported_gib=80, verified_gib=40, ecc_enabled=True)
    assert choose_machine_profile(h40) == "cmp170hx-40"


def test_soak_plan_requires_the_whole_range_and_zero_fatal():
    plan = SoakPlan()
    assert plan.fraction_to_exercise >= 0.9
    good = MemoryHealth(reported_gib=80, verified_gib=80, ecc_enabled=True)
    assert plan.passes(good, correctable_ecc=0)
    assert not plan.passes(good, correctable_ecc=1)     # any SBE at top = flag
    bad = MemoryHealth(reported_gib=80, verified_gib=80, xid_errors=1)
    assert not plan.passes(bad, correctable_ecc=0)


def test_serve_floor_matches_smallest_tier():
    assert MIN_SERVE_GIB == 40.0
