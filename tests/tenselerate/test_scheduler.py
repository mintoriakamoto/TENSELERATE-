"""
Continuous batching + paged KV. These are the invariants that decide whether the
engine survives a real workload: never over-admit, never leak a block, and keep
750,000-token sequences alive on a bounded pool.
"""
from __future__ import annotations

import dataclasses

import pytest

from tenselerate.config import MIN_CONTEXT_TOKENS, RAVENX_27B
from tenselerate.engine.kvpool import KVBlockPool, OutOfBlocks
from tenselerate.engine.scheduler import Scheduler, SeqStatus

GiB = 1024 ** 3


# ---- the block pool ------------------------------------------------------
def test_pool_hands_out_and_reclaims_every_block():
    p = KVBlockPool(block_tokens=256, n_blocks=8)
    a = p.allocate(3)
    b = p.allocate(5)
    assert p.n_free == 0 and p.n_allocated == 8
    assert len(set(a) | set(b)) == 8, "blocks must be unique"
    p.free(a)
    p.free(b)
    assert p.n_free == 8 and p.n_allocated == 0


def test_pool_refuses_over_allocation_instead_of_overcommitting():
    p = KVBlockPool(block_tokens=256, n_blocks=4)
    p.allocate(4)
    assert not p.can_allocate(1)
    with pytest.raises(OutOfBlocks):
        p.allocate(1)


def test_pool_sized_from_a_real_vram_budget():
    kv_per_tok = RAVENX_27B.kv_bytes_per_token()      # 34 KiB
    p = KVBlockPool.from_budget(4.25 * GiB, kv_per_tok, block_tokens=256)
    # 4.25 GiB / (34 KiB * 256) == exactly one 128K window's worth
    assert p.n_blocks == p.blocks_for_tokens(RAVENX_27B.resident_kv_tokens)


def test_pool_rejects_a_budget_too_small_for_one_block():
    with pytest.raises(ValueError):
        KVBlockPool.from_budget(1.0, RAVENX_27B.kv_bytes_per_token(), 256)


def test_blocks_for_tokens_rounds_up():
    p = KVBlockPool(block_tokens=256, n_blocks=100)
    assert p.blocks_for_tokens(1) == 1
    assert p.blocks_for_tokens(256) == 1
    assert p.blocks_for_tokens(257) == 2


# ---- the scheduler -------------------------------------------------------
def _sched(kv_gib=47.0, window=131_072):
    cfg = dataclasses.replace(RAVENX_27B, attention_window=window)
    return Scheduler(cfg, kv_budget_gib=kv_gib)


def test_max_concurrent_matches_the_planner():
    """The CMP at the 750K floor, 128K window: ~11 concurrent sequences."""
    s = _sched(kv_gib=47.0, window=131_072)
    assert s.max_concurrent == 11, s.max_concurrent


def test_narrower_window_buys_more_concurrency():
    wide = _sched(kv_gib=47.0, window=131_072).max_concurrent
    narrow = _sched(kv_gib=47.0, window=32_768).max_concurrent
    assert narrow > wide
    assert narrow == 44, narrow          # the ~638 tok/s row in the docs


def test_never_admits_more_than_capacity():
    s = _sched()
    for _ in range(50):
        s.submit(prompt_len=1000, max_new_tokens=5)
    s.step()
    assert len(s.running) <= s.max_concurrent
    assert s.pool.n_allocated <= s.pool.n_blocks


def test_sequences_join_as_others_finish():
    """The definition of CONTINUOUS batching: the batch refills mid-flight."""
    s = _sched()
    cap = s.max_concurrent
    for _ in range(cap * 3):
        s.submit(prompt_len=1000, max_new_tokens=2)
    s.step()
    first_wave = {q.seq_id for q in s.running}
    assert len(first_wave) == cap
    # after enough steps the early ones retire and later ones take their slots
    for _ in range(6):
        s.step()
    assert s.stats.admitted > cap, "later sequences never got admitted"
    later = {q.seq_id for q in s.running} | {q.seq_id for q in s.finished}
    assert later - first_wave, "no sequence beyond the first wave ran"


def test_every_block_is_returned_when_all_work_drains():
    s = _sched()
    for _ in range(40):
        s.submit(prompt_len=500, max_new_tokens=3)
    s.run_until_idle()
    assert not s.running and not s.waiting
    assert s.pool.n_allocated == 0, "block leak"
    assert s.pool.n_free == s.pool.n_blocks
    assert s.stats.finished == 40


def test_all_submitted_tokens_are_produced():
    s = _sched()
    n, per = 25, 4
    for _ in range(n):
        s.submit(prompt_len=100, max_new_tokens=per)
    s.run_until_idle()
    assert s.stats.tokens_generated == n * per


def test_kv_is_bounded_while_context_runs_past_the_floor():
    """
    The payoff: a sequence's cached KV stops at the window while its real context
    keeps growing past 750,000 tokens on a pool that never grows.
    """
    window = 32_768
    s = _sched(kv_gib=47.0, window=window)
    seq = s.submit(prompt_len=MIN_CONTEXT_TOKENS, max_new_tokens=500)
    s.step()
    assert seq.status is SeqStatus.RUNNING
    # step to just before completion so the sequence is still resident
    for _ in range(498):
        s.step()
    assert seq.status is SeqStatus.RUNNING, "should not have retired yet"
    assert seq.total_tokens > MIN_CONTEXT_TOKENS, seq.total_tokens
    assert seq.cached_tokens == window, seq.cached_tokens
    # a full window of blocks, and no more, for a 750,000+ token context
    assert len(seq.blocks) == s.blocks_per_seq
    assert s.pool.n_allocated <= s.pool.n_blocks
    # and the blocks come back when it finishes
    s.run_until_idle()
    assert seq.status is SeqStatus.FINISHED
    assert seq.blocks == [] and s.pool.n_allocated == 0


def test_utilization_reports_real_occupancy():
    s = _sched()
    assert s.utilization() == 0.0
    for _ in range(s.max_concurrent):
        s.submit(prompt_len=100, max_new_tokens=10)
    s.step()
    assert 0.0 < s.utilization() <= 1.0
    s.run_until_idle()
    assert s.utilization() == 0.0


def test_step_returns_the_effective_batch_size():
    s = _sched()
    for _ in range(5):
        s.submit(prompt_len=100, max_new_tokens=3)
    assert s.step() == 5          # all five admitted and stepped


def test_deadlock_is_raised_not_hung():
    """A sequence needing more than the pool can ever hold must not spin."""
    cfg = dataclasses.replace(RAVENX_27B, attention_window=131_072)
    s = Scheduler(cfg, kv_budget_gib=4.25)     # room for exactly one sequence
    s.blocks_per_seq = s.pool.n_blocks + 1     # force un-admittable work
    s.submit(prompt_len=100, max_new_tokens=1)
    with pytest.raises(RuntimeError, match="deadlock"):
        s.run_until_idle(max_steps=50)


def test_aggregate_throughput_beats_single_stream():
    """
    Batch B costs one weight read and yields B tokens, so tokens-per-step scales
    with concurrency - the whole reason continuous batching is the 600 tok/s lever.
    """
    s = _sched(kv_gib=47.0, window=32_768)
    for _ in range(s.max_concurrent):
        s.submit(prompt_len=1000, max_new_tokens=10)
    s.step()
    tokens_per_step = len(s.running)
    assert tokens_per_step == s.max_concurrent == 44
    assert tokens_per_step > 1
