"""
Attention sinks (StreamingLLM, arXiv:2309.17453): the first few tokens are
pinned in the cache forever, because softmax attention parks surplus
probability mass on them and a sliding window that evicts them collapses in
quality. This is the long-context method the no-RoPE-scaling rule permits -
sink positions are re-anchored to 0..N-1, so the attended span is
window + sinks and never leaves the trained rotary range.
"""
from __future__ import annotations

import dataclasses

from tenselerate.config import (
    ATTENTION_SINK_TOKENS, DEFAULT_ATTENTION_WINDOW, RAVENX_27B, TINY,
)
from tenselerate.engine.scheduler import Scheduler

MiB = 1024 ** 2


def test_sinks_default_to_four():
    # 4 is the StreamingLLM number: enough to absorb the sink mass
    assert ATTENTION_SINK_TOKENS == 4
    assert RAVENX_27B.attention_sink_tokens == 4


def test_sinks_are_resident_alongside_the_window():
    assert RAVENX_27B.resident_kv_tokens == \
        DEFAULT_ATTENTION_WINDOW + ATTENTION_SINK_TOKENS


def test_sinks_cost_almost_nothing():
    with_sinks = RAVENX_27B.kv_bytes_for_context(10_000_000)
    without = dataclasses.replace(RAVENX_27B, attention_sink_tokens=0)
    delta = with_sinks - without.kv_bytes_for_context(10_000_000)
    assert 0 < delta < 1 * MiB, delta      # ~136 KiB at 34 KiB/token


def test_sinks_never_cause_position_extrapolation():
    # window + sinks stays inside the trained range at any context
    assert RAVENX_27B.needs_rope_scaling(10_000_000) is False
    # ...and a window sized exactly to the range now overflows BY the sinks,
    # so the no-extrapolation check must catch it
    edge = dataclasses.replace(
        RAVENX_27B, attention_window=RAVENX_27B.max_position_embeddings)
    assert edge.needs_rope_scaling(1_000_000) is True


def test_scheduler_reserves_blocks_for_the_sinks():
    sched = Scheduler(TINY, kv_budget_gib=0.001)
    assert sched.blocks_per_seq == \
        sched.pool.blocks_for_tokens(TINY.resident_kv_tokens)
    assert TINY.resident_kv_tokens <= TINY.max_position_embeddings


def test_kv_stays_constant_with_sinks_pinned():
    sizes = {RAVENX_27B.kv_bytes_for_context(c)
             for c in (1_000_000, 4_000_000, 10_000_000)}
    assert len(sizes) == 1
