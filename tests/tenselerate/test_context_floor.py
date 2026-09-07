"""
The 1M context floor and the windowed-hybrid property that makes it possible
without RoPE scaling. These are product invariants, not implementation details:
if any of them break, the engine is no longer doing what it claims.
"""
from __future__ import annotations

import dataclasses

import pytest

from tenselerate.config import (
    DEFAULT_ATTENTION_WINDOW, MIN_CONTEXT_TOKENS, QWEN38_27B,
    ContextFloorError, RopeScalingRequired,
)

GiB = 1024 ** 3


def test_floor_is_1m():
    assert MIN_CONTEXT_TOKENS == 1_000_000


def test_context_at_or_above_floor_is_accepted():
    for ctx in (1_000_000, 4_000_000, 10_000_000):
        assert QWEN38_27B.validate_context(ctx) == ctx


def test_context_below_floor_is_rejected():
    for ctx in (8192, 262_144, 749_999, 999_999):
        with pytest.raises(ContextFloorError):
            QWEN38_27B.validate_context(ctx)


def test_default_window_stays_inside_the_trained_rotary_range():
    # this is what makes "1M with no YaRN/RoPE scaling" true rather than a wish
    assert DEFAULT_ATTENTION_WINDOW <= QWEN38_27B.max_position_embeddings


def test_no_rope_scaling_at_any_context_when_windowed():
    for ctx in (1_000_000, 4_000_000, 10_000_000, 100_000_000):
        assert QWEN38_27B.needs_rope_scaling(ctx) is False


def test_unwindowed_config_at_the_floor_demands_rope_scaling_and_is_refused():
    # full attention over 1M would extrapolate past 262,144 trained positions
    unwindowed = dataclasses.replace(QWEN38_27B, attention_window=None)
    assert unwindowed.needs_rope_scaling(1_000_000) is True
    with pytest.raises(RopeScalingRequired):
        unwindowed.validate_context(1_000_000)


def test_kv_is_constant_beyond_the_window():
    """The core payoff: KV size — and so decode speed — stops growing."""
    sizes = {QWEN38_27B.kv_bytes_for_context(c)
             for c in (1_000_000, 4_000_000, 10_000_000)}
    assert len(sizes) == 1, sizes
    # and it is the window's worth (plus the pinned sinks), not the context's
    assert QWEN38_27B.kv_bytes_for_context(10_000_000) == \
        QWEN38_27B.kv_bytes_per_token() * \
        (DEFAULT_ATTENTION_WINDOW + QWEN38_27B.attention_sink_tokens)


def test_kv_at_the_floor_fits_the_supported_box():
    weights_gib = 15.41                      # Qwen3.8-27B Q4_K_M
    kv_gib = QWEN38_27B.kv_bytes_for_context(MIN_CONTEXT_TOKENS) / GiB
    total = weights_gib + kv_gib
    assert 8.0 < kv_gib < 9.0, kv_gib        # ~8.5 GiB at the locked 256K window
    # fits the target box: CMP 170HX 40 GiB + RTX 3060 12 GiB = 52 GiB pooled
    assert total < 52.0, total


def test_a_narrower_window_is_refused_the_window_is_locked():
    """The window is locked at max recall; narrowing it for speed is refused.
    The context floor is independent of that - it holds at the locked window."""
    from tenselerate.config import QualityFloorError, validate_window
    with pytest.raises(QualityFloorError, match="LOCKED"):
        validate_window(32_768)
    # context still holds at the locked window, with no RoPE scaling
    assert QWEN38_27B.validate_context(1_000_000) == 1_000_000
    assert QWEN38_27B.needs_rope_scaling(1_000_000) is False


def test_window_is_clamped_by_the_trained_range():
    silly = dataclasses.replace(QWEN38_27B, attention_window=10_000_000)
    # resident KV never exceeds what the model was actually trained to attend
    assert silly.resident_kv_tokens == QWEN38_27B.max_position_embeddings
    # ...and such a config is refused, because the window itself extrapolates
    assert silly.needs_rope_scaling(1_000_000) is True
