"""
The 750K context floor and the windowed-hybrid property that makes it possible
without RoPE scaling. These are product invariants, not implementation details:
if any of them break, the engine is no longer doing what it claims.
"""
from __future__ import annotations

import dataclasses

import pytest

from tenselerate.config import (
    DEFAULT_ATTENTION_WINDOW, MIN_CONTEXT_TOKENS, RAVENX_27B,
    ContextFloorError, RopeScalingRequired,
)

GiB = 1024 ** 3


def test_floor_is_750k():
    assert MIN_CONTEXT_TOKENS == 750_000


def test_context_at_or_above_floor_is_accepted():
    for ctx in (750_000, 1_000_000, 4_000_000):
        assert RAVENX_27B.validate_context(ctx) == ctx


def test_context_below_floor_is_rejected():
    for ctx in (8192, 262_144, 749_999):
        with pytest.raises(ContextFloorError):
            RAVENX_27B.validate_context(ctx)


def test_default_window_stays_inside_the_trained_rotary_range():
    # this is what makes "750K with no YaRN/RoPE scaling" true rather than a wish
    assert DEFAULT_ATTENTION_WINDOW <= RAVENX_27B.max_position_embeddings


def test_no_rope_scaling_at_any_context_when_windowed():
    for ctx in (750_000, 1_000_000, 10_000_000, 100_000_000):
        assert RAVENX_27B.needs_rope_scaling(ctx) is False


def test_unwindowed_config_at_the_floor_demands_rope_scaling_and_is_refused():
    # full attention over 750K would extrapolate past 262,144 trained positions
    unwindowed = dataclasses.replace(RAVENX_27B, attention_window=None)
    assert unwindowed.needs_rope_scaling(750_000) is True
    with pytest.raises(RopeScalingRequired):
        unwindowed.validate_context(750_000)


def test_kv_is_constant_beyond_the_window():
    """The core payoff: KV size — and so decode speed — stops growing."""
    sizes = {RAVENX_27B.kv_bytes_for_context(c)
             for c in (750_000, 1_000_000, 10_000_000)}
    assert len(sizes) == 1, sizes
    # and it is the window's worth, not the context's
    assert RAVENX_27B.kv_bytes_for_context(10_000_000) == \
        RAVENX_27B.kv_bytes_per_token() * DEFAULT_ATTENTION_WINDOW


def test_kv_at_the_floor_fits_both_target_machines():
    weights_gib = 15.41                      # RavenX Q4_K_M
    kv_gib = RAVENX_27B.kv_bytes_for_context(MIN_CONTEXT_TOKENS) / GiB
    total = weights_gib + kv_gib
    assert kv_gib < 5.0, kv_gib              # ~4.25 GiB at the 128K window
    assert total < 24.0, total               # fits the 24 GiB 5070+3060 box
    assert total < 64.0                      # and the unlocked CMP with room


def test_a_smaller_window_trades_recall_for_speed_not_context():
    """Shrinking the window cuts KV but must not cap context or need scaling."""
    narrow = dataclasses.replace(RAVENX_27B, attention_window=32_768)
    assert narrow.validate_context(1_000_000) == 1_000_000
    assert narrow.needs_rope_scaling(1_000_000) is False
    assert narrow.kv_bytes_for_context(1_000_000) < \
        RAVENX_27B.kv_bytes_for_context(1_000_000)


def test_window_is_clamped_by_the_trained_range():
    silly = dataclasses.replace(RAVENX_27B, attention_window=10_000_000)
    # resident KV never exceeds what the model was actually trained to attend
    assert silly.resident_kv_tokens == RAVENX_27B.max_position_embeddings
    # ...and such a config is refused, because the window itself extrapolates
    assert silly.needs_rope_scaling(750_000) is True
