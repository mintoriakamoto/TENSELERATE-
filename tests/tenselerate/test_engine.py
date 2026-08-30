"""
End-to-end wiring tests for the reference engine: shapes, causality, the hybrid
KV/GDN cache behaviour, and a full generate() run. No GPU. These prove the
pipeline is correct before any CUDA kernel exists.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from tenselerate.config import RAVENX_27B, TINY
from tenselerate.engine.generation import Generator, SamplingParams
from tenselerate.reference.model import ReferenceModel


def test_config_layer_schedule_matches_published_geometry():
    c = RAVENX_27B
    assert c.n_full_attention_layers == 16
    assert c.n_linear_layers == 48
    assert c.is_full_attention(3) and not c.is_full_attention(2)
    # q8_0 KV over the 16 full layers == 34 KiB/token (the number the docs use)
    assert abs(c.kv_bytes_per_token(1.0625) / 1024 - 34.0) < 0.5


def test_step_returns_vocab_logits():
    m = ReferenceModel(TINY, seed=1)
    st = m.new_state()
    logits = m.step(token=5, pos=0, state=st)
    assert logits.shape == (TINY.vocab_size,)
    assert np.all(np.isfinite(logits))


def test_full_layers_grow_cache_linear_layers_do_not():
    m = ReferenceModel(TINY, seed=2)
    st = m.new_state()
    for pos, tok in enumerate([1, 2, 3, 4, 5]):
        m.step(tok, pos, st)
    for li, s in enumerate(st):
        if TINY.is_full_attention(li):
            # full-attention layer cached one K/V per token
            assert len(s.k_cache) == 5 and len(s.v_cache) == 5
            assert s.gdn_state is None
        else:
            # linear layer holds a FIXED-size state, no growing cache
            assert s.k_cache == [] and s.v_cache == []
            assert s.gdn_state is not None
            # GDN state geometry is the layer's OWN (linear_*) dims, not the
            # full-attention layers' n_head/head_dim - see bug fix below.
            assert s.gdn_state.shape == (
                TINY.linear_num_value_heads, TINY.linear_value_head_dim,
                TINY.linear_key_head_dim)


def test_full_attention_cache_is_bounded_by_the_window():
    """
    Regression for the missing sliding-window eviction: a full-attention
    layer's K/V cache must stop growing once it reaches cfg.resident_kv_tokens,
    no matter how many more tokens are generated - that is the entire premise
    of serving 750K+ tokens on a constant-size KV cache.
    """
    cfg = replace(TINY, attention_window=3, max_position_embeddings=4096)
    m = ReferenceModel(cfg, seed=6)
    st = m.new_state()
    n_steps = 10
    assert n_steps > cfg.resident_kv_tokens
    for pos in range(n_steps):
        m.step(pos % cfg.vocab_size, pos, st)
    for li, s in enumerate(st):
        if cfg.is_full_attention(li):
            assert len(s.k_cache) == cfg.resident_kv_tokens
            assert len(s.v_cache) == cfg.resident_kv_tokens


def test_gdn_state_size_independent_of_sequence_length():
    m = ReferenceModel(TINY, seed=3)
    sizes = []
    for length in (2, 50):
        st = m.new_state()
        for pos in range(length):
            m.step(pos % TINY.vocab_size, pos, st)
        lin = next(s for li, s in enumerate(st) if not TINY.is_full_attention(li))
        assert lin.gdn_state is not None
        sizes.append(lin.gdn_state.nbytes)
    assert sizes[0] == sizes[1]      # fixed regardless of length


def test_generate_is_deterministic_and_greedy():
    m = ReferenceModel(TINY, seed=4)
    gen = Generator(m)
    p = SamplingParams(max_tokens=12, temperature=0.0)
    a = gen.generate_list([1, 2, 3], p)
    b = gen.generate_list([1, 2, 3], p)
    assert a == b and len(a) == 12
    assert all(0 <= t < TINY.vocab_size for t in a)


def test_stop_token_halts_generation():
    m = ReferenceModel(TINY, seed=4)
    gen = Generator(m)
    # discover the first greedy token, then make it a stop token
    first = gen.generate_list([7, 8], SamplingParams(max_tokens=1, temperature=0.0))[0]
    out = gen.generate_list([7, 8], SamplingParams(max_tokens=20, stop_tokens=(first,)))
    assert out == []


def test_temperature_sampling_varies_with_seed():
    m = ReferenceModel(TINY, seed=5)
    gen = Generator(m)
    x = gen.generate_list([3], SamplingParams(max_tokens=10, temperature=1.0, seed=1))
    y = gen.generate_list([3], SamplingParams(max_tokens=10, temperature=1.0, seed=2))
    assert x != y      # different seeds -> different samples
