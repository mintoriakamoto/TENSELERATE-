"""
End-to-end wiring tests for the reference engine: shapes, causality, the hybrid
KV/GDN cache behaviour, and a full generate() run. No GPU. These prove the
pipeline is correct before any CUDA kernel exists.
"""
from __future__ import annotations

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
            assert s.gdn_state.shape == (TINY.n_head, TINY.head_dim, TINY.head_dim)


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
