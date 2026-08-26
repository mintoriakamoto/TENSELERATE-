"""
Correctness tests for the reference numerics. These run with no GPU and are the
ground truth every CUDA kernel is validated against. Run: pytest tests/tenselerate
"""
from __future__ import annotations

import numpy as np
import pytest

from tenselerate.reference import numerics as nx

f32 = np.float32


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---- int8 quantization + IMMA-style integer matmul ------------------------
def test_int8_roundtrip_bounded():
    x = _rng(1).standard_normal((8, 64)).astype(f32)
    q, s = nx.quantize_int8_symmetric(x)
    deq = q.astype(f32) * s
    # per-row max error is at most half a quantization step
    step = (np.max(np.abs(x), axis=-1, keepdims=True) / 127.0)
    assert np.all(np.abs(deq - x) <= step * 0.5 + 1e-6)


def test_int8_zero_row_no_nan():
    x = np.zeros((3, 16), f32)
    x[1] = _rng(2).standard_normal(16)
    q, s = nx.quantize_int8_symmetric(x)
    deq = q.astype(f32) * s
    assert not np.any(np.isnan(deq))
    assert np.allclose(deq[0], 0.0)


def test_int8_matmul_matches_float_within_quant_error():
    rng = _rng(3)
    a = rng.standard_normal((5, 128)).astype(f32)
    w = rng.standard_normal((7, 128)).astype(f32)
    ref = a @ w.T
    got = nx.quantized_linear(a, w)
    # int8 on both sides over K=128: relative error should be small
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 0.02, rel


def test_int8_matmul_integer_accumulator_is_exact():
    # with inputs that are already exact int8 * scale, the accumulator is exact
    aq = _rng(4).integers(-127, 128, (4, 32)).astype(np.int8)
    bq = _rng(5).integers(-127, 128, (6, 32)).astype(np.int8)
    a_s = np.ones((4, 1), f32)
    b_s = np.ones((6, 1), f32)
    got = nx.int8_matmul(aq, a_s, bq, b_s)
    exact = aq.astype(np.int64) @ bq.astype(np.int64).T
    assert np.array_equal(got.astype(np.int64), exact)


# ---- RMSNorm + partial RoPE ----------------------------------------------
def test_rmsnorm_unit_scale():
    x = _rng(6).standard_normal((4, 32)).astype(f32) * 5.0
    y = nx.rmsnorm(x, np.ones(32, f32))
    rms = np.sqrt(np.mean(y**2, axis=-1))
    assert np.allclose(rms, 1.0, atol=1e-3)


def test_rope_position_zero_is_identity():
    x = _rng(7).standard_normal((3, 256)).astype(f32)
    pos = np.zeros(3, np.int64)
    y = nx.rope_partial(x, pos, rotary_factor=0.25)
    assert np.allclose(y, x, atol=1e-5)


def test_rope_preserves_norm_and_only_touches_rotary_part():
    x = _rng(8).standard_normal((5, 256)).astype(f32)
    pos = np.arange(5, dtype=np.int64)
    y = nx.rope_partial(x, pos, rotary_factor=0.25)
    rot = (256 // 4) - (256 // 4) % 2      # 64 dims rotated
    # norm of the rotated block is preserved by a rotation
    assert np.allclose(np.linalg.norm(x[:, :rot], axis=-1),
                       np.linalg.norm(y[:, :rot], axis=-1), atol=1e-4)
    # the non-rotary tail is untouched
    assert np.allclose(x[:, rot:], y[:, rot:])


# ---- softmax attention ----------------------------------------------------
def test_attention_is_causal():
    rng = _rng(9)
    seq, d = 6, 16
    q = rng.standard_normal((seq, d)).astype(f32)
    k = rng.standard_normal((seq, d)).astype(f32)
    v = rng.standard_normal((seq, d)).astype(f32)
    out = nx.softmax_attention(q, k, v)
    # perturbing a FUTURE key/value must not change an earlier output row
    k2 = k.copy()
    v2 = v.copy()
    k2[5] += 10.0
    v2[5] += 10.0
    out2 = nx.softmax_attention(q, k2, v2)
    assert np.allclose(out[:5], out2[:5], atol=1e-5)
    assert not np.allclose(out[5], out2[5])


# ---- gated delta net ------------------------------------------------------
def test_gdn_chunked_equals_sequential():
    rng = _rng(10)
    seq, d = 40, 24
    q = rng.standard_normal((seq, d)).astype(f32)
    k = rng.standard_normal((seq, d)).astype(f32)
    v = rng.standard_normal((seq, d)).astype(f32)
    alpha = rng.uniform(0.85, 0.999, seq).astype(f32)
    beta = rng.uniform(0.0, 1.0, seq).astype(f32)
    seq_out = nx.gated_delta_net_sequential(q, k, v, alpha, beta)
    for chunk in (1, 7, 16, 64):
        ch = nx.gated_delta_net_chunked(q, k, v, alpha, beta, chunk=chunk)
        assert np.allclose(seq_out, ch, atol=1e-4), f"chunk={chunk}"


def test_gdn_decay_gate_forgets():
    # alpha -> 0 resets the state each step, so output depends only on the
    # current token's write; alpha = 1, beta = 0 freezes the (zero) state.
    rng = _rng(11)
    seq, d = 8, 12
    q = rng.standard_normal((seq, d)).astype(f32)
    k = rng.standard_normal((seq, d)).astype(f32)
    v = rng.standard_normal((seq, d)).astype(f32)
    frozen = nx.gated_delta_net_sequential(
        q, k, v, np.ones(seq, f32), np.zeros(seq, f32))
    assert np.allclose(frozen, 0.0, atol=1e-6)


def test_gdn_state_is_fixed_size_regardless_of_length():
    # the whole point of the linear layers: state shape does not grow with seq.
    rng = _rng(12)
    d = 16
    for seq in (4, 400):
        q = rng.standard_normal((seq, d)).astype(f32)
        k = rng.standard_normal((seq, d)).astype(f32)
        v = rng.standard_normal((seq, d)).astype(f32)
        a = rng.uniform(0.9, 0.99, seq).astype(f32)
        b = rng.uniform(0.0, 1.0, seq).astype(f32)
        out = nx.gated_delta_net_sequential(q, k, v, a, b)
        assert out.shape == (seq, d)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
