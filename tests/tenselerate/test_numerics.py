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


def test_rope_matches_float64_reference_at_large_absolute_positions():
    """
    Regression for float32 angle computation: at the positions the 750K context
    floor actually reaches (hundreds of thousands), computing the rotation
    angle entirely in float32 loses enough precision that cos/sin drift from
    the true angle. Compare against an explicit float64 reference.
    """
    x = _rng(70).standard_normal((3, 256)).astype(f32)
    pos = np.array([1, 500_000, 749_999], dtype=np.int64)
    y = nx.rope_partial(x, pos, rotary_factor=0.25, theta=1.0e7)

    rot = (256 // 4) - (256 // 4) % 2
    half = rot // 2
    inv_freq64 = 1.0e7 ** (-np.arange(0, half, dtype=np.float64) / half)
    ang64 = pos.astype(np.float64)[:, None] * inv_freq64[None, :]
    cos64, sin64 = np.cos(ang64), np.sin(ang64)
    x1, x2 = x[:, :half].astype(np.float64), x[:, half:rot].astype(np.float64)
    expect = x.copy().astype(f32)
    expect[:, :half] = (x1 * cos64 - x2 * sin64).astype(f32)
    expect[:, half:rot] = (x1 * sin64 + x2 * cos64).astype(f32)

    assert np.allclose(y, expect, atol=1e-4)


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
def test_int8_qk_scores_match_float_within_quant_error():
    rng = _rng(20)
    q = rng.standard_normal((4, 128)).astype(f32)
    k = rng.standard_normal((64, 128)).astype(f32)
    ref = (q @ k.T) / np.sqrt(f32(128))
    got = nx.int8_qk_scores(q, k)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 0.02, rel


def test_int8_qk_attention_close_to_float_and_still_causal():
    rng = _rng(21)
    seq, hd = 16, 128
    q = rng.standard_normal((seq, hd)).astype(f32)
    k = rng.standard_normal((seq, hd)).astype(f32)
    v = rng.standard_normal((seq, hd)).astype(f32)
    ref = nx.softmax_attention(q, k, v)
    got = nx.softmax_attention(q, k, v, int8_qk=True)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 0.05, rel
    # future keys/values must still not leak into earlier rows
    k2, v2 = k.copy(), v.copy()
    k2[10:] = rng.standard_normal((seq - 10, hd))
    v2[10:] = rng.standard_normal((seq - 10, hd))
    out2 = nx.softmax_attention(q, k2, v2, int8_qk=True)
    assert np.allclose(got[:10], out2[:10], atol=1e-5)


def test_int8_qk_decode_row_over_long_cache_keeps_argmax():
    # the decode shape: one query against a long cache. The int8 scores must
    # still rank the same key first and produce a normalized row.
    rng = _rng(22)
    n_keys, hd = 4096, 128
    q = rng.standard_normal((1, hd)).astype(f32)
    k = rng.standard_normal((n_keys, hd)).astype(f32)
    k[1234] = q[0] * 3.0            # a clearly matching key
    ref = (q @ k.T) / np.sqrt(f32(hd))
    got = nx.int8_qk_scores(q, k)
    assert np.argmax(got) == np.argmax(ref) == 1234
    w = np.exp(got - got.max())
    w /= w.sum()
    assert np.isclose(w.sum(), 1.0, atol=1e-5)
    assert np.argmax(w) == 1234


def _dense_row(q, k, v):
    s = (q / np.sqrt(f32(q.shape[-1]))) @ k.T
    s -= s.max()
    w = np.exp(s)
    w /= w.sum()
    return w @ v


def test_page_skip_bound_is_a_true_upper_bound():
    rng = _rng(30)
    k = rng.standard_normal((1000, 64)).astype(f32)
    q = rng.standard_normal(64).astype(f32)
    kmin, kmax = nx.kv_page_bounds(k, page=64)
    ub = np.maximum(q * kmin, q * kmax).sum(axis=-1)
    exact = q @ k.T
    for p in range(ub.shape[0]):
        assert exact[p * 64:(p + 1) * 64].max() <= ub[p] + 1e-4


def test_page_skip_matches_dense_row_on_random_keys():
    # random keys are the worst case for skipping; correctness must hold regardless
    rng = _rng(31)
    n_k, hd = 4096 + 17, 128           # ragged last page
    q = rng.standard_normal(hd).astype(f32)
    k = rng.standard_normal((n_k, hd)).astype(f32)
    v = rng.standard_normal((n_k, hd)).astype(f32)
    out, pages = nx.decode_attention_page_skip(q, k, v)
    assert np.allclose(out, _dense_row(q, k, v), atol=1e-5)
    assert 1 <= pages <= -(-n_k // 64)


def _peaked_row(rng, n_k, hd, n_hot, hot_logit_scaled):
    """A few keys the query matches at a chosen scaled logit; the rest random."""
    q = rng.standard_normal(hd).astype(f32)
    q /= np.linalg.norm(q)
    k = rng.standard_normal((n_k, hd)).astype(f32) * 0.5
    hot = rng.choice(n_k, n_hot, replace=False)
    k[hot] = q * (hot_logit_scaled * np.sqrt(hd)) + rng.standard_normal((n_hot, hd)).astype(f32) * 0.5
    v = rng.standard_normal((n_k, hd)).astype(f32)
    return q, k, v


def test_page_skip_skips_most_pages_when_peak_clears_the_gap():
    # Peak sits page_skip_gap() above the random pages' box bound (~1 scaled
    # unit here): nearly every page is provably negligible and is skipped.
    rng = _rng(32)
    n_k, hd = 262144 // 8, 128        # 32K keys, 512 pages
    q, k, v = _peaked_row(rng, n_k, hd, 12, nx.page_skip_gap() + 4.0)
    out, pages = nx.decode_attention_page_skip(q, k, v)
    assert np.allclose(out, _dense_row(q, k, v), atol=1e-5)
    assert pages <= 0.1 * (n_k // 64), pages       # >= 90% of the KV read skipped


def test_page_skip_reads_everything_when_peak_is_below_the_gap():
    # The same row with a realistic-looking peak (scaled logit ~4): every page's
    # bound is within the gap, nothing is provably negligible, all pages read.
    # This pins the method's limit; a kernel gets bytes back only past the gap.
    rng = _rng(32)
    n_k, hd = 8192, 128
    q, k, v = _peaked_row(rng, n_k, hd, 12, 4.0)
    out, pages = nx.decode_attention_page_skip(q, k, v)
    assert np.allclose(out, _dense_row(q, k, v), atol=1e-5)
    assert pages == n_k // 64
    assert 20.0 < nx.page_skip_gap() < 21.5


def test_page_skip_skipped_mass_is_below_eps():
    rng = _rng(33)
    n_k, hd = 8192, 128
    q = rng.standard_normal(hd).astype(f32)
    k = rng.standard_normal((n_k, hd)).astype(f32)
    hot = rng.choice(n_k, 4, replace=False)
    k[hot] = q * 3.0
    v = np.eye(n_k, hd, dtype=f32)     # not used for the bound; any v works
    eps = 1e-6
    _, pages = nx.decode_attention_page_skip(q, k, v, eps=eps)
    qs = q / np.sqrt(f32(hd))
    s = qs @ k.T
    w = np.exp(s - s.max())
    w /= w.sum()
    kmin, kmax = nx.kv_page_bounds(k, 64)
    ub = np.maximum(qs * kmin, qs * kmax).sum(axis=-1)
    p_star = int(np.argmax(ub))
    m_lb = (qs @ k[p_star * 64:(p_star + 1) * 64].T).max()
    skipped = np.flatnonzero(64 * np.exp(ub - m_lb) < eps)
    for p in skipped:
        assert w[p * 64:(p + 1) * 64].sum() < eps
    assert pages == ub.shape[0] - len(skipped) or pages == ub.shape[0] - len(skipped) + 1


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
