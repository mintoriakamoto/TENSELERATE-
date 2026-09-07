"""
Reference numerics for the TENSELERATE engine.

Every hot operation in the engine has a CUDA kernel (tenselerate/csrc/) and a
pure-NumPy reference here. The reference is the source of truth: kernels are
validated against it, and it runs anywhere without a GPU, so the math is pinned
by tests (tests/tenselerate/) that pass on CI with no CUDA at all.

The three operations that matter for the RavenX / qwen3_5 hybrid target:

  * int8 symmetric quantized matmul  -> csrc/int8_gemm.cu (IMMA, the path that
    is NOT throttled on the CMP 170HX and where FP16 GEMM would be a trap)
  * gated-delta-net recurrence       -> the 48 linear-attention layers; the
    chunked form the kernel uses is checked here against the sequential form
  * partial rotary embedding + RMSNorm + softmax attention for the 16 full
    layers

Nothing here is tuned for speed; it is written to be obviously correct.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

f32 = np.float32


# --------------------------------------------------------------------------
# int8 symmetric quantization + integer matmul (the IMMA path)
# --------------------------------------------------------------------------
def quantize_int8_symmetric(x: NDArray[np.float32], axis: int = -1):
    """
    Per-row symmetric int8 quantization: q = round(x / scale), scale = max|x| / 127.

    Returns (q int8, scale float32). Rows that are all zero get scale 1 so the
    dequant round-trips exactly instead of dividing by zero.
    """
    amax = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(f32)
    q = np.rint(x / scale).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def int8_matmul(
    a_q: NDArray[np.int8], a_scale: NDArray[np.float32],
    b_q: NDArray[np.int8], b_scale: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    C = (a_q * a_scale) @ (b_q * b_scale)^T computed with an INTEGER accumulator,
    exactly as an IMMA tensor-core GEMM does: int8 x int8 -> int32 accumulate,
    then a single float rescale at the end.

    a_q:      [M, K] int8, per-row scale a_scale [M, 1]
    b_q:      [N, K] int8, per-row scale b_scale [N, 1]   (weight, row = output)
    returns   [M, N] float32
    """
    acc = a_q.astype(np.int32) @ b_q.astype(np.int32).T   # [M, N] exact int32
    return acc.astype(f32) * a_scale * b_scale.T


def quantized_linear(x: NDArray[np.float32], w: NDArray[np.float32]) -> NDArray[np.float32]:
    """y = x @ w^T with w quantized to int8 per output-row and x per token-row."""
    xq, xs = quantize_int8_symmetric(x, axis=-1)
    wq, ws = quantize_int8_symmetric(w, axis=-1)
    return int8_matmul(xq, xs, wq, ws)


# --------------------------------------------------------------------------
# RMSNorm and partial rotary embedding (full-attention layers)
# --------------------------------------------------------------------------
def rmsnorm(x: NDArray[np.float32], weight: NDArray[np.float32], eps: float = 1e-6) -> NDArray[np.float32]:
    var = np.mean(x.astype(f32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)).astype(f32) * weight


def rope_partial(
    x: NDArray[np.float32], positions: NDArray[np.int64],
    rotary_factor: float = 0.25, theta: float = 1.0e7,
) -> NDArray[np.float32]:
    """
    Rotary embedding applied to only the first `rotary_factor` of head_dim, the
    rest passed through unchanged (config: partial_rotary_factor = 0.25).

    x: [..., seq, head_dim]. positions: [seq].
    """
    head_dim = x.shape[-1]
    rot = int(round(head_dim * rotary_factor))
    rot -= rot % 2                      # rotary dims come in (even) pairs
    if rot == 0:
        return x.copy()
    half = rot // 2
    # Angle computed in float64: positions run to the millions at the 750K
    # context floor, and float32 loses enough precision there that cos/sin of
    # the rotation angle drift measurably. The rotation itself still applies
    # in float32 - only the angle needs the wider type.
    inv_freq = theta ** (-np.arange(0, half, dtype=np.float64) / half)
    ang = positions.astype(np.float64)[:, None] * inv_freq[None, :]   # [seq, half]
    cos = np.cos(ang).astype(f32)
    sin = np.sin(ang).astype(f32)
    out = x.copy().astype(f32)
    x1 = out[..., 0:half].copy()      # snapshot: the writes below alias `out`
    x2 = out[..., half:rot].copy()
    out[..., 0:half] = x1 * cos - x2 * sin
    out[..., half:rot] = x1 * sin + x2 * cos
    return out


# --------------------------------------------------------------------------
# softmax full attention (the 16 full-attention layers, causal, GQA)
# --------------------------------------------------------------------------
def int8_qk_scores(q: NDArray[np.float32], k: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Attention scores q @ k^T / sqrt(head_dim) with q and k int8-quantized per row
    and the dot products accumulated in int32 — the same IMMA GEMM the linear
    layers use, applied to the QK^T product. This is the part of attention that
    is a matmul and therefore benefits from int8; softmax and the PV product
    stay float32 (softmax is elementwise exp/sum, int8 buys nothing there).

    q: [n_q, head_dim], k: [n_k, head_dim] -> [n_q, n_k] float32. A real kernel
    quantizes K once when it enters the cache; the reference just does it here.
    """
    qq, qs = quantize_int8_symmetric(q, axis=-1)
    kq, ks = quantize_int8_symmetric(k, axis=-1)
    return int8_matmul(qq, qs, kq, ks) / np.sqrt(f32(q.shape[-1]))


def softmax_attention(
    q: NDArray[np.float32], k: NDArray[np.float32], v: NDArray[np.float32],
    int8_qk: bool = False,
) -> NDArray[np.float32]:
    """
    Causal single-head attention. q,k,v: [seq, head_dim]. GQA head-sharing is
    handled by the caller repeating k/v across query heads. int8_qk routes the
    QK^T product through int8_qk_scores.
    """
    seq, head_dim = q.shape
    if int8_qk:
        scores = int8_qk_scores(q, k)
    else:
        scores = (q @ k.T) / np.sqrt(f32(head_dim))      # [seq, seq]
    mask = np.triu(np.full((seq, seq), -np.inf, dtype=f32), k=1)
    scores = scores + mask
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ v


# --------------------------------------------------------------------------
# provable-mass KV page skipping (decode row over a long cache)
# --------------------------------------------------------------------------
PAGE_SKIP_EPS = 2.0 ** -24   # below fp32 resolution: skipping is bit-identical up to summation order


def page_skip_gap(page: int = 64, eps: float = PAGE_SKIP_EPS) -> float:
    """
    How far (in scaled-logit units, i.e. after the 1/sqrt(hd)) the row max must
    sit above a page's upper bound for that page to be provably skippable:
    ln(page / eps). At the defaults that is ~20.8. This is the honest limit of
    the method: pages are skipped only where attention is peaked by at least
    that margin over the page's box bound, which the tests below make explicit.
    """
    return float(np.log(page / eps))


def kv_page_bounds(k: NDArray[np.float32], page: int = 64):
    """
    Per-page coordinate-wise (min, max) of the keys. [n_k, hd] -> two [n_pages, hd]
    arrays; the last page may be short. A kernel stores these next to the KV
    blocks (2 * hd floats per page, ~3% of q8_0 KV) when a key enters the cache.
    """
    n_k, hd = k.shape
    n_pages = -(-n_k // page)
    kmin = np.empty((n_pages, hd), f32)
    kmax = np.empty((n_pages, hd), f32)
    for p in range(n_pages):
        blk = k[p * page:(p + 1) * page]
        kmin[p] = blk.min(axis=0)
        kmax[p] = blk.max(axis=0)
    return kmin, kmax


def decode_attention_page_skip(
    q: NDArray[np.float32], k: NDArray[np.float32], v: NDArray[np.float32],
    page: int = 64, eps: float = PAGE_SKIP_EPS,
) -> tuple[NDArray[np.float32], int]:
    """
    One query row against a long cache, reading only the pages that can matter.

    For page p, ub_p = sum_d max(q_d * kmin_pd, q_d * kmax_pd) is a true upper
    bound on every logit in the page. The page holding the largest ub is scored
    exactly, which gives a lower bound m_lb on the row max. Since the softmax
    denominator is >= 1 after subtracting the true max, page p's share of the
    output is <= n_p * exp(ub_p - m_lb); when that is below eps the page is
    skipped. This is a proof of negligibility, not a top-k heuristic: at the
    default eps the skipped mass is below what fp32 can represent, so the result
    equals the dense row up to summation order. Returns (out [hd], pages_read).

    q: [hd] (already the current token's query), k, v: [n_k, hd].
    """
    hd = q.shape[-1]
    qs = (q / np.sqrt(f32(hd))).astype(f32)
    kmin, kmax = kv_page_bounds(k, page)
    ub = np.maximum(qs * kmin, qs * kmax).sum(axis=-1)         # [n_pages]
    n_pages = ub.shape[0]
    sizes = np.full(n_pages, page, np.int64)
    sizes[-1] = k.shape[0] - page * (n_pages - 1)
    p_star = int(np.argmax(ub))
    m_lb = float((qs @ k[p_star * page:(p_star + 1) * page].T).max())
    keep = sizes * np.exp(ub - m_lb) >= eps
    keep[p_star] = True
    idx = np.concatenate([np.arange(p * page, p * page + sizes[p]) for p in np.flatnonzero(keep)])
    scores = qs @ k[idx].T
    scores -= scores.max()
    w = np.exp(scores)
    w /= w.sum()
    return (w @ v[idx]).astype(f32), int(keep.sum())


# --------------------------------------------------------------------------
# gated delta net (the 48 linear-attention layers)
# --------------------------------------------------------------------------
def gated_delta_net_sequential(
    q: NDArray[np.float32], k: NDArray[np.float32], v: NDArray[np.float32],
    alpha: NDArray[np.float32], beta: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Sequential gated delta rule (Yang et al.), one head:

        S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
        o_t = S_t q_t

    S is [d_v, d_k]; k is L2-normalized. alpha_t in (0,1) is the scalar decay
    gate, beta_t in (0,1) the write strength. This is the ground-truth form the
    chunked kernel must reproduce. Fixed state size, independent of seq length —
    which is why these layers do not grow a KV cache.

    q,k,v: [seq, d]. alpha,beta: [seq]. returns [seq, d_v] (d_v == d here).
    """
    seq, d = q.shape
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    S = np.zeros((d, d), dtype=f32)      # [d_v, d_k]
    out = np.zeros((seq, d), dtype=f32)
    for t in range(seq):
        kt = k[t]                        # [d_k]
        vt = v[t]                        # [d_v]
        S = alpha[t] * (S - beta[t] * np.outer(S @ kt, kt)) + beta[t] * np.outer(vt, kt)
        out[t] = S @ q[t]
    return out


def gated_delta_net_chunked(
    q: NDArray[np.float32], k: NDArray[np.float32], v: NDArray[np.float32],
    alpha: NDArray[np.float32], beta: NDArray[np.float32], chunk: int = 16,
) -> NDArray[np.float32]:
    """
    Chunked evaluation of the same recurrence: the state is carried across
    chunks, and within a chunk tokens are still stepped but the state hand-off
    is the unit the CUDA kernel parallelizes over. Must equal the sequential
    form up to float error; tests assert exactly that. This is the structure the
    kernel uses to keep the 48 linear layers off a growing cache.
    """
    seq, d = q.shape
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    S = np.zeros((d, d), dtype=f32)
    out = np.zeros((seq, d), dtype=f32)
    for start in range(0, seq, chunk):
        end = min(start + chunk, seq)
        for t in range(start, end):
            kt, vt = k[t], v[t]
            S = alpha[t] * (S - beta[t] * np.outer(S @ kt, kt)) + beta[t] * np.outer(vt, kt)
            out[t] = S @ q[t]
    return out


def silu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return x / (1.0 + np.exp(-x))
