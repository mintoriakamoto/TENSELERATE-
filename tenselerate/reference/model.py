"""
Reference forward pass of the qwen3_5 hybrid, in pure NumPy.

This is not fast and not the real weights — it exists to prove the ENGINE WIRING
is correct: the 16-of-64 full/linear layer schedule, the GQA head sharing, the
partial rotary, the SwiGLU MLP, and — most importantly — that full-attention
layers grow a KV cache while linear layers carry only a fixed state. The CUDA
kernels replace each op in place; this stays as the oracle they are tested
against, and as the thing the dev server serves before the kernels exist.

Weights are random (seeded) unless real ones are loaded, so token output is
meaningless at this stage; the point is that shapes, causality, and the
incremental-decode cache all behave.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tenselerate import nvtx
from tenselerate.config import ModelConfig
from tenselerate.reference import numerics as nx

f32 = np.float32


@dataclass
class LayerState:
    """Per-layer decode state. Exactly one of these is populated per layer."""
    # full-attention layers: growing K/V cache [seq, n_head_kv, head_dim]
    k_cache: list = field(default_factory=list)
    v_cache: list = field(default_factory=list)
    # linear layers: fixed [d_v, d_k] recurrent state, size independent of seq
    gdn_state: NDArray[np.float32] | None = None


class ReferenceModel:
    """A structurally-faithful, weight-random qwen3_5 hybrid for pipeline bring-up."""

    def __init__(self, cfg: ModelConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        h, inter = cfg.hidden_size, cfg.intermediate_size
        scale = 1.0 / np.sqrt(h)

        def w(*shape):
            return (rng.standard_normal(shape) * scale).astype(f32)

        self.embed = w(cfg.vocab_size, h)
        self.norm_f = np.ones(h, f32)
        self.lm_head = w(cfg.vocab_size, h)
        self.layers = []
        for i in range(cfg.n_layer):
            full = cfg.is_full_attention(i)
            layer = {
                "full": full,
                "norm1": np.ones(h, f32),
                "norm2": np.ones(h, f32),
                # SwiGLU MLP
                "gate": w(inter, h),
                "up": w(inter, h),
                "down": w(h, inter),
            }
            if full:
                # attention projections: full GQA geometry
                layer["wq"] = w(cfg.n_head * cfg.head_dim, h)
                layer["wk"] = w(cfg.n_head_kv * cfg.head_dim, h)
                layer["wv"] = w(cfg.n_head_kv * cfg.head_dim, h)
                layer["wo"] = w(h, cfg.n_head * cfg.head_dim)
            else:
                # Gated-DeltaNet projections use their OWN geometry, not the
                # full-attention one - linear_num_key_heads/linear_key_head_dim
                # for Q/K, linear_num_value_heads/linear_value_head_dim for V,
                # repeated the same way GQA repeats K/V across query heads.
                lkh, lkd = cfg.linear_num_key_heads, cfg.linear_key_head_dim
                lvh, lvd = cfg.linear_num_value_heads, cfg.linear_value_head_dim
                layer["wq"] = w(lkh * lkd, h)
                layer["wk"] = w(lkh * lkd, h)
                layer["wv"] = w(lvh * lvd, h)
                layer["wo"] = w(h, lvh * lvd)
                # gates for the delta rule, one per value head
                layer["w_alpha"] = w(lvh, h)
                layer["w_beta"] = w(lvh, h)
            self.layers.append(layer)

    def new_state(self) -> list[LayerState]:
        return [LayerState() for _ in range(self.cfg.n_layer)]

    # -- one decode step: a single token at position `pos` --------------------
    def step(self, token: int, pos: int, state: list[LayerState]) -> NDArray[np.float32]:
        cfg = self.cfg
        x = self.embed[token].copy()                     # [h]

        for li, layer in enumerate(self.layers):
            st = state[li]
            with nvtx.range(f"layer{li}.{'full' if layer['full'] else 'linear'}"):
                h1 = nx.rmsnorm(x[None, :], layer["norm1"])   # [1, h]

                if layer["full"]:
                    nh, nkv, hd = cfg.n_head, cfg.n_head_kv, cfg.head_dim
                    q = nx.quantized_linear(h1, layer["wq"]).reshape(nh, hd)
                    k = nx.quantized_linear(h1, layer["wk"]).reshape(nkv, hd)
                    v = nx.quantized_linear(h1, layer["wv"]).reshape(nkv, hd)
                    mixed = self._full_attention(q, k, v, pos, st)
                    out_dim = nh * hd
                else:
                    lkh, lkd = cfg.linear_num_key_heads, cfg.linear_key_head_dim
                    lvh, lvd = cfg.linear_num_value_heads, cfg.linear_value_head_dim
                    q = nx.quantized_linear(h1, layer["wq"]).reshape(lkh, lkd)
                    k = nx.quantized_linear(h1, layer["wk"]).reshape(lkh, lkd)
                    v = nx.quantized_linear(h1, layer["wv"]).reshape(lvh, lvd)
                    mixed = self._linear_attention(q, k, v, h1, layer, st)
                    out_dim = lvh * lvd

                attn_out = nx.quantized_linear(
                    mixed.reshape(1, out_dim), layer["wo"])[0]
                x = x + attn_out

                h2 = nx.rmsnorm(x[None, :], layer["norm2"])
                g = nx.silu(nx.quantized_linear(h2, layer["gate"]))
                u = nx.quantized_linear(h2, layer["up"])
                x = x + nx.quantized_linear(g * u, layer["down"])[0]

        x = nx.rmsnorm(x[None, :], self.norm_f)
        return nx.quantized_linear(x, self.lm_head)[0]    # [vocab] logits

    def _full_attention(self, q, k, v, pos, st: LayerState):
        cfg = self.cfg
        # rotary on q and k, append k/v to the growing cache
        posarr = np.array([pos], np.int64)
        q = nx.rope_partial(q[None], posarr, cfg.partial_rotary_factor, cfg.rope_theta)[0]
        k = nx.rope_partial(k[None], posarr, cfg.partial_rotary_factor, cfg.rope_theta)[0]
        st.k_cache.append(k)
        st.v_cache.append(v)
        # Sliding window: this is what keeps KV bounded (cfg.resident_kv_tokens)
        # no matter how long the sequence runs - without this eviction the cache
        # grows without bound and the 750K-floor architecture's core claim (a KV
        # cache whose size is constant past the window) does not actually hold.
        window = cfg.resident_kv_tokens
        if len(st.k_cache) > window:
            del st.k_cache[:-window]
            del st.v_cache[:-window]
        K = np.stack(st.k_cache)          # [seq, nkv, hd]
        V = np.stack(st.v_cache)
        rep = cfg.n_head // cfg.n_head_kv                 # GQA sharing
        out = np.empty((cfg.n_head, cfg.head_dim), f32)
        for hh in range(cfg.n_head):
            kvh = hh // rep
            # single query row (the current token) attends to all cached keys
            qh = q[hh][None]                              # [1, hd]
            kh = K[:, kvh]                                # [seq, hd]
            vh = V[:, kvh]
            scores = (qh @ kh.T) / np.sqrt(f32(cfg.head_dim))
            scores -= scores.max()
            wts = np.exp(scores)
            wts /= wts.sum()
            out[hh] = (wts @ vh)[0]
        return out

    def _linear_attention(self, q, k, v, h1, layer, st: LayerState):
        cfg = self.cfg
        # Own geometry, not the full-attention layers' - q/k have
        # linear_num_key_heads heads of linear_key_head_dim, v has
        # linear_num_value_heads heads of linear_value_head_dim, and q/k are
        # repeated across value heads the same way GQA repeats K/V.
        lkh, lkd = cfg.linear_num_key_heads, cfg.linear_key_head_dim
        lvh, lvd = cfg.linear_num_value_heads, cfg.linear_value_head_dim
        if st.gdn_state is None:
            st.gdn_state = np.zeros((lvh, lvd, lkd), f32)   # fixed size, [d_v, d_k]
        alpha = 1.0 / (1.0 + np.exp(-(h1 @ layer["w_alpha"].T)[0]))   # [lvh] in (0,1)
        beta = 1.0 / (1.0 + np.exp(-(h1 @ layer["w_beta"].T)[0]))
        rep = lvh // lkh
        out = np.empty((lvh, lvd), f32)
        for hh in range(lvh):
            kvh = hh // rep
            kt = k[kvh] / (np.linalg.norm(k[kvh]) + 1e-8)
            qh = q[kvh]
            vt = v[hh]
            S = st.gdn_state[hh]
            S = alpha[hh] * (S - beta[hh] * np.outer(S @ kt, kt)) + beta[hh] * np.outer(vt, kt)
            st.gdn_state[hh] = S
            out[hh] = S @ qh
        return out
