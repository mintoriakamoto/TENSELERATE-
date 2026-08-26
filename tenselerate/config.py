"""
Model configuration for the TENSELERATE engine.

The reference target is the RavenX Chaos Agent — architecture `qwen3_5`, a hybrid
of Gated-DeltaNet linear-attention layers and periodic full-attention layers.
`RAVENX_27B` mirrors the published config.json exactly; `TINY` is a smoke-scale
model with the same *structure* (same full-attention period, same partial-rotary
factor) used by the tests and the dev server so the whole pipeline runs end to
end without the 15.7 GB weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layer: int
    hidden_size: int
    n_head: int
    n_head_kv: int
    head_dim: int
    full_attention_interval: int      # every Nth layer is full attention
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    partial_rotary_factor: float
    rope_theta: float
    # linear (Gated DeltaNet) layer dims
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_conv_kernel_dim: int
    mtp_num_layers: int               # built-in multi-token-prediction draft head

    def is_full_attention(self, layer_idx: int) -> bool:
        """The 16-of-64 pattern: layers 3,7,11,... (1-indexed multiples of N)."""
        return (layer_idx + 1) % self.full_attention_interval == 0

    @property
    def n_full_attention_layers(self) -> int:
        return sum(self.is_full_attention(i) for i in range(self.n_layer))

    @property
    def n_linear_layers(self) -> int:
        return self.n_layer - self.n_full_attention_layers

    def kv_bytes_per_token(self, bits_per_elem: float = 1.0625) -> float:
        """
        KV cache growth per token. Only the full-attention layers cache; the
        linear layers keep a fixed state. This is why a 24 GiB box reaches the
        native 256K context. Default bpe is q8_0 (~1.0625).
        """
        per_layer = 2 * self.n_head_kv * self.head_dim   # K and V
        return per_layer * self.n_full_attention_layers * bits_per_elem


# Exact geometry from OBLITERATUS/Qwen3.8-27B-OBLITERATED/config.json
RAVENX_27B = ModelConfig(
    name="ravenx-chaos-agent-27b",
    n_layer=64,
    hidden_size=5120,
    n_head=24,
    n_head_kv=4,
    head_dim=256,
    full_attention_interval=4,
    intermediate_size=17408,
    vocab_size=248320,
    max_position_embeddings=262144,
    partial_rotary_factor=0.25,
    rope_theta=1.0e7,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_conv_kernel_dim=4,
    mtp_num_layers=1,
)

# Same structure, tiny dims — for tests and the dev server without real weights.
TINY = ModelConfig(
    name="tenselerate-tiny",
    n_layer=8,
    hidden_size=64,
    n_head=4,
    n_head_kv=2,
    head_dim=16,
    full_attention_interval=4,       # 2 of 8 layers are full attention
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=4096,
    partial_rotary_factor=0.25,
    rope_theta=1.0e7,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_conv_kernel_dim=4,
    mtp_num_layers=1,
)

CONFIGS = {c.name: c for c in (RAVENX_27B, TINY)}
