"""
Model configuration for the TENSELERATE engine.

TENSELERATE serves exactly ONE model: the RavenX Chaos Agent (Qwen3.8-27B,
architecture `qwen3_5`) — a hybrid of Gated-DeltaNet linear-attention layers and
periodic full-attention layers. That is a product decision, the same kind as the
context floor below: the engine is tuned around this one geometry (its 16-of-64
attention split, its KV footprint, its window math), and every published number
assumes it. Loading anything else is refused, not degraded — `config_from_gguf`
raises `UnsupportedModelError` for any file whose architecture is not `qwen3_5`
or whose geometry differs from `RAVENX_27B` in any field.

`RAVENX_27B` mirrors the published config.json exactly; `TINY` is a smoke-scale
stand-in with the same *structure* (same full-attention period, same
partial-rotary factor) used by the tests and the dev server so the whole
pipeline runs end to end without the 15.7 GB weights. TINY is not a second
supported model — it is never loadable from a GGUF file.

TENSELERATE runs at a HARD FLOOR of 1,000,000 tokens of context (MIN_CONTEXT_TOKENS).
That is a product decision, and it has one unavoidable engineering consequence:
1M is far beyond this model's trained rotary range (262,144), so serving it with
*full* attention would require RoPE scaling (YaRN), which we do not do. The only
way to have both is the hybrid window:

  * the 48 Gated-DeltaNet layers carry long-range memory in a fixed recurrent
    state and have NO positional encoding at all - they are unbounded by
    construction, whether the sequence is 1M or 10M tokens;
  * the 16 full-attention layers attend a bounded WINDOW that never exceeds the
    trained rotary range, so no position is ever extrapolated.

So the floor forces windowing, and windowing pays for itself twice: no RoPE
scaling (no quality loss from position extrapolation) and a KV cache whose size
- and therefore decode speed - is constant no matter how long the context gets.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Hard product floor: the engine never runs below this much context.
MIN_CONTEXT_TOKENS = 1_000_000
# Hard product speed floor: a machine serves TENSELERATE only if it can be
# planned to this much aggregate decode throughput (tok/s) at the context
# floor, at SOME attention window. `tenselerate plan` enforces it: the window
# is the dial that trades exact-recall depth for concurrency, and a box that
# cannot reach the floor at any window is refused, not served slowly.
# 400 is the stated product requirement (with 1M context); the 80 GB CMP
# clears it at a 48K window (~453 tok/s) and does ~681 at the 32K floor.
MIN_DECODE_TOKS = 400
# Hard quality floor: the narrowest attention window the engine will run.
# Speed is bought with concurrency and concurrency with a narrower window,
# but below this much verbatim recall the answers degrade - and quality is
# not for sale. Together with the no-RoPE-scaling rule this bounds the window
# on BOTH sides: MIN_ATTENTION_WINDOW <= window <= trained rotary range.
# 32K is the narrowest window that still meets MIN_DECODE_TOKS on the target
# machine, so the three floors are simultaneously satisfiable by design.
MIN_ATTENTION_WINDOW = 32_768
# Attention sinks (StreamingLLM, arXiv:2309.17453): softmax attention dumps
# surplus probability mass on the first few tokens, so a sliding window that
# evicts them collapses in quality. Keeping the first N tokens resident
# forever fixes it for the cost of N tokens of KV (~136 KiB here). This is a
# long-context method the no-RoPE-scaling rule PERMITS: sink positions are
# 0..N-1 and cache positions are re-anchored, so the attended span is
# window + sinks and stays inside the trained rotary range.
ATTENTION_SINK_TOKENS = 4
# Hard product lock: the only architecture and model this engine will load.
SUPPORTED_ARCH = "qwen3_5"
SUPPORTED_MODEL = "Qwen3.8-27B (RavenX Chaos Agent)"
# Default bounded window for the full-attention layers. Must stay <= the model's
# max_position_embeddings so no position is ever extrapolated (no YaRN/RoPE
# scaling). 128K leaves headroom under the RavenX 256K trained range and keeps
# the KV cache at a constant ~4.25 GiB regardless of total context length.
DEFAULT_ATTENTION_WINDOW = 131_072
# ---------------------------------------------------------------------------


class ContextFloorError(ValueError):
    """Raised when a requested context would drop below MIN_CONTEXT_TOKENS."""


class RopeScalingRequired(ValueError):
    """Raised when a configuration could only work by extrapolating positions."""


class UnsupportedModelError(ValueError):
    """Raised when a file is not the one model this engine serves."""


class QualityFloorError(ValueError):
    """Raised when a configuration would trade model quality for speed."""


def validate_window(window: int) -> int:
    """
    Enforce the quality floor on an attention window. Returns the window on
    success so it can be used inline. The upper bound (the trained rotary
    range) is enforced separately by needs_rope_scaling/validate_context.
    """
    if window < MIN_ATTENTION_WINDOW:
        raise QualityFloorError(
            f"attention window {window:,} is below the TENSELERATE quality "
            f"floor of {MIN_ATTENTION_WINDOW:,} tokens. Speed comes from "
            f"concurrency at a window >= the floor, never from cutting "
            f"recall depth further.")
    return window


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
    # Bounded window for the full-attention layers. None = attend everything,
    # which is only legal while the sequence stays within max_position_embeddings.
    attention_window: int | None = DEFAULT_ATTENTION_WINDOW
    # First N tokens pinned in the cache forever (attention sinks) - the
    # quality fix for sliding-window eviction. Counted into the resident KV
    # and into the no-extrapolation check alongside the window.
    attention_sink_tokens: int = ATTENTION_SINK_TOKENS

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

    # -- the 1M floor, and what makes it possible ------------------------
    @property
    def resident_kv_tokens(self) -> int:
        """
        How many tokens of KV are actually held. With a bounded window this is
        the window, NOT the context length - which is the whole point: KV size,
        and therefore decode speed, stops growing with context.
        """
        if self.attention_window is None:
            return self.max_position_embeddings
        return min(self.attention_window + self.attention_sink_tokens,
                   self.max_position_embeddings)

    def kv_bytes_for_context(self, ctx: int, bits_per_elem: float = 1.0625) -> float:
        """Total KV bytes to serve `ctx` tokens (constant once ctx > window)."""
        held = min(ctx, self.resident_kv_tokens)
        return self.kv_bytes_per_token(bits_per_elem) * held

    def needs_rope_scaling(self, ctx: int) -> bool:
        """
        True if serving `ctx` would extrapolate beyond the trained rotary range.
        A bounded window keeps every position inside that range no matter how
        long the sequence is, so a windowed config answers False at ANY ctx.
        """
        if self.attention_window is not None:
            # sinks sit at re-anchored positions 0..N-1, so the attended span
            # is window + sinks; that whole span must stay inside the range
            return (self.attention_window + self.attention_sink_tokens
                    > self.max_position_embeddings)
        return ctx > self.max_position_embeddings

    def validate_context(self, ctx: int) -> int:
        """
        Enforce the product floor and the no-RoPE-scaling rule. Returns ctx on
        success so it can be used inline.
        """
        if ctx < MIN_CONTEXT_TOKENS:
            raise ContextFloorError(
                f"context {ctx:,} is below the TENSELERATE floor of "
                f"{MIN_CONTEXT_TOKENS:,} tokens")
        if self.needs_rope_scaling(ctx):
            raise RopeScalingRequired(
                f"serving {ctx:,} tokens with attention_window="
                f"{self.attention_window} would extrapolate past the trained "
                f"rotary range ({self.max_position_embeddings:,}). Set an "
                f"attention_window <= {self.max_position_embeddings:,} so the "
                f"GDN layers carry the long range instead.")
        return ctx


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


# The fields that identify Qwen3.8-27B. A file matching all of these IS the
# supported model as far as the engine is concerned; a mismatch in any one of
# them is a different model and is refused.
_IDENTITY_FIELDS = (
    "n_layer", "hidden_size", "n_head", "n_head_kv", "head_dim",
    "full_attention_interval", "intermediate_size", "vocab_size",
    "max_position_embeddings",
)


def validate_model(cfg: ModelConfig) -> ModelConfig:
    """
    Enforce the single-model lock: cfg must match RAVENX_27B in every identity
    field. Returns cfg on success so it can be used inline.
    """
    mismatched = [
        f"{f}={getattr(cfg, f)!r} (expected {getattr(RAVENX_27B, f)!r})"
        for f in _IDENTITY_FIELDS
        if getattr(cfg, f) != getattr(RAVENX_27B, f)
    ]
    if mismatched:
        raise UnsupportedModelError(
            f"TENSELERATE serves only {SUPPORTED_MODEL}; this file's geometry "
            f"does not match: " + ", ".join(mismatched))
    return cfg


def _meta(md: dict, arch: str, *suffixes, default=None):
    """First present of {arch}.{suffix} for the given suffixes, else default."""
    for suf in suffixes:
        key = f"{arch}.{suf}"
        if key in md:
            return md[key]
    return default


def config_from_gguf(reader) -> ModelConfig:
    """
    Build a ModelConfig from a GGUFReader's metadata, using our own reader and
    the standard llama.cpp GGUF key conventions ({arch}.embedding_length, etc.).
    Anything a file omits falls back to the published RAVENX_27B value.

    This is where the single-model lock is enforced: only architecture
    `qwen3_5` is accepted, and the resulting geometry must match RAVENX_27B
    exactly (validate_model), or UnsupportedModelError is raised.
    """
    md = reader.metadata
    arch = md.get("general.architecture", "")
    if arch != SUPPORTED_ARCH:
        raise UnsupportedModelError(
            f"config_from_gguf: unsupported architecture {arch!r} - "
            f"TENSELERATE serves only {SUPPORTED_MODEL} "
            f"(architecture {SUPPORTED_ARCH!r})")
    d = RAVENX_27B
    n_layer = _meta(md, arch, "block_count", default=d.n_layer)
    hidden = _meta(md, arch, "embedding_length", default=d.hidden_size)
    n_head = _meta(md, arch, "attention.head_count", default=d.n_head)
    n_head_kv = _meta(md, arch, "attention.head_count_kv", default=d.n_head_kv)
    head_dim = _meta(md, arch, "attention.key_length", "attention.head_dim",
                     default=d.head_dim)
    interval = _meta(md, arch, "full_attention_interval", default=d.full_attention_interval)
    inter = _meta(md, arch, "feed_forward_length", default=d.intermediate_size)
    ctx = _meta(md, arch, "context_length", default=d.max_position_embeddings)
    vocab = len(md.get("tokenizer.ggml.tokens", [])) or d.vocab_size
    return validate_model(ModelConfig(
        name=md.get("general.name", "gguf-loaded"),
        n_layer=int(n_layer), hidden_size=int(hidden), n_head=int(n_head),
        n_head_kv=int(n_head_kv), head_dim=int(head_dim),
        full_attention_interval=int(interval), intermediate_size=int(inter),
        vocab_size=int(vocab), max_position_embeddings=int(ctx),
        partial_rotary_factor=d.partial_rotary_factor, rope_theta=d.rope_theta,
        linear_key_head_dim=d.linear_key_head_dim,
        linear_value_head_dim=d.linear_value_head_dim,
        linear_num_key_heads=d.linear_num_key_heads,
        linear_num_value_heads=d.linear_num_value_heads,
        linear_conv_kernel_dim=d.linear_conv_kernel_dim,
        mtp_num_layers=d.mtp_num_layers,
    ))
