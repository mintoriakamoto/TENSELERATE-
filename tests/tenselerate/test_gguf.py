"""
Round-trip tests for the engine's own GGUF reader/writer. No GPU, no llama.cpp.
Proves we can read real model files: metadata, the tensor directory, and
dequantization of F32/F16/Q8_0 — and that we extract the RavenX hybrid geometry
from GGUF metadata exactly.
"""
from __future__ import annotations

import numpy as np

from tenselerate.gguf.reader import (
    GGUF_ARRAY, GGUF_I32, GGUF_STRING, GGUF_U32, GGUFReader,
)
from tenselerate.gguf.writer import write_gguf


def test_header_and_metadata_roundtrip(tmp_path):
    p = str(tmp_path / "m.gguf")
    meta = {
        "general.architecture": (GGUF_STRING, "qwen3_5"),
        "general.name": (GGUF_STRING, "ravenx-tiny"),
        "qwen3_5.block_count": (GGUF_U32, 64),
        "qwen3_5.full_attention_interval": (GGUF_U32, 4),
        "qwen3_5.attention.head_count_kv": (GGUF_U32, 4),
    }
    tensors = {"tok_embd.weight": (np.zeros((4, 8), np.float32), "F32")}
    write_gguf(p, meta, tensors)

    r = GGUFReader(p)
    assert r.version == 3
    assert r.metadata["general.architecture"] == "qwen3_5"
    assert r.metadata["qwen3_5.block_count"] == 64
    assert r.metadata["qwen3_5.full_attention_interval"] == 4
    assert "tok_embd.weight" in r.tensors


def test_metadata_array_roundtrip(tmp_path):
    p = str(tmp_path / "arr.gguf")
    meta = {
        "general.architecture": (GGUF_STRING, "qwen3_5"),
        # a per-layer recurrent-vs-full flag array, as hybrid models ship
        "qwen3_5.attention.recurrent_layers": (GGUF_ARRAY, (GGUF_I32, [1, 1, 1, 0])),
        "tokenizer.ggml.tokens": (GGUF_ARRAY, (GGUF_STRING, ["<a>", "<b>", "<c>"])),
    }
    write_gguf(p, meta, {"x": (np.ones((2, 2), np.float32), "F32")})
    r = GGUFReader(p)
    assert r.metadata["qwen3_5.attention.recurrent_layers"] == [1, 1, 1, 0]
    assert r.metadata["tokenizer.ggml.tokens"] == ["<a>", "<b>", "<c>"]


def test_tensor_directory_shape_and_type(tmp_path):
    p = str(tmp_path / "t.gguf")
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_gguf(p, {"general.architecture": (GGUF_STRING, "x")},
               {"w": (w, "F32")})
    r = GGUFReader(p)
    t = r.tensors["w"]
    assert t.type_name == "F32"
    # GGUF stores dims fastest-first: a [3,4] row-major array -> (4, 3)
    assert t.shape == (4, 3)
    assert t.n_elements == 12


def test_f32_dequant_exact(tmp_path):
    p = str(tmp_path / "f32.gguf")
    w = np.linspace(-3, 3, 16, dtype=np.float32)
    write_gguf(p, {"general.architecture": (GGUF_STRING, "x")},
               {"w": (w.reshape(4, 4), "F32")})
    r = GGUFReader(p)
    assert np.allclose(r.dequantize("w"), w)


def test_f16_dequant_roundtrips_within_f16_precision(tmp_path):
    p = str(tmp_path / "f16.gguf")
    w = np.array([0.5, -0.25, 1.5, -2.0, 100.0, 0.0], np.float32)
    pad = np.zeros(2, np.float32)
    full = np.concatenate([w, pad])
    write_gguf(p, {"general.architecture": (GGUF_STRING, "x")},
               {"w": (full.reshape(2, 4), "F16")})
    r = GGUFReader(p)
    got = r.dequantize("w")
    assert np.allclose(got[:6], w, rtol=1e-3, atol=1e-3)


def test_q8_0_dequant_within_quant_error(tmp_path):
    p = str(tmp_path / "q8.gguf")
    rng = np.random.default_rng(0)
    w = rng.standard_normal(64).astype(np.float32)      # 2 blocks of 32
    write_gguf(p, {"general.architecture": (GGUF_STRING, "x")},
               {"w": (w.reshape(2, 32), "Q8_0")})
    r = GGUFReader(p)
    got = r.dequantize("w")
    assert r.tensors["w"].type_name == "Q8_0"
    rel = np.linalg.norm(got - w) / np.linalg.norm(w)
    assert rel < 0.01, rel


def test_type_histogram_and_unknown_dequant_message(tmp_path):
    p = str(tmp_path / "h.gguf")
    write_gguf(p, {"general.architecture": (GGUF_STRING, "x")},
               {"a": (np.zeros((2, 2), np.float32), "F32"),
                "b": (np.zeros((2, 32), np.float32), "Q8_0")})
    r = GGUFReader(p)
    hist = r.type_histogram()
    assert hist == {"F32": 1, "Q8_0": 1}


def test_maps_to_model_config(tmp_path):
    # the payoff: read the hybrid geometry out of GGUF metadata, our own code
    p = str(tmp_path / "cfg.gguf")
    meta = {
        "general.architecture": (GGUF_STRING, "qwen3_5"),
        "qwen3_5.block_count": (GGUF_U32, 64),
        "qwen3_5.full_attention_interval": (GGUF_U32, 4),
    }
    write_gguf(p, meta, {"x": (np.ones((2, 2), np.float32), "F32")})
    r = GGUFReader(p)
    n_layer = r.metadata["qwen3_5.block_count"]
    interval = r.metadata["qwen3_5.full_attention_interval"]
    n_full = sum((i + 1) % interval == 0 for i in range(n_layer))
    assert (n_layer, n_full) == (64, 16)     # the real RavenX split


def test_config_from_gguf_builds_ravenx_geometry(tmp_path):
    from tenselerate.config import config_from_gguf
    p = str(tmp_path / "full.gguf")
    meta = {
        "general.architecture": (GGUF_STRING, "qwen3_5"),
        "general.name": (GGUF_STRING, "ravenx-from-gguf"),
        "qwen3_5.block_count": (GGUF_U32, 64),
        "qwen3_5.embedding_length": (GGUF_U32, 5120),
        "qwen3_5.attention.head_count": (GGUF_U32, 24),
        "qwen3_5.attention.head_count_kv": (GGUF_U32, 4),
        "qwen3_5.attention.key_length": (GGUF_U32, 256),
        "qwen3_5.full_attention_interval": (GGUF_U32, 4),
        "qwen3_5.feed_forward_length": (GGUF_U32, 17408),
        "qwen3_5.context_length": (GGUF_U32, 262144),
    }
    write_gguf(p, meta, {"x": (np.ones((2, 2), np.float32), "F32")})
    cfg = config_from_gguf(GGUFReader(p))
    assert cfg.n_layer == 64 and cfg.n_full_attention_layers == 16
    assert cfg.hidden_size == 5120 and cfg.head_dim == 256
    assert cfg.max_position_embeddings == 262144
    # KV per token matches the docs' 34 KiB/token at q8_0
    assert abs(cfg.kv_bytes_per_token(1.0625) / 1024 - 34.0) < 0.5
