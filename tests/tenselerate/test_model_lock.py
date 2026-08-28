"""
The single-model lock: TENSELERATE loads Qwen3.8-27B (RavenX Chaos Agent) and
nothing else. Any other architecture, and any qwen3_5 file whose geometry
differs from the published 27B config in any field, is refused with
UnsupportedModelError — not degraded, not approximated.
"""
from __future__ import annotations

import numpy as np
import pytest

from tenselerate.config import (
    RAVENX_27B, TINY, SUPPORTED_ARCH, UnsupportedModelError,
    config_from_gguf, validate_model,
)
from tenselerate.gguf.reader import GGUF_STRING, GGUF_U32, GGUFReader
from tenselerate.gguf.writer import write_gguf

# the exact published geometry, as GGUF metadata
RAVENX_META = {
    "general.architecture": (GGUF_STRING, SUPPORTED_ARCH),
    "general.name": (GGUF_STRING, "ravenx-chaos-agent-27b"),
    "qwen3_5.block_count": (GGUF_U32, 64),
    "qwen3_5.embedding_length": (GGUF_U32, 5120),
    "qwen3_5.attention.head_count": (GGUF_U32, 24),
    "qwen3_5.attention.head_count_kv": (GGUF_U32, 4),
    "qwen3_5.attention.key_length": (GGUF_U32, 256),
    "qwen3_5.full_attention_interval": (GGUF_U32, 4),
    "qwen3_5.feed_forward_length": (GGUF_U32, 17408),
    "qwen3_5.context_length": (GGUF_U32, 262144),
}


def _reader(tmp_path, meta):
    p = str(tmp_path / "m.gguf")
    write_gguf(p, meta, {"x": (np.ones((2, 2), np.float32), "F32")})
    return GGUFReader(p)


def test_exact_ravenx_geometry_is_accepted(tmp_path):
    cfg = config_from_gguf(_reader(tmp_path, RAVENX_META))
    assert cfg.n_layer == 64 and cfg.n_full_attention_layers == 16


def test_non_qwen_arch_is_refused(tmp_path):
    meta = dict(RAVENX_META)
    meta["general.architecture"] = (GGUF_STRING, "llama")
    with pytest.raises(UnsupportedModelError, match="unsupported architecture"):
        config_from_gguf(_reader(tmp_path, meta))


def test_other_qwen3_archs_are_refused(tmp_path):
    # the lock is exact - a plain qwen3 file no longer slips through the
    # old startswith("qwen3") check
    for arch in ("qwen3", "qwen3moe", "qwen3_5moe"):
        meta = dict(RAVENX_META)
        meta["general.architecture"] = (GGUF_STRING, arch)
        with pytest.raises(UnsupportedModelError):
            config_from_gguf(_reader(tmp_path, meta))


def test_right_arch_wrong_geometry_is_refused(tmp_path):
    # a qwen3_5 file that is not the 27B (e.g. a smaller sibling) is refused,
    # and the error names the mismatched field
    meta = dict(RAVENX_META)
    meta["qwen3_5.block_count"] = (GGUF_U32, 48)
    with pytest.raises(UnsupportedModelError, match="n_layer=48"):
        config_from_gguf(_reader(tmp_path, meta))


def test_wrong_hidden_size_is_refused(tmp_path):
    meta = dict(RAVENX_META)
    meta["qwen3_5.embedding_length"] = (GGUF_U32, 4096)
    with pytest.raises(UnsupportedModelError, match="hidden_size=4096"):
        config_from_gguf(_reader(tmp_path, meta))


def test_validate_model_passes_the_published_config():
    assert validate_model(RAVENX_27B) is RAVENX_27B


def test_tiny_is_not_a_loadable_model():
    # TINY exists for tests and the dev server only; it must never pass the
    # lock that gates real GGUF loads
    with pytest.raises(UnsupportedModelError):
        validate_model(TINY)
