"""
The vLLM backend launcher: it builds a correct `vllm serve` argv for the RavenX
model on the Ampere box, and it enforces the SAME product floors the native
engine does before vLLM is ever started - so the backend swap changes the
runtime, never the contract.
"""
from __future__ import annotations

import dataclasses

import pytest

from tenselerate.backends.vllm import (
    AMPERE_BOX, KV_CACHE_DTYPE, RAVENX_MODEL_REF, build_vllm_serve_argv,
    vllm_serve_command,
)
from tenselerate.config import (
    MIN_CONTEXT_TOKENS, RAVENX_27B, ContextFloorError, QualityFloorError,
    RopeScalingRequired,
)


def argv(ctx: int = MIN_CONTEXT_TOKENS, **kw) -> list[str]:
    return build_vllm_serve_argv(RAVENX_27B, ctx=ctx, **kw)


def test_serves_the_one_model_via_vllm():
    a = argv()
    assert a[:3] == ["vllm", "serve", RAVENX_MODEL_REF]
    assert "--quantization" in a and a[a.index("--quantization") + 1] == "gguf"


def test_two_ampere_gpus_use_pipeline_not_tensor_parallel():
    # heterogeneous CMP 170HX + RTX 3060 12 GiB, no NVLink -> PP=2, never TP
    a = argv()
    assert a[a.index("--pipeline-parallel-size") + 1] == str(len(AMPERE_BOX))
    assert "--tensor-parallel-size" not in a
    assert len(AMPERE_BOX) == 2


def test_context_floor_flows_to_max_model_len():
    a = argv()
    assert a[a.index("--max-model-len") + 1] == str(MIN_CONTEXT_TOKENS)


def test_kv_bits_maps_to_cache_dtype():
    assert KV_CACHE_DTYPE[8] == "auto" and KV_CACHE_DTYPE[4] == "fp8"
    assert argv(kv_bits=8)[
        argv(kv_bits=8).index("--kv-cache-dtype") + 1] == "auto"
    assert argv(kv_bits=4)[
        argv(kv_bits=4).index("--kv-cache-dtype") + 1] == "fp8"


def test_mtp_adds_speculative_config():
    assert "--speculative-config" not in argv(spec="none")
    a = argv(spec="mtp")
    cfg = a[a.index("--speculative-config") + 1]
    assert "qwen3_next_mtp" in cfg


def test_context_floor_is_enforced_before_vllm_starts():
    with pytest.raises(ContextFloorError):
        argv(ctx=8192)


def test_window_ceiling_is_enforced():
    over = dataclasses.replace(RAVENX_27B, attention_window=300_000)
    with pytest.raises(RopeScalingRequired):
        build_vllm_serve_argv(over, ctx=MIN_CONTEXT_TOKENS)


def test_sub_floor_window_is_refused():
    narrow = dataclasses.replace(RAVENX_27B, attention_window=16_384)
    with pytest.raises(QualityFloorError):
        build_vllm_serve_argv(narrow, ctx=MIN_CONTEXT_TOKENS)


def test_loopback_only():
    with pytest.raises(ValueError, match="loopback"):
        argv(host="0.0.0.0")


def test_bad_kv_bits_rejected():
    with pytest.raises(ValueError, match="kv_bits"):
        argv(kv_bits=2)


def test_command_is_a_shell_line():
    line = vllm_serve_command(RAVENX_27B, MIN_CONTEXT_TOKENS, spec="mtp")
    assert line.startswith("vllm serve ") and "--pipeline-parallel-size 2" in line
