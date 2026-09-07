"""
The vLLM backend: build the `vllm serve` command line for the RavenX model on
the Ampere target box, with the engine's product floors baked in.

Why vLLM here. The Ampere box - **CMP 170HX (GA100, sm_80, HBM2e) + RTX 3060 Ti
(GA104, sm_86, GDDR6)** - is vLLM's home turf: FlashAttention-2, the Flash-
Linear-Attention Triton kernels for Gated-DeltaNet, and int4/Marlin all run on
sm_80/sm_86. So unlike the Turing 2080 Ti, this box runs the `qwen3_5` hybrid on
stock upstream vLLM. TENSELERATE keeps its identity - the single-model lock, the
floors, the `plan` advisor - and drives vLLM underneath.

This module never imports vLLM. It only *builds the argv*, so it is fully unit-
testable with no GPU and no vLLM install. `tenselerate serve --backend vllm`
prints or execs the result.

Honest mapping (what the floors do and do not translate to as vLLM flags):
  * context floor  -> `--max-model-len` (>= MIN_CONTEXT_TOKENS). vLLM's Qwen3-
    Next carries long range in the GDN state exactly as our reference does.
  * no-RoPE window -> NOT a vLLM flag. The window is intrinsic to the model
    config; vLLM manages the hybrid KV itself. We still *validate* the window
    against the quality floor and ceiling so a bad request is refused here,
    before vLLM ever starts.
  * two heterogeneous GPUs, no NVLink -> **pipeline** parallelism (PP=2), never
    tensor parallelism: TP wants ~equal GPUs on a fast link, which this box is
    not. The bigger, faster CMP 170HX holds most layers.
  * `--kv-bits 4` -> vLLM has no int4 KV cache; its concurrency lever is fp8 KV
    (`--kv-cache-dtype fp8`). We map 4 -> fp8, 8 -> auto, and say so, rather
    than pretend the KIVI int4 path exists in vLLM today.
  * `--spec mtp` -> vLLM's built-in Qwen3-Next MTP speculative decode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from tenselerate.config import ModelConfig, validate_window

# The one model, as published (GGUF Q4_K_M). vLLM loads it with --quantization
# gguf; on Ampere the int4 weights run through Marlin.
RAVENX_MODEL_REF = (
    "deadbydawn101/RavenXAiLabs-Chaos-Agent-Qwen3.8-27B-"
    "Frontier-Intelligence-Injected-OBLITERATED-GGUF"
)
RAVENX_GGUF_FILE = "Q4_K_M"
RAVENX_SERVED_NAME = f"{RAVENX_MODEL_REF}:{RAVENX_GGUF_FILE}"

# vLLM KV cache dtype for each --kv-bits value. vLLM has no int4 KV; fp8 is its
# footprint/concurrency lever, and "auto" keeps the compute dtype (fp16 here).
KV_CACHE_DTYPE = {8: "auto", 4: "fp8"}


@dataclass(frozen=True)
class GpuTarget:
    """One GPU in the target box."""
    name: str
    sm: str            # compute capability, e.g. "80"
    vram_gib: float


# The Ampere target: CMP 170HX (GA100) + RTX 3060 Ti (GA104). The CMP's VRAM is
# its *unlocked* figure - stock 8 GiB cannot hold the 15.4 GiB weights, so the
# box only serves 27B with the memory unlock applied (see docs/svmi).
CMP170HX = GpuTarget(name="cmp170hx", sm="80", vram_gib=40.0)
RTX3060TI = GpuTarget(name="rtx3060ti", sm="86", vram_gib=8.0)
AMPERE_BOX = (CMP170HX, RTX3060TI)


def build_vllm_serve_argv(
    cfg: ModelConfig,
    *,
    ctx: int,
    gpus: tuple[GpuTarget, ...] = AMPERE_BOX,
    host: str = "127.0.0.1",
    port: int = 8000,
    kv_bits: int = 8,
    spec: str = "none",
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 8,
) -> list[str]:
    """
    Build the argv for `vllm serve` honoring the engine's floors. Raises the
    same ContextFloorError / QualityFloorError / RopeScalingRequired the rest of
    the engine does, so an out-of-floor request is refused before vLLM starts.
    """
    # Floors first - reuse the engine's own validators so the vLLM path can
    # never serve something the native path would refuse.
    cfg.validate_context(ctx)
    if cfg.attention_window is not None:
        validate_window(cfg.attention_window, cfg.max_attention_window)
    if kv_bits not in KV_CACHE_DTYPE:
        raise ValueError(f"kv_bits must be one of {sorted(KV_CACHE_DTYPE)}")
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("TENSELERATE is loopback-only; refuse off-host bind")

    argv = [
        "vllm", "serve", RAVENX_MODEL_REF,
        "--quantization", "gguf",
        "--served-model-name", RAVENX_SERVED_NAME,
        "--host", host,
        "--port", str(port),
        # heterogeneous GPUs, no NVLink -> pipeline, one stage per card
        "--pipeline-parallel-size", str(len(gpus)),
        "--distributed-executor-backend", "mp",
        "--max-model-len", str(ctx),
        "--kv-cache-dtype", KV_CACHE_DTYPE[kv_bits],
        "--gpu-memory-utilization", f"{gpu_memory_utilization:.2f}",
        "--max-num-seqs", str(max_num_seqs),
        "--enable-prefix-caching",
    ]
    if spec == "mtp":
        argv += ["--speculative-config", json.dumps(
            {"method": "qwen3_next_mtp", "num_speculative_tokens": 2})]
    return argv


def vllm_serve_command(cfg: ModelConfig, ctx: int, **kw) -> str:
    """The argv joined into a copy-pasteable shell line."""
    return " ".join(build_vllm_serve_argv(cfg, ctx=ctx, **kw))
