"""
Runtime backends for the TENSELERATE engine.

TENSELERATE's identity is the layer *above* the runtime: the single-model lock,
the product floors (1M context, the 32K-262K no-RoPE window, the speed floor),
and the `plan` advisor. That layer is backend-agnostic. Two backends exist:

  * `reference` - the from-scratch NumPy/CUDA engine in `tenselerate.engine`,
    the correctness oracle every kernel is checked against. Runs anywhere.
  * `vllm`      - drives an upstream vLLM OpenAI server as the compute runtime.
    vLLM natively runs the `qwen3_5` Gated-DeltaNet hybrid with real Flash-
    Linear-Attention kernels, so on the Ampere target box (CMP 170HX + RTX
    3060 12 GiB, sm_80 + sm_86) it is the production path. This module does not
    import vLLM; it builds the `vllm serve` command line from the engine's
    config and floors so the launcher is unit-testable with no GPU present.
"""
from __future__ import annotations

from tenselerate.backends.vllm import build_vllm_serve_argv

__all__ = ["build_vllm_serve_argv"]
