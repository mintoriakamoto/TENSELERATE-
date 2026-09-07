"""
The llama.cpp backend: build the `llama-server` command line that powers
Hercules on the CMP 170HX + RTX 3060 box, shaped by what that box measured
(benches/cmp170hx-3060/README.md) rather than by a model of it.

This module never launches anything. It builds the argv and the environment,
so it is unit-testable with no GPU and no build; `tenselerate serve --backend
llamacpp` prints or execs the result, and scripts/hercules_serve.sh is a thin
wrapper over the same code.

What the measurements decided, and where each one lands in the argv:
  * one operator = one sequential main loop + up to 3 parallel subagents
    (Hermes' default delegation.max_concurrent_children)  -> `-np 4`
  * `--kv-unified`: one shared KV pool, so the main session can hold the full
    262K window while subagents stay small; per-step cost follows live tokens
  * no speculation: MTP measured 7-11% acceptance on the served merge and
    decoded SLOWER than plain (14.9 vs 33.5 tok/s)  -> no `--spec-type`, ever
  * `reasoning_effort=low`: Qwen3.8's chat template defaults to xhigh; fewer
    thinking tokens outranks any decode lever on an agent loop
  * prefill is 855 tok/s, so a prompt-cache miss on a 60K agent context costs
    ~70 s; `-cb --cache-reuse 256` keep slots sticky and prefixes reusable
  * GGML_CUDA_NO_MMVQ=1 forces the tensor-core MMQ path at every batch width;
    the width sweep says that path is cheaper per sequence on this unit. It is
    an environment variable, exposed as `no_mmvq`, off until the N=1/2/4 runs
    confirm it.

What Hermes needs from the server (docs/integrations/providers.md upstream):
  * `--jinja`: without it llama-server ignores the `tools` parameter entirely,
    so every Hermes tool call silently degrades to text - and the
    `--chat-template-kwargs` reasoning_effort lever is a template kwarg, so it
    needs jinja too. Mandatory.
  * `--reasoning-format deepseek`: thinking comes back as
    `message.reasoning_content`, which Hermes keeps in `assistant_msg["reasoning"]`
    (set `reasoning_content: true` on the Hermes side).
  * `--alias`: a stable model id for Hermes' `model.default` instead of the
    GGUF file name.
  * `--no-context-shift`: an oversized request must fail so Hermes compacts and
    retries; a silent context shift corrupts an agent conversation.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

from tenselerate.config import MIN_ATTENTION_WINDOW

KV_TYPES = ("q8_0", "q4_0", "f16")
REASONING_LEVELS = ("low", "medium", "high", "xhigh")
LOOPBACK = ("127.0.0.1", "localhost", "::1")

DEFAULT_SLOTS = 4            # main loop + 3 subagents
DEFAULT_CTX_POOL = 524_288   # two full 262K windows' worth, shared
DEFAULT_KV = "q8_0"
DEFAULT_REASONING = "low"
DEFAULT_ALIAS = "tenselerate"


def build_llama_server_argv(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    binary: str = "llama-server",
    slots: int = DEFAULT_SLOTS,
    ctx_pool: int = DEFAULT_CTX_POOL,
    kv: str = DEFAULT_KV,
    reasoning: str = DEFAULT_REASONING,
    alias: str = DEFAULT_ALIAS,
    extra: Sequence[str] = (),
) -> list[str]:
    """
    The measured serve configuration as an argv. Refuses off-host binds, KV
    types llama.cpp does not have, a pool too small to hold one locked window,
    and a slot count below one - all before anything is launched.
    """
    if host not in LOOPBACK:
        raise ValueError("TENSELERATE is loopback-only; refuse off-host bind")
    if kv not in KV_TYPES:
        raise ValueError(f"kv must be one of {KV_TYPES}, got {kv!r}")
    if reasoning not in REASONING_LEVELS:
        raise ValueError(f"reasoning must be one of {REASONING_LEVELS}, got {reasoning!r}")
    if slots < 1:
        raise ValueError("slots must be >= 1")
    if ctx_pool < MIN_ATTENTION_WINDOW:
        raise ValueError(
            f"ctx_pool {ctx_pool:,} cannot hold one locked {MIN_ATTENTION_WINDOW:,}-token "
            "window; the pool is shared by all slots but the main session must fit")
    if not alias:
        raise ValueError("alias must be a non-empty model id for the agent to address")
    argv = [
        binary, "-m", model, "--alias", alias,
        "--host", host, "--port", str(port),
        "--jinja", "--reasoning-format", "deepseek", "--no-context-shift",
        "-ngl", "999", "--main-gpu", "0", "-fa", "on",
        "-c", str(ctx_pool), "-np", str(slots), "--kv-unified", "-cb",
        "-ctk", kv, "-ctv", kv,
        "-b", "2048", "-ub", "512", "--cache-reuse", "256",
        "--chat-template-kwargs", json.dumps({"reasoning_effort": reasoning}),
    ]
    argv += list(extra)
    return argv


def llama_server_env(
    *, no_mmvq: bool = False, base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment for the launch: the caller's, plus the MMQ-forcing flag."""
    env = dict(os.environ if base is None else base)
    if no_mmvq:
        env["GGML_CUDA_NO_MMVQ"] = "1"
    return env


def llama_server_command(model: str, *, no_mmvq: bool = False, **kw) -> str:
    """The launch as one copy-pasteable shell line, env prefix included."""
    prefix = "GGML_CUDA_NO_MMVQ=1 " if no_mmvq else ""
    return prefix + " ".join(build_llama_server_argv(model, **kw))
