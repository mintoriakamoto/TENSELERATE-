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
  * MTP only at depth 1: the merge left the head's position-1 prediction
    intact and killed positions 2+. Measured on code decode: n-max 1 = +35%
    (46.6 vs 34.4 tok/s), n-max 2 = +15%, n-max 3 = -15%, n-max 5 = -22%.
    `mtp_draft=1` emits `--spec-type draft-mtp --spec-draft-n-max 1`; deeper
    drafts lose until the head is retrained (docs/mtp-realign-davidau.md)
  * greedy sampling by default (`--temp 0 --repeat-penalty 1.0`): llama.cpp
    accepts a draft token only if the target's *sampled* token equals it
    (common/sampling.cpp, common_sampler_sample_and_accept_n). Measured on the
    production server: greedy 46.2 tok/s at 88% acceptance; the model card's
    temp 0.7 + repeat-penalty 1.15 gives 29.9 tok/s at 22% - slower than no
    MTP at all. These are server defaults; a request that carries its own
    temperature/repeat_penalty overrides them, so the client must not send them
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
from pathlib import Path

from tenselerate.config import MIN_ATTENTION_WINDOW

KV_TYPES = ("q8_0", "q4_0", "f16")
REASONING_LEVELS = ("low", "medium", "high", "xhigh")
LOOPBACK = ("127.0.0.1", "localhost", "::1")

DEFAULT_SLOTS = 4            # main loop + 3 subagents
DEFAULT_CTX_POOL = 524_288   # two full 262K windows' worth, shared
DEFAULT_KV = "q8_0"
DEFAULT_REASONING = "low"
DEFAULT_ALIAS = "tenselerate"
DEFAULT_MTP_DRAFT = None     # None = 1 on an -MTP- GGUF, else 0; 1 = measured +13..38%
SAMPLING_MODES = ("greedy", "client")   # greedy: server defaults temp 0 / rp 1.0; client: leave llama.cpp's
DEFAULT_SAMPLING = "greedy"  # measured: the draft head only pays under greedy (46.2 vs 29.9 tok/s)
MAX_MTP_DRAFT = 8
MMVQ_MAX_BATCH = 8           # ggml's MMVQ_MAX_BATCH_SIZE; GGML_CUDA_MMVQ_MAX clamps to it


def resolve_mtp_draft(model: str, mtp_draft: int | None) -> int:
    """None -> 1 when the GGUF file name carries the MTP head ("-MTP-"), else 0."""
    if mtp_draft is None:
        return 1 if "mtp" in Path(model).name.lower() else 0
    return mtp_draft


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
    mtp_draft: int | None = DEFAULT_MTP_DRAFT,
    sampling: str = DEFAULT_SAMPLING,
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
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of {SAMPLING_MODES}, got {sampling!r}")
    mtp_draft = resolve_mtp_draft(model, mtp_draft)
    if not 0 <= mtp_draft <= MAX_MTP_DRAFT:
        raise ValueError(f"mtp_draft must be 0 (off) .. {MAX_MTP_DRAFT}, got {mtp_draft}")
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
    if sampling == "greedy":
        # request-level defaults; the acceptance test is exact match against the
        # sampled token, so temperature and repeat penalty fight the draft head
        argv += ["--temp", "0", "--repeat-penalty", "1.0"]
    if mtp_draft:
        # the MTP head lives inside the -MTP- GGUF; no -md needed
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(mtp_draft)]
    argv += list(extra)
    return argv


def env_prefix(*, no_mmvq: bool = False, mmvq_max: int | None = None) -> dict[str, str]:
    """
    The matmul-routing environment. `no_mmvq` sends every width to the tensor-core
    MMQ path (GGML_CUDA_NO_MMVQ=1). `mmvq_max=N` keeps batches up to N on the dp4a
    vector path and routes wider ones - speculative verification, multi-slot steps -
    to MMQ (GGML_CUDA_MMVQ_MAX, this fork). N=1 is the measured sweet spot to test:
    single-token decode stays where it is, verification of drafts moves to MMQ.
    """
    if mmvq_max is not None and not 0 <= mmvq_max <= MMVQ_MAX_BATCH:
        raise ValueError(f"mmvq_max must be 0..{MMVQ_MAX_BATCH}, got {mmvq_max}")
    out: dict[str, str] = {}
    if no_mmvq:
        out["GGML_CUDA_NO_MMVQ"] = "1"
    elif mmvq_max is not None:
        out["GGML_CUDA_MMVQ_MAX"] = str(mmvq_max)
    return out


def llama_server_env(
    *, no_mmvq: bool = False, mmvq_max: int | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment for the launch: the caller's, plus the matmul-routing flags."""
    env = dict(os.environ if base is None else base)
    env.update(env_prefix(no_mmvq=no_mmvq, mmvq_max=mmvq_max))
    return env


def llama_server_command(model: str, *, no_mmvq: bool = False, mmvq_max: int | None = None,
                         **kw) -> str:
    """The launch as one copy-pasteable shell line, env prefix included."""
    prefix = "".join(f"{k}={v} " for k, v in env_prefix(no_mmvq=no_mmvq, mmvq_max=mmvq_max).items())
    return prefix + " ".join(build_llama_server_argv(model, **kw))
