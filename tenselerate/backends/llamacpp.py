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
  * `--reasoning-effort low`: Qwen3.8's chat template defaults to xhigh; fewer
    thinking tokens outranks any decode lever on an agent loop. Since the
    upstream sync this is a first-class server flag (it sets the template's
    default kwarg), and a request's own `reasoning_effort` field overrides it
    per turn - `none` disables thinking for that request entirely
  * prefill is 855 tok/s, so a prompt-cache miss on a 60K agent context costs
    ~70 s; `-cb --cache-reuse 256` keep slots sticky and prefixes reusable
  * the ~35K-token Hermes system prompt is shared by the main loop and every
    delegation child. Three server features keep it prefilled once:
    `--cache-ram N` (host-RAM prompt cache; idle slots are saved into it and,
    with `--kv-unified`, cleared), `--cache-idle-slots` (explicit), and
    `--slot-prompt-similarity` (a request lands on the idle slot whose cached
    prompt shares the longest prefix; below the threshold it takes the LRU slot
    and loads the best prefix from the RAM cache). A child that arrives while
    the parent's slot is busy therefore gets the 35K prefix from RAM instead
    of re-prefilling it (~40 s). `--slot-save-path DIR` adds the
    /slots/<id>?action=save|restore endpoints so the prefix survives a server
    restart (scripts/hercules_slots.sh).
  * GGML_CUDA_NO_MMVQ=1 forces the tensor-core MMQ path at every batch width;
    the width sweep says that path is cheaper per sequence on this unit. It is
    an environment variable, exposed as `no_mmvq`, off until the N=1/2/4 runs
    confirm it.

What Hermes needs from the server (docs/integrations/providers.md upstream):
  * `--jinja`: without it llama-server ignores the `tools` parameter entirely,
    so every Hermes tool call silently degrades to text - and reasoning_effort
    is a template kwarg, so it needs jinja too. Mandatory.
  * `--reasoning-format deepseek`: thinking comes back as
    `message.reasoning_content`, which Hermes keeps in `assistant_msg["reasoning"]`
    (set `reasoning_content: true` on the Hermes side).
  * `--alias`: a stable model id for Hermes' `model.default` instead of the
    GGUF file name.
  * `--no-context-shift`: an oversized request must fail so Hermes compacts and
    retries; a silent context shift corrupts an agent conversation.
"""
from __future__ import annotations

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
DEFAULT_CACHE_RAM_MIB = 16384  # host-RAM prompt cache: ~7 x the 35K system prefix at q8_0 (2.3 GiB each)
DEFAULT_SLOT_SIMILARITY = 0.1  # llama.cpp's default; the LCP fraction a slot must share to be chosen
# Server-default sampling. Draft acceptance is exact match against the sampled
# token, so anything that moves the argmax costs MTP; the merge loops under pure
# greedy in <think>, so the loop guard must leave the argmax alone as much as it can.
#   greedy : temp 0, no repeat penalty              46.2 tok/s, 0.88 acceptance; loops
#   dry    : greedy + DRY (fires only on n-grams that extend a repeat; scans 2048 tokens,
#            not the 262K window - dry-penalty-last-n -1 would scan the whole context per token)
#   low    : temp 0.3, min-p 0.1, no repeat penalty (the box's 0.15/0.3/0.5 sweet-spot sweep)
#   client : leave llama.cpp's defaults
SAMPLING_ARGV: dict[str, tuple[str, ...]] = {
    "greedy": ("--temp", "0", "--repeat-penalty", "1.0"),
    "dry": ("--temp", "0", "--repeat-penalty", "1.0",
            "--dry-multiplier", "0.8", "--dry-base", "1.75",
            "--dry-allowed-length", "2", "--dry-penalty-last-n", "2048"),
    "low": ("--temp", "0.3", "--min-p", "0.1", "--repeat-penalty", "1.0"),
    "client": (),
}
SAMPLING_MODES = tuple(SAMPLING_ARGV)
DEFAULT_SAMPLING = "greedy"  # measured: the draft head only pays under greedy (46.2 vs 29.9 tok/s)
MAX_MTP_DRAFT = 8
MMVQ_MAX_BATCH = 32          # ggml's MMVQ_MAX_BATCH_SIZE; GGML_CUDA_MMVQ_MAX clamps to it

# n-gram drafting (`ngram-mod`), off by default.
#
# Why this is not the same bet as a deeper MTP draft. The depth sweep measured
# the head as shallow, not broken: position 1 accepts at 0.88, positions 2+ do
# not, so n-max 1 is +35% and n-max 5 is -22%. Raising the MTP depth buys
# columns that are very unlikely to be accepted, and a rejected draft column is
# paid in full.
#
# `ngram-mod` drafts from a different source entirely: it hashes the last
# `n_match` tokens of the live context and replays whatever followed that same
# run earlier in the same context. It runs no model, so a miss costs a hash
# lookup rather than a forward pass. On an agent loop - a file re-emitted with
# one line changed, a diff quoted back, a tool result repeated - a hit is often
# a long verbatim run, which is where the width sweep's flat MMQ region (55 ms
# from N=2 to N=16) turns into free tokens.
#
# Ordering is load-bearing. common_speculative tries the implementations in the
# order given and takes the FIRST one that returns a non-empty draft for that
# sequence; drafts are never concatenated (common/speculative.cpp, the
# `for (auto & impl : spec->impls)` loop that sets `impl_last`). So `ngram-mod`
# must come before `draft-mtp`: the free drafter gets first refusal, and every
# sequence it declines falls through to the measured depth-1 MTP path
# unchanged. Reversing the order would silence ngram-mod entirely, because the
# MTP head always produces a draft.
DEFAULT_NGRAM_MATCH = 24     # llama.cpp's default context-suffix hash length
MAX_NGRAM_DRAFT = 15         # keeps 1 + n_max inside the flat MMQ region (N<=16)


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
    mtp_model: str | None = None,
    ngram_draft: int | None = None,
    ngram_min: int | None = None,
    ngram_match: int = DEFAULT_NGRAM_MATCH,
    sampling: str = DEFAULT_SAMPLING,
    cache_ram_mib: int = DEFAULT_CACHE_RAM_MIB,
    cache_idle_slots: bool = True,
    slot_similarity: float = DEFAULT_SLOT_SIMILARITY,
    slot_save_path: str | None = None,
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
    if cache_ram_mib < -1:
        raise ValueError("cache_ram_mib must be -1 (unlimited), 0 (off) or a MiB count")
    if not 0.0 <= slot_similarity <= 1.0:
        raise ValueError("slot_similarity must be within 0.0 (off) .. 1.0")
    if cache_idle_slots and cache_ram_mib == 0:
        raise ValueError("cache_idle_slots needs a prompt cache: cache_ram_mib must not be 0")
    if mtp_model is not None and mtp_draft is None:
        mtp_draft = 3          # a retrained head is served at the healthy-curve optimum
    mtp_draft = resolve_mtp_draft(model, mtp_draft)
    if mtp_model is not None and mtp_draft == 0:
        raise ValueError("mtp_model given but mtp_draft is 0; a sidecar head needs a draft depth")
    if not 0 <= mtp_draft <= MAX_MTP_DRAFT:
        raise ValueError(f"mtp_draft must be 0 (off) .. {MAX_MTP_DRAFT}, got {mtp_draft}")
    if ngram_draft is not None:
        if not 1 <= ngram_draft <= MAX_NGRAM_DRAFT:
            raise ValueError(
                f"ngram_draft must be 1 .. {MAX_NGRAM_DRAFT} (or None for off), got "
                f"{ngram_draft}; past {MAX_NGRAM_DRAFT} the verify batch leaves the "
                "flat MMQ region and every extra column is paid in full")
        # llama.cpp's own default is n_min 48 against n_max 64. Carried into a
        # short draft it is a trap: ngram-mod discards the WHOLE draft when the
        # replayed run ends before n_min (common/speculative.cpp clears the
        # result and returns), so n_min > n_max can never draft. It fails
        # silently - every request falls through to the MTP path and the only
        # symptom is that the speedup never arrives - so default it and refuse
        # the impossible combination rather than emit it.
        if ngram_min is None:
            ngram_min = min(4, ngram_draft)
        if not 1 <= ngram_min <= ngram_draft:
            raise ValueError(
                f"ngram_min must be 1 .. ngram_draft ({ngram_draft}), got {ngram_min}; "
                "a minimum above the maximum discards every draft silently")
        if ngram_match < 1:
            raise ValueError("ngram_match must be a positive token count")
    elif ngram_min is not None:
        raise ValueError("ngram_min given but ngram_draft is None (n-gram drafting is off)")
    argv = [
        binary, "-m", model, "--alias", alias,
        "--host", host, "--port", str(port),
        "--jinja", "--reasoning-format", "deepseek", "--no-context-shift",
        "-ngl", "999", "--main-gpu", "0", "-fa", "on",
        "-c", str(ctx_pool), "-np", str(slots), "--kv-unified", "-cb",
        "-ctk", kv, "-ctv", kv,
        "-b", "2048", "-ub", "512", "--cache-reuse", "256",
        "--cache-ram", str(cache_ram_mib),
        "--cache-idle-slots" if cache_idle_slots else "--no-cache-idle-slots",
        "--slot-prompt-similarity", f"{slot_similarity:g}",
        "--reasoning-effort", reasoning,
    ]
    if slot_save_path is not None:
        if not slot_save_path:
            raise ValueError("slot_save_path must be a directory path")
        argv += ["--slot-save-path", slot_save_path]
    # request-level defaults; the acceptance test is exact match against the
    # sampled token, so temperature and repeat penalty fight the draft head
    argv += list(SAMPLING_ARGV[sampling])
    # ngram-mod first: it is tried first and only falls through to the MTP head
    # when it has no run to replay, so the free drafter gets first refusal and
    # the measured depth-1 path is untouched on a miss. See MAX_NGRAM_DRAFT.
    spec_types = (["ngram-mod"] if ngram_draft is not None else []) + \
                 (["draft-mtp"] if mtp_draft else [])
    if spec_types:
        argv += ["--spec-type", ",".join(spec_types)]
    if ngram_draft is not None:
        argv += ["--spec-ngram-mod-n-max", str(ngram_draft),
                 "--spec-ngram-mod-n-min", str(ngram_min),
                 "--spec-ngram-mod-n-match", str(ngram_match)]
    if mtp_draft:
        # the MTP head lives inside the -MTP- GGUF; -md only for a retrained
        # sidecar head exported by convert_hf_to_gguf.py --mtp (scripts/mtp-head-train.py)
        argv += ["--spec-draft-n-max", str(mtp_draft)]
        if mtp_model is not None:
            argv += ["-md", mtp_model]
    argv += list(extra)
    return argv


def env_prefix(*, no_mmvq: bool = False, mmvq_max: int | None = None,
               device: int | None = None) -> dict[str, str]:
    """
    The launch environment. `device=N` pins the server to one CUDA device
    (CUDA_VISIBLE_DEVICES=N, so `--main-gpu 0` inside the process is that card):
    the RTX 3060 side server for Hermes delegation children and the compaction
    summarizer, which otherwise occupy 170HX slots at the 27B's per-token cost.

    The matmul-routing environment. `no_mmvq` sends every width to the tensor-core
    MMQ path (GGML_CUDA_NO_MMVQ=1). `mmvq_max=N` keeps batches up to N on the dp4a
    vector path and routes wider ones - speculative verification, multi-slot steps -
    to MMQ (GGML_CUDA_MMVQ_MAX, this fork). N=1 is the measured sweet spot to test:
    single-token decode stays where it is, verification of drafts moves to MMQ.
    """
    if mmvq_max is not None and not 0 <= mmvq_max <= MMVQ_MAX_BATCH:
        raise ValueError(f"mmvq_max must be 0..{MMVQ_MAX_BATCH}, got {mmvq_max}")
    if device is not None and device < 0:
        raise ValueError(f"device must be a CUDA device index >= 0, got {device}")
    out: dict[str, str] = {}
    if device is not None:
        out["CUDA_VISIBLE_DEVICES"] = str(device)
    if no_mmvq:
        out["GGML_CUDA_NO_MMVQ"] = "1"
    elif mmvq_max is not None:
        out["GGML_CUDA_MMVQ_MAX"] = str(mmvq_max)
    return out


def llama_server_env(
    *, no_mmvq: bool = False, mmvq_max: int | None = None, device: int | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment for the launch: the caller's, plus device pin and routing flags."""
    env = dict(os.environ if base is None else base)
    env.update(env_prefix(no_mmvq=no_mmvq, mmvq_max=mmvq_max, device=device))
    return env


def llama_server_command(model: str, *, no_mmvq: bool = False, mmvq_max: int | None = None,
                         device: int | None = None, **kw) -> str:
    """The launch as one copy-pasteable shell line, env prefix included."""
    prefix = "".join(f"{k}={v} " for k, v in env_prefix(
        no_mmvq=no_mmvq, mmvq_max=mmvq_max, device=device).items())
    return prefix + " ".join(build_llama_server_argv(model, **kw))
