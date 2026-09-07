"""
The `tenselerate` command line.

One entry point for the whole lifecycle of the engine:

    tenselerate install          build from this clone, then check the machine
    tenselerate build            compile the engine end to end (kernels + server)
    tenselerate boot             doctor, then serve — one-command bring-up
    tenselerate serve            run the OpenAI /v1 endpoint
    tenselerate update           check for and apply a new build
    tenselerate info             model geometry, the context floor, KV sizing
    tenselerate plan             what this machine can do at a given context
    tenselerate doctor           hardware/driver check before anything else

Run as `python -m tenselerate <cmd>` (or `tenselerate <cmd>` once installed).

From a bare machine, one line clones and builds:
    curl -fsSL https://raw.githubusercontent.com/mintoriakamoto/TENSELERATE-/main/scripts/tenselerate-build.sh | bash
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import dataclasses

from tenselerate.config import (
    CONFIGS, KV_BITS_PER_ELEM, MAX_ATTENTION_WINDOW, MIN_ATTENTION_WINDOW,
    MIN_CONTEXT_TOKENS, MIN_DECODE_TOKS, MTP_SPECULATIVE_SPEEDUP, QWEN38_27B,
    TINY, ContextFloorError, QualityFloorError, RopeScalingRequired,
    validate_window,
)
from tenselerate.engine.scheduler import Scheduler

GiB = 1024 ** 3
REPO_ROOT = Path(__file__).resolve().parent.parent


def _out(msg: str = "") -> None:
    """Print a message to stdout with a newline (formatted CLI output)."""
    sys.stdout.write(msg + "\n")


# --------------------------------------------------------------------------
# build / install
# --------------------------------------------------------------------------
def _run_build(mode_args: list[str]) -> int:
    """Run the build script with the given flags.

    Args:
        mode_args: List of flags to pass to tenselerate-build.sh
                   (e.g., ['--cpu'], ['--cuda'], ['--kernels']).

    Returns:
        Return code from the build script (0 = success).
    """
    script = REPO_ROOT / "scripts" / "tenselerate-build.sh"
    if not script.is_file():
        _out(f"error: build script not found at {script}")
        _out("Run from a clone, or bootstrap a bare machine with:")
        _out("  curl -fsSL https://raw.githubusercontent.com/mintoriakamoto/"
             "TENSELERATE-/main/scripts/tenselerate-build.sh | bash")
        return 1
    cmd = ["bash", str(script), *mode_args]
    _out(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def cmd_build(args: argparse.Namespace) -> int:
    """Compile the engine end to end: int8 kernels + reference server.

    Can build in three modes:
    - CPU-only (reference numerics, no GPU, no CUDA)
    - CUDA (requires nvcc to compile for sm_80/sm_86)
    - Kernels only (just the int8 GEMM, for testing)

    Args:
        args: Namespace with optional flags: cpu, cuda, kernels.

    Returns:
        Return code from tenselerate-build.sh (0 = success).
    """
    mode_args = []
    if args.cpu:
        mode_args.append("--cpu")
    elif args.cuda:
        mode_args.append("--cuda")
    elif args.kernels:
        mode_args.append("--kernels")
    return _run_build(mode_args)


def cmd_install(args: argparse.Namespace) -> int:
    """Build from this clone, then run the hardware check — zero to ready.

    A one-command bootstrap: compiles the engine, then runs the doctor to verify
    the machine is ready. If the build fails, doctor is not run. If doctor fails,
    a diagnostic report is shown but the install is considered complete (the
    reference backend runs without a GPU).

    Args:
        args: Namespace with optional flags: cpu (force CPU-only build).

    Returns:
        Return code (0 = success, non-zero = error from build or doctor).
    """
    _out("TENSELERATE install: build, then verify the machine")
    _out("")
    rc = _run_build([] if not args.cpu else ["--cpu"])
    if rc != 0:
        _out("")
        _out("build failed; fix the errors above, then re-run `tenselerate install`")
        return rc
    _out("")
    _out("build ok - running doctor")
    _out("")
    return cmd_doctor(args)


# --------------------------------------------------------------------------
# update
# --------------------------------------------------------------------------
def cmd_update(args: argparse.Namespace) -> int:
    """Check for and apply updates from published releases.

    Drives tenselerate-update.sh, which compares the running build against the
    newest published release and either:
    - Fast-forwards the clone + rebuilds locally (--source)
    - Downloads and installs a prebuilt binary (--binary)
    - Lists available releases (--list)
    - Checks for new versions and reports availability (--check, default)

    Args:
        args: Namespace with optional flags: source, binary, list, check.

    Returns:
        Return code from the updater script. 10 = update is available (in --check mode).
    """
    script = REPO_ROOT / "scripts" / "tenselerate-update.sh"
    if not script.is_file():
        _out(f"error: updater not found at {script}")
        _out("This looks like an installed copy without the repo. Re-run from a")
        _out("clone, or fetch the script directly:")
        _out("  curl -fsSL https://raw.githubusercontent.com/mintoriakamoto/"
             "TENSELERATE-/main/scripts/tenselerate-update.sh | bash -s -- --check")
        return 1

    mode = "--check"
    if args.source:
        mode = "--source"
    elif args.binary:
        mode = "--binary"
    elif args.list:
        mode = "--list"

    cmd = ["bash", str(script), mode]
    _out(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    # the script uses exit 10 to mean "an update is available" for --check
    if mode == "--check" and proc.returncode == 10:
        _out("")
        _out("An update is available. Apply it with:")
        _out("  tenselerate update --source     # fast-forward and rebuild")
        _out("  tenselerate update --binary     # download the prebuilt release")
    return proc.returncode


# --------------------------------------------------------------------------
# info
# --------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> int:
    """Show model geometry, context floor, KV sizing, and context-vs-RoPE tradeoffs.

    Displays the model config in human-readable form, KV cache footprint at the
    locked attention window (which does not grow with context), and tables of
    throughput/rope-scaling requirements at different context depths.

    Args:
        args: Namespace with config (model config name).

    Returns:
        Always 0 (success).
    """
    cfg = CONFIGS[args.config]
    _out(f"model            : {cfg.name}")
    _out(f"layers           : {cfg.n_layer}  "
         f"({cfg.n_full_attention_layers} full attention, "
         f"{cfg.n_linear_layers} gated-delta-net linear)")
    _out(f"hidden / heads   : {cfg.hidden_size} / {cfg.n_head} "
         f"({cfg.n_head_kv} KV heads, head_dim {cfg.head_dim})")
    _out(f"trained rotary   : {cfg.max_position_embeddings:,} tokens")
    _out("")
    _out(f"CONTEXT FLOOR    : {MIN_CONTEXT_TOKENS:,} tokens (hard minimum)")
    _out(f"SPEED TARGET     : {MIN_DECODE_TOKS:,} tok/s aggregate (NOT a hard "
         "gate; the locked")
    _out("                   window keeps the box below it - speed takes what "
         "recall leaves)")
    _out(f"QUALITY FLOOR    : window LOCKED at {MIN_ATTENTION_WINDOW:,} tokens "
         "(the max no-RoPE recall),")
    _out("                   and no RoPE scaling, ever. The window never narrows "
         "for speed:")
    _out("                   verbatim recall is pinned at its deepest, and the "
         "box takes")
    _out("                   the throughput that leaves (below the speed target, "
         "by design).")
    _out(f"                   quality holds across the FULL {MIN_CONTEXT_TOKENS:,}"
         "+ context: the GDN")
    _out("                   layers carry long range, the windowed attention "
         "stays inside")
    _out("                   the trained range, and sinks anchor it.")
    win = cfg.attention_window
    _out(f"attention window : {win:,} tokens" if win else
         "attention window : unbounded (full attention)")
    _out(f"resident KV      : {cfg.resident_kv_tokens:,} tokens' worth")
    _out(f"KV per token     : {cfg.kv_bytes_per_token() / 1024:.1f} KiB (q8_0, "
         f"{cfg.n_full_attention_layers} caching layers only)")
    _out("")
    _out("KV is bounded by the window, so it does not grow with context:")
    for ctx in (MIN_CONTEXT_TOKENS, 4_000_000, 10_000_000):
        kv = cfg.kv_bytes_for_context(ctx) / GiB
        scaling = "YES" if cfg.needs_rope_scaling(ctx) else "no"
        _out(f"  ctx {ctx:>12,}  ->  KV {kv:6.2f} GiB   rope scaling: {scaling}")
    _out("")
    _out("The 48 linear layers hold long range in a fixed recurrent state with no")
    _out("positional encoding, so context is unbounded without YaRN or RoPE scaling.")
    return 0


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
# Machine hardware specifications: (VRAM GiB, theoretical memory bandwidth GB/s).
# The engine targets exactly ONE hardware configuration:
#
#   - 2x2080ti (reference dev box): Dual RTX 2080 Ti (Turing, sm_75) with
#     Ryzen 9 9950X, 32 GB DDR5. Pooled 22 GiB VRAM, 2x616 GB/s HBM bandwidth.
#     Fits the 1M context floor (21.16 of 22 GiB) but delivers ~152 tok/s aggregate,
#     BELOW the 400 tok/s product target. The engine serves regardless; `plan`
#     reports this honestly.
#
#   - cmp170hx+3060 (Ampere production box): CMP 170HX + RTX 3060 12 GiB.
#     CMP 170HX: GA100 (sm_80), HBM2e, ~1493 GB/s. Stock 8 GiB; unlocked to 40 GiB
#     (required for 27B weights). RTX 3060: GA106 (sm_86), GDDR6, ~360 GB/s.
#     Pooled 52 GiB, no NVLink. Pipeline parallel (PP=2): bandwidth is the sum under
#     a bandwidth-balanced 2-stage pipeline assumption (both stages' HBM reads
#     overlap under continuous batching).
MACHINE_HW = {
    "2x2080ti": (22.0, 1232.0),
    "cmp170hx+3060": (52.0, 1853.0),
}
# Memory bandwidth efficiency at sustained load. Modeling constant; the 170HX
# weight read measured ~0.60 of nominal on 2026-09-07 (benches/cmp170hx-3060/),
# so the number is close - the per-sequence costs the planner does NOT model
# (see that README) are what separate its output from the measured tok/s.
BW_EFFICIENCY = 0.65


def _accel_path(cfg, vram: float, weights: float, args, bw: float) -> None:
    """Show lossless speedup levers at the locked attention window.

    The window is LOCKED at the max-recall value (no RoPE scaling), so speed
    cannot be bought by narrowing it. The only remaining levers are lossless:
    - q4_0 KV cache (more concurrency in the same VRAM)
    - speculative decode: MTP (built-in, ~1.8x, identical output) or EAGLE-3
      (trained draft, higher acceptance, needs a trained head)

    This function models both levers at the fixed window and reports how close
    the throughput gets to MIN_DECODE_TOKS. Typically does not reach the target
    because quality (verbatim recall via the windowed attention + GDN long-range)
    is pinned at maximum.

    Args:
        cfg: Model config with attention_window and KV sizing.
        vram: Total VRAM available (GiB).
        weights: Weight footprint (GiB).
        args: Command-line args with ctx, overhead_gib, kv_bits, spec.
        bw: Memory bandwidth (GB/s).
    """
    kv_w = cfg.kv_bytes_for_context(args.ctx, KV_BITS_PER_ELEM[4]) / GiB
    mb = int((vram - weights - args.overhead_gib) // kv_w)
    if mb < 1:
        return
    full = (bw * BW_EFFICIENCY / ((weights + kv_w * mb) * 1.074) * mb
            * MTP_SPECULATIVE_SPEEDUP)
    reach = "REACHES" if full >= MIN_DECODE_TOKS else "still under"
    _out("")
    _out("lossless levers at the locked window (they never touch quality):")
    _out("  q4_0 KV   -> more concurrency in the same VRAM "
         "(4-bit is the validated KV")
    _out("               floor - KIVI/KVQuant; an A/B confirms the delta)")
    _out(f"  MTP spec  -> ~{MTP_SPECULATIVE_SPEEDUP:.1f}x, output identical to "
         "plain decode; EAGLE-3 higher")
    _out(f"  together at the {cfg.attention_window:,} window -> ~{full:,.0f} "
         f"tok/s  ({reach} the {MIN_DECODE_TOKS} target)")
    _out("  model it:  tenselerate plan --kv-bits 4 --spec mtp")


def cmd_plan(args: argparse.Namespace) -> int:
    """Plan what a machine can achieve: throughput, batch size, and speedup levers.

    Takes a machine profile (VRAM, bandwidth) and context depth and computes:
    - Whether the model fits in VRAM with KV cache
    - Single-sequence throughput (tokens/sec)
    - Maximum concurrent sequences at the locked window
    - Aggregate throughput with continuous batching
    - Whether lossless speedup levers (q4_0 KV, MTP speculative) reach the target

    Reports gaps honestly rather than lowering expectations to flatter the hardware.

    Args:
        args: Namespace with config, machine, ctx, attention_window, weights_gib,
              overhead_gib, kv_bits, spec.

    Returns:
        0 if the target is met or approach is clear, 1 if the model doesn't fit,
        3 if the target is not reached and speed targets differ from quality floors.
    """
    cfg = CONFIGS[args.config]
    ctx = args.ctx
    try:
        if args.attention_window is not None:
            cfg = dataclasses.replace(
                cfg, attention_window=validate_window(
                    args.attention_window, cfg.max_attention_window))
        cfg.validate_context(ctx)
    except (ContextFloorError, QualityFloorError, RopeScalingRequired) as e:
        _out(f"error: {e}")
        return 2

    vram, bw = MACHINE_HW[args.machine]
    weights = args.weights_gib
    # acceleration dials: KV precision and MTP self-speculation
    kv_bpe = KV_BITS_PER_ELEM[args.kv_bits]
    spec = MTP_SPECULATIVE_SPEEDUP if args.spec == "mtp" else 1.0
    kv = cfg.kv_bytes_for_context(ctx, kv_bpe) / GiB
    total = weights + kv + args.overhead_gib

    _out(f"machine          : {args.machine}  ({vram:.0f} GiB, ~{bw:.0f} GB/s)")
    _out(f"context          : {ctx:,} tokens  (floor {MIN_CONTEXT_TOKENS:,})")
    accel = (f"KV q{args.kv_bits}_0"
             + (f" + MTP spec (x{spec:.1f})" if spec > 1.0 else ""))
    _out(f"acceleration     : {accel}")
    _out(f"weights          : {weights:.2f} GiB")
    _out(f"KV (windowed)    : {kv:.2f} GiB   <- constant beyond the window")
    _out(f"total resident   : {total:.2f} GiB of {vram:.0f} GiB "
         f"({'FITS' if total <= vram else 'DOES NOT FIT'})")
    if total > vram:
        _out("")
        _out("Options: a narrower --attention-window, a smaller quant, or the")
        _out("other machine. Lowering context is not one - the floor is fixed.")
        return 1

    # decode roofline: weights + resident KV read per token, times the MTP
    # speculation multiplier (accepted draft tokens cost no extra weight read)
    per_token_gb = (weights + kv) * 1.074
    single = bw * BW_EFFICIENCY / per_token_gb * spec
    _out("")
    _out(f"decode (batch 1) : ~{single:,.0f} tok/s   at ANY context >= the window")

    # Each concurrent sequence carries its OWN windowed KV, so batch is capped by
    # memory, not just by bandwidth. Never report a batch that cannot be resident.
    free_for_kv = vram - weights - args.overhead_gib
    # Ask the real scheduler, so this table and the engine can never disagree.
    try:
        sched = Scheduler(cfg, kv_budget_gib=free_for_kv, kv_bpe=kv_bpe)
        max_batch = sched.max_concurrent
    except ValueError:
        max_batch = 0
    if max_batch < 1:
        _out("  (no room for even one sequence's KV - narrow the window)")
        return 1

    def agg(b: int) -> float:
        return bw * BW_EFFICIENCY / ((weights + kv * b) * 1.074) * b * spec

    _out(f"max concurrent   : {max_batch} sequences "
         f"({free_for_kv:.1f} GiB free / {kv:.2f} GiB KV each)")
    _out("aggregate with continuous batching (memory-feasible only):")
    shown = [b for b in (2, 4, 8, 16, 32, 64) if b <= max_batch]
    if max_batch not in shown:
        shown.append(max_batch)
    for b in shown:
        mark = f"  <- {MIN_DECODE_TOKS}+" if agg(b) >= MIN_DECODE_TOKS else ""
        _out(f"  batch {b:>3}      : ~{agg(b):,.0f} tok/s{mark}")

    # -- the speed target --------------------------------------------------
    # The window is LOCKED at max recall, so there is no narrowing to chase the
    # target - the box does what it does at this one window. Report the gap
    # honestly; being under the target is the deliberate quality-over-speed lock.
    best = agg(max_batch)
    _out("")
    if best >= MIN_DECODE_TOKS:
        _out(f"SPEED TARGET     : {MIN_DECODE_TOKS} tok/s - met at the locked "
             f"{cfg.attention_window:,} window")
        _out("")
        _out("Numbers are a bandwidth roofline at "
             f"{BW_EFFICIENCY:.0%} efficiency, not a measurement.")
        return 0
    _out(f"SPEED TARGET     : {MIN_DECODE_TOKS} tok/s - the box does ~"
         f"{best:,.0f} tok/s here, BELOW it.")
    _out(f"The window is locked at {cfg.attention_window:,} (max verbatim "
         "recall), so it never")
    _out("narrows for speed - quality is pinned at maximum and the box takes "
         "the throughput")
    _out("that leaves. This is the deliberate quality-over-speed trade, not a "
         "misconfiguration.")
    _accel_path(cfg, vram, weights, args, bw)
    _out("")
    _out("Numbers are a bandwidth roofline at "
         f"{BW_EFFICIENCY:.0%} efficiency, not a measurement.")
    return 3
    return 0


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    """Check hardware and driver readiness before serving.

    Verifies:
    - nvidia-smi is installed (NVIDIA userspace present)
    - A GPU driver is bound (CUDA is reachable)
    - The engine can be imported
    - Lists detected GPUs

    This is a prerequisite check before tenselerate serve or boot. The reference
    backend runs without a GPU, so a failing doctor does not block serve unless
    you're using the vLLM backend.

    Args:
        args: Namespace with optional flag: weights_gib (for full GPU check).

    Returns:
        0 if all checks pass, 1 if any check fails.
    """
    ok = True
    _out("TENSELERATE doctor")
    _out("")
    smi = shutil.which("nvidia-smi")
    if not smi:
        _out("  [!] nvidia-smi not found - no NVIDIA userspace installed")
        ok = False
    else:
        proc = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            _out("  [!] nvidia-smi present but cannot reach a driver.")
            _out("      The kernel module is not bound; nothing CUDA works yet.")
            _out("      On Blackwell (RTX 50-series) the OPEN module is required.")
            ok = False
        else:
            for line in proc.stdout.strip().splitlines():
                _out(f"  [ok] GPU: {line.strip()}")
    gpucheck = REPO_ROOT / "scripts" / "svmi-gpucheck.py"
    if gpucheck.is_file():
        _out("")
        _out(f"  full report: python3 {gpucheck.relative_to(REPO_ROOT)} "
             f"--model-gib {args.weights_gib}")
    _out("")
    _out("  [ok] engine importable; context floor "
         f"{MIN_CONTEXT_TOKENS:,} tokens")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------
def _serve_vllm(args: argparse.Namespace) -> int:
    """Launch vLLM as the compute runtime (Ampere target box: CMP 170HX + RTX 3060).

    Builds the vllm serve argv with the Qwen3.8-27B config, validates context and
    KV floors, then either prints the command (--dry-run) or execs vllm if found
    on PATH.

    Args:
        args: Namespace with ctx, host, port, kv_bits, spec, eagle_model, dry_run.

    Returns:
        0 on success, 1 if vllm not found (and not --dry-run), 2 on validation error.
    """
    from tenselerate.backends.vllm import build_vllm_serve_argv
    try:
        argv = build_vllm_serve_argv(
            QWEN38_27B, ctx=args.ctx, host=args.host, port=args.port,
            kv_bits=args.kv_bits, spec=args.spec, eagle_model=args.eagle_model)
    except (ContextFloorError, QualityFloorError, RopeScalingRequired,
            ValueError) as e:
        _out(f"error: {e}")
        return 2
    _out("$ " + " ".join(argv))
    if args.dry_run:
        _out("")
        _out("(dry run - vLLM not launched. Drop --dry-run to serve.)")
        return 0
    if shutil.which("vllm") is None:
        _out("")
        _out("error: `vllm` not found on PATH. Install it on the Ampere box, or")
        _out("re-run with --dry-run to just print the command.")
        return 1
    return subprocess.run(argv).returncode


def _serve_llamacpp(args: argparse.Namespace) -> int:
    """Launch llama-server with the measured Hercules configuration.

    Builds the argv from tenselerate.backends.llamacpp (the same code
    scripts/hercules_serve.sh wraps), prints it, and either stops (--dry-run)
    or execs the binary. The binary defaults to this clone's build/bin/
    llama-server, falling back to PATH.

    Returns:
        0 on success or dry run, 1 if llama-server is not found, 2 on a
        validation error (no --model, off-host bind, bad KV type or pool).
    """
    from tenselerate.backends.llamacpp import (
        build_llama_server_argv, env_prefix, llama_server_env,
    )
    if not args.model:
        _out("error: --backend llamacpp needs --model <path.gguf>")
        return 2
    binary = args.llama_server
    if binary is None:
        local = REPO_ROOT / "build" / "bin" / "llama-server"
        binary = str(local) if local.is_file() else "llama-server"
    try:
        argv = build_llama_server_argv(
            args.model, host=args.host, port=args.port, binary=binary,
            slots=args.slots, ctx_pool=args.ctx_pool, kv=args.kv,
            reasoning=args.reasoning, alias=args.alias, mtp_draft=args.mtp_draft,
            sampling=args.sampling)
    except ValueError as e:
        _out(f"error: {e}")
        return 2
    try:
        env = llama_server_env(no_mmvq=args.no_mmvq, mmvq_max=args.mmvq_max)
        prefix = "".join(f"{k}={v} " for k, v in env_prefix(
            no_mmvq=args.no_mmvq, mmvq_max=args.mmvq_max).items())
    except ValueError as e:
        _out(f"error: {e}")
        return 2
    _out("$ " + prefix + " ".join(argv))
    if args.dry_run:
        _out("")
        _out("(dry run - llama-server not launched. Drop --dry-run to serve.)")
        return 0
    if shutil.which(binary) is None and not Path(binary).is_file():
        _out("")
        _out(f"error: `{binary}` not found. Build it with `tenselerate build`, or")
        _out("re-run with --dry-run to just print the command.")
        return 1
    return subprocess.run(argv, env=env).returncode


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the OpenAI /v1 endpoint on the chosen backend.

    Three backends:
    - reference (default): The native reference engine (CPU, any box, end-to-end valid).
    - llamacpp: llama-server with the measured Hercules config (the CMP 170HX box).
    - vllm: Upstream vLLM (Ampere box only: CMP 170HX + RTX 3060).

    Binds loopback-only (127.0.0.1 or ::1) for security. Runs until interrupted.

    Args:
        args: Namespace with backend, host, port, config (reference) or
              ctx, kv_bits, spec, eagle_model, dry_run (vllm).

    Returns:
        0 on clean shutdown, 1 or 2 on error (server's returncode or validation error).
    """
    if args.backend == "vllm":
        return _serve_vllm(args)
    if args.backend == "llamacpp":
        return _serve_llamacpp(args)
    from tenselerate.server import build_server
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        _out("refusing to bind off-host: this engine is loopback-only")
        return 2
    srv = build_server(args.host, args.port, args.config)
    sys.stderr.write(
        f"TENSELERATE ({args.config}) on http://{args.host}:{args.port}/v1\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


# --------------------------------------------------------------------------
# boot — one-command bring-up: doctor, then serve
# --------------------------------------------------------------------------
def cmd_boot(args: argparse.Namespace) -> int:
    """
    The single command to bring the engine up: run the hardware check first so
    a bad driver/VRAM fails loud before anything binds, then serve. A failing
    doctor stops the boot unless --force is given.
    """
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        _out("refusing to bind off-host: this engine is loopback-only")
        return 2
    rc = cmd_doctor(args)
    if rc != 0 and not args.force:
        _out("")
        _out("doctor reported a problem; not booting. Re-run with --force to")
        _out("serve anyway (the reference backend runs without a GPU).")
        return rc
    _out("")
    return cmd_serve(args)


# --------------------------------------------------------------------------
def _add_runtime_args(p: argparse.ArgumentParser) -> None:
    """The backend selector and per-backend options shared by serve and boot."""
    from tenselerate.backends.llamacpp import (
        DEFAULT_ALIAS, DEFAULT_CTX_POOL, DEFAULT_KV, DEFAULT_MTP_DRAFT,
        DEFAULT_REASONING, DEFAULT_SAMPLING, DEFAULT_SLOTS, KV_TYPES,
        REASONING_LEVELS, SAMPLING_MODES,
    )
    p.add_argument("--backend", default="reference",
                   choices=("reference", "llamacpp", "vllm"),
                   help="compute runtime: reference (the native engine, any "
                        "box), llamacpp (llama-server with the measured "
                        "Hercules config - the CMP170hx+3060 path), or vllm")
    p.add_argument("--dry-run", action="store_true",
                   help="llamacpp/vllm: print the launch command, do not launch")
    # llamacpp backend - the measured Hercules configuration
    p.add_argument("--model", default=None,
                   help="llamacpp backend: path to the GGUF (required)")
    p.add_argument("--slots", type=int, default=DEFAULT_SLOTS,
                   help=f"llamacpp backend: parallel slots ({DEFAULT_SLOTS} = "
                        "main loop + 3 subagents)")
    p.add_argument("--ctx-pool", type=int, default=DEFAULT_CTX_POOL,
                   help=f"llamacpp backend: shared KV pool in tokens "
                        f"({DEFAULT_CTX_POOL:,}; --kv-unified)")
    p.add_argument("--kv", default=DEFAULT_KV, choices=KV_TYPES,
                   help=f"llamacpp backend: KV cache type ({DEFAULT_KV})")
    p.add_argument("--reasoning", default=DEFAULT_REASONING,
                   choices=REASONING_LEVELS,
                   help="llamacpp backend: Qwen3.8 reasoning_effort "
                        f"({DEFAULT_REASONING}; fewer thinking tokens)")
    p.add_argument("--no-mmvq", action="store_true",
                   help="llamacpp backend: set GGML_CUDA_NO_MMVQ=1 (force the "
                        "tensor-core MMQ path at every batch width)")
    p.add_argument("--alias", default=DEFAULT_ALIAS,
                   help="llamacpp backend: model id the agent addresses "
                        f"({DEFAULT_ALIAS}); Hermes model.default must match")
    p.add_argument("--mtp-draft", type=int, default=DEFAULT_MTP_DRAFT,
                   help="llamacpp backend: MTP draft depth (default: 1 when the GGUF "
                        "name carries -MTP-, else 0; 1 measured +13..38%% on this "
                        "merge, deeper loses until the head is retrained)")
    p.add_argument("--sampling", default=DEFAULT_SAMPLING, choices=SAMPLING_MODES,
                   help="llamacpp backend: server-default sampling. greedy (default): "
                        "--temp 0 --repeat-penalty 1.0, the only sampling under which the "
                        "MTP draft pays (46.2 vs 29.9 tok/s at temp 0.7/rp 1.15) but the "
                        "merge loops in <think>; dry: greedy + DRY loop guard; low: temp 0.3 "
                        "min-p 0.1; client: llama.cpp's defaults")
    p.add_argument("--mmvq-max", type=int, default=None,
                   help="llamacpp backend: GGML_CUDA_MMVQ_MAX - widest batch kept on "
                        "the dp4a vector path (0..8); 1 keeps single-token decode "
                        "there and routes draft verification to MMQ tensor cores")
    p.add_argument("--llama-server", default=None,
                   help="llamacpp backend: binary (default build/bin/llama-server, "
                        "then PATH)")
    # vllm backend
    p.add_argument("--ctx", type=int, default=MIN_CONTEXT_TOKENS,
                   help=f"vllm backend: context tokens (floor "
                        f"{MIN_CONTEXT_TOKENS:,})")
    p.add_argument("--kv-bits", type=int, default=4, choices=(8, 4),
                   help="vllm backend: 4=fp8 KV (default, the recommended "
                        "Ampere config), 8=auto KV dtype")
    p.add_argument("--spec", default="none", choices=("none", "mtp", "eagle3"),
                   help="vllm backend: none = plain decode (default: MTP measured "
                        "7-11%% acceptance on the DavidAU merge, slower than plain), "
                        "mtp = built-in Qwen3-Next draft head (lossless when the "
                        "head matches the trunk), eagle3 = trained draft head "
                        "(needs --eagle-model)")
    p.add_argument("--eagle-model", default=None,
                   help="vllm backend: EAGLE-3 draft-head repo/path "
                        "(required when --spec eagle3)")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser with all subcommands.

    Subcommands:
    - install: Build + hardware check (one-command bootstrap)
    - build: Compile the engine (CPU-only, CUDA, or kernels only)
    - boot: Run doctor, then serve (one-line deployment)
    - update: Check for and apply new releases
    - info: Show model geometry, KV sizing, context-vs-RoPE tradeoffs
    - plan: Predict throughput for a given machine + context
    - doctor: Verify hardware and driver are ready
    - serve: Start the OpenAI /v1 endpoint (reference or vLLM backend)

    Returns:
        ArgumentParser configured with all subcommands and options.
    """
    ap = argparse.ArgumentParser(
        prog="tenselerate", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_inst = sub.add_parser("install", help="build from this clone, then check the machine")
    p_inst.add_argument("--cpu", action="store_true", help="force a CPU-only build")
    p_inst.add_argument("--weights-gib", type=float, default=15.41)
    p_inst.set_defaults(func=cmd_install)

    p_bld = sub.add_parser("build", help="compile the engine end to end")
    gb = p_bld.add_mutually_exclusive_group()
    gb.add_argument("--cpu", action="store_true", help="force a CPU-only build")
    gb.add_argument("--cuda", action="store_true", help="require CUDA (fail without nvcc)")
    gb.add_argument("--kernels", action="store_true", help="only the int8 kernels")
    p_bld.set_defaults(func=cmd_build)

    p_boot = sub.add_parser("boot", help="doctor, then serve (one-command bring-up)")
    p_boot.add_argument("--host", default="127.0.0.1")
    p_boot.add_argument("--port", type=int, default=8080)
    p_boot.add_argument("--config", default=TINY.name, choices=sorted(CONFIGS))
    p_boot.add_argument("--weights-gib", type=float, default=15.41)
    _add_runtime_args(p_boot)
    p_boot.add_argument("--force", action="store_true",
                        help="serve even if doctor reports a problem")
    p_boot.set_defaults(func=cmd_boot)

    p_up = sub.add_parser("update", help="check for / apply a new build")
    g = p_up.add_mutually_exclusive_group()
    g.add_argument("--source", action="store_true", help="fast-forward and rebuild")
    g.add_argument("--binary", action="store_true", help="download the release build")
    g.add_argument("--list", action="store_true", help="show the release assets")
    p_up.set_defaults(func=cmd_update)

    p_info = sub.add_parser("info", help="geometry, context floor, KV sizing")
    p_info.add_argument("--config", default=QWEN38_27B.name, choices=sorted(CONFIGS))
    p_info.set_defaults(func=cmd_info)

    p_plan = sub.add_parser("plan", help="what this machine does at a context")
    p_plan.add_argument("--config", default=QWEN38_27B.name, choices=sorted(CONFIGS))
    p_plan.add_argument("--machine", default="cmp170hx+3060",
                        choices=sorted(MACHINE_HW))
    p_plan.add_argument("--ctx", type=int, default=MIN_CONTEXT_TOKENS,
                        help=f"context tokens (floor {MIN_CONTEXT_TOKENS:,})")
    p_plan.add_argument("--attention-window", type=int, default=None,
                        help=f"full-attention window (LOCKED at "
                             f"{MAX_ATTENTION_WINDOW:,}, the max no-RoPE recall; "
                             "only that value is legal - it never narrows)")
    p_plan.add_argument("--weights-gib", type=float, default=15.41,
                        help="weight footprint (default: Qwen3.8-27B Q4_K_M)")
    p_plan.add_argument("--overhead-gib", type=float, default=1.5)
    p_plan.add_argument("--kv-bits", type=int, default=8, choices=(8, 4),
                        help="KV cache precision: 8=q8_0 (default), "
                             "4=q4_0 (~2x concurrency; 4-bit is the "
                             "literature-validated KV floor)")
    p_plan.add_argument("--spec", default="none", choices=("none", "mtp"),
                        help="speculative decode: mtp = the model's built-in "
                             "draft head (~1.8x, identical output)")
    p_plan.set_defaults(func=cmd_plan)

    p_doc = sub.add_parser("doctor", help="hardware / driver check")
    p_doc.add_argument("--weights-gib", type=float, default=15.41)
    p_doc.set_defaults(func=cmd_doctor)

    p_srv = sub.add_parser("serve", help="run the OpenAI /v1 endpoint")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8080)
    p_srv.add_argument("--config", default=TINY.name, choices=sorted(CONFIGS))
    _add_runtime_args(p_srv)
    p_srv.set_defaults(func=cmd_serve)

    return ap


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and dispatch to the appropriate subcommand handler.

    Entry point for `python -m tenselerate <cmd>` or the installed `tenselerate` CLI.

    Args:
        argv: Command-line arguments. If None, uses sys.argv[1:].

    Returns:
        The exit code from the subcommand handler (0 = success).
    """
    ap = build_parser()
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
