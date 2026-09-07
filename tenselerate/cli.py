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
    sys.stdout.write(msg + "\n")


# --------------------------------------------------------------------------
# build / install
# --------------------------------------------------------------------------
def _run_build(mode_args: list[str]) -> int:
    """Drive scripts/tenselerate-build.sh with the given flags."""
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
    """Compile the engine end to end: the int8 kernels and the llama.cpp server."""
    mode_args = []
    if args.cpu:
        mode_args.append("--cpu")
    elif args.cuda:
        mode_args.append("--cuda")
    elif args.kernels:
        mode_args.append("--kernels")
    return _run_build(mode_args)


def cmd_install(args: argparse.Namespace) -> int:
    """Build from this clone, then run the hardware check — zero to ready."""
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
    """
    Drive scripts/tenselerate-update.sh, which compares the running build against
    the newest published release and either fast-forwards + rebuilds, or pulls
    the prebuilt binary for this machine.
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
# (VRAM GiB, VRAM read GB/s) for the machines this engine targets.
# TENSELERATE supports exactly ONE machine: the dual RTX 2080 Ti box (Ryzen 9
# 9950X, 32 GB DDR5, 1 TB NVMe + 250 GB OS SSD). The two 2080 Ti are Turing
# (sm_75) and run as one pipeline node - 22 GiB pooled, both stages' HBM read
# overlapped under continuous batching (2 x 616 GB/s). This box fits the 1M
# context floor (21.16 of 22 GiB) but tops out ~152 tok/s, BELOW the 400 tok/s
# product standard: `plan` reports that honestly rather than lowering the bar
# to flatter the hardware. Serving still works (serve/boot do not gate on the
# speed floor); `plan` is the advisory that the box is under the target.
# The Ampere target box is the vLLM path (see tenselerate.backends.vllm):
# CMP 170HX (GA100, sm_80, HBM2e, ~1493 GB/s, unlocked to 40 GiB - stock 8 GiB
# cannot hold the weights) + RTX 3060 12 GiB (GA106, sm_86, GDDR6, ~360 GB/s).
# Pooled 52 GiB; the bandwidth is the sum under a bandwidth-balanced 2-stage
# pipeline (PP=2, no NVLink), the same perfectly-overlapped assumption the
# 2x2080ti row makes for its two cards.
MACHINE_HW = {
    "2x2080ti": (22.0, 1232.0),
    "cmp170hx+3060": (52.0, 1853.0),
}
BW_EFFICIENCY = 0.65          # planning assumption; svmi-bwprofile.py measures it


def _accel_path(cfg, vram: float, weights: float, args, bw: float) -> None:
    """
    The window is LOCKED at the max-recall value, so speed is not bought by
    narrowing it - the only levers left are lossless: q4_0 KV (more concurrency)
    and speculative decode (MTP, ~1.8x; EAGLE-3 higher). Model them at the fixed
    window and say honestly how close they get to MIN_DECODE_TOKS - they do not
    reach it here, because quality is pinned at maximum.
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
    """Drive an upstream vLLM OpenAI server as the compute runtime (Ampere box)."""
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


def cmd_serve(args: argparse.Namespace) -> int:
    if args.backend == "vllm":
        return _serve_vllm(args)
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
def build_parser() -> argparse.ArgumentParser:
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
    p_srv.add_argument("--backend", default="reference",
                       choices=("reference", "vllm"),
                       help="compute runtime: reference (the native engine, "
                            "any box) or vllm (the Ampere CMP170hx+3060 path)")
    p_srv.add_argument("--dry-run", action="store_true",
                       help="vllm backend: print the vLLM command, do not launch")
    p_srv.add_argument("--ctx", type=int, default=MIN_CONTEXT_TOKENS,
                       help=f"vllm backend: context tokens (floor "
                            f"{MIN_CONTEXT_TOKENS:,})")
    p_srv.add_argument("--kv-bits", type=int, default=4, choices=(8, 4),
                       help="vllm backend: 4=fp8 KV (default, the recommended "
                            "Ampere config), 8=auto KV dtype")
    p_srv.add_argument("--spec", default="mtp", choices=("none", "mtp", "eagle3"),
                       help="vllm backend: mtp = built-in Qwen3-Next speculative "
                            "(default, lossless), eagle3 = trained draft head "
                            "(needs --eagle-model), none = plain decode")
    p_srv.add_argument("--eagle-model", default=None,
                       help="vllm backend: EAGLE-3 draft-head repo/path "
                            "(required when --spec eagle3)")
    p_srv.set_defaults(func=cmd_serve)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
