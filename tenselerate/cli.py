"""
The `tenselerate` command line.

One entry point for the engine:

    tenselerate update           check for and apply a new build
    tenselerate serve            run the OpenAI /v1 endpoint
    tenselerate info             model geometry, the context floor, KV sizing
    tenselerate plan             what this machine can do at a given context
    tenselerate doctor           hardware/driver check before anything else

Run as `python -m tenselerate <cmd>` (or `tenselerate <cmd>` once installed).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from tenselerate.config import (
    CONFIGS, MIN_CONTEXT_TOKENS, MIN_DECODE_TOKS, RAVENX_27B, TINY,
    ContextFloorError, RopeScalingRequired,
)
from tenselerate.engine.scheduler import Scheduler

GiB = 1024 ** 3
REPO_ROOT = Path(__file__).resolve().parent.parent


def _out(msg: str = "") -> None:
    sys.stdout.write(msg + "\n")


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
    _out(f"SPEED FLOOR      : {MIN_DECODE_TOKS:,} tok/s aggregate (hard minimum, "
         "enforced by `plan`)")
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
# (VRAM GiB, VRAM read GB/s) for the machines this engine targets
MACHINE_HW = {
    "cmp170hx": (64.0, 1490.0),
    "5070+3060": (24.0, 500.0),
    "5070": (12.0, 672.0),
    "3060": (12.0, 360.0),
}
BW_EFFICIENCY = 0.65          # planning assumption; svmi-bwprofile.py measures it


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = CONFIGS[args.config]
    ctx = args.ctx
    try:
        cfg.validate_context(ctx)
    except (ContextFloorError, RopeScalingRequired) as e:
        _out(f"error: {e}")
        return 2

    vram, bw = MACHINE_HW[args.machine]
    weights = args.weights_gib
    kv = cfg.kv_bytes_for_context(ctx) / GiB
    total = weights + kv + args.overhead_gib

    _out(f"machine          : {args.machine}  ({vram:.0f} GiB, ~{bw:.0f} GB/s)")
    _out(f"context          : {ctx:,} tokens  (floor {MIN_CONTEXT_TOKENS:,})")
    _out(f"weights          : {weights:.2f} GiB")
    _out(f"KV (windowed)    : {kv:.2f} GiB   <- constant beyond the window")
    _out(f"total resident   : {total:.2f} GiB of {vram:.0f} GiB "
         f"({'FITS' if total <= vram else 'DOES NOT FIT'})")
    if total > vram:
        _out("")
        _out("Options: a narrower --attention-window, a smaller quant, or the")
        _out("other machine. Lowering context is not one - the floor is fixed.")
        return 1

    # decode roofline: weights + resident KV read per token
    per_token_gb = (weights + kv) * 1.074
    single = bw * BW_EFFICIENCY / per_token_gb
    _out("")
    _out(f"decode (batch 1) : ~{single:,.0f} tok/s   at ANY context >= the window")

    # Each concurrent sequence carries its OWN windowed KV, so batch is capped by
    # memory, not just by bandwidth. Never report a batch that cannot be resident.
    free_for_kv = vram - weights - args.overhead_gib
    # Ask the real scheduler, so this table and the engine can never disagree.
    try:
        sched = Scheduler(cfg, kv_budget_gib=free_for_kv)
        max_batch = sched.max_concurrent
    except ValueError:
        max_batch = 0
    if max_batch < 1:
        _out("  (no room for even one sequence's KV - narrow the window)")
        return 1

    def agg(b: int) -> float:
        return bw * BW_EFFICIENCY / ((weights + kv * b) * 1.074) * b

    _out(f"max concurrent   : {max_batch} sequences "
         f"({free_for_kv:.1f} GiB free / {kv:.2f} GiB KV each)")
    _out("aggregate with continuous batching (memory-feasible only):")
    shown = [b for b in (2, 4, 8, 16, 32, 64) if b <= max_batch]
    if max_batch not in shown:
        shown.append(max_batch)
    for b in shown:
        mark = f"  <- {MIN_DECODE_TOKS}+" if agg(b) >= MIN_DECODE_TOKS else ""
        _out(f"  batch {b:>3}      : ~{agg(b):,.0f} tok/s{mark}")

    # -- the speed floor ---------------------------------------------------
    # The floor is a property of the machine: met if ANY window reaches
    # MIN_DECODE_TOKS at the context floor. A box that cannot is refused.
    best = agg(max_batch)
    _out("")
    if best >= MIN_DECODE_TOKS:
        _out(f"SPEED FLOOR      : {MIN_DECODE_TOKS} tok/s - met at this window")
    else:
        # what window would reach the floor? KV per seq is linear in the window.
        _out(f"ceiling here is ~{best:,.0f} tok/s against the {MIN_DECODE_TOKS} "
             "tok/s floor. More needs more")
        _out("concurrency, which means a narrower attention window (KV per "
             "sequence is what caps batch):")
        floor_met = False
        for w in (65_536, 32_768, 16_384, 8_192):
            kv_w = cfg.kv_bytes_per_token() * w / GiB
            mb = int((vram - weights - args.overhead_gib) // kv_w)
            if mb < 1:
                continue
            a = bw * BW_EFFICIENCY / ((weights + kv_w * mb) * 1.074) * mb
            floor_met = floor_met or a >= MIN_DECODE_TOKS
            flag = f"  <- reaches {MIN_DECODE_TOKS}+" if a >= MIN_DECODE_TOKS else ""
            _out(f"  window {w:>7,} -> KV {kv_w:5.2f} GiB, max batch {mb:>3}, "
                 f"~{a:,.0f} tok/s{flag}")
        _out("  (context stays at the floor in every row - the window trades")
        _out("   exact-recall depth for concurrency, never context length.)")
        if not floor_met:
            _out("")
            _out(f"SPEED FLOOR      : {MIN_DECODE_TOKS} tok/s - NOT reachable on "
                 "this machine at any window.")
            _out("This box cannot serve at the TENSELERATE floors. Use the other "
                 "machine.")
            _out("")
            _out("Numbers are a bandwidth roofline at "
                 f"{BW_EFFICIENCY:.0%} efficiency, not a measurement.")
            return 3
    _out("")
    _out("Numbers are a bandwidth roofline at "
         f"{BW_EFFICIENCY:.0%} efficiency, not a measurement.")
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
def cmd_serve(args: argparse.Namespace) -> int:
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
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="tenselerate", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("update", help="check for / apply a new build")
    g = p_up.add_mutually_exclusive_group()
    g.add_argument("--source", action="store_true", help="fast-forward and rebuild")
    g.add_argument("--binary", action="store_true", help="download the release build")
    g.add_argument("--list", action="store_true", help="show the release assets")
    p_up.set_defaults(func=cmd_update)

    p_info = sub.add_parser("info", help="geometry, context floor, KV sizing")
    p_info.add_argument("--config", default=RAVENX_27B.name, choices=sorted(CONFIGS))
    p_info.set_defaults(func=cmd_info)

    p_plan = sub.add_parser("plan", help="what this machine does at a context")
    p_plan.add_argument("--config", default=RAVENX_27B.name, choices=sorted(CONFIGS))
    p_plan.add_argument("--machine", default="cmp170hx", choices=sorted(MACHINE_HW))
    p_plan.add_argument("--ctx", type=int, default=MIN_CONTEXT_TOKENS,
                        help=f"context tokens (floor {MIN_CONTEXT_TOKENS:,})")
    p_plan.add_argument("--weights-gib", type=float, default=15.41,
                        help="weight footprint (default: RavenX Q4_K_M)")
    p_plan.add_argument("--overhead-gib", type=float, default=1.5)
    p_plan.set_defaults(func=cmd_plan)

    p_doc = sub.add_parser("doctor", help="hardware / driver check")
    p_doc.add_argument("--weights-gib", type=float, default=15.41)
    p_doc.set_defaults(func=cmd_doctor)

    p_srv = sub.add_parser("serve", help="run the OpenAI /v1 endpoint")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8080)
    p_srv.add_argument("--config", default=TINY.name, choices=sorted(CONFIGS))
    p_srv.set_defaults(func=cmd_serve)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
