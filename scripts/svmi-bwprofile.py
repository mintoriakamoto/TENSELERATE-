#!/usr/bin/env python3
"""
svmi-bwprofile — measure what this box actually sustains, and persist it.

Every sizing figure the planners print rides on one number: the fraction of a
card's theoretical memory bandwidth a real decode loop achieves. That was a
hardcoded guess (0.65). This measures it instead, and writes a per-GPU profile
the other planners read.

Method (after FreeToken's `ft bench bw`, Apache-2.0, https://github.com/FlashML-org/FreeToken):

  * decode is bandwidth-bound and reads weights + touched KV once per token, so
    achieved_GBs = bytes_per_token * measured_tok_s. Run llama-bench at a known
    batch size and the achieved bandwidth falls out of the token rate.
  * measure the STREAMED path the same way when the model does not fit resident,
    which gives the PCIe gather rate under real kernels rather than a linear copy.
  * the ratio of the two is what decides resident-vs-streamed, and the ratio is
    what a synthetic copy benchmark gets wrong: it never contends with compute.

Usage:
  python3 scripts/svmi-bwprofile.py -m model-int8.gguf --gpu cmp170hx-40
  python3 scripts/svmi-bwprofile.py -m model.gguf --gpu cmp170hx-40 --streamed
  python3 scripts/svmi-bwprofile.py --show
  python3 scripts/svmi-bwprofile.py --self-test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

GiB = 1024**3

_spec = importlib.util.spec_from_file_location("svmi_cluster",
                                               Path(__file__).parent / "svmi-cluster.py")
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HBM_BW = _mod.HBM_BW
KV_BPE = _mod.KV_BPE


def profile_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tenselerate"


def profile_path(gpu: str) -> Path:
    return profile_dir() / f"bwprofile-{gpu}.json"


def load_profile(gpu: str) -> dict | None:
    """the planners call this; returns None when the box has never been benched"""
    p = profile_path(gpu)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def parse_tok_s(text: str) -> float | None:
    """pull the t/s column out of llama-bench's markdown table (last numeric row)"""
    val = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 3:
            continue
        m = re.match(r"^([0-9]+\.?[0-9]*)", cells[-2].replace("±", " ").strip())
        if m:
            val = float(m.group(1))
    return val


def achieved_gbs(bytes_per_token: float, tok_s: float) -> float:
    return bytes_per_token * tok_s / 1e9


def run_bench(binary: str, model: str, extra: list[str]) -> tuple[float | None, str]:
    cmd = [binary, "-m", model, "-ngl", "999", "-r", "3", *extra]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except OSError as e:
        return None, f"could not run {binary}: {e}"
    return parse_tok_s(out), out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", help="GGUF to bench")
    ap.add_argument("--gpu", default="cmp170hx-40", help="preset name this profile is for")
    ap.add_argument("--bench", default="./build/bin/llama-bench")
    ap.add_argument("--kv-type", choices=sorted(KV_BPE), default="q8_0")
    ap.add_argument("--ctx", type=int, default=8192, help="context the bench runs at")
    ap.add_argument("--streamed", action="store_true",
                    help="also measure the SVMI streamed path (PCIe-bound rather than HBM-bound)")
    ap.add_argument("--show", action="store_true", help="print the stored profile and exit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.show:
        prof = load_profile(args.gpu)
        print(json.dumps(prof, indent=2) if prof else f"no profile for {args.gpu} yet")
        return 0
    if not args.model:
        ap.error("-m/--model is required (or --show / --self-test)")

    sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
    from gguf import GGUFReader

    r = GGUFReader(args.model)
    weights = sum(int(t.n_bytes) for t in r.tensors)
    theoretical = HBM_BW.get(args.gpu)
    if theoretical is None:
        print(f"warning: no theoretical bandwidth known for '{args.gpu}', efficiency omitted")

    print(f"model     : {Path(args.model).name}  ({weights / GiB:.1f} GiB weights)")
    print(f"gpu       : {args.gpu}" + (f"  ({theoretical:.0f} GB/s theoretical)"
                                       if theoretical else ""))
    print("resident  : benching decode ...")
    tok_s, out = run_bench(args.bench, args.model, ["-p", "0", "-n", "64"])
    if tok_s is None:
        print(out if out else "  no t/s parsed from llama-bench output", file=sys.stderr)
        return 1

    # at batch 1 with a short context the KV read is negligible next to the weights,
    # so the token rate is a clean read of achieved weight-streaming bandwidth
    gbs = achieved_gbs(weights, tok_s)
    prof: dict = {"gpu": args.gpu, "model": Path(args.model).name,
                  "weights_bytes": weights, "decode_tok_s": tok_s,
                  "achieved_gbs": round(gbs, 1)}
    print(f"            {tok_s:.2f} tok/s -> {gbs:.0f} GB/s achieved")
    if theoretical:
        eff = gbs / theoretical
        prof["theoretical_gbs"] = theoretical
        prof["bw_efficiency"] = round(eff, 3)
        print(f"            {eff * 100:.0f}% of theoretical"
              + ("  <-- planners assumed 65%" if abs(eff - 0.65) > 0.05 else ""))

    if args.streamed:
        print("streamed  : benching the SVMI path ...")
        env_note = "" if os.environ.get("GGML_CUDA_REGISTER_HOST") else \
            "  (note: GGML_CUDA_REGISTER_HOST unset - uploads may be synchronous)"
        s_tok_s, s_out = run_bench(args.bench, args.model,
                                   ["-p", "0", "-n", "64", "--stream-weights", "8",
                                    "--stream-decode"])
        if s_tok_s:
            s_gbs = achieved_gbs(weights, s_tok_s)
            prof["streamed_tok_s"] = s_tok_s
            prof["streamed_gbs"] = round(s_gbs, 1)
            prof["resident_vs_streamed"] = round(gbs / s_gbs, 2) if s_gbs else None
            print(f"            {s_tok_s:.2f} tok/s -> {s_gbs:.0f} GB/s{env_note}")
            print(f"            resident is {gbs / s_gbs:.1f}x the streamed path"
                  if s_gbs else "")
        else:
            print("            streamed bench produced no t/s", file=sys.stderr)

    profile_dir().mkdir(parents=True, exist_ok=True)
    profile_path(args.gpu).write_text(json.dumps(prof, indent=2) + "\n")
    print(f"\nwrote {profile_path(args.gpu)}")
    print("svmi-cluster.py and svmi-plan.py pick this up automatically.")
    return 0


def self_test() -> int:
    table = "\n".join([
        "| model | size | params | backend | ngl | test | t/s |",
        "| ----- | ---- | ------ | ------- | --- | ---- | --- |",
        "| llama 27B INT8 | 17.6 GiB | 27.0 B | CUDA | 999 | tg64 | 42.50 ± 0.31 |",
    ])
    assert parse_tok_s(table) == 42.5, parse_tok_s(table)
    assert parse_tok_s("no table here") is None
    # 20 GB of weights at 50 tok/s is 1000 GB/s achieved
    assert abs(achieved_gbs(20e9, 50) - 1000.0) < 1e-6
    assert load_profile("no-such-gpu-preset") is None
    print("self-test OK: llama-bench t/s parsing, no-table case,")
    print("              achieved-bandwidth math, missing-profile fallback")
    return 0


if __name__ == "__main__":
    sys.exit(main())
