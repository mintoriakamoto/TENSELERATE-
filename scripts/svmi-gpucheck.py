#!/usr/bin/env python3
"""
svmi-gpucheck — what is this GPU actually capable of, and how should llama.cpp
be built and run for it?

Reads nvidia-smi (or a saved capture with --from-file) and reports, per card:
the PCIe link ACTUALLY negotiated (not the marketing number), the effective
host<->device bandwidth that implies, and any known firmware quirks that
change how you must build llama.cpp.

The motivating case: NVIDIA CMP mining cards. They are the cheapest VRAM per
dollar on the used market, but their firmware cripples things the spec sheet
still advertises:

  CMP 100-210 (Volta GV100, 16 GB HBM2 @ ~830-900 GB/s)
    - PCIe 1.0 x1  (~0.25 GB/s) -- HARDWARE limited (missing PCIe-lane SMD
      parts on many boards); flashing a Tesla V100 vBIOS does NOT restore
      x16, it only nudges clocks. Plan around the x1 link, don't expect a
      flash to fix it.
    - Tensor cores are firmware-gimped: measured FP16 ~5.6 TFLOPS vs FP32
      ~10.6 TFLOPS -- FP16 is SLOWER than FP32 (a healthy Volta is ~8x
      faster). So every FP16 path is a trap on this card.
    - What still works: full HBM2 bandwidth. Decode is memory-bandwidth
      bound, so a Q4/Q8 model that fits in 16 GB decodes FAST -- provided
      you keep the math on integer (MMQ) kernels instead of cuBLAS FP16.

  CMP 30HX/40HX/50HX/90HX/170HX (Ampere) throttle dp4a dispatch ~16x
    -> build with -DGGML_CUDA_DISABLE_DP4A=ON (dp2a emulation, ~2x).

  CMP 170HX (GA100, A100 silicon) also ships with its HBM2e geometry fused down
    to 8 or 10 GiB. cmpunlocker restores it (8 -> 64 GiB, 10 -> 40 GiB) along
    with SM throughput; the unlock is volatile and reverts on driver reload.
    This tool flags a 170HX still reporting the factory size.

Usage:
  python3 scripts/svmi-gpucheck.py
  python3 scripts/svmi-gpucheck.py --from-file nvidia-smi-capture.csv
  python3 scripts/svmi-gpucheck.py --self-test
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

# effective payload GB/s per PCIe lane, per generation (after line encoding)
LANE_GBS = {1: 0.25, 2: 0.50, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}

QUERY = ("name,memory.total,pcie.link.gen.current,pcie.link.width.current,"
         "pcie.link.gen.max,pcie.link.width.max")


class Card:
    def __init__(self, name, vram_mib, gen_cur, width_cur, gen_max, width_max):
        self.name = name
        self.vram_gib = vram_mib / 1024.0
        self.gen_cur = gen_cur
        self.width_cur = width_cur
        self.gen_max = gen_max
        self.width_max = width_max

    @property
    def link_gbs(self) -> float:
        return LANE_GBS.get(self.gen_cur, 0.25) * self.width_cur

    @property
    def is_cmp(self) -> bool:
        return "cmp" in self.name.lower()

    @property
    def is_volta_cmp(self) -> bool:
        return self.is_cmp and "100-210" in self.name

    @property
    def is_170hx(self) -> bool:
        return self.is_cmp and "170hx" in self.name.lower()

    @property
    def hbm_locked(self) -> bool:
        """170HX still reporting its factory HBM2e geometry (8 or 10 GiB)"""
        return self.is_170hx and self.vram_gib < 12.0

    @property
    def link_is_crippled(self) -> bool:
        """negotiated far below what the slot/card should give"""
        return self.link_gbs < 2.0


def parse_csv(text: str) -> list[Card]:
    cards = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        name = parts[0]

        def num(s: str) -> int:
            digits = "".join(ch for ch in s if ch.isdigit())
            return int(digits) if digits else 0

        cards.append(Card(name, num(parts[1]), num(parts[2]), num(parts[3]),
                          num(parts[4]), num(parts[5])))
    return cards


def advise(c: Card) -> tuple[list[str], list[str]]:
    """returns (warnings, build/run flags)"""
    warn, flags = [], []

    if c.is_volta_cmp:
        warn.append("CMP 100-210: tensor cores are firmware-gimped — FP16 is SLOWER "
                    "than FP32. Keep ALL math off FP16/cuBLAS paths.")
        warn.append("A Tesla V100 vBIOS flash does NOT restore PCIe width or tensor "
                    "cores (the x1 link is a board-level limit); it only nudges clocks.")
        flags.append("-DGGML_CUDA_FORCE_MMQ=ON   # integer kernels, never cuBLAS FP16 GEMM")
        flags.append("-DGGML_CUDA_F16=OFF        # (default) no FP16 compute path")
        flags.append("run: -fa off is worth A/B-ing — flash-attn kernels lean on FP16")
    elif c.is_cmp:
        warn.append("Ampere CMP card: dp4a dispatch is throttled ~16x, which is exactly "
                    "what quantized decode uses.")
        flags.append("-DGGML_CUDA_DISABLE_DP4A=ON  # dp2a emulation, ~2x (llama.cpp#24616)")
        flags.append("quantize INT8                # q8_0 attention (resident, INT8/MMQ) "
                     "over a Q4_K_M body (streamed)")
        flags.append("cmake --preset cmp170hx-int8  # or cmp90hx-int8: FORCE_MMQ, no CUDA F16")
        flags.append("run: GGML_CUDA_NO_MMVQ=1      # batch-1 decode on MMQ too, off the dp4a "
                     "path (A/B with scripts/svmi-cmpbench.sh)")

    if c.hbm_locked:
        warn.append(f"CMP 170HX reporting {c.vram_gib:.0f} GiB — that is the factory-fused HBM2e "
                    "geometry, not the silicon. cmpunlocker restores 8 GiB -> 64 GiB and "
                    "10 GiB -> 40 GiB; the unlock is volatile and re-applied by a daemon.")
        flags.append("plan the unlocked card: svmi-auto.py --gpu "
                     + ("cmp170hx-64" if c.vram_gib < 9.0 else "cmp170hx-40"))
    elif c.is_170hx:
        warn.append(f"CMP 170HX reporting {c.vram_gib:.0f} GiB — HBM2e geometry is unlocked. "
                    "Re-check after any driver reload: it reverts to 8/10 GiB.")

    if c.link_is_crippled:
        warn.append(f"PCIe link negotiated at gen{c.gen_cur} x{c.width_cur} "
                    f"(~{c.link_gbs:.2f} GB/s) — SVMI weight streaming is NOT viable here.")
        flags.append("run: keep the model RESIDENT (-ngl 999, no --stream-weights)")
    elif c.gen_cur < c.gen_max or c.width_cur < c.width_max:
        warn.append(f"link running below the card's max (gen{c.gen_cur} x{c.width_cur} "
                    f"vs gen{c.gen_max} x{c.width_max}) — check slot/riser/BIOS; "
                    "idle cards also downtrain, so re-check under load.")
    return warn, flags


def report(cards: list[Card]) -> None:
    for i, c in enumerate(cards):
        print(f"GPU {i}: {c.name}")
        print(f"  VRAM      : {c.vram_gib:.1f} GiB")
        print(f"  PCIe link : gen{c.gen_cur} x{c.width_cur} "
              f"(max gen{c.gen_max} x{c.width_max})  ->  ~{c.link_gbs:.2f} GB/s host<->device")
        load_s = c.vram_gib * 1.074 / c.link_gbs if c.link_gbs else 0.0
        load_txt = f"{load_s:.0f} s" if load_s < 90 else f"{load_s / 60:.1f} min"
        print(f"  model load: ~{load_txt} to fill VRAM once over this link")
        warn, flags = advise(c)
        for w in warn:
            print(f"  ! {w}")
        for f in flags:
            print(f"  + {f}")
        print()

    if any(c.link_is_crippled for c in cards):
        print("planner  : pass these cards to svmi-auto/svmi-net as resident shards, e.g.")
        print("           python3 scripts/svmi-net.py --profile 8b --node <main-gpu>:ram=64 \\")
        print("               --node cmp100-210:ram=16 --nic 1gbe")
        print("           (RPC activations are tiny — a crippled PCIe link hurts the ONE-TIME")
        print("            model load, not per-token traffic. Load once, keep it resident.)")


def self_test() -> int:
    fixture = "\n".join([
        "NVIDIA CMP 100-210, 16384 MiB, 1, 1, 1, 1",
        "NVIDIA CMP 90HX, 10240 MiB, 1, 4, 1, 4",
        "NVIDIA GeForce RTX 3060, 12288 MiB, 3, 16, 3, 16",
        "NVIDIA GeForce RTX 2080 Ti, 11264 MiB, 1, 16, 3, 16",
        "NVIDIA CMP 170HX, 8192 MiB, 1, 4, 1, 16",
        "NVIDIA CMP 170HX, 65536 MiB, 2, 4, 2, 16",
    ])
    cards = parse_csv(fixture)
    assert len(cards) == 6, cards

    volta, ampere_cmp, healthy, downtrained, locked_170, unlocked_170 = cards
    # 1. parsing + link math
    assert abs(volta.vram_gib - 16.0) < 0.01
    assert abs(volta.link_gbs - 0.25) < 1e-6, volta.link_gbs          # gen1 x1
    assert abs(healthy.link_gbs - 0.985 * 16) < 1e-6                  # gen3 x16
    # 2. classification
    assert volta.is_cmp and volta.is_volta_cmp
    assert ampere_cmp.is_cmp and not ampere_cmp.is_volta_cmp
    assert not healthy.is_cmp
    assert volta.link_is_crippled and ampere_cmp.link_is_crippled
    assert not healthy.link_is_crippled
    # 3. advice routing: FP16 trap only for the Volta CMP, dp4a only for Ampere CMP
    vw, vf = advise(volta)
    aw, af = advise(ampere_cmp)
    hw, hf = advise(healthy)
    dw, _ = advise(downtrained)
    # both families want integer kernels; only the Ampere CMPs have the dp4a
    # throttle, so only they get the dp2a emulation and the MMVQ bypass
    assert any("FORCE_MMQ" in f for f in vf), vf
    assert not any("DISABLE_DP4A" in f for f in vf), vf
    assert not any("NO_MMVQ" in f for f in vf), vf
    assert any("DISABLE_DP4A" in f for f in af), af
    assert any("NO_MMVQ" in f for f in af), af
    assert any("cmp170hx-int8" in f for f in af), af
    assert hw == [] and hf == [], (hw, hf)
    assert any("below the card's max" in w for w in dw), dw
    # 4. V100-vBIOS claim is stated honestly (does NOT fix the link)
    assert any("does NOT restore PCIe width" in w for w in vw), vw
    # 5. 170HX HBM2e geometry: flagged while fused down, not flagged once unlocked
    assert locked_170.hbm_locked and not unlocked_170.hbm_locked
    lw, lf = advise(locked_170)
    uw, _ = advise(unlocked_170)
    assert any("factory-fused HBM2e" in w for w in lw), lw
    assert any("cmp170hx-64" in f for f in lf), lf
    assert any("reverts to 8/10 GiB" in w for w in uw), uw
    print("self-test OK: 6 fixtures parsed; link math gen1x1=0.25 / gen3x16=15.76 GB/s;")
    print("              CMP-Volta -> FP16-trap + MMQ advice, CMP-Ampere -> dp4a advice,")
    print("              healthy card -> no advice, downtrained link -> flagged,")
    print("              170HX @ 8 GiB -> unlock advice, @ 64 GiB -> revert warning.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-file",
                    help=f"CSV capture: nvidia-smi --query-gpu={QUERY} --format=csv,noheader")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.from_file:
        with open(args.from_file) as fh:
            text = fh.read()
    else:
        if not shutil.which("nvidia-smi"):
            print("nvidia-smi not found. On the target machine run:")
            print(f"  nvidia-smi --query-gpu={QUERY} --format=csv,noheader > gpus.csv")
            print("then: python3 scripts/svmi-gpucheck.py --from-file gpus.csv")
            return 1
        text = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout

    cards = parse_csv(text)
    if not cards:
        print("no GPUs parsed from nvidia-smi output")
        return 1
    report(cards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
