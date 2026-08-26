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

# VRAM read bandwidth GB/s, matched on the model token in the nvidia-smi name.
# This is what bounds decode once the weights are resident; the PCIe link above
# only bounds the one-time load. Longest key wins, so "5070 ti" beats "5070".
MEM_BW = {
    "cmp 100-210": 830.0, "cmp 170hx": 1490.0, "cmp 90hx": 760.0,
    "3060": 360.0, "3080": 760.0, "3090": 936.0,
    "4070": 504.0, "4080": 717.0, "4090": 1008.0,
    "5060 ti": 448.0, "5070 ti": 896.0, "5070": 672.0,
    "5080": 960.0, "5090": 1792.0,
}

# CUDA compute capability by model token -> the preset that emits its SASS.
# sm_120 (Blackwell GB20x) cannot be compiled by CUDA 12.6 or older at all.
ARCH = {
    "5060": ("120", "rtx-blackwell"), "5070": ("120", "rtx-blackwell"),
    "5080": ("120", "rtx-blackwell"), "5090": ("120", "rtx-blackwell"),
    "4060": ("89", "rtx-ada"), "4070": ("89", "rtx-ada"),
    "4080": ("89", "rtx-ada"), "4090": ("89", "rtx-ada"),
    "3060": ("86", "rtx-ampere"), "3070": ("86", "rtx-ampere"),
    "3080": ("86", "rtx-ampere"), "3090": ("86", "rtx-ampere"),
}

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
    def mem_bw(self) -> float:
        """VRAM read GB/s - what bounds decode once resident. 0.0 = unknown."""
        low = self.name.lower()
        hits = [(k, v) for k, v in MEM_BW.items() if k in low]
        if not hits:
            return 0.0
        return max(hits, key=lambda kv: len(kv[0]))[1]

    @property
    def arch(self) -> tuple[str, str]:
        """(compute capability, preset) or ("", "") when unrecognised"""
        low = self.name.lower()
        if self.is_cmp:
            return ("", "")
        hits = [(k, v) for k, v in ARCH.items() if k in low]
        if not hits:
            return ("", "")
        return max(hits, key=lambda kv: len(kv[0]))[1]

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
        flags.append("-DGGML_CUDA_FORCE_CUBLAS=OFF # (default) never route back to FP16 GEMM")
        flags.append("run: -fa off is worth A/B-ing — flash-attn kernels lean on FP16")
    elif c.is_cmp:
        warn.append("Ampere CMP card: dp4a dispatch is throttled ~16x, which is exactly "
                    "what quantized decode uses.")
        flags.append("-DGGML_CUDA_DISABLE_DP4A=ON  # dp2a emulation, ~2x (llama.cpp#24616)")
        flags.append("quantize INT8                # q8_0 attention (resident, INT8/MMQ) "
                     "over a Q4_K_M body (streamed)")
        flags.append("cmake --preset cmp170hx-int8  # or cmp90hx-int8: FORCE_MMQ + dp2a")
        flags.append("run: GGML_CUDA_NO_MMVQ=1      # batch-1 decode on MMQ too, off the dp4a "
                     "path (A/B with scripts/svmi-cmpbench.sh)")

    cc, preset = c.arch
    if cc:
        flags.append(f"cmake --preset {preset}   # sm_{cc}, cuBLAS left ON")
        if cc == "120":
            warn.append("Blackwell (sm_120) needs CUDA >= 12.8 - nvcc 12.6 and older "
                        "cannot emit this architecture at all, and there is no older "
                        "SASS for it to fall back to.")
        flags.append("do NOT add -DGGML_CUDA_FORCE_MMQ=ON here: this card has working "
                     "FP16/BF16 tensor cores, and forcing MMQ costs prompt throughput. "
                     "llama.cpp already prefers MMQ for batch-1 decode on its own.")

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


def plan_split(cards: list[Card], model_gib: float) -> list[str]:
    """
    How to place `model_gib` of weights across a mixed-bandwidth box.

    With the default --split-mode layer every token walks all layers in order,
    so per-token time is sum(bytes_i / bw_i). Minimising that subject to the
    per-card capacities is a linear program whose optimum is greedy: fill the
    FASTEST card first, spill only the remainder onto slower ones.

    That is not the same as splitting proportionally to bandwidth, and it has a
    blunt consequence - if the model fits on the fast card alone, adding a
    slower card makes decode SLOWER, not faster. A second card buys capacity,
    never speed, under a layer split.
    """
    known = [c for c in cards if c.mem_bw > 0]
    if len(known) < 2:
        return []

    ordered = sorted(known, key=lambda c: c.mem_bw, reverse=True)
    total_vram = sum(c.vram_gib for c in ordered)
    out = []

    spread = ordered[0].mem_bw / ordered[-1].mem_bw
    if spread < 1.15:
        out.append(f"cards are within {(spread - 1) * 100:.0f}% on bandwidth - an even "
                   "split is fine; order barely matters.")
        return out

    if model_gib > total_vram:
        out.append(f"{model_gib:.1f} GiB of weights does not fit in {total_vram:.1f} GiB "
                   "of combined VRAM - offload or stream the remainder.")

    # greedy fill fastest-first
    remaining, share = model_gib, {}
    for c in ordered:
        take = min(c.vram_gib, remaining)
        share[id(c)] = take
        remaining -= take
    placed = model_gib - max(remaining, 0.0)

    fast = ordered[0]
    if model_gib <= fast.vram_gib:
        out.append(f"{model_gib:.1f} GiB fits on {fast.name} alone "
                   f"({fast.mem_bw:.0f} GB/s). Use ONLY that card - adding "
                   f"{ordered[-1].name} ({ordered[-1].mem_bw:.0f} GB/s) to a layer "
                   "split would slow decode down, not speed it up:")
        out.append(f"  CUDA_VISIBLE_DEVICES={cards.index(fast)} llama-server -ngl 999")
        return out

    ts = ",".join(f"{share[id(c)] / placed:.2f}" for c in ordered)
    order_txt = " > ".join(f"{c.name.replace('NVIDIA ', '')} {c.mem_bw:.0f}GB/s"
                           for c in ordered)
    out.append(f"mixed bandwidth ({order_txt}) - fill the fastest card first, "
               "do NOT split proportionally:")
    out.append(f"  --main-gpu {cards.index(fast)} --tensor-split {ts}"
               "   # in fastest-first device order")
    out.append("  (reorder with CUDA_VISIBLE_DEVICES so device 0 is the fastest card; "
               "--tensor-split is indexed by visible-device order, not by speed.)")

    # what the slow card costs, relative to an imaginary all-fast box
    t_mixed = sum(share[id(c)] / c.mem_bw for c in ordered)
    t_fast = placed / fast.mem_bw
    out.append(f"  expect ~{t_mixed / t_fast:.2f}x the per-token weight-read time of an "
               f"all-{fast.name.replace('NVIDIA GeForce ', '')} box of the same capacity.")
    return out


def report(cards: list[Card], model_gib: float = 0.0) -> None:
    for i, c in enumerate(cards):
        print(f"GPU {i}: {c.name}")
        print(f"  VRAM      : {c.vram_gib:.1f} GiB")
        if c.mem_bw:
            print(f"  VRAM bw   : ~{c.mem_bw:.0f} GB/s  (bounds decode once resident)")
        print(f"  PCIe link : gen{c.gen_cur} x{c.width_cur} "
              f"(max gen{c.gen_max} x{c.width_max})  ->  ~{c.link_gbs:.2f} GB/s host<->device")
        load_s = c.vram_gib * 1.074 / c.link_gbs if c.link_gbs else 0.0
        if load_s < 10:
            load_txt = f"{load_s:.1f} s"
        elif load_s < 90:
            load_txt = f"{load_s:.0f} s"
        else:
            load_txt = f"{load_s / 60:.1f} min"
        print(f"  model load: ~{load_txt} to fill VRAM once over this link")
        warn, flags = advise(c)
        for w in warn:
            print(f"  ! {w}")
        for f in flags:
            print(f"  + {f}")
        print()

    # sort NUMERICALLY - as strings "120" sorts before "86". Done by sorting ints
    # and mapping back rather than with key=int: passing the int constructor as a
    # sort key widens the inferred element type to int()'s whole argument union,
    # which then no longer satisfies str.join(Iterable[str]).
    ccs = [str(n) for n in sorted({int(c.arch[0]) for c in cards if c.arch[0]})]
    if len(ccs) > 1:
        arch_list = ";".join(f"{cc}-real" for cc in ccs)
        print(f"mixed architectures (sm_{', sm_'.join(ccs)}) - build ONE fat binary, "
              "not one per card:")
        if ccs == ["86", "120"]:
            print("  cmake --preset rtx-5070+3060")
        print(f'  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="{arch_list}"')
        print("  (a binary missing a card's SASS still runs via PTX JIT, but pays a long")
        print("   first-load compile and can miss arch-specific kernels entirely.)")
        print()

    if model_gib > 0:
        plan = plan_split(cards, model_gib)
        if plan:
            print(f"placement ({model_gib:.1f} GiB of weights):")
            for line in plan:
                print(f"  {line}")
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
        "NVIDIA GeForce RTX 5070, 12288 MiB, 5, 16, 5, 16",
    ])
    cards = parse_csv(fixture)
    assert len(cards) == 7, cards

    volta, ampere_cmp, healthy, downtrained, locked_170, unlocked_170, blackwell = cards
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
    # a healthy consumer card raises no WARNINGS, but does now get build advice
    assert hw == [], hw
    assert any("rtx-ampere" in f for f in hf), hf
    assert any("do NOT add -DGGML_CUDA_FORCE_MMQ" in f for f in hf), hf
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
    # 6. arch routing + the sm_120 toolchain floor
    assert healthy.arch == ("86", "rtx-ampere"), healthy.arch
    assert blackwell.arch == ("120", "rtx-blackwell"), blackwell.arch
    assert volta.arch == ("", ""), volta.arch          # CMP cards keep CMP advice
    bw_, bf_ = advise(blackwell)
    assert any("CUDA >= 12.8" in w for w in bw_), bw_
    assert any("rtx-blackwell" in f for f in bf_), bf_

    # 7. memory-bandwidth lookup: longest key wins, unknown -> 0.0
    assert abs(blackwell.mem_bw - 672.0) < 1e-6, blackwell.mem_bw
    assert abs(healthy.mem_bw - 360.0) < 1e-6, healthy.mem_bw
    assert parse_csv("NVIDIA RTX 5070 Ti, 16384 MiB, 5, 16, 5, 16")[0].mem_bw == 896.0
    assert downtrained.mem_bw == 0.0, downtrained.mem_bw   # 2080 Ti not in the table

    # 8. layer-split placement on a mixed 5070 (672 GB/s) + 3060 (360 GB/s) box.
    #    Fits-on-the-fast-card -> use one card; the second would only slow it down.
    mixed = [blackwell, healthy]
    fits = plan_split(mixed, 10.0)
    assert any("fits on" in ln and "alone" in ln for ln in fits), fits
    assert any("CUDA_VISIBLE_DEVICES=0" in ln for ln in fits), fits
    #    Too big for one card -> greedy fill fastest-first, NOT bandwidth-proportional.
    spill = plan_split(mixed, 18.0)
    assert any("--tensor-split 0.67,0.33" in ln for ln in spill), spill
    assert any("do NOT split proportionally" in ln for ln in spill), spill
    #    12/18 on the 5070 and 6/18 on the 3060; a proportional split would have
    #    been 672:360 = 0.65:0.35, which is a different (and slower) answer.
    assert any("does not fit" in ln for ln in plan_split(mixed, 40.0)), "overflow"
    #    A single known card, or two near-identical ones, yields no split advice.
    assert plan_split([blackwell], 10.0) == []

    # 9. mixed-arch report: compute capabilities must sort NUMERICALLY. A string
    #    sort puts "120" before "86", which silently loses the combined preset.
    import io as _io
    from contextlib import redirect_stdout
    buf = _io.StringIO()
    with redirect_stdout(buf):
        report(mixed)
    out = buf.getvalue()
    assert "mixed architectures (sm_86, sm_120)" in out, out
    assert "cmake --preset rtx-5070+3060" in out, out
    assert '"86-real;120-real"' in out, out
    #    a single-arch box gets no fat-binary section at all
    buf = _io.StringIO()
    with redirect_stdout(buf):
        report([blackwell])
    assert "mixed architectures" not in buf.getvalue()

    print("self-test OK: 7 fixtures parsed; link math gen1x1=0.25 / gen3x16=15.76 GB/s;")
    print("              CMP-Volta -> FP16-trap + MMQ advice, CMP-Ampere -> dp4a advice,")
    print("              healthy card -> preset advice but no warnings, bad link -> flagged,")
    print("              170HX @ 8 GiB -> unlock advice, @ 64 GiB -> revert warning,")
    print("              5070 -> sm_120 + CUDA>=12.8 floor, mixed 5070+3060 -> fill")
    print("              the fastest card first (not a bandwidth-proportional split).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-file",
                    help=f"CSV capture: nvidia-smi --query-gpu={QUERY} --format=csv,noheader")
    ap.add_argument("--model-gib", type=float, default=0.0,
                    help="weight footprint in GiB; prints a placement plan for a "
                         "mixed-bandwidth box (see scripts/svmi-auto.py for the size)")
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
        proc = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader"],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            # nvidia-smi ships with the userspace package, so it exists even when
            # no kernel module is bound - which is exactly what a fresh box or a
            # post-kernel-upgrade box looks like.
            err = (proc.stderr or proc.stdout).strip().splitlines()
            print("nvidia-smi ran but could not talk to a driver:")
            for line in err[:3]:
                print(f"  {line}")
            print()
            print("The kernel module is not bound. Nothing CUDA works until it is -")
            print("check with:  lsmod | grep nvidia   and   inxi -G   (driver: N/A = unbound)")
            print("On a Blackwell card (RTX 50-series) the OPEN module is required;")
            print("the proprietary legacy module does not support GB20x at all.")
            return 1
        text = proc.stdout

    cards = parse_csv(text)
    if not cards:
        print("no GPUs parsed from nvidia-smi output")
        return 1
    report(cards, args.model_gib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
