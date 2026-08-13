#!/usr/bin/env python3
"""
svmi-auto — one command: what should THIS machine do with THIS model?

Reads the model (GGUF or --profile) and the GPU, decides the regime, and prints
ready-to-run llama.cpp flags plus the planner to consult for fine-tuning:

  resident   everything fits: plain flags, nothing clever needed
  offload    close: keep attention resident, stream the FFN tail (svmi-plan.py split)
  streamed   far: full SVMI weight streaming + BitSpec advice (svmi-plan / svmi-arbiter)
  fleet      many agents or long context: MAVM/CTX-VM planning (svmi-fleet.py)

Only flags that exist in this branch today are emitted (--stream-weights,
--stream-decode, -ngl, -ctk/-ctv); CTX-VM paging and expert lookahead are still
planner-level designs and are pointed at, not faked.

Usage:
  python3 scripts/svmi-auto.py model.gguf --gpu 3060
  python3 scripts/svmi-auto.py --profile 70b --gpu 3060 --ctx 32768
  python3 scripts/svmi-auto.py --profile 8b --gpu 1660ti --agents 8 --ctx 131072
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

GiB = 1024**3

# (n_layer, n_embd, n_head, n_head_kv, weights GiB Q4_K_M-class, params B) — svmi-fleet.py shapes
MODEL_PROFILES = {
    "7b":  (32, 4096, 32, 32,  3.9,  6.7),
    "8b":  (32, 4096, 32,  8,  4.6,  8.0),
    "13b": (40, 5120, 40, 40,  7.4, 13.0),
    "70b": (80, 8192, 64,  8, 39.6, 70.6),
}
# extreme low-bit requants that ship in this tree (tools/quantize):
# Q2_0 has CPU x86 AVX-VNNI + ARM NEON-DP dot kernels; Q1_0 adds repack paths
# and (Hopper, opt-in GGML_USE_HOPPER_Q1) a wgmma prefill kernel.
LOWBIT_BPW = {"Q2_0": 2.25, "Q1_0": 1.125}
# mixed INT8/Q4_K_M: Q4_K_M body, q8_0 attention (tools/quantize INT8).
# Attention is ~18% of the weights, so the mix costs ~15% size over Q4_K_M and
# keeps the hot, resident half of the model on the integer kernels.
# Q8_0 is full INT8: every 2D tensor, embeddings and head included. Biggest of the
# three, but nothing in the decode path leaves the integer kernels.
MIXED_BPW = {"INT8": 5.6, "Q8_0": 8.5}
GPU_PRESETS = {  # vram GiB, effective PCIe GB/s (card's own link generation)
    "1660ti": (6, 12.0), "2060": (6, 12.0), "2070": (8, 12.0), "2080": (8, 12.0),
    "2080ti": (11, 12.0),
    "3060": (12, 24.0), "3060ti": (8, 24.0), "3070": (8, 24.0), "3080": (10, 24.0),
    "3090": (24, 24.0),
    "4060": (8, 12.0), "4060ti": (16, 12.0),   # x8 gen4 links
    "4070": (12, 24.0), "4070ti": (12, 24.0), "4070tis": (16, 24.0),
    "4080": (16, 24.0), "4090": (24, 24.0),
    "5060ti": (16, 24.0), "5070": (12, 48.0), "5070ti": (16, 48.0),
    "5080": (16, 48.0), "5090": (32, 48.0),
    # CMP mining cards: cheap VRAM, but crippled PCIe makes them streaming-
    # hostile - use as RESIDENT shards / RPC workers only. See GPU_QUIRKS and
    # scripts/svmi-gpucheck.py for the per-card build flags they need.
    "cmp90hx": (10, 0.8), "cmp170hx": (8, 0.8), "cmp170hx-10g": (10, 0.8),
    # 170HX after the cmpunlocker HBM2e geometry unlock: the 8 GB variant comes back
    # as 64 GB, the 10 GB variant as 40 GB, and the link moves to PCIe gen2 (~2 GB/s).
    # At that size the card stops being a streaming problem and just holds the model.
    "cmp170hx-64": (64, 2.0), "cmp170hx-40": (40, 2.0),
    "cmp100-210": (16, 0.25),   # Volta GV100, 16 GB HBM2 @ ~830 GB/s, PCIe 1.0 x1
}
# firmware quirks that change how llama.cpp must be BUILT for a card
DP4A_QUIRK = "throttled dp4a — build -DGGML_CUDA_DISABLE_DP4A=ON (~2x, llama.cpp#24616)"
INT8_QUIRK = "no usable FP16 path - quantize INT8 so attention runs on q8_0/MMQ"
UNLOCK_QUIRK = ["cmpunlocker unlock is VOLATILE: a daemon rewrites it every second, and a "
                "driver reload drops the card back to its factory 8/10 GB",
                "link stays narrow (gen1 x4 ~1 GB/s, gen2 after unlock; the capacitor mod "
                "restores gen1 x16 ~4 GB/s) - that is the one-time model load, not per-token"]
GPU_QUIRKS = {
    "cmp90hx":  [DP4A_QUIRK, INT8_QUIRK,
                 "the 90HX unlock is compute-only - VRAM stays 10 GB, link unchanged"],
    "cmp170hx": [DP4A_QUIRK, INT8_QUIRK,
                 "8 GB stock: cmpunlocker restores HBM2e geometry to 64 GB - plan the "
                 "unlocked card with --gpu cmp170hx-64"],
    "cmp170hx-10g": [DP4A_QUIRK, INT8_QUIRK,
                     "10 GB stock: cmpunlocker restores HBM2e geometry to 40 GB - plan the "
                     "unlocked card with --gpu cmp170hx-40"],
    "cmp170hx-64": [DP4A_QUIRK, INT8_QUIRK] + UNLOCK_QUIRK,
    "cmp170hx-40": [DP4A_QUIRK, INT8_QUIRK] + UNLOCK_QUIRK,
    "cmp100-210": [
        "tensor cores firmware-gimped: FP16 (~5.6 TF) is SLOWER than FP32 (~10.6 TF)",
        "build -DGGML_CUDA_FORCE_MMQ=ON so decode stays on integer kernels, never cuBLAS FP16",
        "a Tesla V100 vBIOS flash does NOT restore PCIe width or tensor cores (board-level x1)",
        "but HBM2 bandwidth (~830 GB/s) is intact — decode is bandwidth-bound, so it is FAST",
        "A/B '-fa on' vs '-fa off' (flash-attn leans on FP16); if you drop FA you must also "
        "drop quantized V: use '-ctk q8_0 -ctv f16' since -ctv q8_0 requires -fa on",
    ],
}
PCIE_BW = {"3.0-x8": 6.0, "3.0-x16": 12.0, "4.0-x8": 12.0, "4.0-x16": 24.0, "5.0-x16": 48.0}
Q8_BPE = 1.0625


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or use --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    ap.add_argument("--gpu", required=True,
                    help="GPU preset, or comma list for mixed rigs (e.g. 3060ti,1660ti); "
                         "known: " + ",".join(sorted(GPU_PRESETS)))
    ap.add_argument("--gpus", type=int, default=1, help="number of identical GPUs (layer split)")
    ap.add_argument("--pcie", choices=sorted(PCIE_BW), default=None,
                    help="platform link override (old boards cap new cards, e.g. 3.0-x16)")
    ap.add_argument("--ctx", type=int, default=8192, help="context length you actually need")
    ap.add_argument("--agents", type=int, default=1, help="concurrent agents/slots planned")
    ap.add_argument("--display-reserve", type=float, default=1.0)
    ap.add_argument("--overhead", type=float, default=1.25, help="activation reserve GiB")
    ap.add_argument("--host-ram", type=float, default=64.0, help="host RAM GiB (for the -nkvo ceiling)")
    args = ap.parse_args()

    if args.model:
        sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
        from gguf import GGUFReader

        reader = GGUFReader(args.model)
        f = reader.get_field("general.architecture")
        arch = bytes(f.parts[f.data[0]]).decode("utf-8") if f else "?"

        def fi(key: str, default: int = 0) -> int:
            fld = reader.get_field(key)
            return int(fld.parts[fld.data[0]][0]) if fld is not None else default

        n_layer = fi(f"{arch}.block_count")
        n_embd = fi(f"{arch}.embedding_length")
        n_head = fi(f"{arch}.attention.head_count") or 1
        n_head_kv = fi(f"{arch}.attention.head_count_kv") or n_head
        weights = sum(int(t.n_bytes) for t in reader.tensors)
        n_params = sum(int(t.n_elements) for t in reader.tensors)
        name = Path(args.model).name
        model_arg = args.model
    elif args.profile:
        n_layer, n_embd, n_head, n_head_kv, w_gib, params_b = MODEL_PROFILES[args.profile]
        weights = int(w_gib * GiB)
        n_params = int(params_b * 1e9)
        name = f"{args.profile} (Q4_K_M-class profile)"
        model_arg = "model.gguf"
    else:
        ap.error("provide a GGUF path or --profile")

    gpu_names = [g.strip() for g in args.gpu.split(",") if g.strip()]
    for g in gpu_names:
        if g not in GPU_PRESETS:
            ap.error(f"unknown GPU '{g}'; known: {', '.join(sorted(GPU_PRESETS))}")
    if len(gpu_names) == 1 and args.gpus > 1:
        gpu_names = gpu_names * args.gpus
    cards = [GPU_PRESETS[g] for g in gpu_names]
    n_gpu = len(gpu_names)
    vram = sum(c[0] for c in cards)
    # platform may cap the card's link (old boards); multi-card desktop slots run
    # x8/x8 so uploads parallelize to ~one full link - use the slowest card's link
    link = min(c[1] for c in cards)
    pcie_agg = PCIE_BW[args.pcie] if args.pcie else link
    mixed = len(set(gpu_names)) > 1
    budget = (vram - args.display_reserve) * GiB
    head_dim = n_embd // n_head
    kv_tok = 2 * head_dim * n_head_kv * n_layer * Q8_BPE
    kv = kv_tok * args.ctx * args.agents
    fixed = args.overhead * GiB
    need = weights + kv + fixed
    fleet = args.agents > 1 or args.ctx > 32768

    print(f"model   : {name}  ({weights / GiB:.1f} GiB weights, {n_layer} layers)")
    gpu_lbl = "+".join(gpu_names) if mixed else (f"{n_gpu}x {gpu_names[0]}" if n_gpu > 1 else gpu_names[0])
    pcie_txt = f"{pcie_agg:.2f}" if pcie_agg < 1 else f"{pcie_agg:.0f}"
    print(f"gpu     : {gpu_lbl}  ({vram} GiB VRAM total, ~{pcie_txt} GB/s aggregate PCIe"
          + (", platform-capped" if args.pcie else "") + ")")
    quirked = [(g, q) for g in dict.fromkeys(gpu_names) for q in GPU_QUIRKS.get(g, [])]
    if quirked:
        for g, q in quirked:
            print(f"quirk   : [{g}] {q}")
        print("          verify the real link + flags on the box: python3 scripts/svmi-gpucheck.py")
    if any(g.startswith("cmp") for g in gpu_names):
        for q, bpw in MIXED_BPW.items():
            est = n_params * bpw / 8
            fits = "resident" if est + fixed <= budget else "does not fit resident"
            out = f"model-{q.lower()}.gguf"
            note = "mixed: q8_0 attention" if q == "INT8" else "full INT8: every 2D tensor"
            print(f"quant   : {q} ({bpw} bpw, {note}) ~{est / GiB:.1f} GiB - {fits}")
            print(f"          llama-quantize {model_arg} {out} {q}")
            if q == "INT8":
                # Q8_0 already stores the head at q8_0, so these overrides only
                # make sense for the mixed quant
                print("          richer head/embeddings (a few % more, the tensors "
                      "quantization hurts most):")
                print("          llama-quantize --output-tensor-type q8_0 "
                      "--token-embedding-type q6_K \\")
                print(f"              {model_arg} {out.replace('.gguf', '-max.gguf')} {q}")
                print("          calibrated: llama-imatrix -m f16.gguf -f calib.txt "
                      "-o imatrix.gguf, then add --imatrix imatrix.gguf")
        print("build   : cmake --preset cmp170hx-int8 && cmake --build build-cmp170hx-int8 -j")
        print("          (all matmuls on MMQ, no cuBLAS FP16; also cmp90hx-int8, cmp100-210-int8)")
    if mixed:
        print("warning : mixed cards - a layer split runs each token at the SLOWEST card's")
        print("          pace for its share. Prefer asymmetric roles (brain on the big card,")
        print("          worker/draft/embeddings on the small one) unless capacity forces a split.")
    print(f"need    : {weights / GiB:.1f} weights + {kv / GiB:.1f} KV (q8_0 @ {args.ctx:,} ctx "
          f"x {args.agents}) + {fixed / GiB:.1f} reserve = {need / GiB:.1f} GiB "
          f"vs {budget / GiB:.1f} budget\n")

    kvflags = "-fa on -ctk q8_0 -ctv q8_0"   # quantized V cache requires flash attention
    if n_gpu > 1:
        split = ",".join(str(c[0]) for c in cards)     # proportional to VRAM
        kvflags += f" --split-mode layer --tensor-split {split}"
    if need <= budget:
        print("regime  : RESIDENT — everything fits, no streaming needed")
        print(f"\n  llama-cli -m {model_arg} -ngl 999 {kvflags} -c {args.ctx}")
    elif weights * 0.55 + kv + fixed <= budget:
        print("regime  : OFFLOAD — attention + head fit; stream the FFN tail on demand")
        print(f"\n  llama-cli -m {model_arg} -ngl 999 --stream-weights 8 --stream-decode \\")
        print(f"            {kvflags} -c {args.ctx}")
        print("\n  exact resident/streamed split (-ot overrides), decode floor:")
        print(f"    python3 scripts/svmi-plan.py {model_arg} --gpu {gpu_names[0]} --ctx {args.ctx}")
    elif args.host_ram * GiB * 0.9 < weights:
        print("regime  : DOES NOT FIT — too big for VRAM, and host RAM "
              f"({args.host_ram:.0f} GiB) cannot pin {weights / GiB:.1f} GiB of weights for streaming")
        print("          options: a smaller model/quant that fits VRAM, or more host RAM.")
        print(f"          largest resident-friendly weights on this rig: ~{max(0.0, (budget - fixed) / GiB - 1.0):.1f} GiB + KV")
        for q, bpw in LOWBIT_BPW.items():
            est = n_params * bpw / 8
            fits = "fits resident" if est + kv + fixed <= budget else "still does not fit"
            print(f"          requant {q} ({bpw} bpw): ~{est / GiB:.1f} GiB — {fits}")
        print("          (Q2_0 is the sane floor for agent-quality output; Q1_0 is draft/experiment territory)")
    else:
        streamed = max(weights * 0.05, need - budget)   # bytes/pass that do not fit resident
        floor = pcie_agg * 1e9 / streamed
        print("regime  : STREAMED — the model mostly lives in pinned host RAM (SVMI)")
        print(f"          PCIe decode floor ~{floor:.1f} tok/s, ~{floor * 7:.0f} with BitSpec-class speculation")
        print(f"\n  llama-cli -m {model_arg} -ngl 999 --stream-weights 8 --stream-decode \\")
        print(f"            {kvflags} -c {args.ctx}")
        for q, bpw in LOWBIT_BPW.items():
            est = n_params * bpw / 8
            if est + kv + fixed <= budget:
                print(f"\n  or requant {q} ({bpw} bpw, ~{est / GiB:.1f} GiB) to go RESIDENT and skip PCIe entirely")
                print("  (Q2_0 is the sane floor for agent-quality output; Q1_0 is draft/experiment territory)")
                break
        print(f"\n  split + floor detail : python3 scripts/svmi-plan.py {model_arg} --gpu {gpu_names[0]}")
        prof = args.profile or "70b"
        print(f"  CPU-assisted decode  : python3 scripts/svmi-arbiter.py --profile {prof} "
              f"--gpu {gpu_names[0]} --cpu ddr5-2ch --cores 16")
    if fleet:
        print(f"\nfleet   : {args.agents} agent(s) @ {args.ctx:,} ctx — full-VRAM KV stops scaling here;")
        print("          plan capacity with CTX-VM paging (planner-level today, engine phase 5):")
        print(f"    python3 scripts/svmi-fleet.py {'--profile ' + args.profile if args.profile else model_arg} "
              f"--gpu {gpu_names[0]} --agents {args.agents} --ctx {args.ctx} \\")
        print("        --cold-kv-type q4_0 --landmarks pq16 --prefetch-hit 0.7")
    # --- max context on this rig, per engine-real KV mode ---
    # weights the KV must coexist with: full model when resident, else the
    # non-streamable floor (embeddings/norms/head, ~5%) since SVMI pages the rest
    w_gpu = weights if weights + fixed <= budget else weights * 0.05
    kv_room_vram = max(0.0, budget - w_gpu - fixed)
    host_room = max(0.0, args.host_ram * GiB * 0.85 - (weights if w_gpu < weights else 0))
    kv_tok1 = kv_tok / Q8_BPE            # per-token bytes at 1.0 bpe, single agent basis
    per = args.agents
    print("\nmax ctx : engine-real modes on this rig"
          + (f" ({args.agents} agents)" if per > 1 else "") + ":")
    for label, bpe, room, flags in (
        ("f16 KV, VRAM",  2.0,    kv_room_vram, "-fa on"),
        ("q8_0 KV, VRAM", 1.0625, kv_room_vram, "-fa on -ctk q8_0 -ctv q8_0"),
        ("q4_0 KV, VRAM", 0.5625, kv_room_vram, "-fa on -ctk q4_0 -ctv q4_0"),
        ("q4_0 KV, host RAM", 0.5625, host_room, "-fa on -ctk q4_0 -ctv q4_0 -nkvo (slower/step)"),
    ):
        cmax = int(room / (kv_tok1 * bpe * per))
        print(f"  {label:<18} ~{cmax:>9,} tok   {flags}")
    print("  q4_0 K rows lose fidelity vs q8_0 — claw most of it back (attention-invariant")
    print("  per-channel mean shift) with a one-time bias file:")
    print(f"    llama-kv-mean-center -m {model_arg} -f calib.txt -o kbias.gguf   # tools/kv-mean-center")
    print("    then add: --kv-mean-center kbias.gguf   (requires -ctk q4_0)")
    print("  beyond training ctx: --rope-scaling yarn --yarn-orig-ctx <train-ctx>; long")
    print("  sessions reuse the window via --cache-reuse / context shift; SWA models cap")
    print("  KV at their window (--swa-full disables). CTX-VM paging past all of these")
    print("  is planner-level today (svmi-fleet.py).")
    print("\nspeculate: llama-server now ships the DSpark block-diffusion drafter —")
    print("  build a drafter GGUF for your target family (conversion/dspark.py), then:")
    print("    llama-server ... --spec-type draft-dspark -md drafter.gguf --spec-draft-n-max <block>")
    print("  (server engages target-layer capture automatically; <block> must equal the")
    print("  drafter's block_size. CLI llama-cli does not engage capture yet.)")

    if n_gpu > 1:
        print("\nmulti-gpu: layer split shown (safest); 2080 Ti-class pairs with an NVLink")
        print("bridge can try --split-mode row for tensor parallelism on the resident share.")
    print("\nmore VRAM on another machine? combine rigs over the LAN (RPC layer split):")
    print("    python3 scripts/svmi-net.py " + (f"--profile {args.profile}" if args.profile else model_arg)
          + f" --node {','.join(gpu_names)}:ram={args.host_ram:.0f} --node <other-rig-gpus>:ram=<GiB> --nic 1gbe")
    print("\nverify any streamed setup with scripts/svmi-verify.sh (token identity) before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
