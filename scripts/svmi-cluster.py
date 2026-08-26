#!/usr/bin/env python3
"""
svmi-cluster — how do N nodes reach a target tokens/second, and in what topology?

svmi-net.py answers "does this model FIT across these machines" (one model, layer
split over RPC). This answers the other question: given a throughput target and a
context depth, should the nodes be independent REPLICAS or pipeline GROUPS, and
does the plan actually reach the number?

The decision is not a preference, it falls out of two ratios:

  weights vs one node's VRAM   ->  if the model fits on one card, splitting it
                                   across nodes buys nothing and costs a network
                                   hop per token per boundary. Replicate instead.
  KV at target ctx vs VRAM     ->  at deep context KV dwarfs the weights, and THAT
                                   is what forces a split (or KV offload to host
                                   RAM, which these nodes have a lot of).

Decode is memory-bandwidth bound, so per-node throughput at B concurrent slots is

    tok/s  =  B * HBM_BW / (weights_bytes + B * kv_bytes_touched_per_seq)

- weights are read once per step no matter how many slots share it, which is why
  batching is the whole game on a high-bandwidth card.

Usage:
  python3 scripts/svmi-cluster.py --profile 27b --nodes 10 --gpu cmp170hx-40 \\
      --ctx 262144 --slots 16 --target 600
  python3 scripts/svmi-cluster.py model.gguf --nodes 10 --gpu cmp170hx-40 --ctx 1048576
  python3 scripts/svmi-cluster.py --self-test
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

GiB = 1024**3

_spec = importlib.util.spec_from_file_location("svmi_auto", Path(__file__).parent / "svmi-auto.py")
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MODEL_PROFILES = _mod.MODEL_PROFILES
GPU_PRESETS = _mod.GPU_PRESETS

# HBM/GDDR read bandwidth GB/s - what actually bounds decode once resident
HBM_BW = {
    "cmp170hx": 1490.0, "cmp170hx-10g": 1560.0,
    "cmp170hx-64": 1490.0, "cmp170hx-40": 1560.0,
    "cmp90hx": 760.0, "cmp100-210": 830.0,
    "3090": 936.0, "4090": 1008.0, "5090": 1792.0, "3060": 360.0,
    # GDDR6X/GDDR7 consumer parts: bus_width x data_rate / 8
    "3080": 760.0,                                   # 320-bit GDDR6X @ 19 Gbps
    "4070": 504.0, "4080": 717.0,                    # 192/256-bit GDDR6X
    "5060ti": 448.0,                                 # 128-bit GDDR7 @ 28 Gbps
    "5070": 672.0,                                   # 192-bit GDDR7 @ 28 Gbps
    "5070ti": 896.0, "5080": 960.0,                  # 256-bit GDDR7 @ 28/30 Gbps
}
DEFAULT_BW = 500.0
KV_BPE = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
# fraction of theoretical bandwidth a real decode loop sustains. This is a
# PLANNING ASSUMPTION - run scripts/svmi-bwprofile.py to replace it with a
# measurement of the actual box, which the loader below picks up automatically.
BW_EFFICIENCY = 0.65


def measured_efficiency(gpu: str) -> tuple[float, bool]:
    """(efficiency, measured?) - a benched profile for this GPU wins over the guess"""
    import json
    import os
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    path = Path(base) / "tenselerate" / f"bwprofile-{gpu}.json"
    if path.is_file():
        try:
            eff = json.loads(path.read_text()).get("bw_efficiency")
            if isinstance(eff, (int, float)) and 0.0 < eff <= 1.0:
                return float(eff), True
        except (OSError, ValueError):
            pass
    return BW_EFFICIENCY, False


def model_shape(args, ap):
    if args.model:
        sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
        from gguf import GGUFReader

        r = GGUFReader(args.model)
        f = r.get_field("general.architecture")
        arch = bytes(f.parts[f.data[0]]).decode("utf-8") if f else "?"

        def fi(key, default=0):
            fld = r.get_field(key)
            return int(fld.parts[fld.data[0]][0]) if fld is not None else default

        n_layer = fi(f"{arch}.block_count")
        n_embd = fi(f"{arch}.embedding_length")
        n_head = fi(f"{arch}.attention.head_count") or 1
        n_head_kv = fi(f"{arch}.attention.head_count_kv") or n_head
        weights = sum(int(t.n_bytes) for t in r.tensors)
        # hybrid (SSM + attention) models keep a KV cache only on the full-attention
        # layers; the rest carry a fixed-size recurrent state. Resolve it exactly the
        # way src/models/qwen35.cpp does: the explicit per-layer array wins, then the
        # interval, and the engine's own fallback is 4 - not 1. Reading only the
        # interval silently reverts to the dense assumption on a model that ships
        # the array, which is the whole error this is here to avoid.
        n_attn = 0
        fld = r.get_field(f"{arch}.attention.recurrent_layers")
        if fld is not None:
            recr = [bool(fld.parts[i][0]) for i in fld.data]
            n_attn = sum(1 for i, v in enumerate(recr[:n_layer]) if not v)
        if n_attn == 0:
            interval = fi(f"{arch}.full_attention_interval")
            if interval == 0:
                interval = (args.full_attn_interval if args.full_attn_interval > 1
                            else (4 if str(arch).startswith("qwen3") else 1))
            n_attn = max(1, n_layer // max(1, interval))
        return n_layer, n_embd, n_head, n_head_kv, weights, Path(args.model).name, n_attn
    if not args.profile:
        ap.error("provide a GGUF path or --profile")
    n_layer, n_embd, n_head, n_head_kv, w_gib, params_b = MODEL_PROFILES[args.profile]
    weights = int(params_b * 1e9 * args.bpw / 8)
    n_attn = max(1, n_layer // max(1, args.full_attn_interval))
    return (n_layer, n_embd, n_head, n_head_kv, weights,
            f"{args.profile} @ {args.bpw} bpw", n_attn)


def node_throughput(weights_b: float, kv_per_seq_b: float, bw_gbs: float, slots: int,
                    efficiency: float = BW_EFFICIENCY) -> float:
    """tok/s for one node at `slots` concurrent sequences (bandwidth-bound)"""
    bytes_per_step = weights_b + slots * kv_per_seq_b
    return slots * (bw_gbs * 1e9 * efficiency) / bytes_per_step


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    ap.add_argument("--bpw", type=float, default=5.6,
                    help="bits/weight when using --profile (5.6 = INT8 mixed, 8.5 = Q8_0)")
    ap.add_argument("--nodes", type=int, default=10)
    ap.add_argument("--gpu", default="cmp170hx-40", help="GPU preset per node")
    ap.add_argument("--host-ram", type=float, default=128.0, help="GiB per node")
    ap.add_argument("--ctx", type=int, default=262144)
    ap.add_argument("--slots", type=int, default=16, help="concurrent sequences per node")
    ap.add_argument("--kv-type", choices=sorted(KV_BPE), default="q8_0")
    ap.add_argument("--target", type=float, default=0.0, help="aggregate tok/s to hit")
    ap.add_argument("--overhead", type=float, default=2.0, help="activation reserve GiB/node")
    ap.add_argument("--spec-gain", type=float, default=1.0,
                    help="MEASURED end-to-end speedup from MTP speculation, not the raw "
                         "acceptance rate. 1.0 = off. Anchor: a 27B Q4_K_S on a 5090 goes "
                         "75 -> 90 tok/s with 2-token MTP at 60% acceptance, i.e. x1.20 - "
                         "the verify pass is not free, so accepted-tokens-per-pass (x1.6 "
                         "there) badly overstates it. Vendor figures of 2-3x are document "
                         "throughput under their own conditions, not this ratio")
    ap.add_argument("--full-attn-interval", type=int, default=1,
                    help="hybrid models keep KV on 1 layer in N (Qwen3.5/3.6, Qwen3-Next: "
                         "typically 4); 1 = every layer, i.e. a dense model. Read from the "
                         "GGUF when one is given")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.gpu not in GPU_PRESETS:
        ap.error(f"unknown GPU '{args.gpu}'; known: {', '.join(sorted(GPU_PRESETS))}")

    n_layer, n_embd, n_head, n_head_kv, weights, name, n_attn_layer = model_shape(args, ap)
    vram = GPU_PRESETS[args.gpu][0] * GiB
    bw = HBM_BW.get(args.gpu, DEFAULT_BW)
    eff, eff_measured = measured_efficiency(args.gpu)
    head_dim = n_embd // n_head
    kv_tok = 2 * head_dim * n_head_kv * n_attn_layer * KV_BPE[args.kv_type]
    kv_seq = kv_tok * args.ctx
    fixed = args.overhead * GiB

    print(f"model   : {name}  ({weights / GiB:.1f} GiB weights, {n_layer} layers)")
    print(f"nodes   : {args.nodes}x {args.gpu}  ({vram / GiB:.0f} GiB VRAM, "
          f"{args.host_ram:.0f} GiB RAM, ~{bw:.0f} GB/s HBM each)")
    if n_attn_layer < n_layer:
        print(f"hybrid  : {n_attn_layer} of {n_layer} layers hold KV; the rest carry a "
              "fixed-size recurrent state")
    print(f"context : {args.ctx:,} tok, KV {args.kv_type} = {kv_tok / 1024:.0f} KiB/tok "
          f"-> {kv_seq / GiB:.1f} GiB per sequence\n")

    # --- how many sequences fit one node, and is the model itself the problem?
    room = vram - weights - fixed
    slots_fit = int(room // kv_seq) if kv_seq > 0 else 0
    weights_fit = weights + fixed < vram

    if not weights_fit:
        need = weights + fixed
        groups = int(-(-need // (vram - fixed)))
        print(f"topology: PIPELINE - weights alone ({weights / GiB:.1f} GiB) exceed one node; "
              f"split over {groups} nodes per replica")
        print("          layer split over RPC, see svmi-net.py for the exact commands")
        replicas = args.nodes // max(1, groups)
    elif slots_fit >= 1:
        print(f"topology: REPLICAS - the model fits one node ({weights / GiB:.1f} GiB of "
              f"{vram / GiB:.0f} GiB), so every node runs a full copy")
        print("          splitting it across nodes would add a network hop per token "
              "and buy nothing")
        groups, replicas = 1, args.nodes
    else:
        # weights fit, KV does not - THIS is what forces a split at deep context
        per_node_kv = room
        groups = int(-(-kv_seq // per_node_kv)) if per_node_kv > 0 else args.nodes
        print(f"topology: PIPELINE (KV-bound) - the weights fit one node but a single "
              f"{args.ctx:,}-token sequence needs {kv_seq / GiB:.1f} GiB of KV vs "
              f"{room / GiB:.1f} GiB free")
        print(f"          -> {groups} nodes per replica just to hold one context")
        print(f"          cheaper alternative: -nkvo puts KV in host RAM "
              f"({args.host_ram:.0f} GiB/node), slower per token but no extra nodes")
        replicas = args.nodes // max(1, groups)

    eff_slots = max(1, min(args.slots, slots_fit)) if slots_fit else 1
    if slots_fit and args.slots > slots_fit:
        print(f"          note: asked for {args.slots} slots, only {slots_fit} fit in VRAM "
              f"at this context")

    # --- throughput
    print()
    per_node = node_throughput(weights, kv_seq, bw, eff_slots, eff) * args.spec_gain
    single = node_throughput(weights, kv_seq, bw, 1, eff) * args.spec_gain
    print(f"bandwidth: {bw:.0f} GB/s x {eff * 100:.0f}% = {bw * eff:.0f} GB/s effective "
          + ("(MEASURED on this box)" if eff_measured
             else "(assumed - run scripts/svmi-bwprofile.py to measure)"))
    aggregate = per_node * replicas / max(1, groups) if groups > 1 else per_node * replicas
    if args.spec_gain != 1.0:
        print(f"spec    : MTP x{args.spec_gain:.2f} accepted/pass - costs no KV, so it is the "
              "lever that survives deep context")
    print(f"decode  : {single:6.1f} tok/s single-stream per node "
          f"(bandwidth roof {bw * BW_EFFICIENCY / (weights / 1e9):.0f} at zero KV)")
    print(f"          {per_node:6.1f} tok/s per node at {eff_slots} slots")
    print(f"          {aggregate:6.1f} tok/s aggregate over {replicas} replica(s)")
    if args.target:
        verdict = "MEETS" if aggregate >= args.target else "MISSES"
        print(f"target  : {args.target:.0f} tok/s -> {verdict}")
        if aggregate < args.target:
            need_slots = eff_slots
            while need_slots < 256 and node_throughput(weights, kv_seq, bw, need_slots, eff) \
                    * replicas * args.spec_gain < args.target:
                need_slots *= 2
            print(f"          would need ~{need_slots} slots/node (VRAM permitting), "
                  f"a smaller KV type, or more nodes")

    # --- commands
    print("\nrun     :")
    m = args.model or "model-int8.gguf"
    if groups == 1:
        print("  # one of these per node, behind any HTTP load balancer")
        print(f"  GGML_CUDA_NO_MMVQ=1 llama-server -m {m} -ngl 999 \\")
        print(f"      -c {args.ctx} -np {eff_slots} -cb -fa on "
              f"-ctk {args.kv_type} -ctv {args.kv_type} \\")
        print("      --host 0.0.0.0 --port 8080")
    else:
        print(f"  # {groups} nodes per replica: workers run the RPC server,")
        print("  ggml-rpc-server -H 0.0.0.0 -p 50052        # on each worker")
        print(f"  GGML_CUDA_NO_MMVQ=1 llama-server -m {m} -ngl 999 \\")
        print("      --rpc worker1:50052,worker2:50052 \\")
        print(f"      -c {args.ctx} -np {eff_slots} -cb -fa on "
              f"-ctk {args.kv_type} -ctv {args.kv_type}")
        print("  # sizing + per-device tensor split: scripts/svmi-net.py")
    print("  build   : cmake --preset cmp170hx-int8   (all-integer, MMQ)")
    print("  A/B     : scripts/svmi-cmpbench.sh -m " + m + "   # is NO_MMVQ actually helping?")
    return 0


def self_test() -> int:
    # 1. batching model: weights dominate at low slot counts, KV at high ones
    w, kv, bw = 20e9, 1e9, 1000.0
    t1 = node_throughput(w, kv, bw, 1)
    t8 = node_throughput(w, kv, bw, 8)
    assert t8 > t1 * 4, (t1, t8)          # near-linear while weights dominate
    t256 = node_throughput(w, kv, bw, 256)
    assert t256 < 256 * t1, (t1, t256)    # sublinear once KV dominates
    # 2. a KV read that dwarfs the weights caps throughput near the KV roof
    tk = node_throughput(1e9, 100e9, bw, 4)
    assert tk < node_throughput(1e9, 1e9, bw, 4)
    # 3. profile shapes are present for the models the docs reference
    for p in ("27b", "70b"):
        assert p in MODEL_PROFILES, p
    # 4. anchor the bandwidth model to a published measurement: a 27B Q4_K_S on an
    #    RTX 5090 (1792 GB/s) decodes at 75 tok/s single-stream. Predicted throughput
    #    must land within 10% of that, or BW_EFFICIENCY has drifted from reality.
    w_q4ks = 27e9 * 4.37 / 8
    pred = node_throughput(w_q4ks, 0.0, 1792.0, 1, BW_EFFICIENCY)
    assert abs(pred - 75.0) / 75.0 < 0.10, f"bandwidth model off the 5090 anchor: {pred:.1f}"
    print("self-test OK: batching curve (near-linear then KV-bound), KV-dominated cap,")
    print(f"              27b/70b profiles, 5090 anchor {pred:.0f} vs 75 tok/s measured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
