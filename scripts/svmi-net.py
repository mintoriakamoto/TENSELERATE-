#!/usr/bin/env python3
"""
svmi-net — distributed network inference planner: combine VRAM across machines.

llama.cpp's RPC backend (built into this repo's release binaries, GGML_RPC=ON)
lets one main host drive GPU devices on other machines over TCP: run
`ggml-rpc-server` on each worker, pass `--rpc host:port,...` on the main host,
and the scheduler layer-splits the model across ALL devices, local + remote.
This planner does the sizing and prints the exact commands:

  * does the model + KV fit the COMBINED VRAM of every node?
  * per-device --tensor-split proportions (local devices first, then RPC
    devices in --rpc order)
  * the network decode/prefill overhead of each node boundary (per token,
    one activation row of n_embd floats crosses each boundary — decode is
    RTT-bound, prefill is bandwidth-bound)

Usage (node = one machine; first --node is the MAIN host, workers follow):
  python3 scripts/svmi-net.py --profile 70b \\
      --node 2080ti,2080ti:ram=8 --node 3060ti,1660ti:ram=64 --nic 1gbe
  python3 scripts/svmi-net.py model.gguf --node 3090 --node 3060 --nic 2.5gbe

Security note (from tools/rpc): the RPC protocol is unauthenticated —
run it ONLY on a trusted LAN, never on an open network.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

GiB = 1024**3

# reuse svmi-auto's tables (dash in the filename forces a spec-based import)
_spec = importlib.util.spec_from_file_location("svmi_auto", Path(__file__).parent / "svmi-auto.py")
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MODEL_PROFILES = _mod.MODEL_PROFILES
GPU_PRESETS = _mod.GPU_PRESETS
Q8_BPE = _mod.Q8_BPE

# effective GB/s on the wire (payload after TCP/IP overhead) and typical LAN RTT
NIC_PRESETS = {
    "1gbe":   (0.112, 0.00035),
    "2.5gbe": (0.28,  0.00030),
    "5gbe":   (0.56,  0.00025),
    "10gbe":  (1.12,  0.00020),
    "wifi6":  (0.09,  0.00250),
    "tb4":    (2.50,  0.00010),  # thunderbolt/usb4 networking
}


def parse_node(spec: str) -> tuple[list[str], float]:
    """'2080ti,2080ti:ram=8' -> (['2080ti','2080ti'], 8.0)"""
    ram = 16.0
    parts = spec.split(":")
    gpus = [g.strip() for g in parts[0].split(",") if g.strip()]
    for p in parts[1:]:
        if p.startswith("ram="):
            ram = float(p[4:])
    for g in gpus:
        if g not in GPU_PRESETS:
            raise SystemExit(f"unknown GPU '{g}'; known: {', '.join(sorted(GPU_PRESETS))}")
    return gpus, ram


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or use --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    ap.add_argument("--node", action="append", required=True, metavar="GPUS[:ram=GiB]",
                    help="one machine: comma list of GPU presets, optional :ram=GiB "
                         "(first --node is the main host; repeat per machine)")
    ap.add_argument("--nic", choices=sorted(NIC_PRESETS), default="1gbe",
                    help="slowest link between the nodes (default 1gbe)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--n-batch", type=int, default=512, help="prefill batch (network sizing)")
    ap.add_argument("--display-reserve", type=float, default=1.0, help="GiB held back on the main host GPU 0")
    ap.add_argument("--overhead", type=float, default=1.25, help="activation reserve GiB per node")
    ap.add_argument("--port-base", type=int, default=50052)
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
        name = Path(args.model).name
        model_arg = args.model
    elif args.profile:
        n_layer, n_embd, n_head, n_head_kv, w_gib, _params_b = MODEL_PROFILES[args.profile]
        weights = int(w_gib * GiB)
        name = f"{args.profile} (Q4_K_M-class profile)"
        model_arg = "model.gguf"
    else:
        ap.error("provide a GGUF path or --profile")

    nodes = [parse_node(s) for s in args.node]
    if len(nodes) < 2:
        ap.error("need at least two --node entries (main host + 1 worker) — "
                 "for a single machine use svmi-auto.py")

    net_bw, net_rtt = NIC_PRESETS[args.nic]

    # per-device VRAM budget; display reserve only on the main host's first GPU
    devices = []   # (node_idx, gpu_name, vram_budget_bytes)
    for ni, (gpus, _ram) in enumerate(nodes):
        for gi, g in enumerate(gpus):
            vram = GPU_PRESETS[g][0]
            if ni == 0 and gi == 0:
                vram -= args.display_reserve
            devices.append((ni, g, max(0.0, vram) * GiB))

    vram_total = sum(d[2] for d in devices)
    head_dim = n_embd // n_head
    kv_tok = 2 * head_dim * n_head_kv * n_layer * Q8_BPE
    kv = kv_tok * args.ctx
    fixed = args.overhead * GiB * len(nodes)
    need = weights + kv + fixed

    print(f"model   : {name}  ({weights / GiB:.1f} GiB weights, {n_layer} layers, n_embd {n_embd})")
    for ni, (gpus, ram) in enumerate(nodes):
        role = "main" if ni == 0 else f"worker{ni}"
        v = sum(GPU_PRESETS[g][0] for g in gpus)
        print(f"node {ni}  : [{role}] {'+'.join(gpus)}  ({v} GiB VRAM, {ram:.0f} GiB RAM)")
    print(f"network : {args.nic}  (~{net_bw * 1e3:.0f} MB/s effective, ~{net_rtt * 1e3:.2f} ms RTT)")
    print(f"need    : {weights / GiB:.1f} weights + {kv / GiB:.1f} KV (q8_0 @ {args.ctx:,}) "
          f"+ {fixed / GiB:.1f} reserve = {need / GiB:.1f} GiB vs {vram_total / GiB:.1f} GiB combined\n")

    # ---- network cost of the layer-split pipeline -------------------------
    # decode: one f32 activation row (n_embd * 4 B) crosses each node boundary
    # per token; each crossing pays RTT + payload/bw.
    n_bound = len(nodes) - 1
    act = n_embd * 4
    per_tok = n_bound * (net_rtt + act / (net_bw * 1e9))
    net_cap = 1.0 / per_tok if per_tok > 0 else float("inf")
    # prefill: n_batch rows cross per boundary per ubatch
    pre_bytes = act * args.n_batch
    pre_s = n_bound * (net_rtt + pre_bytes / (net_bw * 1e9))
    print(f"net cost: {n_bound} node boundary(ies); decode +{per_tok * 1e3:.2f} ms/tok "
          f"(caps at ~{net_cap:,.0f} tok/s — {'negligible' if net_cap > 200 else 'SIGNIFICANT'} for this class)")
    print(f"          prefill +{pre_s * 1e3:.0f} ms per {args.n_batch}-token batch "
          f"({pre_bytes / 1e6:.1f} MB/boundary)")
    print("          model load ships weight shards once over the same link "
          f"(~{weights / (net_bw * 1e9) / 60:.1f} min at {args.nic}; use rpc-server -c to cache on workers)\n")

    if need > vram_total:
        print("verdict : DOES NOT FIT the combined VRAM — options:")
        print("          - smaller quant (see svmi-auto.py requant lines: Q2_0 / Q1_0)")
        print("          - fewer ctx / q4_0 KV (-ctk q4_0 -ctv q4_0 --kv-mean-center kbias.gguf)")
        print("          - keep the overflow streamed from the MAIN host's RAM (SVMI):")
        print("            add --stream-weights 8 --stream-decode on the main command below")
        print("            (RPC workers hold their shard resident; only the local shard streams)")
        main_ram = nodes[0][1]
        best_ram = max(r for _, r in nodes)
        local_share = weights * (sum(d[2] for d in devices if d[0] == 0) / vram_total)
        if main_ram * GiB * 0.9 < local_share and best_ram > main_ram:
            print(f"            NOTE: main host has only {main_ram:.0f} GiB RAM — too small to pin its")
            print(f"            ~{local_share / GiB:.1f} GiB streamed shard. Make the {best_ram:.0f} GiB-RAM node the")
            print("            main host (reorder --node) so the overflow streams from big RAM.")
        print()
    else:
        print("verdict : FITS the combined VRAM — fully resident distributed split\n")

    # ---- commands ----------------------------------------------------------
    print("commands (trusted LAN only — the RPC protocol is unauthenticated!):\n")
    rpc_list = []
    port = args.port_base
    for ni in range(1, len(nodes)):
        gpus, _ = nodes[ni]
        print(f"  # worker{ni} ({'+'.join(gpus)}) — release binaries already include this")
        print(f"  ggml-rpc-server -H 0.0.0.0 -p {port} -c")
        rpc_list.append(f"<worker{ni}-ip>:{port}")
        port += 1
    split = ",".join(str(round(d[2] / GiB, 1)) for d in devices)
    print("\n  # main host")
    print(f"  llama-cli -m {model_arg} --rpc {','.join(rpc_list)} -ngl 999 \\")
    print(f"            --split-mode layer --tensor-split {split} \\")
    print(f"            -fa on -ctk q8_0 -ctv q8_0 -c {args.ctx}")
    print("\n  # --tensor-split order = local devices first, then RPC devices in --rpc order;")
    print("  #   proportions above mirror each device's VRAM. Verify placement with -v.")
    print("  # llama-server takes the same --rpc/--tensor-split flags for an API endpoint.")
    print("\nnotes   : put the SLOWEST card last in the pipeline (fewer layers land on it);")
    print("          decode crosses the network once per boundary per token, so 2 nodes on")
    print("          gigabit is usually compute-bound, not network-bound — but prefill of")
    print("          long prompts feels the link. Wire > WiFi, always.")
    print("\ngo further:")
    print(f"  block speculation over this split : python3 scripts/svmi-distspec.py --nodes {len(nodes)} --nic {args.nic}")
    print("  dropout-tolerant weight swarm     : python3 scripts/svmi-swarm.py --weights-gib "
          f"{weights / GiB:.1f} --nic {args.nic}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
