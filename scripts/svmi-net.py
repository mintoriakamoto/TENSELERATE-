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
import re
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
MACHINES = _mod.MACHINES
Q8_BPE = _mod.Q8_BPE

# VRAM read bandwidth, for "what would this site do on its own?"
_cspec = importlib.util.spec_from_file_location("svmi_cluster", Path(__file__).parent / "svmi-cluster.py")
assert _cspec is not None and _cspec.loader is not None
_cmod = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(_cmod)
HBM_BW = _cmod.HBM_BW
DEFAULT_BW = _cmod.DEFAULT_BW
BW_EFFICIENCY = _cmod.BW_EFFICIENCY

# effective GB/s on the wire (payload after TCP/IP overhead) and round-trip time.
NIC_PRESETS = {
    "1gbe":   (0.112, 0.00035),
    "2.5gbe": (0.28,  0.00030),
    "5gbe":   (0.56,  0.00025),
    "10gbe":  (1.12,  0.00020),
    "wifi6":  (0.09,  0.00250),
    "tb4":    (2.50,  0.00010),  # thunderbolt/usb4 networking
    # WAN links between separate SITES. Bandwidth is almost irrelevant here -
    # one decode activation row is a few KB - and RTT is everything, because a
    # layer split pays a full round trip per token per node boundary. Speed of
    # light in fibre is ~200,000 km/s, so ~1 ms per 100 km each way BEFORE any
    # router hop; these are typical measured figures, not the physical floor.
    "wan-metro":       (0.30, 0.010),   # same city
    "wan-regional":    (0.30, 0.030),   # ~1000 km
    "wan-continental": (0.30, 0.070),   # coast to coast
}
# above this RTT the link is not a LAN and an RPC layer split is the wrong shape
WAN_RTT_FLOOR = 0.005


def site_standalone_tok_s(gpus: list[str], weights_bytes: float) -> tuple[float, float]:
    """
    (tok/s, VRAM GiB) this site would reach running the whole model ALONE.

    Decode is bandwidth-bound once resident, and under a layer split a token
    walks every card in order, so the site's effective read rate is the
    capacity-weighted harmonic mean of its cards' bandwidths - not their sum.
    """
    vram = sum(GPU_PRESETS[g][0] for g in gpus)
    if vram <= 0:
        return 0.0, 0.0
    # time to read `share` of the weights off each card, filling fastest first
    ordered = sorted(gpus, key=lambda g: HBM_BW.get(g, DEFAULT_BW), reverse=True)
    remaining = weights_bytes / GiB
    seconds = 0.0
    for g in ordered:
        take = min(GPU_PRESETS[g][0], remaining)
        bw = HBM_BW.get(g, DEFAULT_BW) * BW_EFFICIENCY
        seconds += (take * 1.074) / bw          # GiB -> GB, then / (GB/s)
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0 or seconds <= 0:
        return 0.0, vram                        # does not fit on this site
    return 1.0 / seconds, vram


def parse_node(spec: str) -> tuple[list[str], float]:
    """
    '2080ti,2080ti:ram=8' -> (['2080ti','2080ti'], 8.0)

    A bare machine name from MACHINES also works ('fallen', 'cmp-rig'), with an
    explicit ':ram=' still winning over the machine's recorded RAM.
    """
    ram = 16.0
    parts = spec.split(":")
    head = parts[0].strip()
    if head in MACHINES:
        m_gpus, m_ram, _desc = MACHINES[head]
        gpus, ram = list(m_gpus), m_ram
        for p in parts[1:]:
            if p.startswith("ram="):
                ram = float(p[4:])
        return gpus, ram
    gpus = [g.strip() for g in parts[0].split(",") if g.strip()]
    for p in parts[1:]:
        if p.startswith("ram="):
            ram = float(p[4:])
    for g in gpus:
        if g not in GPU_PRESETS:
            raise SystemExit(f"unknown GPU '{g}'; known: {', '.join(sorted(GPU_PRESETS))}")
    return gpus, ram


def self_test() -> int:
    import io as _io
    from contextlib import redirect_stdout

    def run(argv: list[str]) -> str:
        buf = _io.StringIO()
        argv_save = sys.argv
        sys.argv = ["svmi-net.py"] + argv
        try:
            with redirect_stdout(buf):
                main()
        finally:
            sys.argv = argv_save
        return buf.getvalue()

    # 1. machine names resolve, and ':ram=' still overrides the recorded RAM
    for mname, (gpus, ram, _d) in MACHINES.items():
        assert parse_node(mname) == (list(gpus), ram), mname
        for g in gpus:
            assert g in GPU_PRESETS, (mname, g)
    assert parse_node("fallen:ram=96")[1] == 96.0
    assert parse_node("3090,3060:ram=32") == (["3090", "3060"], 32.0)

    # 2. every WAN preset is above the floor, every LAN preset below it
    for nic, (_bw, rtt) in NIC_PRESETS.items():
        wan = nic.startswith("wan-")
        assert (rtt >= WAN_RTT_FLOOR) == wan, (nic, rtt)

    # 3. standalone throughput: fills the fastest card first, and reports
    #    0 tok/s when the model cannot fit on that site at all
    solo_cmp, vram_cmp = site_standalone_tok_s(["cmp170hx-64"], 15.4 * GiB)
    solo_mix, vram_mix = site_standalone_tok_s(["5070", "3060"], 15.4 * GiB)
    assert vram_cmp == 64 and vram_mix == 24, (vram_cmp, vram_mix)
    assert solo_cmp > solo_mix > 0, (solo_cmp, solo_mix)
    #    15.4 GiB fits inside the 5070's 12 GiB + 3.4 on the 3060, so the mixed
    #    site must be SLOWER than an imaginary all-5070 site of the same size
    solo_fast, _ = site_standalone_tok_s(["5070", "5070"], 15.4 * GiB)
    assert solo_fast > solo_mix, (solo_fast, solo_mix)
    assert site_standalone_tok_s(["3060"], 40.0 * GiB) == (0.0, 12), "overflow -> 0 tok/s"

    # 4. a WAN link refuses to print a tensor-split, and says why
    wan_out = run(["--profile", "27b", "--node", "cmp-rig", "--node", "fallen",
                   "--nic", "wan-continental"])
    assert "SEPARATE SITES" in wan_out, wan_out
    #    match an emitted split ("--tensor-split 23.0,12.0"), not the prose that
    #    explains one is deliberately withheld
    assert not re.search(r"--tensor-split\s+[\d.]", wan_out), "WAN must not emit a split"
    assert "ggml-rpc-server" not in wan_out, "WAN must not emit worker commands"
    assert "unauthenticated" in wan_out and "round trip" in wan_out
    assert "FITS the combined VRAM" not in wan_out, "combined VRAM is not a pool over a WAN"
    #    the replica commands it prints must be runnable: profile KEY, not the
    #    display name ("27b", never "27b (Q4_K_M-class profile)")
    assert "--profile 27b --gpu" in wan_out, wan_out
    assert "(Q4_K_M-class profile)" not in wan_out.split("What to do instead")[1]

    # 5. --force-rpc emits commands, bound to a tunnel rather than 0.0.0.0
    forced = run(["--profile", "27b", "--node", "cmp-rig", "--node", "fallen",
                  "--nic", "wan-continental", "--force-rpc"])
    assert "ggml-rpc-server -H <tunnel-ip>" in forced, forced
    assert "-H 0.0.0.0" not in forced, "must not tell a WAN worker to bind 0.0.0.0"

    # 6. LAN behaviour is untouched: split + workers on 0.0.0.0
    lan = run(["--profile", "70b", "--node", "3090", "--node", "3060", "--nic", "1gbe"])
    assert "SEPARATE SITES" not in lan
    assert re.search(r"--tensor-split\s+[\d.]", lan), lan
    assert "ggml-rpc-server -H 0.0.0.0" in lan

    print("self-test OK: machine names resolve (+ram= override); WAN/LAN preset split")
    print("              matches WAN_RTT_FLOOR; standalone tok/s fills fastest-first and")
    print("              returns 0 on overflow; a WAN prints no tensor-split and no worker")
    print("              commands; --force-rpc binds a tunnel, never 0.0.0.0; LAN unchanged.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or use --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    ap.add_argument("--node", action="append", metavar="GPUS[:ram=GiB]",
                    help="one machine: comma list of GPU presets, optional :ram=GiB "
                         "(first --node is the main host; repeat per machine)")
    ap.add_argument("--nic", choices=sorted(NIC_PRESETS), default="1gbe",
                    help="slowest link between the nodes (default 1gbe)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--n-batch", type=int, default=512, help="prefill batch (network sizing)")
    ap.add_argument("--display-reserve", type=float, default=1.0, help="GiB held back on the main host GPU 0")
    ap.add_argument("--overhead", type=float, default=1.25, help="activation reserve GiB per node")
    ap.add_argument("--port-base", type=int, default=50052)
    ap.add_argument("--force-rpc", action="store_true",
                    help="print RPC commands even on a WAN link (use only inside a "
                         "private tunnel - the protocol is unauthenticated)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--list-machines", action="store_true",
                    help="list the named machines usable as --node values")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.list_machines:
        print("named machines (use as --node <name>, ':ram=' still overrides):\n")
        for mname, (gpus, ram, desc) in sorted(MACHINES.items()):
            vram = sum(GPU_PRESETS[g][0] for g in gpus)
            print(f"  {mname:12} {'+'.join(gpus):16} {vram:3.0f} GiB VRAM  {ram:3.0f} GiB RAM")
            print(f"  {'':12} {desc}")
        return 0

    if not args.node:
        ap.error("at least one --node is required (or --list-machines)")

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

    is_wan = net_rtt >= WAN_RTT_FLOOR
    if is_wan:
        # combined VRAM across sites is not a pool: reaching it costs per-token
        # traffic this link cannot carry. Sizing is per site, below.
        pass
    elif need > vram_total:
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
    elif not is_wan:
        print("verdict : FITS the combined VRAM — fully resident distributed split\n")

    # ---- a WAN link is not a slow LAN: do not shard across it ---------------
    if is_wan:
        print("verdict : SEPARATE SITES — do NOT layer-split across this link.\n")
        print("  1. Security. The RPC protocol is unauthenticated and unencrypted: any host")
        print("     that can reach the port can load a model file and execute on the worker.")
        print("     Over a WAN that is a remote-code-execution surface, not a config choice.")
        print("     No --tensor-split is printed for this link on purpose.\n")
        print("  2. Physics. A layer split pays a full round trip per node boundary per token,")
        print(f"     so this link alone caps decode at ~{net_cap:,.1f} tok/s before any compute:")
        for ni, (gpus, _ram) in enumerate(nodes):
            solo, vram = site_standalone_tok_s(gpus, weights)
            label = f"site {ni} ({'+'.join(gpus)}, {vram:.0f} GiB)"
            if solo <= 0:
                print(f"       {label:38} cannot hold the model alone")
            else:
                verdict = "FASTER alone" if solo > net_cap else "slower alone"
                print(f"       {label:38} ~{solo:5,.1f} tok/s standalone  -> {verdict}")
        print("     Sharding is only worth the link when EVERY site is slower alone than")
        print(f"     ~{net_cap:,.1f} tok/s and none can hold the model - otherwise it is a pure loss.\n")
        print("  3. What to do instead — one full replica per site, no cross-site tensor traffic:")
        for ni, (gpus, ram) in enumerate(nodes):
            solo, _ = site_standalone_tok_s(gpus, weights)
            if solo > 0:
                sel = f"--profile {args.profile}" if args.profile else args.model
                print(f"       site {ni}: python3 scripts/svmi-auto.py {sel} "
                      f"--gpu {gpus[0]} --ctx {args.ctx} --host-ram {ram:.0f}")
                if len(set(gpus)) > 1:
                    # sized above against the FASTEST card, which is the right default:
                    # under a layer split a slower second card buys capacity, not speed
                    print(f"       {'':6}  ^ mixed cards ({'+'.join(gpus)}) - that plans the fastest one."
                          f" For placement across both:")
                    print(f"       {'':8}python3 scripts/svmi-gpucheck.py --model-gib "
                          f"{weights / GiB:.1f}")
            else:
                sel = f"--profile {args.profile}" if args.profile else args.model
                print(f"       site {ni}: does not fit on this site alone — requant or stream: "
                      f"python3 scripts/svmi-auto.py {sel} --gpu {gpus[0]} --ctx {args.ctx}")
        print("     Route users to their nearest site; replicas share nothing at run time, so")
        print("     each one's throughput is its own and a site going down costs capacity, not")
        print("     correctness. Only the GGUF has to reach both, once, by any file transfer.")
        print("     If you genuinely need one logical endpoint, put a proxy in front of the two")
        print("     llama-server instances - that routes whole REQUESTS (RTT paid once), which")
        print("     is what a WAN can carry, unlike per-token activations.\n")
        print("     Needing the two boxes as ONE larger pool of VRAM is the case this cannot")
        print("     serve: that requires per-token traffic, and the link cannot carry it.")
        print("     Put the big model on whichever site holds it, or shrink it to fit.\n")
        if args.force_rpc:
            print("  --force-rpc given: commands below anyway. Tunnel them (WireGuard/SSH)")
            print("  and bind the workers to the tunnel interface, never 0.0.0.0.\n")
        else:
            print("  (--force-rpc prints the RPC commands anyway, for a private tunnel.)")
            return 0

    # ---- commands ----------------------------------------------------------
    print("commands (trusted LAN only — the RPC protocol is unauthenticated!):\n")
    rpc_list = []
    port = args.port_base
    for ni in range(1, len(nodes)):
        gpus, _ = nodes[ni]
        print(f"  # worker{ni} ({'+'.join(gpus)}) — release binaries already include this")
        # on a tunnelled WAN link the worker must bind the tunnel interface only;
        # 0.0.0.0 would expose an unauthenticated port to everything that can route to it
        bind = "<tunnel-ip>" if is_wan else "0.0.0.0"
        print(f"  ggml-rpc-server -H {bind} -p {port} -c")
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
