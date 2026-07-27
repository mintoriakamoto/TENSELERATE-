#!/usr/bin/env python3
"""
svmi-swarm — RAID-W: parity-coded weight sharding for a LAN inference swarm.

svmi-net.py splits LAYERS across machines; every node is then a single point
of failure — one worker reboots and the whole pipeline is down until its
shard reloads. RAID-W applies RAID-5 thinking to the WEIGHT DISTRIBUTION
tier instead: cut the streamed weight set into N data shards held in N
peers' pinned RAM plus one XOR parity shard on a spare peer. Any single
node dropout is survivable: the missing shard is reconstructed on the fly
as the XOR of the other N-1 data shards + parity, at the cost of reading
N-1 streams instead of 1 (degraded mode) until the peer returns.

This tool is two things:
  1. a WORKING parity codec (numpy XOR) with a --self-test that shards a
     synthetic weight blob, kills a random shard, reconstructs it bit-exact,
     and reports codec throughput measured on this machine;
  2. the capacity/degradation planner: RAM pool after parity overhead,
     serve bandwidth healthy vs degraded, rebuild time per shard.

Engine status (honest): the codec + sizing are real and tested here; wiring
reconstruction into the SVMI staging ring (fetch layer -> if peer dead, XOR
the survivors) is engine phase 6. Today rpc-server -c already gives every
worker a local shard cache, which is the non-redundant baseline.

Usage:
  python3 scripts/svmi-swarm.py --weights-gib 39.6 --peers 8,8,16,32 --nic 2.5gbe
  python3 scripts/svmi-swarm.py --self-test
"""

from __future__ import annotations

import argparse
import sys
import time

NIC_PRESETS = {
    "1gbe": 0.112, "2.5gbe": 0.28, "5gbe": 0.56, "10gbe": 1.12,
    "wifi6": 0.09, "tb4": 2.50,
}
GiB = 1024**3


def make_parity(shards):
    import numpy as np
    parity = np.zeros_like(shards[0])
    for s in shards:
        np.bitwise_xor(parity, s, out=parity)
    return parity


def reconstruct(survivors, parity):
    import numpy as np
    out = parity.copy()
    for s in survivors:
        np.bitwise_xor(out, s, out=out)
    return out


def self_test() -> int:
    import numpy as np
    rng = np.random.default_rng(7)
    n_shards = 4
    shard_bytes = 32 * 1024 * 1024   # 32 MiB shards -> 128 MiB "weights"
    shards = [rng.integers(0, 256, shard_bytes, dtype=np.uint8) for _ in range(n_shards)]

    t0 = time.perf_counter()
    parity = make_parity(shards)
    t_enc = time.perf_counter() - t0

    lost = int(rng.integers(0, n_shards))
    survivors = [s for i, s in enumerate(shards) if i != lost]

    t0 = time.perf_counter()
    rebuilt = reconstruct(survivors, parity)
    t_dec = time.perf_counter() - t0

    assert np.array_equal(rebuilt, shards[lost]), "reconstruction NOT bit-exact"
    # parity must not be a copy of any shard (i.e. XOR actually mixed them)
    assert not any(np.array_equal(parity, s) for s in shards)
    total = n_shards * shard_bytes
    print(f"self-test OK: {n_shards}x{shard_bytes // (1024 * 1024)} MiB shards, "
          f"killed shard {lost}, rebuilt bit-exact")
    print(f"              encode {total / t_enc / 1e9:.1f} GB/s, "
          f"reconstruct {total / t_dec / 1e9:.1f} GB/s on this CPU "
          "(XOR is never the bottleneck; the wire is)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-gib", type=float, default=39.6,
                    help="streamed weight set to distribute (GiB)")
    ap.add_argument("--peers", default="8,8,16,32",
                    help="comma list of donor RAM GiB per peer (excl. the main host)")
    ap.add_argument("--nic", choices=sorted(NIC_PRESETS), default="2.5gbe")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rams = [float(x) for x in args.peers.split(",") if x.strip()]
    n = len(rams)
    if n < 2:
        raise SystemExit("need >= 2 peers for parity to mean anything")
    bw = NIC_PRESETS[args.nic]
    w = args.weights_gib

    # RAID-5 layout: shard size set by the SMALLEST donor (uniform stripes);
    # one shard-equivalent of the pool is parity.
    shard = min(rams) * 0.85                      # 15% headroom per peer
    data_capacity = shard * (n - 1)               # one shard's worth is parity
    fits = w <= data_capacity
    per_peer_share = w / (n - 1)

    print(f"weights : {w:.1f} GiB streamed set, {n} peers (RAM donors: "
          + ", ".join(f'{r:.0f}' for r in rams) + f" GiB) over {args.nic}")
    print(f"layout  : uniform stripes sized by smallest donor -> shard {shard:.1f} GiB,")
    print(f"          capacity {data_capacity:.1f} GiB data + {shard:.1f} GiB parity"
          f"  ->  {'FITS' if fits else 'DOES NOT FIT'}")
    if not fits:
        need = w / (n - 1) / 0.85
        print(f"          smallest donor must reach {need:.1f} GiB (or add peers)")
    print(f"\nserving : healthy   — read 1 stream/layer      : {bw * 1e3:.0f} MB/s per fetch")
    print(f"          degraded  — 1 peer down, XOR of {n - 1}   : "
          f"{bw * 1e3 / (n - 1):.0f} MB/s effective per fetch (streams share the NIC)")
    print(f"          rebuild   — re-host lost {per_peer_share:.1f} GiB shard: "
          f"~{per_peer_share * GiB / (bw * 1e9) / 60:.1f} min background copy")
    print("\nfailure : any SINGLE peer can vanish with zero downtime (degraded rate);")
    print("          two simultaneous losses = reload from disk, same as today.")
    print("\nstatus  : codec + sizing verified here (--self-test); staging-ring fetch")
    print("          fallback (peer dead -> XOR survivors) is engine phase 6. Baseline")
    print("          today: rpc-server -c local caches (no redundancy, full reload).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
