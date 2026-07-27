#!/usr/bin/env python3
"""
svmi-distspec — DIST-SPEC: block speculation as a network-bubble filler.

In a distributed layer split (svmi-net.py), plain autoregressive decode pays
every node-boundary cost (RTT + activation payload) once PER TOKEN, serially:
token t+1 cannot enter the pipeline until token t left it. Block speculation
(the DSpark drafter this repo ships, or any draft model) changes the shape of
the traffic: draft k tokens cheaply on the main host, then push ONE k-token
verify batch through the pipeline — the boundary cost is paid once per ROUND
of ~E[accepted] tokens instead of once per token.

This tool contains both a closed-form model and a discrete-event simulator
(geometric acceptance, per-round sampling) and cross-checks them; --self-test
asserts the two agree and that the speedup is real for LAN-class parameters.

  net gain per boundary  : AR pays (rtt + act/bw) * 1        per token
                           SPEC pays (rtt + k*act/bw) * 1/E[acc] per token
  E[accepted] for accept rate a, block k :  (1 - a^(k+1)) / (1 - a)

Usage:
  python3 scripts/svmi-distspec.py --nodes 2 --nic 1gbe --alpha 0.75 --k 8 \\
      --compute-ms 45 --n-embd 8192
  python3 scripts/svmi-distspec.py --self-test
"""

from __future__ import annotations

import argparse
import random
import sys

NIC_PRESETS = {
    "1gbe":   (0.112, 0.00035),
    "2.5gbe": (0.28,  0.00030),
    "5gbe":   (0.56,  0.00025),
    "10gbe":  (1.12,  0.00020),
    "wifi6":  (0.09,  0.00250),
    "tb4":    (2.50,  0.00010),
}

# marginal GPU cost of adding one token to a verify batch, as a fraction of
# the single-token cost (decode is bandwidth-bound: weights are read once for
# the whole batch, so extra tokens are nearly free until compute saturates)
BATCH_MARGINAL = 0.12


def expected_accepted(alpha: float, k: int) -> float:
    # E[tokens emitted per round] = E[accepted drafts] + 1: every round also
    # emits the verify pass's own next token (the correction on a miss, or
    # the k+1-th token when the whole block is accepted).
    return (1 - alpha ** (k + 1)) / (1 - alpha) if alpha < 1.0 else float(k + 1)


def round_time(compute_s: float, boundaries: int, rtt: float, bw: float,
               act_bytes: int, k: int, draft_s_tok: float) -> float:
    """time for one draft+verify round of a k-token block"""
    batch_compute = compute_s * (1 + (k - 1) * BATCH_MARGINAL)
    net = boundaries * (rtt + k * act_bytes / (bw * 1e9))
    return k * draft_s_tok + batch_compute + net


def ar_tok_s(compute_s: float, boundaries: int, rtt: float, bw: float, act_bytes: int) -> float:
    return 1.0 / (compute_s + boundaries * (rtt + act_bytes / (bw * 1e9)))


def spec_tok_s(alpha: float, k: int, compute_s: float, boundaries: int,
               rtt: float, bw: float, act_bytes: int, draft_s_tok: float) -> float:
    acc = expected_accepted(alpha, k)
    return acc / round_time(compute_s, boundaries, rtt, bw, act_bytes, k, draft_s_tok)


def simulate(alpha: float, k: int, compute_s: float, boundaries: int, rtt: float,
             bw: float, act_bytes: int, draft_s_tok: float,
             n_tokens: int = 20000, seed: int = 42) -> float:
    """discrete-event: geometric acceptance per round, returns tok/s"""
    rng = random.Random(seed)
    produced = 0
    t = 0.0
    while produced < n_tokens:
        acc = 0
        while acc < k and rng.random() < alpha:
            acc += 1
        # accepted drafts + the verify pass's own token (correction / k+1-th)
        produced += acc + 1
        t += round_time(compute_s, boundaries, rtt, bw, act_bytes, k, draft_s_tok)
    return produced / t


def self_test() -> int:
    rtt, bw = NIC_PRESETS["1gbe"][1], NIC_PRESETS["1gbe"][0]
    act = 8192 * 4
    compute = 0.045
    draft = 0.002
    for alpha in (0.6, 0.75, 0.9):
        for k in (4, 8, 16):
            model = spec_tok_s(alpha, k, compute, 1, rtt, bw, act, draft)
            sim = simulate(alpha, k, compute, 1, rtt, bw, act, draft)
            # same accounting on both sides now -> tight agreement
            assert abs(sim - model) <= model * 0.03, (alpha, k, model, sim)
    # the headline claim: speculation beats AR on a 2-node gigabit split
    base = ar_tok_s(compute, 1, rtt, bw, act)
    good = spec_tok_s(0.75, 8, compute, 1, rtt, bw, act, draft)
    assert good > base * 1.5, (base, good)
    # what DIST-SPEC actually amortizes is the RTT component of a boundary --
    # payload crosses the wire either way. So the network must AMPLIFY the
    # speculation win on an RTT-dominated link (wifi6: 2.5 ms RTT) ...
    # (13B-class 15 ms/tok so the boundary is a meaningful share of the token)
    wbw, wrtt = NIC_PRESETS["wifi6"]
    fast = 0.015
    r0 = spec_tok_s(0.75, 8, fast, 0, wrtt, wbw, act, draft) / ar_tok_s(fast, 0, wrtt, wbw, act)
    r2 = spec_tok_s(0.75, 8, fast, 2, wrtt, wbw, act, draft) / ar_tok_s(fast, 2, wrtt, wbw, act)
    assert r2 > r0 * 1.05, (r0, r2)
    # ... while on wired gigabit (payload-dominated boundary) the ratio must
    # stay within a few percent of the standalone win -- no amplification.
    g0 = spec_tok_s(0.75, 8, compute, 0, rtt, bw, act, draft) / ar_tok_s(compute, 0, rtt, bw, act)
    g2 = spec_tok_s(0.75, 8, compute, 2, rtt, bw, act, draft) / ar_tok_s(compute, 2, rtt, bw, act)
    assert abs(g2 - g0) < g0 * 0.05, (g0, g2)
    print("self-test OK: sim matches closed form; spec >1.5x AR on 2-node 1GbE;")
    print(f"              RTT-dominated links amplify the win (wifi6 x{r0:.2f} -> x{r2:.2f} w/ 2 hops),")
    print(f"              payload-dominated wired links do not (1gbe x{g0:.2f} -> x{g2:.2f}).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=2)
    ap.add_argument("--nic", choices=sorted(NIC_PRESETS), default="1gbe")
    ap.add_argument("--alpha", type=float, default=0.75, help="draft acceptance rate")
    ap.add_argument("--k", type=int, default=8, help="draft block size")
    ap.add_argument("--compute-ms", type=float, default=45.0,
                    help="single-token full-pipeline GPU time (all stages summed)")
    ap.add_argument("--draft-ms", type=float, default=2.0, help="draft cost per token (main host)")
    ap.add_argument("--n-embd", type=int, default=8192)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    bw, rtt = NIC_PRESETS[args.nic]
    act = args.n_embd * 4
    boundaries = args.nodes - 1
    compute = args.compute_ms / 1e3
    draft = args.draft_ms / 1e3

    base = ar_tok_s(compute, boundaries, rtt, bw, act)
    print(f"pipeline: {args.nodes} nodes over {args.nic}, n_embd {args.n_embd}, "
          f"compute {args.compute_ms:.0f} ms/tok, draft accept {args.alpha:.2f}")
    print(f"AR decode          : {base:6.2f} tok/s "
          f"(network share {boundaries * (rtt + act / (bw * 1e9)) * base * 100:.1f}% of each token)")
    best = (0.0, 0)
    for k in (2, 4, 8, 12, 16, 24, 32):
        s = spec_tok_s(args.alpha, k, compute, boundaries, rtt, bw, act, draft)
        sim = simulate(args.alpha, k, compute, boundaries, rtt, bw, act, draft, n_tokens=5000)
        tag = ""
        if s > best[0]:
            best = (s, k)
            tag = "  <- best"
        print(f"DIST-SPEC k={k:<3d}    : {s:6.2f} tok/s (sim {sim:6.2f})   x{s / base:.2f}{tag}")
    print(f"\nrun it   : draft with --spec-type draft-dspark (block == {best[1]}) or a small")
    print("           draft model via -md on the svmi-net.py main-host command; the RPC")
    print("           verify batch pays each boundary once per block instead of per token.")
    print("model    : closed form, cross-checked by discrete-event sim (--self-test);")
    print(f"           batch marginal cost {BATCH_MARGINAL:.0%}/token — calibrate on your rig with")
    print("           llama-batched-bench and pass --compute-ms from measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
