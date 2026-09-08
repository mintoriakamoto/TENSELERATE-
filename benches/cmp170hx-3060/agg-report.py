#!/usr/bin/env python3
"""Report aggregate throughput from a concurrent burst - and refuse to lie about it.

Reads the JSON responses of N completion requests plus the wall-clock time the
burst took, and prints per-request rates, the aggregate, and two guards that
catch the ways an "aggregate" number goes wrong:

  concurrency factor = sum(per-request elapsed) / wall
      ~N  the requests really overlapped; the aggregate means something
      ~1  they ran one after another; every per-request rate is a SINGLE-STREAM
          rate and there is no aggregate to report

  aggregate <= sum(per-request rates)
      Always true for streams sharing one window. A larger "aggregate" means the
      token count and the elapsed time came from different clocks - the classic
      one is dividing every request's tokens by the fastest request's duration.

Usage: agg-report.py --wall <seconds> response1.json response2.json ...
"""
from __future__ import annotations

import argparse
import json
import sys


def timings(path: str) -> tuple[int, float]:
    """(completion tokens, seconds this request spent generating)."""
    with open(path) as fh:
        d = json.load(fh)
    n = (d.get("usage") or {}).get("completion_tokens")
    t = d.get("timings") or {}
    if n is None:
        n = t.get("predicted_n", 0)
    # predicted_ms is generation only; it is the right clock for a per-request
    # rate and the WRONG one for an aggregate denominator
    ms = t.get("predicted_ms")
    if ms is None:
        ms = 0.0
    return int(n or 0), float(ms) / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall", type=float, required=True,
                    help="wall-clock seconds the whole burst took (measured by the caller)")
    ap.add_argument("--label", default="burst")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    rows = [timings(f) for f in args.files]
    rows = [(n, t) for n, t in rows if n > 0]
    if not rows:
        print("no completions with tokens - nothing to report", file=sys.stderr)
        return 2

    n_req = len(rows)
    total = sum(n for n, _ in rows)
    rates = [n / t if t > 0 else float("nan") for n, t in rows]
    sum_rates = sum(r for r in rates if r == r)
    busy = sum(t for _, t in rows)
    agg = total / args.wall if args.wall > 0 else float("nan")
    conc = busy / args.wall if args.wall > 0 else float("nan")

    print(f"== {args.label}: {n_req} requests, {total} tokens, wall {args.wall:.2f}s")
    for i, (n, t) in enumerate(rows):
        print(f"   req {i}: {n:5d} tok in {t:6.2f}s = {n / t if t > 0 else 0:6.1f} tok/s")
    print(f"   aggregate          : {agg:.1f} tok/s")
    print(f"   sum of per-req rate: {sum_rates:.1f} tok/s   (the ceiling)")
    print(f"   concurrency factor : {conc:.2f}  (~{n_req} = overlapped, ~1 = sequential)")

    rc = 0
    if agg > sum_rates * 1.02:
        print(f"   !! aggregate {agg:.1f} EXCEEDS the sum of per-request rates {sum_rates:.1f}.")
        print("      Impossible for concurrent streams: the token count and the elapsed")
        print("      time came from different clocks. Do not report this number.")
        rc = 1
    if conc < 1.5 and n_req > 1:
        print(f"   !! concurrency factor {conc:.2f}: these requests did NOT overlap.")
        print("      Each rate above is a single-stream rate; there is no aggregate here.")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
