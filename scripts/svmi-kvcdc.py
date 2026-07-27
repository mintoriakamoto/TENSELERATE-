#!/usr/bin/env python3
"""
svmi-kvcdc — KV-CDC: content-defined KV dedup for agent fleets.

Prefix reuse (--cache-reuse / svmi-fleet's shared-prompt dedup) only saves KV
when agents share a common PREFIX. Real fleets share content in the MIDDLE:
the same RAG snippet, tool schema, or few-shot block pasted at different
offsets in every agent's context. Byte-identical middles produce zero prefix
savings.

KV-CDC cuts token streams into content-defined chunks (Gear rolling hash —
boundaries decided by content, so an inserted token shifts ONE chunk, not
all downstream boundaries) and dedups chunks by content hash. A reused chunk
can share one KV copy across agents IF the engine re-rotates its K rows to
the new position — the same per-row RoPE rotation --cache-reuse already
applies to shifted spans, generalized from "one contiguous shift" to "per
chunk" (planner-level today: engine phase 5.5).

--self-test builds a synthetic fleet (unique chatter + shared snippets at
random offsets), and asserts the two properties that make KV-CDC worth it:
  1. CDC finds nearly all cross-agent shared content where prefix dedup
     finds ~none;
  2. boundary stability: prepending tokens to one agent re-chunks only the
     insertion neighborhood (this is what fixed-size blocks fundamentally
     cannot do).

Usage:
  python3 scripts/svmi-kvcdc.py --agents 8 --ctx 8192 --shared-frac 0.35
  python3 scripts/svmi-kvcdc.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys

# 256 random 64-bit gear constants (fixed seed -> reproducible boundaries)
_g = random.Random(0xC0FFEE)
GEAR = [_g.getrandbits(64) for _ in range(256)]
MASK64 = (1 << 64) - 1


def cdc_chunks(tokens: list[int], target: int = 128, min_len: int = 32, max_len: int = 512):
    """content-defined chunking over a token stream (Gear hash).
    boundary when the rolling hash's top bits hit zero at ~1/target rate."""
    mask = (1 << (target.bit_length() - 1)) - 1  # ~1/target boundary probability
    chunks = []
    h = 0
    start = 0
    for i, t in enumerate(tokens):
        h = ((h << 1) + GEAR[t & 0xFF]) & MASK64
        ln = i - start + 1
        if (ln >= min_len and (h & mask) == 0) or ln >= max_len:
            chunks.append(tuple(tokens[start:i + 1]))
            start = i + 1
            h = 0
    if start < len(tokens):
        chunks.append(tuple(tokens[start:]))
    return chunks


def chunk_id(c: tuple) -> bytes:
    return hashlib.blake2b(repr(c).encode(), digest_size=16).digest()


def fleet_savings(streams: list[list[int]], target: int = 128):
    """returns (total_tokens, prefix_dedup_saved, cdc_saved)"""
    total = sum(len(s) for s in streams)
    # prefix dedup: tokens in the longest common prefix shared by >=2 streams,
    # counted once per extra stream sharing it (pairwise vs first stream)
    prefix_saved = 0
    for i in range(1, len(streams)):
        a, b = streams[0], streams[i]
        j = 0
        while j < min(len(a), len(b)) and a[j] == b[j]:
            j += 1
        prefix_saved += j
    # CDC dedup: every repeat occurrence of a chunk id is saved
    seen: set[bytes] = set()
    cdc_saved = 0
    for s in streams:
        for c in cdc_chunks(s, target=target):
            cid = chunk_id(c)
            if cid in seen:
                cdc_saved += len(c)
            else:
                seen.add(cid)
    return total, prefix_saved, cdc_saved


def synth_fleet(n_agents: int, ctx: int, shared_frac: float, seed: int = 1):
    """unique chatter per agent + a pool of shared snippets pasted at random
    offsets (the RAG/tool-schema pattern). vocab kept byte-ish (0..255)."""
    rng = random.Random(seed)
    n_snippets = 6
    snip_len = max(64, int(ctx * shared_frac / n_snippets))
    snippets = [[rng.randrange(256) for _ in range(snip_len)] for _ in range(n_snippets)]
    streams = []
    for _ in range(n_agents):
        s: list[int] = []
        while len(s) < ctx:
            if snippets and rng.random() < shared_frac:
                s.extend(rng.choice(snippets))
            else:
                s.extend(rng.randrange(256) for _ in range(rng.randrange(48, 200)))
        streams.append(s[:ctx])
    return streams


def self_test() -> int:
    streams = synth_fleet(8, 8192, 0.35)
    total, pre, cdc = fleet_savings(streams)
    # 1. CDC must find real cross-agent sharing; prefix must find ~none
    #    (agents start with unique chatter)
    assert pre < total * 0.01, f"prefix unexpectedly high: {pre}/{total}"
    assert cdc > total * 0.10, f"CDC too low: {cdc}/{total}"
    assert cdc > pre * 10
    # 2. boundary stability: shift one stream by prepending 7 tokens; the
    #    chunk set must stay almost identical (only the head region re-cuts)
    s = streams[0]
    before = {chunk_id(c) for c in cdc_chunks(s)}
    shifted = [251, 17, 93, 5, 201, 44, 128] + s
    after = {chunk_id(c) for c in cdc_chunks(shifted)}
    kept = len(before & after) / len(before)
    assert kept > 0.90, f"CDC boundary stability broken: {kept:.2f}"
    # 3. sanity: chunks reassemble to the exact stream (no token lost)
    assert [t for c in cdc_chunks(s) for t in c] == s
    print(f"self-test OK: fleet 8x8192 tok, shared 35% -> prefix dedup {pre / total:.1%}, "
          f"KV-CDC {cdc / total:.1%};")
    print(f"              boundary stability {kept:.1%} of chunks survive a 7-token prepend;")
    print("              chunking is lossless (exact reassembly).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--shared-frac", type=float, default=0.35,
                    help="fraction of each context drawn from the shared snippet pool")
    ap.add_argument("--target-chunk", type=int, default=128, help="target chunk length (tokens)")
    ap.add_argument("--kv-mib-per-ktok", type=float, default=68.0,
                    help="KV MiB per 1k tokens (8B q8_0-class default)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    streams = synth_fleet(args.agents, args.ctx, args.shared_frac)
    total, pre, cdc = fleet_savings(streams, target=args.target_chunk)
    kv_per_tok = args.kv_mib_per_ktok / 1000.0
    print(f"fleet   : {args.agents} agents x {args.ctx:,} tok, ~{args.shared_frac:.0%} shared snippets")
    print(f"          total {total:,} tok = {total * kv_per_tok:,.0f} MiB KV at "
          f"{args.kv_mib_per_ktok:.0f} MiB/ktok")
    print(f"prefix  : {pre:,} tok saved ({pre / total:.1%}) — what --cache-reuse-class prefix dedup sees")
    print(f"KV-CDC  : {cdc:,} tok saved ({cdc / total:.1%}) = {cdc * kv_per_tok:,.0f} MiB "
          f"(target chunk {args.target_chunk})")
    print("\nhow     : Gear-hash content-defined chunks, dedup by content hash; a reused")
    print("          chunk shares one K/V copy and re-rotates K per position (same RoPE")
    print("          shift --cache-reuse applies today, applied per chunk).")
    print("status  : savings + boundary stability validated here (--self-test); the")
    print("          per-chunk RoPE re-rotation fetch is engine phase 5.5 — planner-level")
    print("          today, like CTX-VM paging (svmi-fleet.py --shared-prompt handles the")
    print("          prefix-shaped share NOW).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
